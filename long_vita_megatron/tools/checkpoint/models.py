import abc
import os
import sys
import re
import json
from types import SimpleNamespace
import logging as logger
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from megatron.core import mpu
from megatron.training.arguments import validate_args
from megatron.legacy.model import module
from megatron.core.enums import ModelType
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.checkpointing import load_checkpoint
from megatron.core import tensor_parallel
# from pretrain_gpt import model_provider
# from modellink.utils import parse_args
from modellink.training import model_provider_func_wrapper
from modellink.checkpointing import load_checkpoint_wrapper

from pretrain_long_vita import model_provider, extra_args_provider
from megatron.training.arguments import parse_args

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

model_provider = model_provider_func_wrapper(model_provider)
load_checkpoint = load_checkpoint_wrapper(load_checkpoint)


def tensor_info(tensor):
    shape = tensor.shape
    mean_val = tensor.mean().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    return f"shape: {shape} mean_val: {mean_val} min_val: {min_val} max_val: {max_val}"


class ModelBase(abc.ABC):
    def __init__(self, args_cmd=None):
        self.args_cmd = args_cmd
        self.args = None
        self.args_megatron_checkpoint = None
        self.module = None
        self.module_mapping = None
        self.model_cfg = self.read_model_cfg()
        self.__register_functions()
        self.kwargs_idx = OrderedDict({
            "vp_rank": 0,
            "ep_rank": 0,
            "tp_rank": 0,
            "layer_idx": 0,
            "expert_idx": 0
        })

    def update_kwargs_idx(self, **kwargs):
        for key in self.kwargs_idx:
            if key in kwargs:
                self.kwargs_idx[key] = kwargs[key]
            else:
                self.kwargs_idx[key] = 0

    def __register_functions(self):
        self.get_module_mapping()

        def _get_obj(self, value, **kwargs):
            pattern = r'(\w+)(?:\[(\w+)\])?'
            matches = re.findall(pattern, value)
            self.update_kwargs_idx(**kwargs)
            obj = self.get_model_item(**kwargs)
            for attr, attr_ident in matches:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    return None
                if attr_ident:
                    if attr_ident in self.kwargs_idx:
                        attr_idx = self.kwargs_idx[attr_ident]
                        obj = obj[attr_idx]
                    else:
                        raise AssertionError(f"check {self.__class__.__name__}.module_mapping **{attr_ident}**.")
            return obj

        def _func_generator_get_module(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs)
            return func

        def _func_generator_get_weight(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs).weight.data
            return func

        def _func_generator_get_bias(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs).bias.data
            return func

        def _func_generator_set_weight(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs).weight.data.copy_(kwargs.get('data'))
            return func

        def _func_generator_set_module(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs).data.copy_(kwargs.get('data'))
            return func

        def _func_generator_set_bias(value):
            def func(self, **kwargs):
                return _get_obj(self, value, **kwargs).bias.data.copy_(kwargs.get('data'))
            return func

        def _func_generator_has_module(value):
            def func(self, **kwargs):
                obj = _get_obj(self, value, **kwargs)
                return True if obj else False
            return func
        
        def _func_generator_has_bias(value):
            def func(self, **kwargs):
                bias = getattr(_get_obj(self, value, **kwargs), 'bias', None)
                return bias is not None
            return func

        if self.module_mapping:
            for key, value in self.module_mapping.items():
                setattr(self, "get_" + key + "_module", _func_generator_get_module(value).__get__(self, ModelBase))
                setattr(self, "set_" + key + "_module", _func_generator_set_module(value).__get__(self, ModelBase))
                setattr(self, "get_" + key + "_weight", _func_generator_get_weight(value).__get__(self, ModelBase))
                setattr(self, "get_" + key + "_bias", _func_generator_get_bias(value).__get__(self, ModelBase))
                setattr(self, "set_" + key + "_weight", _func_generator_set_weight(value).__get__(self, ModelBase))
                setattr(self, "set_" + key + "_bias", _func_generator_set_bias(value).__get__(self, ModelBase))
                setattr(self, "has_" + key + "_module", _func_generator_has_module(value).__get__(self, ModelBase))
                setattr(self, "has_" + key + "_bias", _func_generator_has_bias(value).__get__(self, ModelBase))

    def update_module(self, src_model):
        self.set_preprocess_state(src_model)
        self.set_postprocess_state(src_model)
        for layer_idx in tqdm(range(self.args.num_layers), "set layer states"):
            self.set_layer_state(src_model, layer_idx)

    def set_preprocess_state(self, src_model):
        '''Set embedding params.'''
        embeddings_weight = src_model.get_embedding_word_embeddings_weight()
        self.set_embedding_word_embeddings_weight(data=embeddings_weight)
        if src_model.has_embedding_word_embeddings_norm_module():
            embd_norm_weight = src_model.get_embedding_word_embeddings_norm_weight()
            embd_norm_bias = src_model.get_embedding_word_embeddings_norm_bias()
            self.set_embedding_word_embeddings_norm_weight(data=embd_norm_weight)
            self.set_embedding_word_embeddings_norm_bias(data=embd_norm_bias)

    def set_postprocess_state(self, src_model):
        final_layernorm_weight = src_model.get_final_layernorm_weight()
        self.set_final_layernorm_weight(data=final_layernorm_weight)
        if self.args.untie_embeddings_and_output_weights:
            output_layer_weight = src_model.get_output_layer_weight()
            self.set_output_layer_weight(data=output_layer_weight)
        if self.has_final_layernorm_bias():
            final_layernorm_bias = src_model.get_final_layernorm_bias()
            self.set_final_layernorm_bias(data=final_layernorm_bias)

    def set_layer_state(self, src_model, layer_idx):
        self.set_attn_state(layer_idx, src_model)
        self.set_mlp_state(layer_idx, src_model)
        input_layernorm_weight = src_model.get_layers_input_layernorm_weight(layer_idx=layer_idx)
        self.set_layers_input_layernorm_weight(layer_idx=layer_idx, data=input_layernorm_weight)
        if self.args.post_norm:
            post_attn_layernorm_weight = src_model.get_layers_self_attention_post_attention_layernorm_weight(
                layer_idx=layer_idx)
            self.set_layers_self_attention_post_attention_layernorm_weight(layer_idx=layer_idx,
                                                                           data=post_attn_layernorm_weight)
        else:
            pre_mlp_layernorm_weight = src_model.get_layers_self_attention_pre_mlp_layernorm_weight(layer_idx=layer_idx)
            self.set_layers_self_attention_pre_mlp_layernorm_weight(layer_idx=layer_idx, data=pre_mlp_layernorm_weight)

        if self.has_layers_input_layernorm_bias(layer_idx=layer_idx):
            input_layernorm_bias = src_model.get_layers_input_layernorm_bias(layer_idx=layer_idx)
            self.set_layers_input_layernorm_bias(layer_idx=layer_idx, data=input_layernorm_bias)
        if self.has_layers_self_attention_pre_mlp_layernorm_bias(layer_idx=layer_idx):
            pre_mlp_layernorm_bias = src_model.get_layers_self_attention_pre_mlp_layernorm_bias(layer_idx=layer_idx)
            self.set_layers_self_attention_pre_mlp_layernorm_bias(layer_idx=layer_idx, data=pre_mlp_layernorm_bias)

    def set_attn_state(self, layer_idx, src_model):
        '''Set self-attention params.'''
        # Get attention layer & state.
        if getattr(src_model.get_args(), "qk_layernorm", False):
            q_layernorm = src_model.get_layers_self_attention_q_layernorm_weight(layer_idx=layer_idx)
            k_layernorm = src_model.get_layers_self_attention_k_layernorm_weight(layer_idx=layer_idx)
            self.set_layers_self_attention_q_layernorm_weight(layer_idx=layer_idx, data=q_layernorm)
            self.set_layers_self_attention_k_layernorm_weight(layer_idx=layer_idx, data=k_layernorm)
        
        if getattr(src_model.get_args(), "multi_head_latent_attention", False):
            linear_qb = src_model.get_layers_self_attention_linear_qb_weight(layer_idx=layer_idx)
            linear_kvb = src_model.get_layers_self_attention_linear_kvb_weight(layer_idx=layer_idx)
            self.set_layers_self_attention_linear_qb_weight(layer_idx=layer_idx, data=linear_qb)
            self.set_layers_self_attention_linear_kvb_weight(layer_idx=layer_idx, data=linear_kvb)
        
        qkv_weight = src_model.get_layers_self_attention_linear_qkv_weight(layer_idx=layer_idx)
        proj_weight = src_model.get_layers_self_attention_linear_proj_weight(layer_idx=layer_idx)
        self.set_layers_self_attention_linear_qkv_weight(layer_idx=layer_idx, data=qkv_weight)
        self.set_layers_self_attention_linear_proj_weight(layer_idx=layer_idx, data=proj_weight)
        if self.args.add_qkv_bias:
            qkv_bias = src_model.get_layers_self_attention_linear_qkv_bias(layer_idx=layer_idx)
            self.set_layers_self_attention_linear_qkv_bias(layer_idx=layer_idx, data=qkv_bias)
        if self.args.add_dense_bias:
            proj_bias = src_model.get_layers_self_attention_linear_proj_bias(layer_idx=layer_idx)
            self.set_layers_self_attention_linear_proj_bias(layer_idx=layer_idx, data=proj_bias)

    def _set_mlp_state(self, src_model, **kwargs):
        '''Set MLP params.'''
        fc1_weight = src_model.get_layers_mlp_linear_fc1_weight(**kwargs)
        fc2_weight = src_model.get_layers_mlp_linear_fc2_weight(**kwargs)
        self.set_layers_mlp_linear_fc1_weight(data=fc1_weight, **kwargs)
        self.set_layers_mlp_linear_fc2_weight(data=fc2_weight, **kwargs)
        if src_model.has_layers_mlp_linear_fc1_bias(**kwargs):
            fc1_bias = src_model.get_layers_mlp_linear_fc1_bias(**kwargs)
            self.set_layers_mlp_linear_fc1_bias(data=fc1_bias, **kwargs)
        if src_model.has_layers_mlp_linear_fc2_bias(**kwargs):
            fc2_bias = src_model.get_layers_mlp_linear_fc2_bias(**kwargs)
            self.set_layers_mlp_linear_fc2_bias(data=fc2_bias, **kwargs)
    
    def _set_mlp_experts_state(self, src_model, **kwargs):
        '''Set MLP experts params.'''
        fc1_weight = src_model.get_layers_mlp_experts_linear_fc1_weight(**kwargs)
        fc2_weight = src_model.get_layers_mlp_experts_linear_fc2_weight(**kwargs)
        self.set_layers_mlp_experts_linear_fc1_weight(data=fc1_weight, **kwargs)
        self.set_layers_mlp_experts_linear_fc2_weight(data=fc2_weight, **kwargs)

    def _set_mlp_shared_experts_state(self, src_model, **kwargs):
        '''Set MLP shared experts params.'''
        fc1_weight = src_model.get_layers_mlp_shared_experts_linear_fc1_weight(**kwargs)
        fc2_weight = src_model.get_layers_mlp_shared_experts_linear_fc2_weight(**kwargs)
        self.set_layers_mlp_shared_experts_linear_fc1_weight(data=fc1_weight, **kwargs)
        self.set_layers_mlp_shared_experts_linear_fc2_weight(data=fc2_weight, **kwargs)

    def _set_moe_grouped_gemm_state(self, src_model, **kwargs):
        '''Set MOE grouped gemm params.'''
        weight1 = src_model.get_layers_mlp_experts_weight1_module(**kwargs)
        weight2 = src_model.get_layers_mlp_experts_weight2_module(**kwargs)
        self.set_layers_mlp_experts_weight1_module(data=weight1, **kwargs)
        self.set_layers_mlp_experts_weight2_module(data=weight2, **kwargs)

        if self.args.post_norm:
            pre_mlp_layernorm_weight = src_model.get_layers_self_attention_pre_mlp_layernorm_weight(**kwargs)
            post_mlp_layernorm_weight = src_model.get_layers_self_attention_post_mlp_layernorm_weight(**kwargs)
            self.set_layers_self_attention_pre_mlp_layernorm_weight(data=pre_mlp_layernorm_weight, **kwargs)
            self.set_layers_self_attention_post_mlp_layernorm_weight(data=post_mlp_layernorm_weight, **kwargs)

    def set_mlp_state(self, layer_idx, src_model):
        args = src_model.get_args()
        kwargs = {'layer_idx': layer_idx}
        num_experts = getattr(args, 'num_experts', None) or getattr(args, 'num_local_experts', None)
        first_k_dense_replace = getattr(args, 'first_k_dense_replace', None)
        moe_layer_freq = getattr(args, 'moe_layer_freq', None)
        if (num_experts
                and first_k_dense_replace is not None
                and moe_layer_freq is not None
        ):
            if layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0:
                router_weight = src_model.get_layers_mlp_router_weight(**kwargs)
                self.set_layers_mlp_router_weight(**kwargs, data=router_weight)
                if getattr(self.args, "n_shared_experts", None) is not None:
                    self._set_mlp_shared_experts_state(src_model, **kwargs)
                if args.moe_grouped_gemm:
                    self._set_moe_grouped_gemm_state(src_model, **kwargs)
                else:
                    for expert_idx in range(num_experts):
                        kwargs['expert_idx'] = expert_idx
                        self._set_mlp_experts_state(src_model, **kwargs)
            else:
                self._set_mlp_state(src_model, **kwargs)

        elif num_experts:
            router_weight = src_model.get_layers_mlp_router_weight(**kwargs)
            self.set_layers_mlp_router_weight(**kwargs, data=router_weight)
            for expert_idx in range(num_experts):
                kwargs['expert_idx'] = expert_idx
                self._set_mlp_state(src_model, **kwargs)
        else:
            self._set_mlp_state(src_model, **kwargs)


    def get_args(self):
        return self.args

    def get_args_cmd(self):
        return self.args_cmd

    def get_metadata(self):
        return self.md

    def get_modules_count(self):
        return len(self.module)

    @staticmethod
    def read_model_cfg():
        def merge_configs(base_config, specific_config):
            merged_config = base_config.copy()
            for key, value in specific_config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key] = merge_configs(merged_config[key], value)
                else:
                    merged_config[key] = value
            return merged_config

        current_directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_directory, 'model_cfg.json'), 'r') as file:
            config = json.load(file)
        final_configs = {}

        for model_name, model_config in config["model_mappings"].items():
            if "__base__" in model_config:
                base_model_name = model_config["__base__"]
                base_config = config["model_mappings"][base_model_name]
                specific_config = model_config.copy()
                specific_config.pop("__base__", None)
                final_config = merge_configs(base_config, specific_config)
            else:
                final_config = model_config
            final_configs[model_name] = final_config

        return final_configs

    @abc.abstractmethod
    def get_module_mapping(self):
        pass

    @abc.abstractmethod
    def get_model_item(self, **kwargs):
        pass


class HuggingfaceModel(ModelBase):
    def __init__(self, args_cmd):
        super(HuggingfaceModel, self).__init__(args_cmd)
        self.initialize_args()
        self.layers_self_attention_linear_qkv_caches = {"layer_idx": -1, "weight": None, "bias": None}

    def initialize_args(self):
        # Read huggingface args.
        if self.args_cmd.save_model_type == 'huggingface':
            cfg_dir = self.args_cmd.save_dir
        else:
            cfg_dir = self.args_cmd.load_dir
        llama_args_path = os.path.join(cfg_dir, "config.json")
        with open(llama_args_path) as f:
            self.args = json.load(f)

        config_key_mapping = self.model_cfg.get(self.args_cmd.model_type_hf).get('config_hf_key_mapping')
        config_value = self.model_cfg.get(self.args_cmd.model_type_hf).get('config_set_value')
        for key_target in config_key_mapping:
            key_hf = config_key_mapping[key_target]
            if self.args.get(key_hf, None) is not None:
                self.args[key_target] = self.args[key_hf]
            else:
                logger.warning(f"{key_target} was not found in the config file.")
        for key_target in config_value:
            self.args[key_target] = config_value[key_target]

        if (
                "num_key_value_heads" in self.args and
                self.args["num_attention_heads"] != self.args["num_key_value_heads"]
        ):
            if self.args["num_attention_heads"] == 1:
                raise AssertionError("Number of attention heads should be greater than 1!")
            self.args['group_query_attention'] = True

        self.args['untie_embeddings_and_output_weights'] = not self.args.get("tie_word_embeddings", False)
        self.args = SimpleNamespace(**self.args)
        self.args.add_qkv_bias = self.args_cmd.add_qkv_bias
        self.args.add_dense_bias = self.args_cmd.add_dense_bias
        self.args.post_norm = self.args_cmd.post_norm

    def get_modules_from_pretrained(self, device_map="cpu", trust_remote_code=True):
        # Load Huggingface model.
        if self.args_cmd.save_model_type == "huggingface":
            load_dir = self.args_cmd.save_dir
        else:
            load_dir = self.args_cmd.load_dir
        self.module = [AutoModelForCausalLM.from_pretrained(load_dir, device_map=device_map, trust_remote_code=trust_remote_code)]
        if self.args.torch_dtype in ["float16", "bfloat16"]:
            self.module[0] = self.module[0].to(eval(f'torch.{self.args.torch_dtype}'))

    def get_module_mapping(self):
        self.module_mapping = self.model_cfg.get(self.args_cmd.model_type_hf).get('model_hf_key_mapping')

    def __get_layers_self_attention_linear_qkv_module(self, layer_idx=0):
        if self.layers_self_attention_linear_qkv_caches["layer_idx"] == layer_idx:
            return
        self.layers_self_attention_linear_qkv_caches["layer_idx"] = layer_idx
        # Reshape loaded weights.
        nh = self.args.num_attention_heads
        ng = (self.args.num_key_value_heads if self.args.group_query_attention else self.args.num_attention_heads)
        dim = self.args.kv_channels if hasattr(self.args, "kv_channels") else self.args.hidden_size // self.args.num_attention_heads
        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")

        def qkv_concatenate_weight(qkv):
            return torch.cat([
                qkv[0].reshape((ng, dim * nh // ng, -1)),
                qkv[1].reshape((ng, dim, -1)),
                qkv[2].reshape((ng, dim, -1)),
            ], dim=1).reshape((-1, self.args.hidden_size))

        def qkv_concatenate_bias(qkv):
            return torch.cat([
                qkv[0].reshape((ng, dim * nh // ng)),
                qkv[1].reshape((ng, dim)),
                qkv[2].reshape((ng, dim)),
            ], dim=1).reshape((-1))

        qkv_type = self.args.qkv_type
        if qkv_type == "unpack":
            q_proj = self.get_layers_self_attention_linear_q_proj_module(layer_idx=layer_idx)
            k_proj = self.get_layers_self_attention_linear_k_proj_module(layer_idx=layer_idx)
            v_proj = self.get_layers_self_attention_linear_v_proj_module(layer_idx=layer_idx)
            query_key_value_weight = [q_proj.weight, k_proj.weight, v_proj.weight]
            query_key_value_bias = [q_proj.bias, k_proj.bias, v_proj.bias]
            self.layers_self_attention_linear_qkv_caches["weight"] = (qkv_concatenate_weight(query_key_value_weight))
            if self.args_cmd.add_qkv_bias:
                self.layers_self_attention_linear_qkv_caches["bias"] = (qkv_concatenate_bias(query_key_value_bias))
        elif qkv_type == "pack_mla":
            q_proj = self.get_layers_self_attention_linear_q_proj_module(layer_idx=layer_idx)
            kv_proj = self.get_layers_self_attention_linear_kv_proj_module(layer_idx=layer_idx)
            query_key_value_weight = [q_proj.weight.reshape((-1, self.args.hidden_size)),
                                      kv_proj.weight.reshape((-1, self.args.hidden_size))]
            self.layers_self_attention_linear_qkv_caches["weight"] = (torch.cat(query_key_value_weight, dim=0))
            if self.args_cmd.add_qkv_bias:
                query_key_value_bias = [q_proj.bias, kv_proj.bias]
                self.layers_self_attention_linear_qkv_caches["bias"] = (qkv_concatenate_bias(query_key_value_bias))
        elif qkv_type == "pack_gqa":
            qkv_pack = self.get_layers_self_attention_linear_qkv_pack_module(layer_idx=layer_idx)
            qkv_pack_weight = qkv_pack.weight
            full_q = dim * nh
            end_k = full_q + ng * dim
            q_weight = qkv_pack_weight[:full_q, :]
            k_weight = qkv_pack_weight[full_q:end_k, :]
            v_weight = qkv_pack_weight[end_k:, :]
            query_key_value_weight = [q_weight, k_weight, v_weight]
            self.layers_self_attention_linear_qkv_caches["weight"] = (qkv_concatenate_weight(query_key_value_weight))
            if self.args_cmd.add_qkv_bias:
                qkv_pack_bias = qkv_pack.bias
                q_bias = qkv_pack_bias[:full_q]
                k_bias = qkv_pack_bias[full_q:end_k]
                v_bias = qkv_pack_bias[end_k:]
                query_key_value_bias = [q_bias, k_bias, v_bias]
                self.layers_self_attention_linear_qkv_caches["bias"] = (qkv_concatenate_bias(query_key_value_bias))
        elif qkv_type == "pack_self":
            qkv_pack = self.get_layers_self_attention_linear_qkv_pack_module(layer_idx=layer_idx)
            qkv_pack_weight = qkv_pack.weight
            self.layers_self_attention_linear_qkv_caches["weight"] = qkv_pack_weight
            if self.args_cmd.add_qkv_bias:
                qkv_pack_bias = qkv_pack.bias
                full_q = dim * nh
                end_k = full_q + ng * dim
                q_bias = qkv_pack_bias[:full_q, :]
                k_bias = qkv_pack_bias[full_q:end_k, :]
                v_bias = qkv_pack_bias[end_k:, :]
                query_key_value_bias = [q_bias, k_bias, v_bias]
                self.layers_self_attention_linear_qkv_caches["bias"] = (qkv_concatenate_bias(query_key_value_bias))        
        else:
            raise ValueError(f"Unsupported types. {qkv_type}")
            
    def has_layers_mlp_linear_fc1_bias(self, **kwargs):
        return False

    def get_layers_mlp_linear_fc1_weight(self, **kwargs):
        fc_type = self.args.fc_type
        if fc_type == "h_to_4h":
            return self.get_layers_mlp_linear_fc1_module(**kwargs).weight
        elif fc_type == "gate_up_down":
            gate_proj = self.get_layers_mlp_gate_proj_weight(**kwargs)
            up_proj = self.get_layers_mlp_up_proj_weight(**kwargs)
            return torch.cat([gate_proj, up_proj], dim=0)
        else:
            raise ValueError(f"Unsupported fc_type {fc_type}")

    def get_layers_self_attention_linear_qkv_weight(self, layer_idx):
        self.__get_layers_self_attention_linear_qkv_module(layer_idx=layer_idx)
        return self.layers_self_attention_linear_qkv_caches["weight"]

    def get_layers_self_attention_linear_qkv_bias(self, layer_idx):
        self.__get_layers_self_attention_linear_qkv_module(layer_idx=layer_idx)
        return self.layers_self_attention_linear_qkv_caches["bias"]

    def set_layers_mlp_linear_fc1_weight(self, data=None, **kwargs):
        gate_proj, up_proj = torch.chunk(data, 2, dim=0)
        self.set_layers_mlp_gate_proj_weight(data=gate_proj, **kwargs)
        self.set_layers_mlp_up_proj_weight(data=up_proj, **kwargs)

    def set_layers_mlp_experts_linear_fc1_weight(self, data=None, **kwargs):
        gate_proj, up_proj = torch.chunk(data, 2, dim=0)
        self.set_layers_mlp_experts_gate_proj_weight(data=gate_proj, **kwargs)
        self.set_layers_mlp_experts_up_proj_weight(data=up_proj, **kwargs)

    def set_layers_mlp_shared_experts_linear_fc1_weight(self, data=None, **kwargs):
        gate_proj, up_proj = torch.chunk(data, 2, dim=0)
        self.set_layers_mlp_shared_experts_gate_proj_weight(data=gate_proj, **kwargs)
        self.set_layers_mlp_shared_experts_up_proj_weight(data=up_proj, **kwargs)

    def set_layers_mlp_experts_weight1_module(self, data=None, **kwargs):
        args = self.get_args()
        num_experts = getattr(args, 'num_experts', None) or getattr(args, 'num_local_experts', None)
        experts_linear_fc1_list = torch.chunk(data.view(-1), num_experts)
        for expert_idx in range(num_experts):
            kwargs['expert_idx'] = expert_idx
            fc1_weight = experts_linear_fc1_list[expert_idx].view(args.hidden_size, -1).t()
            self.set_layers_mlp_experts_linear_fc1_weight(data=fc1_weight, **kwargs)

    def set_layers_mlp_experts_weight2_module(self, data=None, **kwargs):
        args = self.get_args()
        num_experts = getattr(args, 'num_experts', None) or getattr(args, 'num_local_experts', None)
        experts_linear_fc2_list = torch.chunk(data.view(-1), num_experts)
        for expert_idx in range(num_experts):
            kwargs['expert_idx'] = expert_idx
            fc2_weight = experts_linear_fc2_list[expert_idx].view(-1, args.hidden_size).t()
            self.set_layers_mlp_experts_linear_fc2_weight(data=fc2_weight, **kwargs)

    def get_layers_mlp_experts_linear_fc1_weight(self, **kwargs):
        fc_type = self.args.fc_type
        if fc_type == "h_to_4h":
            return self.get_layers_mlp_experts_linear_fc1_module(**kwargs).weight
        elif fc_type == "gate_up_down":
            gate_proj = self.get_layers_mlp_experts_gate_proj_weight(**kwargs)
            up_proj = self.get_layers_mlp_experts_up_proj_weight(**kwargs)
            return torch.cat([gate_proj, up_proj], dim=0)
        else:
            raise ValueError(f"Unsupported fc_type {fc_type}")

    def get_layers_mlp_shared_experts_linear_fc1_weight(self, **kwargs):
        fc_type = self.args.fc_type
        if fc_type == "h_to_4h":
            return self.get_layers_mlp_experts_linear_fc1_module(**kwargs).weight
        elif fc_type == "gate_up_down":
            gate_proj = self.get_layers_mlp_shared_experts_gate_proj_weight(**kwargs)
            up_proj = self.get_layers_mlp_shared_experts_up_proj_weight(**kwargs)
            return torch.cat([gate_proj, up_proj], dim=0)
        else:
            raise ValueError(f"Unsupported fc_type {fc_type}")

    def get_layers_mlp_experts_weight1_module(self, **kwargs):
        args = self.get_args()
        num_experts = getattr(args, 'num_experts', None) or getattr(args, 'num_local_experts', None)
        experts_linear_fc1_list = []
        for expert_idx in range(num_experts):
            kwargs['expert_idx'] = expert_idx
            fc1_weight = self.get_layers_mlp_experts_linear_fc1_weight(**kwargs)
            experts_linear_fc1_list.append(fc1_weight.t().view(-1))
        return torch.cat(experts_linear_fc1_list).view(args.hidden_size, -1)

    def get_layers_mlp_experts_weight2_module(self, **kwargs):
        args = self.get_args()
        num_experts = getattr(args, 'num_experts', None) or getattr(args, 'num_local_experts', None)
        experts_linear_fc2_list = []
        for expert_idx in range(num_experts):
            kwargs['expert_idx'] = expert_idx
            fc2_weight = self.get_layers_mlp_experts_linear_fc2_weight(**kwargs)
            experts_linear_fc2_list.append(fc2_weight.t().view(-1))
        return torch.cat(experts_linear_fc2_list).view(-1, args.hidden_size)

    def set_layers_self_attention_linear_qkv_weight(self, layer_idx=0, data=None):
        def qkv_split_weight(query_key_value):
            qkv_weight = query_key_value.reshape(
                ng,
                repeats + 2,
                query_key_value.shape[0] // ng // (repeats + 2),
                query_key_value.shape[1],
            )
            hidden_size = qkv_weight.shape[-1]
            qw = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
            kw = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
            vw = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)
            return qw, kw, vw

        nh = self.args.num_attention_heads
        ng = (self.args.num_key_value_heads if self.args.group_query_attention else self.args.num_attention_heads)
        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")
        repeats = nh // ng

        qkv_type = self.args.qkv_type
        if qkv_type == "unpack":
            q_weight, k_weight, v_weight = qkv_split_weight(data)
            self.set_layers_self_attention_linear_q_proj_weight(layer_idx=layer_idx, data=q_weight)
            self.set_layers_self_attention_linear_k_proj_weight(layer_idx=layer_idx, data=k_weight)
            self.set_layers_self_attention_linear_v_proj_weight(layer_idx=layer_idx, data=v_weight)
        elif qkv_type == "pack_gqa":
            qw, k_weight, v_weight = qkv_split_weight(data)
            qkv = torch.cat((qw, k_weight, v_weight), dim=0)
            self.set_layers_self_attention_linear_qkv_pack_weight(layer_idx=layer_idx, data=qkv)
        elif qkv_type == "pack_mla":
            if self.args.q_lora_rank is None:
                q_proj = data[:self.args.num_attention_heads * self.args.q_head_dim, :]
                kv_proj = data[self.args.num_attention_heads * self.args.q_head_dim:, :]
            else:
                q_proj = data[:self.args.q_lora_rank, :]
                kv_proj = data[self.args.q_lora_rank:, :]
            self.set_layers_self_attention_linear_q_proj_weight(layer_idx=layer_idx, data=q_proj)
            self.set_layers_self_attention_linear_kv_proj_weight(layer_idx=layer_idx, data=kv_proj)
        else:
            raise ValueError(f"Unsupported types. {qkv_type}")

    def set_layers_self_attention_linear_qkv_bias(self, layer_idx, data=None):
        def qkv_split_bias(query_key_value):
            bias_weight = query_key_value.reshape(
                ng, repeats + 2, query_key_value.shape[0] // ng // (repeats + 2)
            )
            qw = bias_weight[:, :repeats, ...].reshape(-1)
            kw = bias_weight[:, repeats: repeats + 1, ...].reshape(-1)
            vw = bias_weight[:, repeats + 1:, ...].reshape(-1)
            return qw, kw, vw

        nh = self.args.num_attention_heads
        ng = (self.args.num_key_value_heads if self.args.group_query_attention else self.args.num_attention_heads)
        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")
        repeats = nh // ng

        qkv_type = self.args.qkv_type
        if qkv_type == "unpack":
            if self.args_cmd.add_qkv_bias:
                q_bias, k_bias, v_bias = qkv_split_bias(data)
                self.set_layers_self_attention_linear_q_proj_bias(layer_idx=layer_idx, data=q_bias)
                self.set_layers_self_attention_linear_k_proj_bias(layer_idx=layer_idx, data=k_bias)
                self.set_layers_self_attention_linear_v_proj_bias(layer_idx=layer_idx, data=v_bias)
        elif qkv_type == "pack_gqa":
            if self.args_cmd.add_qkv_bias:
                q_bias, k_bias, v_bias = qkv_split_bias(data)
                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                self.set_layers_self_attention_linear_qkv_pack_bias(layer_idx=layer_idx, data=qkv_bias)
        else:
            raise ValueError(f"Unsupported types. {qkv_type}")

    def get_model_item(self, **kwargs):
        return self.module[0]


class MegatronModel(ModelBase):
    def __init__(self, args_cmd, md=None):
        super(MegatronModel, self).__init__(args_cmd)
        self.md = md
        self.pp_stage_cache = []

    def initialize_megatron_args(self, hf_args=None, queue=None, loader_megatron=False, saver_megatron=False):
        sys.argv = self.get_sys_argv()
        self.args = parse_args(extra_args_provider)
        # # self.args.kv_channels = 128
        self.args.vision_seq_length = 576
        self.args.rotary_base = 1e6
        self.args.vision_projector_type = "mlp"
        self.args.vision_model_type = "eva"
        self.args.image_size = 336

        self.update_megatron_args_from_megatron_checkpoint(loader_megatron)
        self.update_megatron_args_from_cmd_config(loader_megatron)
        self.update_megatron_args_from_huggingface_config(hf_args)

        # Arguments do sanity checks on the world size, but we don't care,
        # so trick it into thinking we are plenty of processes.
        self.args.world_size = self.args.tensor_model_parallel_size * self.args.pipeline_model_parallel_size
        self.update_megatron_args_from_loader_margs()
        self.args = validate_args(self.args)
        self.check_for_args(queue, saver_megatron)

        self.args.model_type = ModelType.encoder_or_decoder
        # Suppress warning about torch.distributed not being initialized.
        module.MegatronModule.embedding_warning_printed = True
        set_args(self.args)
        self.set_megatron_parallel_state(saver_megatron)

    def update_megatron_args_from_loader_margs(self):
        if self.md and hasattr(self.md, 'checkpoint_args'):
            # These are arguments that we are either changing, or cause problems for validation if they are set
            args_to_keep = [
                'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                'bias_gelu_fusion', 'bias_dropout_fusion', 'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng', 'vocab_file', 'tokenizer_model',
                'save_interval', 'save', 'perform_initialization', 'use_cpu_initialization', 'recompute_granularity',
                'recompute_num_layers', 'recompute_method', 'encoder_num_layers', 'encoder_seq_length',
                'distribute_saved_activations', 'train_iters', 'lr_decay_iters', 'lr_warmup_iters',
                'lr_warmup_fraction', 'start_weight_decay', 'end_weight_decay', 'make_vocab_size_divisible_by',
                'masked_softmax_fusion', 'num_layer_list', 'lora_target_modules', 'expert_model_parallel_size'
            ]

            for arg, value in vars(self.md.checkpoint_args).items():
                if arg in args_to_keep:
                    continue
                if not hasattr(self.args, arg):
                    logger.warning(f"Checkpoint had argument {arg} but new arguments does not have this.")
                    continue
                if getattr(self.args, arg) != value:
                    logger.warning(
                        f"Overwriting default {arg} value {getattr(self.args, arg)} with value from checkpoint {value}."
                    )
                    setattr(self.args, arg, value)

            if hasattr(self.md, 'consumed_train_samples'):
                self.args.consumed_train_samples = self.md.consumed_train_samples
                self.args.consumed_valid_samples = self.md.consumed_valid_samples
                logger.info(f"Setting consumed_train_samples to {self.args.consumed_train_samples} "
                            f"and consumed_valid_samples to {self.args.consumed_valid_samples}")
            else:
                logger.warning("consumed_train_samples not provided.")

    def update_megatron_args_from_huggingface_config(self, hf_args):
        if hf_args is None:
            return
        self.args.seq_length = hf_args.max_position_embeddings
        self.args.max_position_embeddings = hf_args.max_position_embeddings
        self.args.hidden_size = hf_args.hidden_size
        self.args.num_attention_heads = hf_args.num_attention_heads
        self.args.num_layers = hf_args.num_layers
        self.args.global_batch_size = 1024
        self.args.norm_epsilon = hf_args.norm_epsilon
        self.args.iteration = 1  # '0', 'release' don't work
        self.args.add_position_embedding = hf_args.add_position_embedding
        self.args.use_rotary_position_embeddings = hf_args.use_rotary_position_embeddings
        self.args.swiglu = hf_args.swiglu
        self.args.tokenizer_type = hf_args.tokenizer_type
        self.args.normalization = hf_args.normalization
        self.args.add_bias_linear = hf_args.add_bias_linear
        self.args.untie_embeddings_and_output_weights = not hf_args.tie_word_embeddings
        self.args.vocab_size = hf_args.vocab_size
        self.args.padded_vocab_size = hf_args.vocab_size
        self.args.llama = hf_args
        self.args.ffn_hidden_size = hf_args.intermediate_size
        self.args.gradient_accumulation_fusion = hf_args.gradient_accumulation_fusion
        self.args.kv_channels = hf_args.kv_channels if hasattr(hf_args, "kv_channels") else None
        self.args.moe_grouped_gemm = hf_args.moe_grouped_gemm
        self.args.num_experts = getattr(hf_args, "num_experts", None)
        self.args.n_shared_experts = getattr(hf_args, "n_shared_experts", None)
        self.args.qk_layernorm = getattr(hf_args, "qk_layernorm", False)
        self.args.moe_intermediate_size = getattr(hf_args, "moe_intermediate_size", None)
        self.args.first_k_dense_replace = getattr(hf_args, "first_k_dense_replace", None)
        self.args.moe_layer_freq = getattr(hf_args, "moe_layer_freq", None)
        self.args.multi_head_latent_attention = getattr(hf_args, "multi_head_latent_attention", False)
        if self.args.multi_head_latent_attention:
            self.args.qk_rope_head_dim = getattr(hf_args, "qk_rope_head_dim", None)
            self.args.qk_nope_head_dim = getattr(hf_args, "qk_nope_head_dim", None)
            self.args.q_lora_rank = getattr(hf_args, "q_lora_rank", None)
            self.args.kv_lora_rank = getattr(hf_args, "kv_lora_rank", None)
            self.args.v_head_dim = getattr(hf_args, "v_head_dim", None)

        if self.args.add_dense_bias:
            self.args.skip_bias_add = False

        if (
                hasattr(hf_args, "num_key_value_heads") and
                hf_args.num_attention_heads != hf_args.num_key_value_heads
        ):
            if hf_args.num_attention_heads == 1:
                raise AssertionError("Number of attention heads should be greater than 1!")
            self.args.group_query_attention = True
            self.args.num_query_groups = hf_args.num_key_value_heads
        if hasattr(hf_args, 'num_local_experts'):
            self.args.num_experts = hf_args.num_local_experts

    def update_megatron_args_from_megatron_checkpoint(self, loader_megatron):
        if not loader_megatron:
            return
        set_args(self.args)
        self.args, self.args_megatron_checkpoint = load_args_from_checkpoint(self.args)

    def update_megatron_args_from_cmd_config(self, loader_megatron):
        self.args.w_pack = self.args_cmd.w_pack
        self.args.add_qkv_bias = self.args_cmd.add_qkv_bias
        self.args.add_dense_bias = self.args_cmd.add_dense_bias
        self.args.post_norm = self.args_cmd.post_norm
        self.args.tokenizer_model = getattr(self.args_cmd, 'tokenizer_model', None)
        self.args.make_vocab_size_divisible_by = getattr(self.args_cmd, 'make_vocab_size_divisible_by', None)
        if self.args_cmd.params_dtype == 'bf16':
            self.args.bf16 = True
        elif self.args_cmd.params_dtype == 'fp16':
            self.args.fp16 = True
        if self.args_cmd.add_dense_bias:
            self.args.skip_bias_add = False

        if loader_megatron:
            self.args.lora_target_modules = self.args_cmd.lora_target_modules
            self.args.lora_load = self.args_cmd.lora_load
            self.args.lora_r = self.args_cmd.lora_r
            self.args.lora_alpha = self.args_cmd.lora_alpha
        # Determine how to make our models.
        if not self.args_cmd.model_type == 'GPT':
            raise ValueError("Llama-2 is a GPT model.")

        if self.md and self.args_cmd.num_layer_list:
            self.args.num_layer_list = self.args_cmd.num_layer_list

    def set_megatron_parallel_state(self, saver_megatron):
        if saver_megatron:
            self.set_tensor_model_parallel_world_size(self.args_cmd.target_tensor_parallel_size)
            self.set_expert_model_parallel_world_size(self.args_cmd.target_expert_parallel_size)
            self.set_pipeline_model_parallel_world_size(self.args_cmd.target_pipeline_parallel_size)
            if self.args_cmd.num_layers_per_virtual_pipeline_stage:
                vp_size = (self.args.num_layers //
                           self.args_cmd.target_pipeline_parallel_size //
                           self.args_cmd.num_layers_per_virtual_pipeline_stage)
                self.set_virtual_pipeline_model_parallel_world_size(vp_size)
        else:
            self.set_tensor_model_parallel_world_size(self.args.tensor_model_parallel_size)
            self.set_pipeline_model_parallel_world_size(self.args.pipeline_model_parallel_size)
            self.set_virtual_pipeline_model_parallel_world_size(self.args.virtual_pipeline_model_parallel_size)

        # Get first pipe stage.
        self.set_tensor_model_parallel_rank(0)
        self.set_pipeline_model_parallel_rank(0)

    def get_modules_from_config(self, pp_stage_cache_flag=False):
        self.__get_modules(pp_stage_cache_flag=pp_stage_cache_flag)

    def get_modules_from_pretrained(self, pp_stage_cache_flag=False):
        self.__get_modules(from_pretrained=True, pp_stage_cache_flag=pp_stage_cache_flag)

    def __get_modules(self, from_pretrained=False, pp_stage_cache_flag=False):
        if self.args.num_experts:
            tensor_parallel.model_parallel_cuda_manual_seed(123)
        # Initialize the dictionary for the parallel mode of the model
        pp_rank = self.get_pipeline_model_parallel_rank()
        if pp_stage_cache_flag and pp_rank < len(self.pp_stage_cache):
            self.module = self.pp_stage_cache[pp_rank]
            return

        virtual_pipeline_model_parallel_size = self.args.virtual_pipeline_model_parallel_size
        if virtual_pipeline_model_parallel_size is None:
            virtual_pipeline_model_parallel_size = 1

        models = [
            [
                [
                    None for _ in range(self.args.tensor_model_parallel_size)
                ]
                for _ in range(self.args.expert_model_parallel_size)
            ]
            for _ in range(virtual_pipeline_model_parallel_size)
        ]

        for ep_rank in range(self.args.expert_model_parallel_size):
            if self.args.expert_model_parallel_size > 1:
                self.set_expert_model_parallel_rank(ep_rank)
            for tp_rank in range(self.args.tensor_model_parallel_size):
                self.set_tensor_model_parallel_rank(tp_rank)
                if self.args.virtual_pipeline_model_parallel_size is not None:
                    model_ = []
                    for vp_rank in range(self.args.virtual_pipeline_model_parallel_size):
                        self.set_virtual_pipeline_model_parallel_rank(vp_rank)
                        # Set pre_process and post_process only after virtual rank is set.
                        pre_process = mpu.is_pipeline_first_stage()
                        post_process = mpu.is_pipeline_last_stage()
                        expert_parallel_size = mpu.get_expert_model_parallel_world_size()
                        this_model = model_provider(
                            pre_process=pre_process,
                            post_process=post_process
                        ).to(self.args.params_dtype)
                        model_.append(this_model)
                else:
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    model_ = [model_provider(pre_process, post_process).to(self.args.params_dtype)]
                self.args.consumed_train_samples = 0
                self.args.consumed_valid_samples = 0
                if from_pretrained:
                    load_checkpoint(model_, None, None)
                for vp_rank in range(virtual_pipeline_model_parallel_size):
                    models[vp_rank][ep_rank][tp_rank] = model_[vp_rank]
                    if self.args.lora_target_modules and from_pretrained:
                        if virtual_pipeline_model_parallel_size > 1:
                            raise AssertionError("Virtual pipeline and LoRA weight merging "
                                                 "are not supported simultaneously")
                        models[vp_rank][ep_rank][tp_rank].merge_and_unload()

        self.module = models

        if pp_stage_cache_flag:
            self.pp_stage_cache.append(models)


    def check_for_args(self, queue, saver_megatron):
        if saver_megatron:
            return 
        check_args_list = {
            'tensor_model_parallel_size': None, 'pipeline_model_parallel_size': None, 'num_layers': None,
            'hidden_size': None, 'seq_length': None, 'num_attention_heads': None, 'max_position_embeddings': None,
            'position_embedding_type': None, 'tokenizer_type': None, 'iteration': 1, 'bert_binary_head': None,
            'disable_bias_linear': False, 'params_dtype': None, 'swiglu': False
        }
        # if hasattr(self.args, 'add_bias_linear'):
        #     check_args_list['disable_bias_linear'] = self.args.add_bias_linear

        def check_for_arg(arg_name, default=None):
            if getattr(self.args, arg_name, None) is None:
                if default is not None:
                    setattr(self.args, arg_name, default)
                elif queue is not None:
                    logger.error(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
                    logger.info(f"Arguments: {self.args}")
                    queue.put("exit")
                    exit(1)

        for check_arg in check_args_list:
            check_for_arg(check_arg, check_args_list[check_arg])

    def get_sys_argv(self):
        sys_argv = [
            'script.py',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--no-async-tensor-model-parallel-allreduce',
            '--use-cpu-initialization',
            '--micro-batch-size', '1',
            '--no-load-optim',
            '--no-load-rng',
            '--no-save-optim',
            '--no-save-rng',
            '--no-initialization',
            '--save-interval', '1',
            '--mock-data',  # To pass the "blend data checks" in arguments.py
            '--load', self.args_cmd.load_dir,
            '--finetune',
            # '--disable-bias-linear'
        ]
        
        if hasattr(self.args_cmd, 'add_bias_linear') and not self.args_cmd.add_bias_linear:
            sys_argv.append('--disable-bias-linear')

        if self.args_cmd.use_mcore_models:
            sys_argv.append('--use-mcore-models')

        if self.model_cfg.get(self.args_cmd.model_type_hf).get('config_set_value').get('embed_layernorm', False):
            sys_argv.append('--embed-layernorm')

        if self.md is None:
            return sys_argv

        sys_argv.extend([
            '--num-layers', str(self.md.num_layers),
            '--hidden-size', str(self.md.hidden_size),
            '--seq-length', str(self.md.seq_length),
            '--num-attention-heads', str(self.md.num_attention_heads),
            '--max-position-embeddings', str(self.md.max_position_embeddings),
            '--position-embedding-type', str(self.md.position_embedding_type),
            '--tokenizer-type', str(self.md.tokenizer_type),
            '--tensor-model-parallel-size', str(self.args_cmd.target_tensor_parallel_size),
            '--pipeline-model-parallel-size', str(self.args_cmd.target_pipeline_parallel_size),
            '--expert-model-parallel-size', str(self.args_cmd.target_expert_parallel_size),
            '--save', self.args_cmd.save_dir
        ])

        if self.args_cmd.num_layers_per_virtual_pipeline_stage:
            sys_argv.extend(['--num-layers-per-virtual-pipeline-stage',
                             str(self.args_cmd.num_layers_per_virtual_pipeline_stage)])

        num_experts = getattr(self.md.checkpoint_args, 'num_experts', None)
        if self.args_cmd.target_tensor_parallel_size > 1 and num_experts is not None and num_experts > 1:
            sys_argv.append('--sequence-parallel')

        if self.md.make_vocab_size_divisible_by is not None:
            sys_argv.extend(['--make-vocab-size-divisible-by', str(self.md.make_vocab_size_divisible_by)])
        if self.md.params_dtype == torch.float16:
            sys_argv.append('--fp16')
        elif self.md.params_dtype == torch.bfloat16:
            sys_argv.append('--bf16')

        if self.md.output_layer:
            sys_argv.append('--untie-embeddings-and-output-weights')
        if not self.md.linear_bias:
            sys_argv.append('--disable-bias-linear')

        if self.md.model_type == 'BERT' and not self.md.bert_binary_head:
            sys_argv.append('--bert-no-binary-head')

        return sys_argv

    def get_model_item(self, **kwargs):
        self.update_kwargs_idx(**kwargs)
        _module = self.module
        for key in self.kwargs_idx:
            if "rank" in key:
                _module = _module[self.kwargs_idx[key]]
        return _module

    @staticmethod
    def set_tensor_model_parallel_world_size(tensor_model_parallel_size):
        mpu.set_tensor_model_parallel_world_size(tensor_model_parallel_size)

    @staticmethod
    def set_expert_model_parallel_world_size(expert_model_parallel_size):
        mpu.set_expert_model_parallel_world_size(expert_model_parallel_size)

    @staticmethod
    def set_pipeline_model_parallel_world_size(pipeline_model_parallel_size):
        mpu.set_pipeline_model_parallel_world_size(pipeline_model_parallel_size)

    @staticmethod
    def set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size):
        mpu.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)

    @staticmethod
    def set_tensor_model_parallel_rank(tensor_model_parallel_rank):
        mpu.set_tensor_model_parallel_rank(tensor_model_parallel_rank)

    @staticmethod
    def set_pipeline_model_parallel_rank(pipeline_model_parallel_rank):
        mpu.set_pipeline_model_parallel_rank(pipeline_model_parallel_rank)

    @staticmethod
    def set_expert_model_parallel_rank(pipeline_model_parallel_rank):
        mpu.set_expert_model_parallel_rank(pipeline_model_parallel_rank)

    @staticmethod
    def set_virtual_pipeline_model_parallel_rank(pipeline_model_parallel_rank):
        mpu.set_virtual_pipeline_model_parallel_rank(pipeline_model_parallel_rank)

    @staticmethod
    def get_pipeline_model_parallel_rank():
        return mpu.get_pipeline_model_parallel_rank()


class MegatronLegacyModel(MegatronModel):
    def __init__(self, args_cmd, md=None):
        super(MegatronLegacyModel, self).__init__(args_cmd, md)

    def get_module_mapping(self):
        module_layer = "language_model.encoder.layers[layer_idx]."
        self.module_mapping = {
            "embedding": "language_model.embedding",
            "embedding_word_embeddings": "language_model.embedding.word_embeddings",
            "embedding_word_embeddings_norm": "language_model.embedding.word_embeddings.norm",
            "embedding_position_embeddings": "language_model.embedding.position_embeddings",
            "model": "module",
            "layers_input_layernorm": module_layer + "input_norm",
            "layers": "language_model.encoder.layers",
            "layers_self_attention_linear_proj": module_layer + "self_attention.dense",
            "layers_self_attention_linear_qkv": module_layer + "self_attention.query_key_value",
            "layers_self_attention_post_attention_layernorm": module_layer + "post_attention_norm",
            "layers_self_attention_pre_mlp_layernorm": module_layer + "post_attention_norm",
            "layers_mlp_linear_fc1": module_layer + "mlp.dense_h_to_4h",
            "layers_mlp_linear_fc2": module_layer + "mlp.dense_4h_to_h",
            "layers_self_attention_post_mlp_layernorm": module_layer + "post_mlp_layernorm",
            "final_layernorm": "language_model.encoder.final_norm",
            "output_layer": "language_model.output_layer",
            "word_embeddings": "word_embeddings"
        }


class MegatronMCoreModel(MegatronModel):
    def __init__(self, args_cmd, md=None):
        super(MegatronMCoreModel, self).__init__(args_cmd, md)

    def get_module_mapping(self):
        module_layer = "decoder.layers[layer_idx]."
        self.module_mapping = {
            "embedding": "embedding",
            "embedding_word_embeddings": "embedding.word_embeddings",
            "embedding_word_embeddings_norm": "embedding.word_embeddings.norm",
            "embedding_position_embeddings": "embedding.position_embeddings",
            "model": "module",
            "layers_input_layernorm": module_layer + "input_layernorm",
            "layers": "decoder.layers",
            "layers_self_attention_linear_proj": module_layer + "self_attention.linear_proj",
            "layers_self_attention_linear_qkv": module_layer + "self_attention.linear_qkv",
            "layers_self_attention_q_layernorm": module_layer + "self_attention.q_layernorm",
            "layers_self_attention_k_layernorm": module_layer + "self_attention.k_layernorm",
            "layers_self_attention_post_attention_layernorm": module_layer + "post_attn_norm",
            "layers_self_attention_pre_mlp_layernorm": module_layer + "pre_mlp_layernorm",
            "layers_mlp_linear_fc1": module_layer + "mlp.linear_fc1",
            "layers_mlp_linear_fc2": module_layer + "mlp.linear_fc2",
            "layers_self_attention_post_mlp_layernorm": module_layer + "post_mlp_layernorm",
            "final_layernorm": "decoder.final_layernorm",
            "output_layer": "output_layer"
        }

        config_value = self.model_cfg.get(self.args_cmd.model_type_hf).get('config_set_value')
        if config_value.get('moe_flag', False):
            self.module_mapping["layers_mlp_router"] = module_layer + "mlp.router"
            self.module_mapping["layers_mlp_linear_fc1"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc1"
            self.module_mapping["layers_mlp_linear_fc2"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2"

        if config_value.get('mlp_experts_flag', False):
            self.module_mapping["layers_mlp_router"] = module_layer + "mlp.router"
            self.module_mapping[
                "layers_mlp_experts_linear_fc1"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc1"
            self.module_mapping[
                "layers_mlp_experts_linear_fc2"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2"
        
        # MLP
        self.module_mapping["layers_self_attention_linear_qb"] = module_layer + "self_attention.linear_qb"
        self.module_mapping["layers_self_attention_linear_kvb"] = module_layer + "self_attention.linear_kvb"
        
        # shared experts
        self.module_mapping[
            "layers_mlp_shared_experts_linear_fc1"] = module_layer + "mlp.shared_experts.linear_fc1"
        self.module_mapping[
            "layers_mlp_shared_experts_linear_fc2"] = module_layer + "mlp.shared_experts.linear_fc2"
        
        # moe grouped gemm
        self.module_mapping[
            "layers_mlp_experts_weight1"] = module_layer + "mlp.experts.weight1"
        self.module_mapping[
            "layers_mlp_experts_weight2"] = module_layer + "mlp.experts.weight2"


def get_megatron_model(args_cmd, md=None):
    if args_cmd.use_mcore_models:
        return MegatronMCoreModel(args_cmd=args_cmd, md=md)
    else:
        return MegatronLegacyModel(args_cmd=args_cmd, md=md)


def get_huggingface_model(args_cmd):
    return HuggingfaceModel(args_cmd)
