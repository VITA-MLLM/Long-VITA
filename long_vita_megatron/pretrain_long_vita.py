# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT VLM."""

import os
import torch
from functools import partial

from typing import Union

from copy import deepcopy
from contextlib import nullcontext
import long_vita_megatron.megatron_adaptor

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from long_vita_megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

from long_vita_megatron.training.checkpointing import load_checkpoint

# from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from long_vita_megatron.core.models.vision.clip_vit_model import CLIPViTModel
from long_vita_megatron.core.models.vision.eva_vit_model import EVA2ViTModel
from long_vita_megatron.core.models.vision.intern_vit_model import InternViTModel
from long_vita_megatron.core.models.vision.siglip_vit_model import SigLIPViTModel
from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_spec
# from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec_for_eva
from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_local_spec_for_eva
from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_local_spec_for_intern
from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec_for_intern
from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_local_spec_for_siglip
from long_vita_megatron.core.models.vision.vit_layer_specs import get_mlp_module_spec

from long_vita_megatron.core.transformer.transformer_config import VisionTransformerConfig

from long_vita import build_supervised_dataset_megatron
import logging
logger = logging.getLogger("__name__")


def get_vision_model_args(args):

    args.sequence_parallel = False
    if not args.vision_context_parallel:
        args.context_parallel_size = 1
    args.expert_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.virtual_pipeline_model_parallel_size = None
    # args.independent_parallel = True

    args.overlap_grad_reduce = False
    args.gradient_accumulation_fusion = False
    args.bias_dropout_fusion = False

    # args.load = args.vit_load

    args.num_layer_list = ""

    args.recompute_method = None
    args.recompute_granularity = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_method = "block"
        args.recompute_granularity = "full"
        args.recompute_num_layers = 9999

    args.img_h = args.image_size
    args.img_w = args.image_size

    args.encoder_seq_length = args.vision_seq_length
    args.decoder_seq_length = args.vision_seq_length
    args.max_position_embeddings = args.vision_seq_length
    args.seq_length = args.vision_seq_length

    return args

def get_vision_model_args_openai_300m(args):
    # from megatron.training.activations import quick_gelu, squared_relu
    # args.add_class_token = False
    # args.add_class_token = True

    args.recompute_granularity = None
    args.recompute_method = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_granularity = 'full'
        args.recompute_method = 'block'
        args.recompute_num_layers = 24

    args.num_layers = 24
    args.add_bias_linear = True
    args.add_qkv_bias = True
    args.hidden_size = 1024
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.ffn_hidden_size = 4096
    args.gated_linear_unit = False
    # args.activation_func = quick_gelu
    args.activation_func = torch.nn.functional.gelu
    args.kv_channels = 64
    args.num_attention_heads = 16
    args.group_query_attention = False
    args.num_query_groups = 16
    args.layernorm_zero_centered_gamma = False
    # args.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    args.bias_activation_fusion = False
    args.bias_dropout_fusion = False
    args.attention_softmax_in_fp32 = True
    args.normalization = 'LayerNorm'
    args.apply_rope_fusion = False

    args.patch_dim = 14

    return args

def get_vision_model_args_eva_4b(args):

    # args.add_class_token = False
    # args.add_class_token = True

    args.recompute_granularity = None
    args.recompute_method = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_granularity = 'full'
        args.recompute_method = 'block'
        args.recompute_num_layers = 63

    args.num_layers = 63
    args.encoder_num_layers = 63
    args.ffn_hidden_size = 15360
    args.hidden_size = 1792
    args.kv_channels = 112
    args.normalization = 'LayerNorm'
    args.num_attention_heads = 16
    args.patch_dim = 14
    args.position_embedding_type = 'learned_absolute'
    args.swiglu = False
    args.vocab_size = 1
    args.add_dense_bias = False
    args.add_bias_linear = True

    args.norm_epsilon = 1e-06
    # args.attention_dropout = 0.1
    # args.hidden_dropout = 0.1

    args.perform_initialization = False
    args.init_method_std = 0.02
    args.transformer_impl = 'transformer_engine'
    args.transformer_pipeline_model_parallel_size = 1

    args.group_query_attention = False
    args.num_query_groups = 16
    args.padded_vocab_size = 1

    return args


def get_vision_model_args_intern_300m(args):
    # from megatron.training.activations import quick_gelu, squared_relu

    # args.add_class_token = False
    # args.add_class_token = True

    args.recompute_method = None
    args.recompute_granularity = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_method = "block"
        args.recompute_granularity = "full"
        args.recompute_num_layers = 24

    args.num_layers = 24
    args.add_bias_linear = True
    args.add_qkv_bias = True
    args.hidden_size = 1024
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.ffn_hidden_size = 4096
    args.gated_linear_unit = False
    # args.activation_func = quick_gelu
    args.activation_func = torch.nn.functional.gelu
    args.kv_channels = 64
    args.num_attention_heads = 16
    args.group_query_attention = False
    args.num_query_groups = 16
    args.layernorm_zero_centered_gamma = False
    # args.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    args.bias_activation_fusion = False
    args.bias_dropout_fusion = False
    args.attention_softmax_in_fp32 = True
    args.normalization = 'LayerNorm'
    args.apply_rope_fusion = False

    args.patch_dim = 14

    args.swiglu = False

    return args

def get_vision_model_args_intern_6b(args):
    # from megatron.training.activations import quick_gelu, squared_relu

    # args.add_class_token = False
    # args.add_class_token = True

    args.recompute_method = None
    args.recompute_granularity = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_method = "block"
        args.recompute_granularity = "full"
        args.recompute_num_layers = 45

    args.num_layers = 45
    args.add_bias_linear = True
    args.add_qkv_bias = True
    args.hidden_size = 3200
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.ffn_hidden_size = 12800
    args.gated_linear_unit = False
    # args.activation_func = quick_gelu
    args.activation_func = torch.nn.functional.gelu
    args.kv_channels = 128
    args.num_attention_heads = 25
    args.group_query_attention = False
    args.num_query_groups = 25
    args.layernorm_zero_centered_gamma = False
    # args.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    args.bias_activation_fusion = False
    args.bias_dropout_fusion = False
    args.attention_softmax_in_fp32 = True
    args.normalization = 'RMSNorm'
    args.apply_rope_fusion = False

    args.patch_dim = 14

    args.swiglu = False

    return args


def get_vision_model_args_siglip_400m(args):
    # from megatron.training.activations import quick_gelu, squared_relu

    # args.add_class_token = False
    # args.add_class_token = True

    args.recompute_method = None
    args.recompute_granularity = None
    args.recompute_num_layers = None
    if args.vision_model_recompute and not args.vision_model_freeze:
        args.recompute_method = "block"
        args.recompute_granularity = "full"
        args.recompute_num_layers = 27

    args.num_layers = 27
    args.add_bias_linear = True
    args.add_qkv_bias = True
    args.hidden_size = 1152
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.ffn_hidden_size = 4304
    args.gated_linear_unit = False
    # args.activation_func = quick_gelu
    args.activation_func = partial(torch.nn.functional.gelu, approximate="tanh")
    args.kv_channels = 72
    args.num_attention_heads = 16
    args.group_query_attention = False
    args.layernorm_zero_centered_gamma = False
    # args.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    args.bias_activation_fusion = False
    args.bias_dropout_fusion = False
    args.attention_softmax_in_fp32 = True
    args.normalization = 'LayerNorm'
    args.apply_rope_fusion = False

    args.patch_dim = 14

    args.swiglu = False

    return args


class MegatronVisionModel(torch.nn.Module):
    def __init__(self, pre_process):
        super().__init__()
        args = get_args()
        self.vision_seq_length = args.vision_seq_length
        self.image_token_length = args.image_token_length
        self.vision_model_type = args.vision_model_type
        self.vision_context_parallel = args.vision_context_parallel
        self.vision_downsample_ratio = args.vision_downsample_ratio
        self.vision_downsample_stride = args.vision_downsample_stride
        self.add_class_token = args.add_class_token
        self.vision_model_freeze = args.vision_model_freeze
        self.vision_projector_freeze = args.vision_projector_freeze

        self.vision_model_recompute = args.vision_model_recompute
        self.vision_projector_recompute = args.vision_projector_recompute
        print(f'vision_projector_recompute {self.vision_projector_recompute}')

        vit_args = deepcopy(args)
        vit_args = get_vision_model_args(vit_args)

        print_rank_0(f'Building {self.vision_model_type} model ...')
        use_te = args.transformer_impl == "transformer_engine"
        use_te = False
        if self.vision_model_type == "eva":
            vit_args = get_vision_model_args_eva_4b(vit_args)

            # vision_model_layer_spec = get_vit_layer_with_transformer_engine_spec_for_eva()
            vision_model_layer_spec = get_vit_layer_local_spec_for_eva()

            vit_module = EVA2ViTModel

        if self.vision_model_type == "openai":
            vit_args = get_vision_model_args_openai_300m(vit_args)

            vision_model_layer_spec = get_vit_layer_spec(use_te=False)

            vit_module = CLIPViTModel

        if self.vision_model_type == "intern_300m":
            vit_args = get_vision_model_args_intern_300m(vit_args)

            if use_te:
                vision_model_layer_spec = get_vit_layer_with_transformer_engine_spec_for_intern()
            else:
                vision_model_layer_spec = get_vit_layer_local_spec_for_intern()

            vit_module = InternViTModel

        if self.vision_model_type == "intern_6b":
            vit_args = get_vision_model_args_intern_6b(vit_args)

            vision_model_layer_spec = get_vit_layer_local_spec_for_intern()

            vit_module = InternViTModel

        if self.vision_model_type == "siglip_400m":
            vit_args = get_vision_model_args_siglip_400m(vit_args)

            vision_model_layer_spec = get_vit_layer_local_spec_for_siglip()

            vit_module = SigLIPViTModel

        vision_model_config = core_transformer_config_from_args(vit_args, VisionTransformerConfig)
        # assert vision_model_config.independent_parallel

        from long_vita_megatron.training.utils import print_args
        print_args('vit_args', vit_args)
        print_rank_0(f"MegatronVisionModel vision_model_config {vision_model_config}")
        print_rank_0(f"MegatronVisionModel vision_model_layer_spec {vision_model_layer_spec}")

        self.vit = vit_module(
            vision_model_config,
            vision_model_layer_spec,
            add_class_token=vit_args.add_class_token,
            patch_dim=vit_args.patch_dim,
            img_h=vit_args.image_size,
            img_w=vit_args.image_size,
            vision_context_parallel=vit_args.vision_context_parallel,
        )

        # print(f"MegatronVisionModel vit {vit}")
        if vit_args.vit_load is not None:
            load_checkpoint([self.vit], None, None, load_arg="vit_load", args=vit_args)

        # self.linear_proj = torch.nn.Linear(vit_args.hidden_size, args.hidden_size)
        # return

        from long_vita_megatron.core.models.vision.multimodal_projector import MultimodalProjector

        base_config = core_transformer_config_from_args(args)
        # base_config.language_model_type = args.language_model_type

        vision_projector_config = deepcopy(base_config)
        vision_projector_config.gated_linear_unit = False
        vision_projector_config.bias_activation_fusion = False
        vision_projector_config.add_bias_linear = False
        vision_projector_config.hidden_size = args.hidden_size
        # vision_projector_config.ffn_hidden_size = 14336
        vision_projector_config.ffn_hidden_size = vision_model_config.hidden_size
        # vision_projector_config.ffn_hidden_size = args.hidden_size
        # vision_projector_config.activation_func = torch.nn.functional.silu
        vision_projector_config.activation_func = torch.nn.functional.gelu

        # vision_projector_config.sequence_parallel = True

        vision_projector_layer_spec = get_mlp_module_spec(use_te=False).submodules
        if hasattr(args, "vision_projector_type"):
            vision_projector_type = args.vision_projector_type
        else:
            vision_projector_type = "mlp"

        proj_input_size = vision_model_config.hidden_size
        if self.add_class_token:
            assert args.vision_seq_length == int(args.image_size // vit_args.patch_dim) ** 2 + 1
        else:
            assert args.vision_seq_length == int(args.image_size // vit_args.patch_dim) ** 2
        if self.vision_downsample_ratio != 1:
            proj_input_size = vision_model_config.hidden_size * int(1 / self.vision_downsample_ratio) ** 2
            assert args.image_token_length == int(self.vision_seq_length * self.vision_downsample_ratio ** 2)
        if self.vision_downsample_stride != 1:
            proj_input_size = vision_model_config.hidden_size * int(self.vision_downsample_stride) ** 2
            assert args.image_token_length == int(-(-args.image_size // vit_args.patch_dim)) ** 2 / self.vision_downsample_stride ** 2

        print_rank_0(f"vision_projector_config {vision_projector_config}")
        print_rank_0(f"vision_projector_layer_spec {vision_projector_layer_spec}")
        self.vision_projection = MultimodalProjector(
            vision_projector_config,
            vision_projector_layer_spec,
            vision_projector_type,
            proj_input_size,
        )

        if args.vision_projector_pre_norm:
            self.pre_proj_layernorm = torch.nn.LayerNorm(proj_input_size)
        else:
            self.pre_proj_layernorm = torch.nn.Identity()


        args.vit_num_layers = vit_args.num_layers

    def forward_projection(self, vit_output):
        # print("MegatronVisionModel vit_output2", vit_output.size())
        vit_output = self.pre_proj_layernorm(vit_output)
        # print("MegatronVisionModel vit_output3", vit_output.size())
        # print(f"vit_output {vit_output.size()}")
        vit_output = self.vision_projection(vit_output)

        # from megatron.core.utils import make_viewless_tensor
        # vit_output = make_viewless_tensor(
        #     inp=vit_output, requires_grad=True, keep_graph=True
        # )

        return vit_output


    def forward_downsample(self, vit_output):
        if self.add_class_token:
            vit_output = vit_output[:, 1:, :]

        if self.vision_downsample_ratio != 1:
            h = w = int(vit_output.shape[1] ** 0.5)
            vit_output = vit_output.reshape(vit_output.shape[0], h, w, -1)
            vit_output = self.pixel_shuffle(vit_output, scale_factor=self.vision_downsample_ratio)
            vit_output = vit_output.reshape(vit_output.shape[0], -1, vit_output.shape[-1])

        if self.vision_downsample_stride != 1:
            h = w = int(vit_output.shape[1] ** 0.5)
            vit_output = vit_output.reshape(vit_output.shape[0], h, w, -1)
            vit_output = self.pixel_shuffle_v2(vit_output, scale_stride=self.vision_downsample_stride)
            vit_output = vit_output.reshape(vit_output.shape[0], -1, vit_output.shape[-1])

        return vit_output

    def forward_once(self, images, attention_mask):
        vit_context = nullcontext()
        if self.vision_model_freeze:
            vit_context = torch.no_grad()

        with vit_context:
            vit_output = self.vit(images, attention_mask)
            # print("MegatronVisionModel vit_output1", vit_output.size())
            # torch.cuda.empty_cache()

            if self.vision_model_recompute:
                vit_output = tensor_parallel.checkpoint(
                    self.forward_downsample,
                    False,
                    vit_output
                )
            else:
                vit_output = self.forward_downsample(vit_output)

        proj_context = nullcontext()
        if self.vision_projector_freeze:
            proj_context = torch.no_grad()
            # torch.cuda.empty_cache()
        # print(f"vit_output {vit_output.size()}")

        with proj_context:
            if self.vision_projector_recompute:
                vit_output = tensor_parallel.checkpoint(
                    self.forward_projection,
                    False,
                    vit_output
                )
            else:
                vit_output = self.forward_projection(vit_output)

        return vit_output

    def forward_chunk(self, images, attention_mask):
        chunk_size = 256
        images = torch.split(images, chunk_size, dim=0)
        chunk_num = len(images)

        vit_output_all = []
        for chunk_idx in range(chunk_num):
            vit_output = self.forward_once(images[chunk_idx], attention_mask)
            vit_output_all.append(vit_output)
        vit_output = torch.cat(vit_output_all, dim=0)

        return vit_output

    def forward(self, **kw_args):
        images = kw_args['images']
        attention_mask = None
        # print("images", images.size())

        vit_output = self.forward_chunk(images, attention_mask)

        # print("MegatronVisionModel vit_output4", vit_output.size())

        if mpu.get_context_parallel_world_size() != 1 and self.vision_context_parallel:
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()

            calibration_index = torch.arange(self.image_token_length, device='cuda').view(2 * cp_size, self.image_token_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)

            ci_list = [torch.zeros_like(calibration_index) for _ in range(cp_size)]
            torch.distributed.all_gather(ci_list, calibration_index, group=mpu.get_context_parallel_group())
            calibration_index = torch.cat(ci_list)

            vo_list = [torch.zeros_like(vit_output) for _ in range(cp_size)]
            torch.distributed.all_gather(vo_list, vit_output, group=mpu.get_context_parallel_group())
            vo_list[cp_rank] = vit_output
            # vit_output_all = torch.cat(vo_list)
            vit_output_all = torch.cat(vo_list, dim=1)

            vit_output = torch.zeros_like(vit_output_all)
            # vit_output[calibration_index] = vit_output_all
            vit_output[:, calibration_index] = vit_output_all

        # print("MegatronVisionModel vit_output3", vit_output.size())
        # torch.set_printoptions(precision=10)
        # print("MegatronVisionModel linear_proj", self.linear_proj.weight)
        # vit_output =  self.linear_proj(vit_output.transpose(0, 1))
        # vit_output = self.vision_projection(vit_output.transpose(0, 1).contiguous())
        # print("MegatronVisionModel vit_output4", vit_output.size())
        return vit_output

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def pixel_shuffle_v2(self, x, scale_stride=2):
        n, w, h, c = x.size()
        assert w == h
        pl = (scale_stride - (h % scale_stride)) % scale_stride
        x = torch.nn.functional.pad(x, (0, 0, 0, pl, 0, pl), "constant", 0)
        h += pl
        w += pl

        x = x.reshape(n, w // scale_stride, scale_stride, h // scale_stride, scale_stride, c)
        x = x.permute(0, 1, 3, 2, 4, 5) 
        x = x.flatten(3)
        x = x.reshape(n, -1, scale_stride * scale_stride * c)
        return x


def vision_model_provider(config):
    model = MegatronVisionModel(True)
    return model


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        from long_vita_megatron.core.models.multimodal.gpt_vl_model import GPTVLModel
        print_rank_0("Building megatron mcore vision language model ...")

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        print_rank_0(f"model_provider args {args}")
        print_rank_0(f"model_provider config {config}")
        print_rank_0(f"model_provider transformer_layer_spec {transformer_layer_spec}")
        model = GPTVLModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            external_feature_model_provider=vision_model_provider,
            allow_missing_keys=['external_feature_model'],
        )
    else:
        raise NotImplementedError

    if args.language_model_freeze:
        model.language_model_freeze()
    if args.vision_model_freeze:
        model.vision_model_freeze()
    if args.vision_projector_freeze:
        model.vision_projector_freeze()

    print(f"model {model}")
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if args.reset_position_ids or args.reset_attention_mask:
        pass
    elif (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    external_inputs = {}
    for k in list(batch.keys()):
        if 'external_' in k:
            external_inputs[k[len('external_'):]] = batch.pop(k)
    batch['external_inputs'] = external_inputs

    print_batch(batch)

    return batch.values()


DATA_PRINT_ONCE = True
BATCH = None
def print_batch(batch):

    global DATA_PRINT_ONCE
    global BATCH

    if batch is not None:
        BATCH = batch
    else:
        batch = BATCH
        DATA_PRINT_ONCE = True

    if batch is None:
        return

    if DATA_PRINT_ONCE and mpu.is_pipeline_first_stage():

        args = get_args()
        global_rank = torch.distributed.get_rank()
        f = open(os.path.join(args.save, f"print_batch_{global_rank}.log"), "a")

        tokenizer = get_tokenizer().tokenizer
        torch.set_printoptions(threshold=100_000)

        if "loss_mask" in batch and batch["loss_mask"] is not None:
            loss_mask = batch["loss_mask"]
            print(f"loss_mask {loss_mask} {loss_mask.size()}", file=f)

        if "position_ids" in batch and batch["position_ids"] is not None:
            position_ids = batch["position_ids"]
            print(f"position_ids {position_ids} {position_ids.size()}", file=f)

        if "attention_mask" in batch and batch["attention_mask"] is not None:
            attention_mask = batch["attention_mask"]
            if isinstance(attention_mask, list):
                attention_mask = attention_mask[0]
            print(f"attention_mask {attention_mask} {attention_mask.size()}", file=f)

        if "tokens" in batch and batch["tokens"] is not None:
            tokens = batch["tokens"]
            print(f"tokens {tokens} {tokens.size()}", file=f)

            tokens_ = tokens.cpu().clone().detach()
            tokens_ = tokenizer.batch_decode(tokens_.tolist(), skip_special_tokens=False)
            print(f"tokens_ {tokens_[:]}", file=f)

        if "labels" in batch and batch["labels"] is not None:
            labels = batch["labels"]
            print(f"labels {labels} {labels.size()}", file=f)

            labels_ = labels.cpu().clone().detach()
            labels_[labels_==-100] = tokenizer("-", add_special_tokens=False).input_ids[0]
            labels_ = tokenizer.batch_decode(labels_.tolist(), skip_special_tokens=False)
            print(f"labels {labels_}", file=f)

            labels__ = labels.cpu().clone().detach()
            labels__[loss_mask.to(torch.int64)==0] = tokenizer("-", add_special_tokens=False).input_ids[0]
            labels__ = tokenizer.batch_decode(labels__.tolist(), skip_special_tokens=False)
            print(f"labels__ {labels__}", file=f)

        from long_vita_megatron.training.utils import get_actual_seq_len
        actual_seq_len = get_actual_seq_len()
        print(f"actual_seq_len {actual_seq_len}", file=f)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} {v} {v.size()}", file=f)
            else:
                print(f"{k} {v}", file=f)
        for k, v in batch['external_inputs'].items():
            print(f"{k} {v} {v.size()}", file=f)

        f.close()

    DATA_PRINT_ONCE = False


LOSS_PRINT_ONCE = True
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    if args.is_instruction_dataset:
        loss_mask = loss_mask[..., 1:].view(-1).float()
    else:
        loss_mask = loss_mask.view(-1).float()

    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    global LOSS_PRINT_ONCE
    if LOSS_PRINT_ONCE:
        args = get_args()
        global_rank = torch.distributed.get_rank()
        f = open(os.path.join(args.save, f"print_loss_{global_rank}.log"), "a")

        torch.set_printoptions(threshold=100_000)
        print(f"context_parallel_size {args.context_parallel_size}", file=f)
        print(f"output_tensor {output_tensor} {output_tensor.size()}", file=f)
        print(f"loss_mask {loss_mask} {loss_mask.size()}", file=f)
        print(f"loss {loss}", file=f)

        f.close()

    LOSS_PRINT_ONCE = False

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, external_inputs = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    logit_mask = None
    if args.logit_mask:
        logit_mask = loss_mask.bool()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, external_inputs=external_inputs, logit_mask=logit_mask)
    if args.logit_mask:
        loss_mask = torch.ones(output_tensor.size(0), output_tensor.size(1) + 1, dtype=output_tensor.dtype, device=output_tensor.device)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    args = get_args()
    if args.reset_position_ids or args.reset_attention_mask:
        return mpu.get_tensor_model_parallel_rank() == 0

    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    from long_vita.tokenizer import update_tokenizer
    tokenizer = get_tokenizer().tokenizer
    tokenizer = update_tokenizer(tokenizer)
    print_rank_0(f"tokenizer {tokenizer}")

    global_rank = torch.distributed.get_rank()
    if is_dataset_built_on_rank():
        print(f"> rank {global_rank} is creating GPT datasets ...")

        train_ds, valid_ds, test_ds = build_supervised_dataset_megatron(
            args,
            tokenizer,
            create_position_ids=True,
            create_loss_mask=True,
            shift_token=False,
        )
        print_rank_0("> finished creating GPT datasets ...")

        return train_ds, valid_ds, test_ds
    else:
        print(f"> rank {global_rank} does not create GPT datasets ...")
        return None, None, None

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def extra_args_provider(parser):
    parser = add_data_args(parser)
    parser = add_memory_args(parser)
    parser = add_vlm_args(parser)

    return parser


def add_data_args(parser):
    group = parser.add_argument_group(title='data')

    group.add_argument('--data-seq-length', type=int, default=4096, help='data_seq_length.')

    return parser


def add_memory_args(parser):
    group = parser.add_argument_group(title='memory')

    group.add_argument('--enable-chunk-sequence', action='store_true', help='enable_chunk_sequence.')
    group.add_argument('--enable-chunk-memory', action='store_true', help='enable_chunk_memory.')
    group.add_argument('--chunk-size', type=int, default=4096, help='chunk_size.')

    return parser


def add_vlm_args(parser):

    group = parser.add_argument_group(title='vit load')

    group.add_argument("--image-token-length", type=int, help='image_token_length')
    group.add_argument("--image-size", type=int, default=448, help='vit image size')
    # group.add_argument("--max-num-image", type=int, default=8, help='max_num_image')
    group.add_argument("--max-num-frame", type=int, default=8, help='max_num_frame')
    group.add_argument("--max-fps", type=int, default=1, help='max_fps')
    group.add_argument("--max-patch-grid", type=int, default=6, help='max_patch_grid')
    group.add_argument("--min-patch-grid", type=int, default=1, help='min_patch_grid')

    group.add_argument('--cross-dataset-joint', action='store_true', help='cross_dataset_joint')

    group.add_argument('--first-pipeline-num-layers', type=int, default=0,
                       help='Used when you want to split pipeline parallel unevenly. 0 means even partition.')
    group.add_argument('--independent-parallel', action='store_true',
                       help='Set to True if you want to disable pipeline parallel for the model.')

    group.add_argument('--vision-model-recompute', action='store_true', help='vision_model_recompute')
    group.add_argument('--language-model-freeze', action='store_true', help='language_model_freeze.')

    group.add_argument("--vision-projector-type", type=str, default="affine", help='vision_projector_type')
    group.add_argument('--vision-projector-pre-norm', action='store_true', help='vision_projector_pre_norm')
    group.add_argument('--vision-projector-recompute', action='store_true', help='vision_projecttion_recompute')
    group.add_argument('--vision-projector-freeze', action='store_true', help='vision_projector_freeze.')

    group.add_argument('--vision-model-type', type=str, default="openai", help='vision_model_type')
    group.add_argument('--vision-model-lr-mult', type=float, default=1.0, help='vision_model_lr_mult.')
    group.add_argument('--vision-model-lr-decay-rate', type=float, default=1.0, help='vision_model_lr_decay_rate.')
    group.add_argument('--vision-model-freeze', action='store_true', help='vision_model_freeze.')

    group.add_argument("--vision-seq-length", type=int, help='vision_seq_length')
    group.add_argument('--vision-downsample-ratio', type=float, default=1.0, help='vision_downsample_ratio')
    group.add_argument('--vision-downsample-stride', type=int, default=1.0, help='vision_downsample_stride')
    group.add_argument('--vision-process-type', type=str, default="anyres", help='vision_process_type')
    group.add_argument('--vision-normalize-type', type=str, default="imagenet", help='vision_normalize_type')

    group.add_argument("--vit-load", type=str, help='path of vit')
    group.add_argument('--prompt-format', type=str, default="llama2", help='prompt_format')

    group.add_argument('--vision-context-parallel', action='store_true', help='vision_context_parallel')
    group.add_argument('--add-class-token', action='store_true', help='add_class_token')

    group.add_argument('--logit-mask', action='store_true', help='logit_mask')

    group.add_argument('--tp-2d', action='store_true', default=False,
                       help='use use-2d-tp to replace megatron-style tensor parallel')
    group.add_argument('--tp-x', type=int, default=1,
                       help='the fist dim tensor parallel size for Linear')
    group.add_argument('--tp-y', type=int, default=1,
                       help='the second dim tensor parallel size for Linear')

    group.add_argument('--moe-without-activation', action='store_true', default=False,
                       help='save all the memory occupied by activations in moe layer.')

    group.add_argument("--moe-zero-memory", type=str, default='disable',
                       choices=['disable', 'level0', 'level1'],
                       help="Save activation memory in moe layer.")

    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # try:
    if True:
        pretrain(train_valid_test_datasets_provider,
                 model_provider,
                 ModelType.encoder_or_decoder,
                 forward_step,
                 args_defaults={},
                 extra_args_provider=extra_args_provider,
                 )
    # except Exception as error:
    #     print_batch(None)
    #     print(error)

