

import os
import numpy as np
import torch
import json
from transformers.modeling_utils import (
    WEIGHTS_INDEX_NAME, 
    WEIGHTS_NAME, 
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    # shard_checkpoint, 
)
import safetensors
from collections.abc import Mapping, Sequence

def save_hfmodel(args, model, max_shard_size='10GB'):
    print("Saving")
    model.save_pretrained(args.save)
    return

    output_state_dict = model.state_dict()
    weight_file = SAFE_WEIGHTS_NAME if args.save_safetensors else WEIGHTS_NAME
    index_file = SAFE_WEIGHTS_INDEX_NAME if args.save_safetensors else WEIGHTS_INDEX_NAME
    # NOTE: remove all old index files
    if os.path.exists(os.path.join(args.save, SAFE_WEIGHTS_INDEX_NAME)):
        os.remove(os.path.join(args.save, SAFE_WEIGHTS_INDEX_NAME))
    if os.path.exists(os.path.join(args.save, WEIGHTS_INDEX_NAME)):
        os.remove(os.path.join(args.save, WEIGHTS_INDEX_NAME))

    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size, weights_name=weight_file)
    os.makedirs(args.save, exist_ok=True)
    for shard_file, shard in shards.items():
        target_file = os.path.join(args.save, shard_file)
        print(f'huggingface model is save to {target_file}')
        if args.save_safetensors:
            safetensors.torch.save_file(clone_state_dict(shard), target_file, metadata={"format": "pt"})
        else:
            torch.save(clone_state_dict(shard), target_file)

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save, weight_file)}") # do nothing
    else:
        save_index_file = os.path.join(args.save, index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

@torch.inference_mode()
def clone_state_dict(elem):
    """clone all tensors in the elem to cpu device.
    """
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        elem = elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        pass
    elif isinstance(elem, Mapping):
        elem = dict(elem)
        for k, v in elem.items():
            elem[k] = clone_state_dict(v)
        elem = elem_type(elem)
    elif isinstance(elem, Sequence):
        elem = list(elem)
        for i in range(len(elem)):
            elem[i] = clone_state_dict(elem[i])
        elem = elem_type(elem)
    return elem

def build_layer_id_mapping(args):
    """
        global layer id <--> local layer id
    """
    ltog, gtol = dict(), dict()

    assert args.target_decoder_first_pipeline_num_layers is None or args.target_num_layers_per_virtual_pipeline_stage is None, "Currently uneven VPP not supported"

    if args.target_decoder_first_pipeline_num_layers is not None:
        remained_layers = args.num_layers - args.target_decoder_first_pipeline_num_layers
        remained_stages = args.pipeline_model_parallel_size - 1
        assert remained_layers % remained_stages == 0
        pp_layers_per_stage = [args.target_decoder_first_pipeline_num_layers] +([remained_layers // remained_stages] * remained_stages)

        for pp_id, num_layers in enumerate(pp_layers_per_stage):
            for global_layer_id in range(offset, offset + num_layers):
                # NOTE: map a global transformer layer to a local pp rank
                # global_id <--> (pp_id, vpp_id, local_id)
                local_layer_id = global_layer_id - offset
                ltog[(pp_id, 0, local_layer_id)] = global_layer_id
                gtol[global_layer_id] = (pp_id, 0, local_layer_id)
            offset += num_layers
    else:
        n_chunks = args.pipeline_model_parallel_size
        pp_size = args.pipeline_model_parallel_size
        if args.target_num_layers_per_virtual_pipeline_stage is not None:
            assert args.num_layers % (args.target_num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size) == 0
            n_chunks = args.num_layers // args.target_num_layers_per_virtual_pipeline_stage
        num_layer_per_chunk = args.num_layers // n_chunks
        pp_layers_per_stage = [num_layer_per_chunk] * n_chunks

        offset = 0        
        for chunk_id, num_layers in enumerate(pp_layers_per_stage):
            for global_layer_id in range(offset, offset + num_layers):
                # NOTE: map a global transformer layer to a local pp rank
                # global_id <--> (pp_id, vpp_id, local_id)
                pp_id = chunk_id % pp_size
                vpp_id = chunk_id // pp_size
                local_layer_id = global_layer_id - offset
                ltog[(pp_id, vpp_id, local_layer_id)] = global_layer_id
                gtol[global_layer_id] = (pp_id, vpp_id, local_layer_id)
            offset += num_layers
    return ltog, gtol

def safe_copy(src_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    assert src_tensor.dtype == dst_tensor.dtype
    assert src_tensor.shape == dst_tensor.shape, f"{src_tensor.shape} {dst_tensor.shape}"
    dst_tensor.data.copy_(src_tensor.data)
    return src_tensor.numel()

def save_state_dict(args, model_chunks, checkpoint_name, has_vpp: bool=False):
    """
    Save some model chunks to a megatron checkpoint file
    """
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    state_dict['iteration'] = 0    
    if not has_vpp:
        state_dict['model'] = model_chunks[0]
    else:
        for vpp_id in range(len(model_chunks)):
            state_dict[f"model{vpp_id}"] = model_chunks[vpp_id]
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)














import os
import re
import json
import torch
import copy
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from long_vita.models.long_vita_qwen2_intern import LongVITAForCausalLM

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
from functools import partial
from megatron.training.utils import get_ltor_masks_and_position_ids
import sys

from long_vita_megatron.pretrain_long_vita import model_provider
from long_vita_megatron.pretrain_long_vita import extra_args_provider
# from toolkits.model_checkpoints_convertor.utils import (
#     build_layer_id_mapping,
#     safe_copy,
#     save_state_dict,
#     save_hfmodel
# )

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

def add_model_args(parser):

    parser.add_argument(
        "--target-tensor-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-pipeline-model-parallel-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--target-num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None
    )

    parser.add_argument(
        "--target-decoder-first-pipeline-num-layers",
        type=int,
        default=None
    )

    parser.add_argument(
        "--config_name",
        type=str
    )

    parser.add_argument(
        "--hf-ckpt-path",
        type=str
    )

    parser.add_argument(
        "--save-safetensors",
        action='store_true',
    )

    parser.add_argument(
        "--convert-checkpoint-from-megatron-to-transformers",
        action='store_true',
    )

    return parser

def load_megatron_model(args):
    """load a TPxPPx checkpoint into a TP1PP1 model."""
    os.makedirs(args.save, exist_ok=True)

    model = model_provider()
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    if args.target_num_layers_per_virtual_pipeline_stage is not None:
        args.num_layers_per_virtual_pipeline_stage = args.target_num_layers_per_virtual_pipeline_stage
        num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage

    model_path = args.load
    tracker_filename = get_checkpoint_tracker_filename(model_path)
    iteration, release = read_metadata(tracker_filename)

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.tensor_model_parallel_size
    
    vision_state_dicts = defaultdict(dict)
    state_dict = {}
    mid_state = defaultdict(list)
    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, None, None, None, None)
        state_dict = torch.load(checkpoint_name)['model']

    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
    ):  
        for tp_rank in range(args.tensor_model_parallel_size):
            checkpoint_name = get_checkpoint_name(model_path, iteration, release, None, tp_rank, None, None, None)
            print(f'load {checkpoint_name}')
            split_state = torch.load(checkpoint_name, map_location="cpu")['model']
            for k, v in split_state.items():
                if k.startswith('external_feature_model'):
                    vision_state_dicts[(tp_rank, 0)][k] = v
                else:
                    mid_state[k].append(v)
        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            elif 'unused' in k:
                continue
            else:
                print(k, v)
                raise ValueError
            state_dict[k] = target_v
    elif (
        args.pipeline_model_parallel_size > 1
    ):  
        ltog, _ = build_layer_id_mapping(args)
        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                checkpoint_name = get_checkpoint_name(model_path, iteration, release, True, tp_rank, pp_rank, None, None)
                print(f'load {checkpoint_name}')
                keys = ['model']
                if args.virtual_pipeline_model_parallel_size is not None:
                    keys = [f'model{i}' for i in range(args.virtual_pipeline_model_parallel_size)]
                split_state = torch.load(checkpoint_name, map_location="cpu")
                for vpp_id, key in enumerate(keys):
                    for k, v in split_state[key].items():
                        if k.startswith('vision_model'):
                            assert pp_rank == 0
                            vision_state_dicts[(tp_rank, 0)][k] = v
                            continue
                        try:
                            pattern = re.compile(r'\d+')
                            local_id = int(pattern.findall(k)[0])
                            global_id = ltog[(pp_rank, vpp_id, local_id)]
                            tgt = re.sub(r"decoder.layers.\d+", f"decoder.layers.{global_id}", k)
                            mid_state[tgt].append(v)
                        except Exception as e:
                            print(f"Skipping {k} with exception {e}")
                            mid_state[k].append(v)

        for k, v in mid_state.items():
            if not isinstance(v[0], torch.Tensor) or 'norm' in k:
                target_v = v[0]
            elif 'embedding' in k or 'output_layer' in k:
                target_v = torch.cat(v, dim=0)
            elif 'linear_proj' in k or 'linear_fc2' in k:
                target_v = torch.cat(v, dim=1)
            elif 'linear_qkv.weight' in k:
                viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1, args.hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = [x.view(group_per_split, -1) for x in v]
                target_v = torch.cat(viewed, dim=0).view(-1)
            elif 'linear_fc1' in k:
                viewed = [x.view(2, -1, args.hidden_size) for x in v]
                target_v = torch.cat(viewed, dim=1).view(-1, args.hidden_size)
            else:
                raise ValueError
            state_dict[k] = target_v

    else:
        raise ValueError('not support yet')

    for k, v in state_dict.items():
        if v is None:
            print(f"{k} {v}")
        else:
            print(f"{k} {v.size()}")

    load_split_state_dict_to_vision_model(vision_state_dicts, model.external_feature_model, args)
    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict, strict=True)
    print("Load LLM finish")
    return model

"""
    The following two functions convert a TP1PP1 MG/HF model to HF/MG format.
"""

@torch.inference_mode()
def convert_checkpoint_from_megatron_to_transformers(mgmodel, hfmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // args.num_attention_heads
    use_te = args.transformer_impl == "transformer_engine"
    value_num_per_group = args.num_attention_heads // num_query_groups
    q_dim_per_group = hidden_size // num_query_groups
    kv_dim_per_group = head_dim

    # 1. vision model
    hfvision = hfmodel.model.vision_model
    mgvision = mgmodel.external_feature_model
    vision_hidden_size = args.vit_args.hidden_size
    vision_num_query_groups = args.vit_args.num_query_groups
    vision_head_dim = vision_hidden_size // args.vit_args.num_attention_heads

    kv_channels = 64
    hidden_dim = 1024
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels * 3
        ub = lb + kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
    for i in range(num_heads):
        lb = i * kv_channels * 3 + kv_channels * 1
        ub = lb + kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
    for i in range(num_heads):
        lb = i * kv_channels * 3 + kv_channels * 2
        ub = lb + kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))

    indices = torch.cat(indices)

    copied_numel = 0
    # safe_copy(mgvision.rotary_pos_emb.inv_freq, hfvision.rotary_pos_emb.inv_freq)
    # copied_numel += safe_copy(mgvision.patch_embed.proj.weight, hfvision.patch_embed.proj.weight)
    copied_numel += safe_copy(mgvision.vit.position_embeddings.weight.unsqueeze(0), hfvision.embeddings.position_embedding)
    copied_numel += safe_copy(mgvision.vit.class_token, hfvision.embeddings.class_embedding)
    copied_numel += safe_copy(mgvision.vit.conv1.weight, hfvision.embeddings.patch_embedding.weight)
    copied_numel += safe_copy(mgvision.vit.conv1.bias, hfvision.embeddings.patch_embedding.bias)
    for hfblock, mgblock in zip(hfvision.encoder.layers, mgvision.vit.decoder.layers):
        # linear_qkv.norm --> norm1
        copied_numel += safe_copy(mgblock.input_layernorm.weight, hfblock.norm1.weight)
        copied_numel += safe_copy(mgblock.input_layernorm.bias, hfblock.norm1.bias)
        # mlp.linear_fc1.norm --> norm2
        copied_numel += safe_copy(mgblock.pre_mlp_layernorm.weight, hfblock.norm2.weight)
        copied_numel += safe_copy(mgblock.pre_mlp_layernorm.bias, hfblock.norm2.bias)
        copied_numel += safe_copy(mgblock.ls1, hfblock.ls1)
        copied_numel += safe_copy(mgblock.ls2, hfblock.ls2)
        # self_attention.linear_qkv --> qkv
        # converted_weight = (
        #     mgblock.self_attention.linear_qkv.weight
        #     .view(vision_num_query_groups, 3, -1, vision_head_dim, vision_hidden_size)
        #     .transpose(0, 1)
        #     .reshape(-1, vision_hidden_size)
        #     .contiguous()
        # )
        # copied_numel += safe_copy(converted_weight, hfblock.attn.qkv.weight)
        copied_numel += safe_copy(mgblock.self_attention.linear_qkv.weight[indices], hfblock.attn.qkv.weight)
        # converted_bias = (
        #     mgblock.self_attention.linear_qkv.bias
        #     .view(vision_num_query_groups, 3, -1)
        #     .transpose(0, 1)
        #     .reshape(-1)
        #     .contiguous()
        # )
        # copied_numel += safe_copy(converted_bias, hfblock.attn.qkv.bias)
        copied_numel += safe_copy(mgblock.self_attention.linear_qkv.bias[indices], hfblock.attn.qkv.bias)
        # self_attention.linear_proj --> proj
        copied_numel += safe_copy(mgblock.self_attention.linear_proj.weight, hfblock.attn.proj.weight)
        copied_numel += safe_copy(mgblock.self_attention.linear_proj.bias, hfblock.attn.proj.bias)
        # mlp --> mlp: no gate
        copied_numel += safe_copy(mgblock.mlp.linear_fc1.weight, hfblock.mlp.fc1.weight)
        copied_numel += safe_copy(mgblock.mlp.linear_fc1.bias, hfblock.mlp.fc1.bias)
        copied_numel += safe_copy(mgblock.mlp.linear_fc2.weight, hfblock.mlp.fc2.weight)
        copied_numel += safe_copy(mgblock.mlp.linear_fc2.bias, hfblock.mlp.fc2.bias)        

    print("hfvision", sum([t.numel() for t in hfvision.state_dict().values() if isinstance(t, torch.Tensor)]))
    print("copied_numel", copied_numel)

    hfprojector = hfmodel.model.vision_projection
    mgprojector = mgvision.vision_projection
    copied_numel += safe_copy(mgvision.pre_proj_layernorm.weight, hfprojector.pre_proj_layernorm.weight)
    copied_numel += safe_copy(mgvision.pre_proj_layernorm.bias, hfprojector.pre_proj_layernorm.bias)   

    copied_numel += safe_copy(mgprojector.encoder.linear_fc1.weight, hfprojector.mlp[0].weight)
    # copied_numel += safe_copy(mgprojector.encoder.linear_fc1.bias, hfprojector.mlp[0].bias)
    copied_numel += safe_copy(mgprojector.encoder.linear_fc2.weight, hfprojector.mlp[2].weight)
    # copied_numel += safe_copy(mgprojector.encoder.linear_fc2.bias, hfprojector.mlp[2].bias)
    n_params = sum([t.numel() for t in mgvision.state_dict().values() if isinstance(t, torch.Tensor)])
    print(n_params, copied_numel)
    assert n_params == copied_numel

    print("hfvision + hfprojector", sum([t.numel() for t in hfvision.state_dict().values() if isinstance(t, torch.Tensor)]) + sum([t.numel() for t in hfprojector.state_dict().values() if isinstance(t, torch.Tensor)]))
    print("copied_numel", copied_numel)

    # 3. llm [just Qwen2]
    hfllm = hfmodel.model
    mgllm = mgmodel
    # copied_numel = 0
    copied_numel += safe_copy(mgllm.embedding.word_embeddings.weight, hfllm.embed_tokens.weight)
    for mglayer, hflayer in zip(mgllm.decoder.layers, hfllm.layers):
        # print(copied_numel)
        copied_numel += safe_copy(mglayer.self_attention.linear_qkv.layer_norm_weight, hflayer.input_layernorm.weight)
        
        qkv_weight = mglayer.self_attention.linear_qkv.weight.view(num_query_groups, -1, head_dim, hidden_size)
        q_weight, k_weight, v_weight = torch.split(qkv_weight, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
        copied_numel += safe_copy(q_weight.reshape(-1, hidden_size), hflayer.self_attn.q_proj.weight)
        copied_numel += safe_copy(k_weight.reshape(-1, hidden_size), hflayer.self_attn.k_proj.weight)
        copied_numel += safe_copy(v_weight.reshape(-1, hidden_size), hflayer.self_attn.v_proj.weight)
        
        qkv_bias = mglayer.self_attention.linear_qkv.bias.view(num_query_groups, -1)
        q_bias, k_bias, v_bias = torch.split(qkv_bias, split_size_or_sections=[q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1)
        copied_numel += safe_copy(q_bias.contiguous().view(-1), hflayer.self_attn.q_proj.bias)
        copied_numel += safe_copy(k_bias.contiguous().view(-1), hflayer.self_attn.k_proj.bias)
        copied_numel += safe_copy(v_bias.contiguous().view(-1), hflayer.self_attn.v_proj.bias)

        copied_numel += safe_copy(mglayer.self_attention.linear_proj.weight, hflayer.self_attn.o_proj.weight)

        gate_weight, fc1_weight = torch.split(mglayer.mlp.linear_fc1.weight, split_size_or_sections=args.ffn_hidden_size)
        copied_numel += safe_copy(gate_weight, hflayer.mlp.gate_proj.weight)
        copied_numel += safe_copy(fc1_weight, hflayer.mlp.up_proj.weight)
        copied_numel += safe_copy(mglayer.mlp.linear_fc2.weight, hflayer.mlp.down_proj.weight)

        copied_numel += safe_copy(mglayer.mlp.linear_fc1.layer_norm_weight, hflayer.post_attention_layernorm.weight)

    copied_numel += safe_copy(mgllm.decoder.final_layernorm.weight, hfllm.norm.weight)
    if args.untie_embeddings_and_output_weights:
        safe_copy(mgllm.output_layer.weight, hfmodel.lm_head.weight)
    
    n_params = sum([t.numel() for t in hfllm.state_dict().values() if isinstance(t, torch.Tensor)])
    print(n_params, copied_numel)
    print(n_params - copied_numel)
    assert n_params == copied_numel

@torch.inference_mode()
def convert_checkpoint_from_transformers_to_megatron(hfmodel, mgmodel, args):
    if args.fp16:
        mgmodel = mgmodel.half()
        hfmodel = hfmodel.half()
    elif args.bf16:
        mgmodel = mgmodel.bfloat16()
        hfmodel = hfmodel.bfloat16()

    # assert args.num_query_groups >= args.target_tensor_model_parallel_size

    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    hidden_size = args.hidden_size
    head_dim = hidden_size // num_attention_heads

    # 1. vision model
    hfvision = hfmodel.visual
    mgvision = mgmodel.vision_model
    vision_hidden_size = mgvision.config.hidden_size
    vision_num_query_groups = mgvision.config.num_query_groups
    vision_head_dim = vision_hidden_size // mgvision.config.num_attention_heads
    copied_numel = 0
    safe_copy(hfvision.rotary_pos_emb.inv_freq, mgvision.rotary_pos_emb.inv_freq)
    copied_numel += safe_copy(hfvision.patch_embed.proj.weight, mgvision.patch_embed.proj.weight)
    for hfblock, mgblock in zip(hfvision.blocks, mgvision.decoder.layers):
        # norm1 --> linear_qkv.norm
        copied_numel += safe_copy(hfblock.norm1.weight, mgblock.self_attention.linear_qkv.layer_norm_weight)
        copied_numel += safe_copy(hfblock.norm1.bias, mgblock.self_attention.linear_qkv.layer_norm_bias)
        # norm2 --> mlp.linear_fc1.norm
        copied_numel += safe_copy(hfblock.norm2.weight, mgblock.mlp.linear_fc1.layer_norm_weight)
        copied_numel += safe_copy(hfblock.norm2.bias, mgblock.mlp.linear_fc1.layer_norm_bias)       
        # qkv --> self_attention.linear_qkv
        converted_weight = (
            hfblock.attn.qkv.weight
            .view(3, vision_num_query_groups, -1, vision_head_dim, vision_hidden_size)
            .transpose(0, 1)
            .flatten(1, 2)
            .reshape(-1, vision_hidden_size)
            .contiguous()
        )
        copied_numel += safe_copy(converted_weight, mgblock.self_attention.linear_qkv.weight)
        converted_bias = (
            hfblock.attn.qkv.bias
            .view(3, vision_num_query_groups, -1)
            .transpose(0, 1)
            .flatten(1, 2)
            .view(-1)
            .contiguous()
        )
        copied_numel += safe_copy(converted_bias, mgblock.self_attention.linear_qkv.bias)
        # proj --> self_attention.linear_proj
        copied_numel += safe_copy(hfblock.attn.proj.weight, mgblock.self_attention.linear_proj.weight)
        copied_numel += safe_copy(hfblock.attn.proj.bias, mgblock.self_attention.linear_proj.bias)
        # mlp --> mlp: no gate
        copied_numel += safe_copy(hfblock.mlp.fc1.weight, mgblock.mlp.linear_fc1.weight)
        copied_numel += safe_copy(hfblock.mlp.fc1.bias, mgblock.mlp.linear_fc1.bias)
        copied_numel += safe_copy(hfblock.mlp.fc2.weight, mgblock.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hfblock.mlp.fc2.bias, mgblock.mlp.linear_fc2.bias)        

    # 2. vision projector
    hfprojector = hfvision.merger
    mgprojector = mgvision.projection
    copied_numel += safe_copy(hfprojector.ln_q.weight, mgvision.decoder.final_layernorm.weight)
    copied_numel += safe_copy(hfprojector.ln_q.bias, mgvision.decoder.final_layernorm.bias)   

    copied_numel += safe_copy(hfprojector.mlp[0].weight, mgprojector.encoder.linear_fc1.weight)
    copied_numel += safe_copy(hfprojector.mlp[0].bias, mgprojector.encoder.linear_fc1.bias)
    copied_numel += safe_copy(hfprojector.mlp[2].weight, mgprojector.encoder.linear_fc2.weight)
    copied_numel += safe_copy(hfprojector.mlp[2].bias, mgprojector.encoder.linear_fc2.bias)
    n_params = sum([t.numel() for t in hfvision.state_dict().values()])
    assert n_params == copied_numel

    # 3. llm [just Qwen2]
    hfllm = hfmodel.model
    mgllm = mgmodel.language_model
    copied_numel = 0
    copied_numel += safe_copy(hfllm.embed_tokens.weight, mgllm.embedding.word_embeddings.weight)
    for mglayer, hflayer in zip(mgllm.decoder.layers, hfllm.layers):
        copied_numel += safe_copy(hflayer.input_layernorm.weight, mglayer.self_attention.linear_qkv.layer_norm_weight)
        
        q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, hidden_size).contiguous()
        copied_numel += safe_copy(qkv_proj, mglayer.self_attention.linear_qkv.weight)

        q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
        k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
        v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
        qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
        copied_numel += safe_copy(qkv_bias, mglayer.self_attention.linear_qkv.bias)
        copied_numel += safe_copy(hflayer.self_attn.o_proj.weight, mglayer.self_attention.linear_proj.weight)

        fc1_weight = torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight])
        copied_numel += safe_copy(fc1_weight, mglayer.mlp.linear_fc1.weight)

        copied_numel += safe_copy(hflayer.mlp.down_proj.weight, mglayer.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hflayer.post_attention_layernorm.weight, mglayer.mlp.linear_fc1.layer_norm_weight)

    copied_numel += safe_copy(hfllm.norm.weight, mgllm.decoder.final_layernorm.weight)
    if args.untie_embeddings_and_output_weights:
        safe_copy(hfmodel.lm_head.weight, mgllm.output_layer.weight)
    
    n_params = sum([t.numel() for t in hfllm.state_dict().values()])
    assert n_params == copied_numel

def check_layer(layers_to_copy, k):
    if 'vision_model' in k:
        return False
    pattern = re.compile(r"decoder.layers.(\d+)")
    res = pattern.findall(k)
    return res and int(res[0]) in layers_to_copy.keys()

def split_vision_model(mgvision, args, prefix="vision_model") -> Dict[Tuple, Dict]:
    state_dicts = {}
    tp = args.tensor_model_parallel_size
    ENCODER_NUM_ATTENTION_HEADS = 16

    full_model = mgvision.state_dict_for_save_checkpoint()
    num_query_groups = mgvision.config.num_query_groups
    group_per_split = num_query_groups // tp

    vision_hidden_size = mgvision.config.hidden_size
    vision_ffn_hidden_size = mgvision.config.ffn_hidden_size
    head_dim = vision_hidden_size // ENCODER_NUM_ATTENTION_HEADS
    assert ENCODER_NUM_ATTENTION_HEADS % tp == 0
    # split model with ETP
    for etp_rank in range(tp):
        d = {}
        for k, v in full_model.items():
            if not isinstance(v, torch.Tensor):
                target_v = v
            elif 'patch_embed' in k:
                target_v = v
            elif 'linear_qkv.weight' in k:
                viewed = v.view(num_query_groups, -1, head_dim, vision_hidden_size)
                viewed = viewed[group_per_split*etp_rank : group_per_split*(etp_rank + 1)]
                target_v = viewed.view(-1, vision_hidden_size)
            elif 'linear_qkv.bias' in k:
                viewed = v.view(num_query_groups, -1, head_dim)
                viewed = viewed[group_per_split * etp_rank: group_per_split * (etp_rank + 1)]
                target_v = viewed.view(-1)
            elif ('linear_proj' in k or 'linear_fc2' in k) and 'bias' not in k:
                seg = v.shape[-1] // tp
                target_v = v[..., seg*etp_rank : seg*(etp_rank + 1)]
            elif 'linear_fc1.weight' in k and 'projection' not in k:
                viewed = v.view(-1, vision_ffn_hidden_size, vision_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1), :].reshape(-1, vision_hidden_size)
            elif 'linear_fc1.weight' in k:
                viewed = v.view(-1, vision_ffn_hidden_size, vision_ffn_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1), :].reshape(-1, vision_ffn_hidden_size)                    
            elif 'linear_fc1.bias' in k:
                viewed = v.view(-1, vision_ffn_hidden_size)
                seg = vision_ffn_hidden_size // tp
                target_v = viewed[:, seg*etp_rank: seg*(etp_rank+1)].flatten()
            else:
                target_v = v                
            d[prefix + '.' + k] = target_v
        state_dicts[(etp_rank, 0)] = d
    return state_dicts

def load_split_state_dict_to_vision_model(state_dicts, mgvision, args):

    tp = args.tensor_model_parallel_size
    ENCODER_NUM_ATTENTION_HEADS = args.vit_args.num_attention_heads

    num_query_groups = args.vit_args.num_query_groups
    group_per_split = num_query_groups // tp

    vision_hidden_size = args.vit_args.hidden_size
    vision_ffn_hidden_size = args.vit_args.ffn_hidden_size
    head_dim = vision_hidden_size // ENCODER_NUM_ATTENTION_HEADS

    merged_dict = defaultdict(list)
    # merge model by etp
    for etp_rank in range(tp):
        d = state_dicts[(etp_rank, 0)]
        for k, v in d.items():
            # if v is None:
            #     print(f"{k} {v}")
            # else:
            #     print(f"{k} {v.size()}")
            # NOTE: remove prefix
            k = '.'.join(k.split('.')[1:])
            if not isinstance(v, torch.Tensor):
                if etp_rank == 0:
                    merged_dict[k].append(v)
            elif 'patch_embed' in k:
                if etp_rank == 0:
                    merged_dict[k].append(v)
            elif 'linear_qkv.weight' in k:
                merged_dict[k].append(v.view(group_per_split, -1, head_dim, vision_hidden_size))
            elif 'linear_qkv.bias' in k:
                merged_dict[k].append(v.view(group_per_split, -1, head_dim))
            elif ('linear_proj' in k or 'linear_fc2' in k) and 'bias' not in k:
                merged_dict[k].append(v)
            elif 'linear_fc1.weight' in k and 'projection' not in k:
                seg = vision_ffn_hidden_size // tp
                merged_dict[k].append(v.view(-1, seg, vision_hidden_size))
            elif 'linear_fc1.weight' in k and 'projection' in k:
                seg = vision_hidden_size // tp
                merged_dict[k].append(v.view(-1, seg, vision_hidden_size * 4))  
            elif 'linear_fc1.bias' in k:
                seg = vision_ffn_hidden_size // tp
                merged_dict[k].append(v.view(-1, seg))  
            elif etp_rank == 0:
                merged_dict[k].append(v)

    for k, v in merged_dict.items():
        if not isinstance(v[0], torch.Tensor):
            merged_dict[k] = v[0]
        elif 'patch_embed' in k:
            merged_dict[k] = v[0]
        elif 'linear_qkv.weight' in k:
            merged_dict[k] = torch.cat(v, dim=0).view(-1, vision_hidden_size)
        elif 'linear_qkv.bias' in k:
            merged_dict[k] = torch.cat(v, dim=0).view(-1)
        elif ('linear_proj' in k or 'linear_fc2' in k) and 'bias' not in k:
            merged_dict[k] = torch.cat(v, dim=-1)
        elif 'linear_fc1.weight' in k and 'projection' not in k:
            merged_dict[k] = torch.cat(v, dim=1).view(-1, vision_hidden_size)
        elif 'linear_fc1.weight' in k and 'projection' in k:
            merged_dict[k] = torch.cat(v, dim=1).view(-1, vision_ffn_hidden_size)
        elif 'linear_fc1.bias' in k:
            merged_dict[k] = torch.cat(v, dim=1).view(-1)
        else:
            merged_dict[k] = v[0]

    # print(f"mgvision {mgvision}")
    # for k, v in merged_dict.items():
    #     if v is None:
    #         print(f"{k} {v}")
    #     else:
    #         print(f"{k} {v.size()}")

    # mgvision.load_state_dict(merged_dict, strict=False)
    mgvision.load_state_dict(merged_dict, strict=True)
    print("Load vision model finish")

def save_mgmodel(mgmodel, args):
    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    vpp_size = 1 # NOTE: vpp_size=1 if vpp is not used
    if args.target_num_layers_per_virtual_pipeline_stage is not None:
        args.num_layers_per_virtual_pipeline_stage = args.target_num_layers_per_virtual_pipeline_stage
        num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
            args.num_layers_per_virtual_pipeline_stage
        vpp_size = args.virtual_pipeline_model_parallel_size

    os.makedirs(args.save, exist_ok=True)

    tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
    with open(tracker_filepath, "w") as f:
        f.write("release")

    head_dim = args.hidden_size // args.num_attention_heads
    group_per_split = args.num_query_groups // args.target_tensor_model_parallel_size
    full_model = mgmodel.state_dict_for_save_checkpoint()
    
    for k in list(full_model.keys()):
        if full_model[k] is None:
            full_model.pop(k)

    if (
        args.tensor_model_parallel_size == 1
        and args.pipeline_model_parallel_size == 1
    ):
        checkpoint_name = get_checkpoint_name(args.save, 0, True)
        save_state_dict(args, full_model, checkpoint_name)
    elif (
        args.tensor_model_parallel_size > 1
        and args.pipeline_model_parallel_size == 1
    ):
        vision_state_dicts = split_vision_model(mgmodel.vision_model, args)
        for tp_rank in range(args.tensor_model_parallel_size):
            model_part = {}
            checkpoint_name = get_checkpoint_name(args.save, 0, True, None, tp_rank)
            print(f'tensor_parallel, save model to {checkpoint_name}')
            for k, v in full_model.items():
                if not isinstance(v, torch.Tensor):
                    target_v = v
                elif 'vision_model' in k:
                    vision_part = vision_state_dicts[(tp_rank, 0)]
                    assert k in vision_part, f"Cannot find key {k} in vision model split!"
                    target_v = vision_part[k]
                elif 'linear_qkv.weight' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                    viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                    target_v = viewed.view(-1, args.hidden_size)
                elif 'linear_qkv.bias' in k:
                    viewed = v.view(args.num_query_groups, -1, head_dim)
                    viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                    target_v = viewed.view(-1)
                elif 'linear_proj' in k or 'linear_fc2' in k:
                    seg = v.shape[1] // args.tensor_model_parallel_size
                    target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                elif 'embedding' in k or 'output_layer' in k:
                    seg = v.shape[0] // args.tensor_model_parallel_size
                    target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                elif 'linear_fc1' in k and 'norm' not in k:
                    viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                    seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                    target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                else:
                    target_v = v
                model_part[k] = target_v
            save_state_dict(args, [model_part], checkpoint_name)
    elif (
        args.pipeline_model_parallel_size > 1
    ):
        vision_state_dicts = split_vision_model(mgmodel.vision_model, args)
        ltog, _ = build_layer_id_mapping(args)

        for tp_rank in range(args.tensor_model_parallel_size):
            for pp_rank in range(args.pipeline_model_parallel_size):
                model_chunk = []
                checkpoint_name = get_checkpoint_name(args.save, 0, True, True, tp_rank, pp_rank)
                print(f'tensor_parallel & pipeline_parallel, save model to {checkpoint_name}')
                for vpp_id in range(vpp_size):
                    layers_to_copy = {}
                    local_id = 0
                    while (pp_rank, vpp_id, local_id) in ltog:
                        gloabl_layer_id = ltog[(pp_rank, vpp_id, local_id)]
                        layers_to_copy[gloabl_layer_id] = local_id
                        local_id += 1
                    model_part = {}
                    for k, v in full_model.items():
                        if check_layer(layers_to_copy, k):
                            pattern = re.compile(r'\d+')
                            res = pattern.findall(k)
                            k = re.sub(r"decoder.layers.\d+", f"decoder.layers.{layers_to_copy[int(res[0])]}", k)
                        elif not ("word_embeddings" in k or "output_layer" in k or "final_layernorm" in k or 'vision_model' in k):
                            continue
                        if 'vision_model' in k:
                            if pp_rank > 0  or vpp_id > 0:
                                # NOTE: The vision model will only be attached to the first model_part of pp stage 0
                                continue
                            vision_part = vision_state_dicts[(tp_rank, 0)]
                            assert k in vision_part, f"Cannot find key {k} in vision model split!"
                            target_v = vision_part[k]
                        elif not isinstance(v, torch.Tensor):
                            target_v = v
                        elif 'linear_qkv.weight' in k:
                            viewed = v.view(args.num_query_groups, -1, head_dim, args.hidden_size)
                            viewed = viewed[group_per_split*tp_rank : group_per_split*(tp_rank + 1)]
                            target_v = viewed.view(-1, args.hidden_size)
                        elif 'linear_qkv.bias' in k:
                            viewed = v.view(args.num_query_groups, -1, head_dim)
                            viewed = viewed[group_per_split * tp_rank: group_per_split * (tp_rank + 1)]
                            target_v = viewed.view(-1)
                        elif 'linear_proj' in k or 'linear_fc2' in k:
                            seg = v.shape[1] // args.tensor_model_parallel_size
                            target_v = v[:, seg*tp_rank : seg*(tp_rank + 1)]
                        elif 'embedding' in k or 'output_layer' in k:
                            seg = v.shape[0] // args.tensor_model_parallel_size
                            target_v = v[seg*tp_rank : seg*(tp_rank + 1)]
                        elif 'linear_fc1' in k and 'norm' not in k:
                            viewed = v.view(-1, args.ffn_hidden_size, args.hidden_size)
                            seg = args.ffn_hidden_size // args.tensor_model_parallel_size
                            target_v = viewed[:, seg*tp_rank: seg*(tp_rank+1), :].reshape(-1, args.hidden_size)
                        else:
                            target_v = v
                        if "word_embeddings" in k:
                            if pp_rank == 0 and vpp_id == 0:
                                model_part[k] = target_v
                        elif 'vision_model' not in k and ("output_layer" in k or "final_layernorm" in k):
                            if pp_rank == args.pipeline_model_parallel_size - 1 and vpp_id == vpp_size - 1:
                                model_part[k] = target_v
                        else:
                            model_part[k] = target_v
                    model_chunk.append(model_part)
                save_state_dict(args, model_chunk, checkpoint_name, args.target_num_layers_per_virtual_pipeline_stage is not None)
    else:
        raise ValueError(f'Got invalid TP/PP: {args.tensor_model_parallel_size}/{args.pipeline_model_parallel_size}')

    print(f'megatron model is save to {args.save}')



def add_extra_args(parser):
    parser = extra_args_provider(parser)
    parser = add_model_args(parser)
    return parser

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        config_kwargs = {
        }
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_ckpt_path, config=config, torch_dtype=config.torch_dtype)
        mg_model = load_megatron_model(args)
        convert_checkpoint_from_megatron_to_transformers(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)

        from long_vita.tokenizer import update_tokenizer
        from megatron.training import get_tokenizer
        tokenizer = get_tokenizer().tokenizer
        tokenizer = update_tokenizer(tokenizer)
        print(f"tokenizer {tokenizer}")
        tokenizer.save_pretrained(args.save)
    else:
        assert NotImplementedError
        config = AutoConfig.from_pretrained(args.load)
        hf_model = LongVITAForCausalLM.from_pretrained(args.load, torch_dtype=config.torch_dtype)
        mg_model = model_provider()
        convert_checkpoint_from_transformers_to_megatron(hf_model, mg_model, args)
        save_mgmodel(mg_model, args)

if __name__ == "__main__":
    main()
