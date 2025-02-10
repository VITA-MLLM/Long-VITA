# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import logging as logger
import torch
from megatron.training.checkpointing import save_checkpoint
from megatron.core import mpu
from models import get_megatron_model

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int, default=1,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                            'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--save-model-type', type=str, default='megatron',
                       choices=['megatron', 'huggingface'], help='Save model type')
    group.add_argument("--w-pack", type=bool,
                       help='True is w_pack weight for llm',
                       default=False)
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, seperated by comma; e.g., 4,4,4,4')
    group.add_argument('--use-mcore-models', action='store_true',
                       help='Use the implementation from megatron core')
    group.add_argument('--moe-grouped-gemm', action='store_true',
                       help='Usr moe grouped gemm.')


def update_padded_vocab_size(md, model_mg, orig_tensor, orig_word_embed):
    # figure out what our padded vocab size is
    if md.true_vocab_size is not None:
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        margs = model_mg.get_args()
        padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)
        model_mg.set_padded_vocab_size(padded_vocab_size)
    else:
        logger.warning("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        model_mg.set_padded_vocab_size(orig_word_embed.shape[0])
    margs = model_mg.get_args()
    padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)
    model_mg.set_padded_vocab_size(padded_vocab_size)


def vocab_padding(orig_vocab_size, padded_vocab_size, orig_tensor):
    # figure out what our padded vocab size is

    # Cut out extra padding we don't need
    if orig_vocab_size > padded_vocab_size:
        full_word_embed = orig_tensor[0:padded_vocab_size, :]

    # Expanding embedding to larger size by replicating final entry
    elif orig_vocab_size < padded_vocab_size:
        padding_size = padded_vocab_size - orig_vocab_size

        full_word_embed = torch.cat((
            orig_tensor,
            orig_tensor[-1].unsqueeze(0).expand(padding_size, -1)))

    # Same size!
    else:
        full_word_embed = orig_tensor

    return full_word_embed


def reset_cmd_args_from_md(args, md):
    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            logger.warning("loader did not provide a tensor parallel size and "
                  "--target-tensor-parallel-size not provided on command line. Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            logger.warning(
                "loader did not provide a pipeline parallel size and "
                "--target-pipeline-parallel-size not provided on command line. Default to 1.")
            args.target_pipeline_parallel_size = 1


def set_model_preprocess(model, embeddings_msg):
    md = model.get_metadata()
    margs = model.get_args()
    pos_embed = None
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop(f"position embeddings")
    orig_word_embed = embeddings_msg.pop(f"word embeddings")
    orig_word_embed_n_w, orig_word_embed_n_b = None, None
    if "word embeddings norm_w" in embeddings_msg:
        orig_word_embed_n_w = embeddings_msg.pop(f"word embeddings norm_w")
        if "word embeddings norm_b" in embeddings_msg:
            orig_word_embed_n_b = embeddings_msg.pop(f"word embeddings norm_b")
    out_word_embed_list = []
    for ep_rank in range(ep_size):
        if md.true_vocab_size is not None:
            orig_vocab_size = orig_word_embed.shape[0]
            full_word_embed = vocab_padding(orig_vocab_size, margs.padded_vocab_size, orig_word_embed)
        else:
            full_word_embed = orig_word_embed

        # Split into new tensor model parallel sizes  tensor_model_parallel_size
        out_word_embed = torch.chunk(full_word_embed, margs.tensor_model_parallel_size, dim=0)
        for tp_rank in range(tp_size):
            model.set_embedding_word_embeddings_weight(ep_rank=ep_rank, tp_rank=tp_rank, data=out_word_embed[tp_rank])
            if orig_word_embed_n_w is not None:
                model.set_embedding_word_embeddings_norm_weight(ep_rank=ep_rank, tp_rank=tp_rank, data=orig_word_embed_n_w)
                if orig_word_embed_n_b is not None:
                    model.set_embedding_word_embeddings_norm_bias(ep_rank=ep_rank, tp_rank=tp_rank, data=orig_word_embed_n_b)
            if pos_embed is not None:
                model.set_embedding_position_embeddings_weight(ep_rank=ep_rank, tp_rank=tp_rank, data=pos_embed)
            else:
                if hasattr(model.get_embedding_module(), 'position_embeddings'):
                    raise ValueError("model should have position_embeddings")

        out_word_embed_list.append(out_word_embed)

    return out_word_embed_list


def set_model_layer_norm(model_mg, msg, md, **kwargs):
    # duplicated tensors
    input_norm_weight = msg.pop("input norm weight")
    post_norm_weight = msg.pop("post norm weight")
    input_norm_bias = None
    post_norm_bias = None
    if md.norm_has_bias:
        input_norm_bias = msg.pop("input norm bias")
    if md.norm_has_bias:
        post_norm_bias = msg.pop("post norm bias")

    margs = model_mg.get_args()

    post_norm = margs.post_norm
    if post_norm:
        pre_mlp_norm_weight = msg.pop("pre mlp norm weight")
        post_mlp_norm_weight = msg.pop("post mlp norm weight")
    # Save them to the model
    for ep_rank in range(margs.expert_model_parallel_size):
        kwargs["ep_rank"] = ep_rank
        for tp_rank in range(margs.tensor_model_parallel_size):
            kwargs["tp_rank"] = tp_rank

            model_mg.set_layers_input_layernorm_weight(**kwargs, data=input_norm_weight)
            if input_norm_bias is not None:
                model_mg.set_layers_input_layernorm_bias(**kwargs, data=input_norm_bias)
            model_mg.set_layers_self_attention_pre_mlp_layernorm_weight(**kwargs, data=post_norm_weight)
            if post_norm:
                model_mg.set_layers_self_attention_pre_mlp_layernorm_weight(**kwargs, data=pre_mlp_norm_weight)
                model_mg.set_layers_self_attention_post_attention_layernorm_weight(**kwargs, data=post_norm_weight)
                model_mg.set_layers_self_attention_post_mlp_layernorm_weight(**kwargs, data=post_mlp_norm_weight)
            if post_norm_bias is not None:
                model_mg.set_layers_self_attention_pre_mlp_layernorm_bias(**kwargs, data=post_norm_bias)


def set_model_layer_attn(model_mg, msg, md, **kwargs):
    # duplicated tensors
    margs = model_mg.get_args()
    if md.linear_bias or margs.add_dense_bias:
        dense_bias = msg.pop("dense bias")
    if md.linear_bias or margs.add_qkv_bias:
        qkv_bias = torch.chunk(msg.pop("qkv bias"), margs.tensor_model_parallel_size, dim=0)

    qkv_org = msg.pop("qkv weight")
    qkv_weight = torch.chunk(qkv_org, margs.tensor_model_parallel_size, dim=0)

    if getattr(md, "qk_layernorm", False):
        q_layernorm = msg.pop("q layernorm")
        k_layernorm = msg.pop("k layernorm")

    if getattr(md, "multi_head_latent_attention", False):
        linear_qb = msg.pop("linear qb weight")
        linear_kvb = msg.pop("linear kvb weight")

    # Split up the parallel tensors
    dense_weight = torch.chunk(msg.pop("dense weight"), margs.tensor_model_parallel_size, dim=1)

    # Save them to the model
    for ep_rank in range(margs.expert_model_parallel_size):
        kwargs["ep_rank"] = ep_rank
        for tp_rank in range(margs.tensor_model_parallel_size):
            kwargs["tp_rank"] = tp_rank
            model_mg.set_layers_self_attention_linear_qkv_weight(**kwargs, data=qkv_weight[tp_rank])
            model_mg.set_layers_self_attention_linear_proj_weight(**kwargs, data=dense_weight[tp_rank])
            
            if getattr(md, "qk_layernorm", False):
                model_mg.set_layers_self_attention_q_layernorm_weight(**kwargs, data=q_layernorm)
                model_mg.set_layers_self_attention_k_layernorm_weight(**kwargs, data=k_layernorm)

            if getattr(md, "multi_head_latent_attention", False):
                model_mg.set_layers_self_attention_linear_qb_weight(**kwargs, data=linear_qb)
                model_mg.set_layers_self_attention_linear_kvb_weight(**kwargs, data=linear_kvb)

            if md.linear_bias:
                model_mg.set_layers_self_attention_linear_qkv_bias(**kwargs, data=qkv_bias[tp_rank])
                model_mg.set_layers_self_attention_linear_proj_bias(**kwargs, data=dense_bias)

            if margs.add_qkv_bias:
                model_mg.set_layers_self_attention_linear_qkv_bias(**kwargs, data=qkv_bias[tp_rank])
            if margs.add_dense_bias:
                model_mg.set_layers_self_attention_linear_proj_bias(**kwargs, data=dense_bias)


def _set_set_model_layer_mlp(model_mg, msg, md, pop_flag=True, is_moe_mlp=False, **kwargs):
    margs = model_mg.get_args()
    func = msg.pop if pop_flag else msg.get
    num_experts_local = 1
    if margs.num_experts:
        num_experts_local = margs.num_experts // margs.expert_model_parallel_size
    # Save them to the model

    if md.linear_bias:
        mlp_l1_bias = func(f"mlp l1 bias")
    # Split up the parallel tensors
    mlp_l1_weight = torch.chunk(func(f"mlp l1 weight"), margs.tensor_model_parallel_size, dim=1)

    # Special handling for swiglu
    if md.swiglu:
        mlp_l0_weight_W = torch.chunk(func(f"mlp l0 weight W"), margs.tensor_model_parallel_size, dim=0)
        mlp_l0_weight_V = torch.chunk(func(f"mlp l0 weight V"), margs.tensor_model_parallel_size, dim=0)
        mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
    else:
        mlp_l0_weight = torch.chunk(func(f"mlp l0 weight"), margs.tensor_model_parallel_size, dim=0)
    if md.linear_bias:
        if md.swiglu:
            mlp_l0_bias_W = torch.chunk(func(f"mlp l0 bias W"), margs.tensor_model_parallel_size, dim=0)
            mlp_l0_bias_V = torch.chunk(func(f"mlp l0 bias V"), margs.tensor_model_parallel_size, dim=0)
            mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
        else:
            mlp_l0_bias = torch.chunk(func(f"mlp l0 bias"), margs.tensor_model_parallel_size, dim=0)

    # duplicated tensors
    for tp_rank in range(margs.tensor_model_parallel_size):
        kwargs["tp_rank"] = tp_rank
        if is_moe_mlp:
            model_mg.set_layers_mlp_experts_linear_fc1_weight(**kwargs, data=mlp_l0_weight[tp_rank])
            model_mg.set_layers_mlp_experts_linear_fc2_weight(**kwargs, data=mlp_l1_weight[tp_rank])
        else:
            model_mg.set_layers_mlp_linear_fc1_weight(**kwargs, data=mlp_l0_weight[tp_rank])
            model_mg.set_layers_mlp_linear_fc2_weight(**kwargs, data=mlp_l1_weight[tp_rank])

        if md.linear_bias:
            if is_moe_mlp:
                model_mg.set_layers_mlp_experts_linear_fc1_bias(**kwargs, data=mlp_l0_bias[tp_rank])
                model_mg.set_layers_mlp_experts_linear_fc2_bias(**kwargs, data=mlp_l1_bias)
            else:
                model_mg.set_layers_mlp_linear_fc1_bias(**kwargs, data=mlp_l0_bias[tp_rank])
                model_mg.set_layers_mlp_linear_fc2_bias(**kwargs, data=mlp_l1_bias)


def set_model_layer_mlp(model_mg, msg, md, total_layer_num, **kwargs):
    margs = model_mg.get_args()
    first_k_dense_replace = getattr(margs, 'first_k_dense_replace', None)
    moe_layer_freq = getattr(margs, 'moe_layer_freq', None)
    if (
            margs.num_experts
            and first_k_dense_replace is not None
            and moe_layer_freq is not None
    ):
        if total_layer_num >= first_k_dense_replace and total_layer_num % moe_layer_freq == 0:
            num_experts_local = margs.num_experts // margs.expert_model_parallel_size
            mlp_moe = msg.pop("mlp_moe")
            mlp_router_weight = mlp_moe.pop("mlp router weight")
            if getattr(margs, "n_shared_experts", None) is not None:
                shared_experts_linear_fc1_weight = mlp_moe.pop("mlp shared experts linear fc1 weight")
                shared_experts_linear_fc2_weight = mlp_moe.pop("mlp shared experts linear fc2 weight")
            if margs.moe_grouped_gemm:
                # TODO: check TP
                weight1 = torch.chunk(mlp_moe.pop("mlp experts weight1 module").view(margs.hidden_size, -1),
                                      margs.expert_model_parallel_size, dim=0)
                weight2 = torch.chunk(mlp_moe.pop("mlp experts weight2 module").view(-1, margs.hidden_size),
                                      margs.expert_model_parallel_size, dim=0)
            for ep_rank in range(margs.expert_model_parallel_size):
                kwargs["ep_rank"] = ep_rank
                for tp_rank in range(margs.tensor_model_parallel_size):
                    kwargs['tp_rank'] = tp_rank
                    model_mg.set_layers_mlp_router_weight(**kwargs, data=mlp_router_weight)
                    if getattr(margs, "n_shared_experts", None) is not None:
                        model_mg.set_layers_mlp_shared_experts_linear_fc1_weight(**kwargs,
                                                                                 data=shared_experts_linear_fc1_weight)
                        model_mg.set_layers_mlp_shared_experts_linear_fc2_weight(**kwargs,
                                                                                 data=shared_experts_linear_fc2_weight)
                if margs.moe_grouped_gemm:
                    # TODO: check TP
                    model_mg.set_layers_mlp_experts_weight1_module(**kwargs,
                                                                   data=weight1[ep_rank].view(margs.hidden_size, -1))
                    model_mg.set_layers_mlp_experts_weight2_module(**kwargs,
                                                                   data=weight2[ep_rank].view(-1, margs.hidden_size))
                else:
                    for expert_idx in range(num_experts_local):
                        kwargs["expert_idx"] = expert_idx
                        global_expert_idx = expert_idx + ep_rank * num_experts_local
                        expert = mlp_moe.pop(f"expert {global_expert_idx}")
                        _set_set_model_layer_mlp(model_mg, expert, md, is_moe_mlp=True, **kwargs)
        else:
            for ep_rank in range(margs.expert_model_parallel_size):
                kwargs["ep_rank"] = ep_rank
                pop_flag = ep_rank == margs.expert_model_parallel_size - 1
                _set_set_model_layer_mlp(model_mg, msg, md, pop_flag=pop_flag, **kwargs)
    elif margs.num_experts:
        num_experts_local = margs.num_experts // margs.expert_model_parallel_size
        mlp_moe = msg.pop("mlp_moe")
        mlp_router_weight = mlp_moe.pop("mlp router weight")
        for ep_rank in range(margs.expert_model_parallel_size):
            kwargs["ep_rank"] = ep_rank
            for tp_rank in range(margs.tensor_model_parallel_size):
                kwargs['tp_rank'] = tp_rank
                model_mg.set_layers_mlp_router_weight(**kwargs, data=mlp_router_weight)
            for expert_idx in range(num_experts_local):
                kwargs["expert_idx"] = expert_idx
                global_expert_idx = expert_idx + ep_rank * num_experts_local
                expert = mlp_moe.pop(f"expert {global_expert_idx}")
                _set_set_model_layer_mlp(model_mg, expert, md, **kwargs)
    else:
        _set_set_model_layer_mlp(model_mg, msg, md, **kwargs)


def set_model_postprocess(model_mg, msg, md, out_word_embed_list, **kwargs):
    margs = model_mg.get_args()
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    final_norm_weight = msg.pop(f"weight")
    final_norm_bias = None
    if md.norm_has_bias:
        final_norm_bias = msg.pop(f"bias")
    for ep_rank in range(ep_size):
        kwargs["ep_rank"] = ep_rank
        for tp_rank in range(tp_size):
            kwargs["tp_rank"] = tp_rank
            model_mg.set_final_layernorm_weight(**kwargs, data=final_norm_weight)
            if final_norm_bias is not None:
                model_mg.set_final_layernorm_bias(**kwargs, data=final_norm_bias)
            if kwargs.get("pp_rank", 0) != 0 and not md.output_layer:
                # Copy word embeddings to final pipeline rank
                model_mg.set_word_embeddings_weight(**kwargs, data=out_word_embed_list[ep_rank][tp_rank])
    del final_norm_weight
    if final_norm_bias is not None:
        del final_norm_bias


def set_model_output_layer(model_mg, msg, md, **kwargs):
    margs = model_mg.get_args()
    tp_size = margs.tensor_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    output_layer = msg.pop(f"weight")
    for ep_rank in range(ep_size):
        kwargs["ep_rank"] = ep_rank
        if md.true_vocab_size is not None:
            orig_vocab_size = output_layer.shape[0]
            full_word_embed = vocab_padding(orig_vocab_size, margs.padded_vocab_size, output_layer)
        else:
            full_word_embed = output_layer
        output_layer_weight = torch.chunk(full_word_embed, margs.tensor_model_parallel_size, dim=0)
        for tp_rank in range(tp_size):
            kwargs["tp_rank"] = tp_rank
            model_mg.set_output_layer_weight(**kwargs, data=output_layer_weight[tp_rank])


def save_model(model_mg, md, **kwargs):
    margs = model_mg.get_args()
    args_cmd = model_mg.get_args_cmd()
    virtual_pipeline_model_parallel_size = margs.virtual_pipeline_model_parallel_size
    if virtual_pipeline_model_parallel_size is None:
        virtual_pipeline_model_parallel_size = 1
    for ep_rank in range(margs.expert_model_parallel_size):
        model_mg.set_expert_model_parallel_rank(ep_rank)
        kwargs["ep_rank"] = ep_rank
        for tp_rank in range(margs.tensor_model_parallel_size):
            model_mg.set_tensor_model_parallel_rank(tp_rank)
            kwargs["tp_rank"] = tp_rank
            vp_models = []
            for vp_rank in range(virtual_pipeline_model_parallel_size):
                kwargs["vp_rank"] = vp_rank
                vp_models.append(model_mg.get_model_item(**kwargs))
            if args_cmd.save_model_type == 'megatron':
                # Split the PP into multiple VPPs and select the corresponding layers for each VPP by copying and deleting
                save_checkpoint(md.iteration, vp_models, None, None, 0)
            elif args_cmd.save_model_type == "huggingface":
                save_huggingface(args_cmd, model_mg)


def save_huggingface(args, model):
    '''Set model params.'''
    from models import get_huggingface_model
    model_hf = get_huggingface_model(args)
    model_hf.get_modules_from_pretrained()
    args_cmd = model_hf.get_args_cmd()

    model_hf.update_module(model)

    save_dir = os.path.join(args_cmd.save_dir, 'mg2hf')
    logger.info(f'save weight to {save_dir}')
    model_hf.get_model_item().save_pretrained(save_dir)


def save_model_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            logger.error("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            logger.error(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            logger.info(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            logger.error(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                logger.error(f"   {key}")
            logger.error(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    md = queue_get()
    reset_cmd_args_from_md(args, md)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    model_mg = get_megatron_model(args_cmd=args, md=md)
    model_mg.initialize_megatron_args(queue=queue, saver_megatron=True)

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    model_mg.get_modules_from_config(pp_stage_cache_flag=True)

    # Embeddings
    embeddings_msg = queue_get("embeddings")
    out_word_embed_list = set_model_preprocess(model_mg, embeddings_msg)
    check_message(embeddings_msg)
    margs = model_mg.get_args()

    logger.info(f"model_mg {dir(model_mg)}")
    logger.info(f"model_mg {model_mg.module}")
    # -------------------------------------------------
    # EVA ViT
    # Embeddings
    embeddings_msg = queue_get("vit embeddings")

    vit_pos_embed = embeddings_msg.pop("position embeddings")
    vit_word_embed = embeddings_msg.pop("word embeddings")
    vit_conv1_weight = embeddings_msg.pop("conv1 weight")
    vit_conv1_bias = embeddings_msg.pop("conv1 bias")
    # vit_proj_weight = embeddings_msg.pop("proj weight")
    # vit_proj_bias = embeddings_msg.pop("proj bias")
    check_message(embeddings_msg)

    for tp_rank in range(args.target_tensor_parallel_size):
        model_mg.module[0][0][tp_rank].external_feature_model.vit.embedding.word_embeddings.weight.data.copy_(vit_word_embed)
        assert model_mg.module[0][0][tp_rank].external_feature_model.vit.embedding.word_embeddings.weight.data.size() == vit_word_embed.size()

        model_mg.module[0][0][tp_rank].external_feature_model.vit.conv1.weight.data.copy_(vit_conv1_weight)
        assert model_mg.module[0][0][tp_rank].external_feature_model.vit.conv1.weight.data.size() == vit_conv1_weight.size()
        model_mg.module[0][0][tp_rank].external_feature_model.vit.conv1.bias.data.copy_(vit_conv1_bias)
        assert model_mg.module[0][0][tp_rank].external_feature_model.vit.conv1.bias.data.size() == vit_conv1_bias.size()

        # model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.weight.data.copy_(vit_proj_weight)
        # assert model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.weight.data.size() == vit_proj_weight.size()
        # model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.bias.data.copy_(vit_proj_bias)
        # assert model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.bias.data.size() == vit_proj_bias.size()

        model_mg.module[0][0][tp_rank].external_feature_model.vit.embedding.position_embeddings.weight.data.copy_(vit_pos_embed)
        assert model_mg.module[0][0][tp_rank].external_feature_model.vit.embedding.position_embeddings.weight.data.size() == vit_pos_embed.size()

    vision_projector_msg = queue_get("vision projection")
    if margs.vision_projector_type == "affine":
        vit_proj_weight = vision_projector_msg.pop("proj weight")
        vit_proj_bias = vision_projector_msg.pop("proj bias")
    if margs.vision_projector_type == "mlp":
        vision_projector_fc1_weight = vision_projector_msg.pop("fc1 weight")
        vision_projector_fc2_weight = vision_projector_msg.pop("fc2 weight")
        vision_projector_fc1_weight = torch.chunk(vision_projector_fc1_weight, args.target_tensor_parallel_size, dim=0)
        vision_projector_fc2_weight = torch.chunk(vision_projector_fc2_weight, args.target_tensor_parallel_size, dim=1)
    for tp_rank in range(args.target_tensor_parallel_size):

        if margs.vision_projector_type == "affine":
            model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.weight.data.copy_(vit_proj_weight)
            assert model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.weight.data.size() == vit_proj_weight.size()
            model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.bias.data.copy_(vit_proj_bias)
            assert model_mg.module[0][0][tp_rank].external_feature_model.linear_proj.bias.data.size() == vit_proj_bias.size()

        if margs.vision_projector_type == "mlp":
            model_mg.module[0][0][tp_rank].external_feature_model.vision_projection.encoder.linear_fc1.weight.data.copy_(vision_projector_fc1_weight[tp_rank])
            assert model_mg.module[0][0][tp_rank].external_feature_model.vision_projection.encoder.linear_fc1.weight.data.size() == vision_projector_fc1_weight[tp_rank].size()
            model_mg.module[0][0][tp_rank].external_feature_model.vision_projection.encoder.linear_fc2.weight.data.copy_(vision_projector_fc2_weight[tp_rank])
            assert model_mg.module[0][0][tp_rank].external_feature_model.vision_projection.encoder.linear_fc2.weight.data.size() == vision_projector_fc2_weight[tp_rank].size()

    # Transformer layers
    # -------------------
    total_layer_num = 0
    for pp_rank in range(1):
        for layer_id, _ in enumerate(model_mg.module[0][0][0].external_feature_model.vit.encoder.layers):
            msg = queue_get(f"vit transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            input_norm_bias = msg.pop("input norm bias")

            pre_norm_weight = msg.pop("pre norm weight")
            pre_norm_bias = msg.pop("pre norm bias")

            proj_bias = msg.pop("proj bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            proj_weight = torch.chunk(msg.pop("proj weight"), args.target_tensor_parallel_size, dim=1)

            qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)

            mlp_fc1_weight = msg.pop("mlp fc1 weight")
            mlp_fc2_weight = msg.pop("mlp fc2 weight")
            mlp_fc1_bias = msg.pop("mlp fc1 bias")
            mlp_fc2_bias = msg.pop("mlp fc2 bias")

            mlp_fc1_weight = torch.chunk(mlp_fc1_weight, args.target_tensor_parallel_size, dim=0)
            mlp_fc2_weight = torch.chunk(mlp_fc2_weight, args.target_tensor_parallel_size, dim=1)

            mlp_fc1_bias = torch.chunk(mlp_fc1_bias, args.target_tensor_parallel_size, dim=0)

            for tp_rank in range(args.target_tensor_parallel_size):
                layer_chunk = model_mg.module[0][0][tp_rank].external_feature_model.vit.encoder.layers[layer_id]

                layer_chunk.input_layernorm.weight.data.copy_(input_norm_weight)
                assert layer_chunk.input_layernorm.weight.data.size() == input_norm_weight.size()
                layer_chunk.pre_mlp_layernorm.weight.data.copy_(pre_norm_weight)
                assert layer_chunk.pre_mlp_layernorm.weight.data.size() == pre_norm_weight.size()

                layer_chunk.input_layernorm.bias.data.copy_(input_norm_bias)
                assert layer_chunk.input_layernorm.bias.data.size() == input_norm_bias.size()
                layer_chunk.pre_mlp_layernorm.bias.data.copy_(pre_norm_bias)
                assert layer_chunk.pre_mlp_layernorm.bias.data.size() == pre_norm_bias.size()

                layer_chunk.mlp.linear_fc1.weight.data.copy_(mlp_fc1_weight[tp_rank])
                assert layer_chunk.mlp.linear_fc1.weight.data.size() == mlp_fc1_weight[tp_rank].size()
                layer_chunk.mlp.linear_fc2.weight.data.copy_(mlp_fc2_weight[tp_rank])
                assert layer_chunk.mlp.linear_fc2.weight.data.size() == mlp_fc2_weight[tp_rank].size()

                layer_chunk.mlp.linear_fc1.bias.data.copy_(mlp_fc1_bias[tp_rank])
                assert layer_chunk.mlp.linear_fc1.bias.data.size() == mlp_fc1_bias[tp_rank].size()
                layer_chunk.mlp.linear_fc2.bias.data.copy_(mlp_fc2_bias)
                assert layer_chunk.mlp.linear_fc2.bias.data.size() == mlp_fc2_bias.size()

                layer_chunk.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
                assert layer_chunk.self_attention.linear_qkv.weight.data.size() == qkv_weight[tp_rank].size()
                layer_chunk.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
                assert layer_chunk.self_attention.linear_proj.weight.data.size() == proj_weight[tp_rank].size()

                layer_chunk.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
                assert layer_chunk.self_attention.linear_qkv.bias.data.size() == qkv_bias[tp_rank].size()
                layer_chunk.self_attention.linear_proj.bias.data.copy_(proj_bias)
                assert layer_chunk.self_attention.linear_proj.bias.data.size() == proj_bias.size()

            total_layer_num = total_layer_num + 1
            check_message(msg)

    # -------------------------------------------------

    # Transformer layers
    # -------------------
    total_layer_num = 0

    virtual_pipeline_model_parallel_size = margs.virtual_pipeline_model_parallel_size
    if virtual_pipeline_model_parallel_size is None:
        virtual_pipeline_model_parallel_size = 1

    for vp_rank in range(virtual_pipeline_model_parallel_size):
        model_mg.set_virtual_pipeline_model_parallel_rank(vp_rank)
        kwargs = {"vp_rank": vp_rank}
        for pp_rank in range(args.target_pipeline_parallel_size):
            # For later pipeline parallel ranks, make the new models
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            model_mg.get_modules_from_config(pp_stage_cache_flag=True)
            kwargs["pp_rank"] = pp_rank
            for layer in range(len(model_mg.get_layers_module())):
                kwargs["layer_idx"] = layer
                msg = queue_get(f"transformer layer {total_layer_num}")
                set_model_layer_norm(model_mg, msg, md, **kwargs)
                set_model_layer_attn(model_mg, msg, md, **kwargs)
                set_model_layer_mlp(model_mg, msg, md, total_layer_num, **kwargs)

                total_layer_num = total_layer_num + 1
                check_message(msg)

            post_process = (
                    (pp_rank == args.target_pipeline_parallel_size - 1) &
                    (vp_rank == virtual_pipeline_model_parallel_size - 1)
            )
            if post_process:
                msg = queue_get("final norm")
                set_model_postprocess(model_mg, msg, md, out_word_embed_list, **kwargs)
                check_message(msg)

                if md.output_layer:
                    msg = queue_get("output layer")
                    set_model_output_layer(model_mg, msg, md, **kwargs)
                    check_message(msg)

            if vp_rank == virtual_pipeline_model_parallel_size - 1:
                save_model(model_mg, md, **kwargs)
    logger.info("Done!")
