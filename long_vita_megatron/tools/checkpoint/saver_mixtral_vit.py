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
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import os
import sys
import logging
from tqdm import tqdm

import torch
import torch_npu

logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = '1'


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
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--save-model-type', type=str, default='megatron',
                       help='Save model type')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, '
                                'seperated by comma; e.g., 4,4,4,4')


def save_huggingface(args, margs, model):
    import shutil
    from transformers import AutoModelForCausalLM

    # Create Huggingface model.
    hf_model = AutoModelForCausalLM.from_pretrained(args.save_dir, device_map="cpu", torch_dtype=torch.float16)

    # Set embedding params.
    hf_model.model.embed_tokens.weight.data.copy_(
        model.language_model.embedding.word_embeddings.weight)

    # Set output layer & norm params.
    hf_model.model.norm.weight.data.copy_(model.language_model.encoder.final_norm.weight)
    hf_model.lm_head.weight.data.copy_(model.language_model.output_layer.weight)

    for layer_idx in tqdm(range(margs.num_layers), "set layer states"):
        # Set self-attention params.
        layer = model.language_model.encoder.layers[layer_idx]
        hf_layer = hf_model.model.layers[layer_idx]

        # Get attention layer & state.
        attn = layer.self_attention
        hf_attn = hf_layer.self_attn

        # Reshape loaded weights.
        nh = margs.num_attention_heads
        ng = margs.num_query_groups if margs.group_query_attention else margs.num_attention_heads
        dim = margs.kv_channels

        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")

        # Set layer norm params.
        hf_layer.input_layernorm.weight.data.copy_(layer.input_norm.weight)
        hf_layer.post_attention_layernorm.weight.data.copy_(layer.post_attention_norm.weight)

        # Copy weights (re-order dimensions for Megatron).
        query_key_value = attn.query_key_value.weight.reshape((ng, -1, margs.hidden_size))
        hf_attn.q_proj.weight.data.copy_(
            query_key_value[:, :dim * nh // ng, ...].reshape((-1, margs.hidden_size)))
        hf_attn.k_proj.weight.data.copy_(
            query_key_value[:, dim * nh // ng: dim * nh // ng + dim, ...].reshape((-1, margs.hidden_size)))
        hf_attn.v_proj.weight.data.copy_(
            query_key_value[:, dim * nh // ng + dim:, ...].reshape((-1, margs.hidden_size)))
        hf_attn.o_proj.weight.data.copy_(attn.dense.weight)

        # Set Mixtral MoE params.
        hf_layer.block_sparse_moe.gate.weight.data.copy_(layer.mlp.block.moe_layer.gate.weight.weight)
        for idx in range(margs.num_experts):
            w1 = hf_layer.block_sparse_moe.experts[idx].w1.weight.data.copy_
            w2 = hf_layer.block_sparse_moe.experts[idx].w2.weight.data.copy_
            w3 = hf_layer.block_sparse_moe.experts[idx].w3.weight.data.copy_

            mg_w1_weight = layer.mlp.block.moe_layer.experts.experts[idx].w1.weight
            mg_w2_weight = layer.mlp.block.moe_layer.experts.experts[idx].w2.weight
            mg_w3_weight = layer.mlp.block.moe_layer.experts.experts[idx].w3.weight

            w1(mg_w1_weight)
            w2(mg_w2_weight)
            w3(mg_w3_weight)

    save_dir = os.path.join(args.save_dir, 'mg2hg')
    os.mkdir(save_dir)

    for file in os.listdir(args.save_dir):
        if all([not file.endswith(".safetensors"),
                not file.endswith(".pt"),
                os.path.isfile(os.path.join(args.save_dir, file))]):
            shutil.copy(os.path.join(args.save_dir, file), os.path.join(args.save_dir, 'mg2hg', file))

    logging.info('save weight to %s', save_dir)
    hf_model.save_pretrained(save_dir)


def save_model_checkpoint(queue, args):

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import modellink
        from megatron.training.arguments import validate_args
        from modellink.utils import parse_args
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError as e:
        logging.info("Unable to import Megatron, please specify the path to Megatron using --megatron-path.")
        raise e

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
            logging.info("Loader exited, exiting saver")
            raise ValueError
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            logging.info('Unexpected message. Expecting "%s" but got "%s". Exiting saver.', name, val_name)
            raise AttributeError
        if name is not None:
            logging.info("received %s", name)
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            logging.info("Unexpected values in %s:", msg_name)
            for key in msg.keys():
                logging.info("   %s", key)
            logging.info(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            raise AttributeError

    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            logging.info(
                "loader did not provide a tensor parallel size and --target-tensor-parallel-size "
                "not provided on command line. Default to 1."
            )
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            logging.info(
                "loader did not provide a pipeline parallel size and --target-pipeline-parallel-size "
                "not provided on command line. Default to 1."
            )
            args.target_pipeline_parallel_size = 1

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = str(
        args.target_tensor_parallel_size * args.target_pipeline_parallel_size * args.target_expert_parallel_size)

    # We want all arguments to come from us
    sys.argv = [
        'script.py',
        '--num-layers', str(md.num_layers),
        '--hidden-size', str(md.hidden_size),
        '--seq-length', str(md.seq_length),
        '--num-attention-heads', str(md.num_attention_heads),
        '--group-query-attention',
        '--num-query-groups', str(md.num_query_groups),
        '--max-position-embeddings', str(md.max_position_embeddings),
        '--position-embedding-type', str(md.position_embedding_type),
        '--tokenizer-type', str(md.tokenizer_type),
        '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
        '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
        '--expert-model-parallel-size', str(args.target_expert_parallel_size),
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--global-batch-size', str(args.target_expert_parallel_size),
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--save-interval', '1',
        '--save', args.save_dir,
        '--make-vocab-size-divisible-by', '1',
        '--sequence-parallel',
        '--bf16',
        '--num-layer-list', str(args.num_layer_list),
        '--vision-projector-type', 'mlp',
    ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    margs = parse_args()

    if hasattr(md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = [
            'tensor_model_parallel_size', 'pipeline_model_parallel_size', 'expert_model_parallel_size',
            'world_size', 'params_dtype', 'global_batch_size',
            'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
            'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
            'sequence_parallel', 'async_tensor_model_parallel_allreduce',
            'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
            'vocab_file', 'tokenizer_model', 'save_interval', 'save',
            'perform_initialization', 'use_cpu_initialization',
            'recompute_granularity', 'recompute_num_layers', 'recompute_method',
            'encoder_num_layers', 'encoder_seq_length',
            'distribute_saved_activations',
            'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
            'start_weight_decay', 'end_weight_decay'
        ]

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                logging.info("Checkpoint had argument %s but new arguments does not have this.", args)
                continue
            if getattr(margs, arg) != value:
                logging.info(
                    "Overwriting default %s value %s with value from checkpoint {value}.", args, getattr(margs, arg))
                setattr(margs, arg, value)

    validate_args(margs)
    margs.global_batch_size = args.target_expert_parallel_size

    set_global_variables(margs, build_tokenizer=False)

    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        logging.info(
            "Setting consumed_train_samples to %s and consumed_valid_samples to %s",
            margs.consumed_train_samples, margs.consumed_valid_samples
        )
    else:
        logging.info("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from long_vita_modellink.pretrain_long_vita import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    def get_models(args, dtype, pre_process, post_process):
        tp = args.target_tensor_parallel_size
        ep = args.target_expert_parallel_size
        return [[model_provider(pre_process, post_process).to(dtype) for _ in range(tp)] for _ in range(ep)]

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_expert_model_parallel_world_size(args.target_expert_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)

    # Embeddings
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # Deal with padding
    if md.true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

        # Cut out extra padding we don't need
        if orig_vocab_size > margs.padded_vocab_size:
            full_word_embed = orig_word_embed[0:margs.padded_vocab_size, :]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < margs.padded_vocab_size:
            padding_size = margs.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_word_embed
    else:
        logging.info(
            "Original vocab size not specified, leaving embedding table as-is. "
            "If you've changed the tensor parallel size this could cause problems."
        )
        margs.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = get_models(args, md.params_dtype, True, post_process)

    for ep_rank in range(args.target_expert_parallel_size):
        for tp_rank in range(args.target_tensor_parallel_size):
            models[ep_rank][tp_rank].language_model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            if pos_embed is not None:
                models[ep_rank][tp_rank].language_model.embedding.position_embeddings.weight.data.copy_(pos_embed)
            else:
                if hasattr(models[ep_rank][tp_rank].language_model.embedding, 'position_embeddings'):
                    raise ValueError("model should have position_embeddings")

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

    for ep_rank in range(args.target_expert_parallel_size):
        for tp_rank in range(args.target_tensor_parallel_size):
            models[ep_rank][tp_rank].language_model.external_feature_model.vit.embedding.word_embeddings.weight.data.copy_(vit_word_embed)
            assert models[ep_rank][tp_rank].language_model.external_feature_model.vit.embedding.word_embeddings.weight.data.size() == vit_word_embed.size()
            models[ep_rank][tp_rank].language_model.external_feature_model.vit.conv1.weight.data.copy_(vit_conv1_weight)
            assert models[ep_rank][tp_rank].language_model.external_feature_model.vit.conv1.weight.data.size() == vit_conv1_weight.size()
            models[ep_rank][tp_rank].language_model.external_feature_model.vit.conv1.bias.data.copy_(vit_conv1_bias)
            assert models[ep_rank][tp_rank].language_model.external_feature_model.vit.conv1.bias.data.size() == vit_conv1_bias.size()
            # models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.weight.data.copy_(vit_proj_weight)
            # assert models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.weight.data.size() == vit_proj_weight.size()
            # models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.bias.data.copy_(vit_proj_bias)
            # assert models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.bias.data.size() == vit_proj_bias.size()
            models[ep_rank][tp_rank].language_model.external_feature_model.vit.embedding.position_embeddings.weight.data.copy_(vit_pos_embed)
            assert models[ep_rank][tp_rank].language_model.external_feature_model.vit.embedding.position_embeddings.weight.data.size() == vit_pos_embed.size()

    vision_projector_msg = queue_get("vision projection")
    if margs.vision_projector_type == "affine":
        vit_proj_weight = vision_projector_msg.pop("proj weight")
        vit_proj_bias = vision_projector_msg.pop("proj bias")
    if margs.vision_projector_type == "mlp":
        vision_projector_fc1_weight = vision_projector_msg.pop("fc1 weight")
        vision_projector_fc2_weight = vision_projector_msg.pop("fc2 weight")
        vision_projector_fc1_weight = torch.chunk(vision_projector_fc1_weight, args.target_tensor_parallel_size, dim=0)
        vision_projector_fc2_weight = torch.chunk(vision_projector_fc2_weight, args.target_tensor_parallel_size, dim=1)
    for ep_rank in range(args.target_expert_parallel_size):
        for tp_rank in range(args.target_tensor_parallel_size):

            if margs.vision_projector_type == "affine":
                models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.weight.data.copy_(vit_proj_weight)
                assert models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.weight.data.size() == vit_proj_weight.size()
                models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.bias.data.copy_(vit_proj_bias)
                assert models[ep_rank][tp_rank].language_model.external_feature_model.linear_proj.bias.data.size() == vit_proj_bias.size()

            if margs.vision_projector_type == "mlp":
                models[ep_rank][tp_rank].language_model.external_feature_model.vision_projection.encoder.linear_fc1.weight.data.copy_(vision_projector_fc1_weight[tp_rank])
                assert models[ep_rank][tp_rank].language_model.external_feature_model.vision_projection.encoder.linear_fc1.weight.data.size() == vision_projector_fc1_weight[tp_rank].size()
                models[ep_rank][tp_rank].language_model.external_feature_model.vision_projection.encoder.linear_fc2.weight.data.copy_(vision_projector_fc2_weight[tp_rank])
                assert models[ep_rank][tp_rank].language_model.external_feature_model.vision_projection.encoder.linear_fc2.weight.data.size() == vision_projector_fc2_weight[tp_rank].size()

    # Transformer layers
    total_layer_num = 0
    for pp_rank in range(1):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args, md.params_dtype, False, post_process)

        for layer_id, _ in enumerate(models[0][0].language_model.external_feature_model.vit.encoder.layers):
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

            for ep_rank in range(args.target_expert_parallel_size):
                for tp_rank in range(args.target_tensor_parallel_size):
                    layer_chunk = models[ep_rank][tp_rank].language_model.external_feature_model.vit.encoder.layers[layer_id]

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
    total_layer_num = 0
    num_local_experts = md.num_experts // args.target_expert_parallel_size
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            post_process = pp_rank == args.target_pipeline_parallel_size - 1
            models = get_models(args, md.params_dtype, False, post_process)

        for layer_id, _ in enumerate(models[0][0].language_model.encoder.layers):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_gate_weight = msg.pop("mlp gate weight")

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                mlp_gate_bias = msg.pop("mlp gate bias")

            mlp_w1_weights = []
            mlp_w2_weights = []
            mlp_w3_weights = []
            mlp_w1_biases = []
            mlp_w2_biases = []
            mlp_w3_biases = []
            for expert_idx in range(md.num_experts):
                mlp_w1_weights.append(torch.chunk(
                    msg.pop(f"mlp {expert_idx} w1 weight"), args.target_tensor_parallel_size, dim=0))
                mlp_w2_weights.append(torch.chunk(
                    msg.pop(f"mlp {expert_idx} w2 weight"), args.target_tensor_parallel_size, dim=1))
                mlp_w3_weights.append(torch.chunk(
                    msg.pop(f"mlp {expert_idx} w3 weight"), args.target_tensor_parallel_size, dim=0))

                if md.linear_bias:
                    mlp_w1_biases.append(msg.pop(f"mlp {expert_idx} w1 bias"))
                    mlp_w2_biases.append(msg.pop(f"mlp {expert_idx} w2 bias"))
                    mlp_w3_biases.append(msg.pop(f"mlp {expert_idx} w3 bias"))

            for ep_rank in range(args.target_expert_parallel_size):
                for tp_rank in range(args.target_tensor_parallel_size):
                    layer_chunk = models[ep_rank][tp_rank].language_model.encoder.layers[layer_id]
                    for local_expert_idx in range(num_local_experts):
                        global_expert_idx = ep_rank * num_local_experts + local_expert_idx
                        w1_weight = getattr(
                            layer_chunk.mlp.block.moe_layer.experts.experts,
                            f"{local_expert_idx}").w1.weight.data.copy_
                        w2_weight = getattr(
                            layer_chunk.mlp.block.moe_layer.experts.experts,
                            f"{local_expert_idx}").w2.weight.data.copy_
                        w3_weight = getattr(
                            layer_chunk.mlp.block.moe_layer.experts.experts,
                            f"{local_expert_idx}").w3.weight.data.copy_

                        w1_weight(mlp_w1_weights[global_expert_idx][tp_rank])
                        w2_weight(mlp_w2_weights[global_expert_idx][tp_rank])
                        w3_weight(mlp_w3_weights[global_expert_idx][tp_rank])

                        if md.linear_bias:
                            w1_bias = getattr(
                                layer_chunk.mlp.block.moe_layer.experts.experts,
                                f"{local_expert_idx}").w1.bias.data.copy_
                            w2_bias = getattr(
                                layer_chunk.mlp.block.moe_layer.experts.experts,
                                f"{local_expert_idx}").w2.bias.data.copy_
                            w3_bias = getattr(
                                layer_chunk.mlp.block.moe_layer.experts.experts,
                                f"{local_expert_idx}").w3.bias.data.copy_

                            w1_bias(mlp_w1_biases[global_expert_idx])
                            w2_bias(mlp_w2_biases[global_expert_idx])
                            w3_bias(mlp_w3_biases[global_expert_idx])

            # Save them to the model
            for ep_rank in range(args.target_expert_parallel_size):
                for tp_rank in range(args.target_tensor_parallel_size):
                    layer_chunk = models[ep_rank][tp_rank].language_model.encoder.layers[layer_id]
                    layer_chunk.input_norm.weight.data.copy_(input_norm_weight)
                    if md.norm_has_bias:
                        layer_chunk.input_norm.bias.data.copy_(input_norm_bias)
                    layer_chunk.self_attention.query_key_value.weight.data.copy_(qkv_weight[tp_rank])
                    layer_chunk.self_attention.dense.weight.data.copy_(dense_weight[tp_rank])
                    layer_chunk.post_attention_norm.weight.data.copy_(post_norm_weight)
                    layer_chunk.mlp.block.moe_layer.gate.weight.weight.data.copy_(mlp_gate_weight)

                    if md.norm_has_bias:
                        layer_chunk.post_attention_norm.bias.data.copy_(post_norm_bias)

                    if md.linear_bias:
                        layer_chunk.self_attention.query_key_value.bias.data.copy_(qkv_bias[tp_rank])
                        layer_chunk.self_attention.dense.bias.data.copy_(dense_bias)
                        layer_chunk.mlp.block.moe_layer.gate.weight.bias.data.copy_(mlp_gate_bias)

            total_layer_num = total_layer_num + 1
            check_message(msg)

        if post_process:
            msg = queue_get("final norm")
            final_norm_weight = msg.pop("weight")
            if md.norm_has_bias:
                final_norm_bias = msg.pop("bias")
            for ep_rank in range(args.target_expert_parallel_size):
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[ep_rank][tp_rank].language_model.encoder.final_norm.weight.data.copy_(final_norm_weight)
                    if md.norm_has_bias:
                        models[ep_rank][tp_rank].language_model.encoder.final_norm.bias.data.copy_(final_norm_bias)
                    if pp_rank != 0 and not md.output_layer:
                        # Copy word embeddings to final pipeline rank
                        models[ep_rank][tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_norm_weight
            if md.norm_has_bias:
                del final_norm_bias
            check_message(msg)

            if md.output_layer:
                msg = queue_get("output layer")
                if not hasattr(models[0][0].language_model, 'output_layer'):
                    logging.info("ERROR: got an output layer, but model does not have one")
                    raise AttributeError
                output_layer_weight = torch.chunk(msg.pop("weight"), args.target_tensor_parallel_size, dim=0)
                for ep_rank in range(args.target_expert_parallel_size):
                    for tp_rank in range(args.target_tensor_parallel_size):
                        models[ep_rank][tp_rank].language_model.output_layer.weight.data.copy_(
                            output_layer_weight[tp_rank])
                del output_layer_weight
                check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                if not hasattr(models[0][0].language_model, 'pooler'):
                    logging.info("ERROR: got a pooler, but model does not have one")
                    raise AttributeError
                logging.info("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for ep_rank in range(args.target_expert_parallel_size):
                    for tp_rank in range(args.target_tensor_parallel_size):
                        models[ep_rank][tp_rank].language_model.pooler.dense.weight.data.copy_(pooler_weight)
                        models[ep_rank][tp_rank].language_model.pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[0][0], 'lm_head'):
                    logging.info("ERROR: got an lm head, but model does not have one")
                    raise AttributeError
                logging.info("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_norm_weight = msg.pop("norm weight")
                if md.norm_has_bias:
                    lm_head_norm_bias = msg.pop("norm bias")
                for ep_rank in range(args.target_expert_parallel_size):
                    for tp_rank in range(args.target_tensor_parallel_size):
                        models[ep_rank][tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                        models[ep_rank][tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                        models[ep_rank][tp_rank].lm_head.norm.weight.data.copy_(lm_head_norm_weight)
                        if md.norm_has_bias:
                            models[ep_rank][tp_rank].lm_head.norm.bias.data.copy_(lm_head_norm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    logging.info("ERROR: got a binary head, but model does not have one")
                    raise AttributeError
                logging.info("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for ep_rank in range(args.target_expert_parallel_size):
                    for tp_rank in range(args.target_tensor_parallel_size):
                        models[ep_rank][tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                        models[ep_rank][tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                logging.info("ERROR: got some more data but was expecting to be done")

        for ep_rank in range(args.target_expert_parallel_size):
            for tp_rank in range(args.target_tensor_parallel_size):
                mpu.set_tensor_model_parallel_rank(tp_rank)
                mpu.set_expert_model_parallel_rank(ep_rank)

                if args.save_model_type == 'megatron':
                    save_checkpoint(md.iteration, [models[ep_rank][tp_rank]], None, None, 0)
                elif args.save_model_type == 'huggingface':
                    save_huggingface(args, margs, models[ep_rank][tp_rank])

    logging.info("Done!")
