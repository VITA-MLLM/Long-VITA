# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import json
import os
import sys
import types
import logging

import torch


logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def check_for_arg(arg_name, margs, queue, default=None):
    if getattr(margs, arg_name, None) is None:
        if default is not None:
            setattr(margs, arg_name, default)
        else:
            logging.info("Checkpoint does not specify the argument %s. Exiting.", arg_name)
            logging.info("Arguments: %s", margs)
            queue.put("exit")


def get_models(margs, mpu, model_provider, load_checkpoint_mg, dtype):
    model_array_len = margs.virtual_pipeline_model_parallel_size
    if model_array_len is None:
        model_array_len = 1
    vp_models = []
    ep_models = [[] for _ in range(margs.expert_model_parallel_size)]
    for ep_rank in range(margs.expert_model_parallel_size):
        mpu.set_expert_model_parallel_rank(ep_rank)
        for tp_rank in range(margs.tensor_model_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            if margs.virtual_pipeline_model_parallel_size is not None:
                for i in range(margs.virtual_pipeline_model_parallel_size):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    # Set pre_process and post_process only after virtual rank is set.
                    pre_process = mpu.is_pipeline_first_stage()
                    post_process = mpu.is_pipeline_last_stage()
                    this_model = model_provider(
                        pre_process=pre_process,
                        post_process=post_process
                    ).to(dtype)
                    ep_models[ep_rank].append(this_model)
            else:
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()
                ep_models[ep_rank].append(model_provider(pre_process, post_process).to(dtype))
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint_mg([ep_models[ep_rank][-1]], None, None)

    for _ in range(model_array_len):
        vp_models.append(ep_models)

    return vp_models


def _load_checkpoint(queue, args):

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import modellink
        from megatron.training.arguments import validate_args
        from modellink.utils import parse_args
        from megatron.training.global_vars import set_args, set_global_variables
        from megatron.training.checkpointing import load_args_from_checkpoint
        from megatron.training.checkpointing import load_checkpoint as load_checkpoint_mg
        from megatron.legacy.model import module
        from megatron.core import mpu
        from megatron.core.enums import ModelType
    except ModuleNotFoundError:
        logging.info("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        queue.put("exit")

    # We want all arguments to come from us
    sys.argv = [
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
        '--load', args.load_dir,
        '--micro-batch-size', str(1),
        '--global-batch-size', str(1),
        '--bf16',
        '--make-vocab-size-divisible-by', '1',
        '--sequence-parallel',
    ]

    margs = parse_args()
    margs.data_parallel_size = 1
    set_global_variables(margs, build_tokenizer=False)
    margs, checkpoint_args = load_args_from_checkpoint(margs)
    margs.expert_model_parallel_size = checkpoint_args.expert_model_parallel_size
    margs.num_experts = checkpoint_args.num_experts
    margs.moe_router_topk = checkpoint_args.moe_router_topk

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * \
        margs.pipeline_model_parallel_size * \
        margs.expert_model_parallel_size

    margs = validate_args(margs)

    check_for_arg('tensor_model_parallel_size', margs, queue)
    check_for_arg('pipeline_model_parallel_size', margs, queue)
    check_for_arg('expert_model_parallel_size', margs, queue)
    check_for_arg('num_layers', margs, queue)
    check_for_arg('hidden_size', margs, queue)
    check_for_arg('seq_length', margs, queue)
    check_for_arg('num_attention_heads', margs, queue)
    check_for_arg('max_position_embeddings', margs, queue)
    check_for_arg('position_embedding_type', margs, queue)
    check_for_arg('tokenizer_type', margs, queue)
    check_for_arg('iteration', margs, queue)
    check_for_arg('disable_bias_linear', margs, queue, False)
    check_for_arg('params_dtype', margs, queue)
    check_for_arg('swiglu', margs, queue, False)

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(margs.virtual_pipeline_model_parallel_size)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vb_file = open(args.vocab_file)
        vocab = json.load(vb_file)
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            logging.info(
                "Both --true-vocab-size and --vocab-file specified and the vocab size does not match, aborting.")
            queue.put("exit")
        vb_file.close()
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size
    ep_size = margs.expert_model_parallel_size
    vp_size = margs.virtual_pipeline_model_parallel_size
    if vp_size is None:
        vp_size = 1

    # Layernorm has bias; RMSNorm does not.
    if hasattr(checkpoint_args, 'normalization'):
        norm_has_bias = checkpoint_args.normalization == "LayerNorm"
    else:
        # older models only supported LayerNorm
        norm_has_bias = True

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.num_experts = margs.num_experts
    md.moe_router_topk = margs.moe_router_topk
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.group_query_attention = margs.group_query_attention
    md.num_query_groups = margs.num_query_groups
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = norm_has_bias
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = checkpoint_args

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    all_models = [get_models(margs, mpu, model_provider, load_checkpoint_mg, md.params_dtype)]
    models = all_models[0][0]

    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    queue.put(md)

    def queue_put(name, msg):
        logging.info("sending %s", name)
        msg["name"] = name
        queue.put(msg)

    # Send embeddings
    message = {
        "word embeddings": torch.cat(
            [models[0][tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim=0)
    }
    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0][0].language_model.embedding.position_embeddings.weight.data

    queue_put("embeddings", message)

    total_layer_num = 0
    for vp_rank in range(vp_size):
        mpu.set_virtual_pipeline_model_parallel_rank(vp_rank)
        for pp_rank in range(pp_size):
            if pp_rank > 0:
                mpu.set_pipeline_model_parallel_rank(pp_rank)
                if vp_rank == 0:
                    all_models.append(get_models(margs, mpu, model_provider, load_checkpoint_mg, md.params_dtype))
            models = all_models[pp_rank][vp_rank]
            for layer_num, _ in enumerate(models[0][0].language_model.encoder.layers):
                message = {}

                # Get non-parallel tensors from tp_rank 0
                layer = models[0][0].language_model.encoder.layers[layer_num]
                message["input norm weight"] = layer.input_norm.weight.data
                if norm_has_bias:
                    message["input norm bias"] = layer.input_norm.bias.data
                message["post norm weight"] = layer.post_attention_norm.weight.data
                if norm_has_bias:
                    message["post norm bias"] = layer.post_attention_norm.bias.data
                if md.linear_bias:
                    message["dense bias"] = layer.self_attention.dense.bias.data
                    message["mlp gate bias"] = layer.mlp.block.moe_layer.gate.weight.bias.data

                message["mlp gate weight"] = layer.mlp.block.moe_layer.gate.weight.weight.data
                # Grab all parallel tensors for this layer
                qkv_weight = []
                qkv_bias = []
                dense_weight = []

                for tp_rank in range(tp_size):
                    layer = models[0][tp_rank].language_model.encoder.layers[layer_num]
                    qkv_weight.append(layer.self_attention.query_key_value.weight.data)
                    dense_weight.append(layer.self_attention.dense.weight.data)
                    if md.linear_bias:
                        qkv_bias.append(layer.self_attention.query_key_value.bias.data)

                message["qkv weight"] = torch.cat(qkv_weight, dim=0)
                message["dense weight"] = torch.cat(dense_weight, dim=1)

                # Experts relative
                global_w1_weight, global_w1_bias = [], []
                global_w2_weight, global_w2_bias = [], []
                global_w3_weight, global_w3_bias = [], []
                for ep_rank in range(ep_size):
                    w1_weight, w1_bias = [], []
                    w2_weight, w2_bias = [], []
                    w3_weight, w3_bias = [], []

                    for tp_rank in range(tp_size):
                        layer = models[ep_rank][tp_rank].language_model.encoder.layers[layer_num]
                        local_w1_weight, local_w1_bias = [], []
                        local_w2_weight, local_w2_bias = [], []
                        local_w3_weight, local_w3_bias = [], []
                        for expert_idx in range(margs.num_experts // ep_size):
                            local_w1_weight.append(
                                getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w1.weight.data)
                            local_w2_weight.append(
                                getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w2.weight.data)
                            local_w3_weight.append(
                                getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w3.weight.data)

                            if md.linear_bias:
                                local_w1_bias.append(
                                    getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w1.bias.data)
                                local_w2_bias.append(
                                    getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w2.bias.data)
                                local_w3_bias.append(
                                    getattr(layer.mlp.block.moe_layer.experts.experts, f"{expert_idx}").w3.bias.data)

                        w1_weight.append(local_w1_weight)
                        w2_weight.append(local_w2_weight)
                        w3_weight.append(local_w3_weight)
                        w1_bias.append(local_w1_bias)
                        w2_bias.append(local_w2_bias)
                        w3_bias.append(local_w3_bias)

                    for expert_idx in range(margs.num_experts // ep_size):
                        global_w1_weight.append(torch.cat([lp[expert_idx] for lp in w1_weight], dim=0))
                        global_w2_weight.append(torch.cat([lp[expert_idx] for lp in w2_weight], dim=1))
                        global_w3_weight.append(torch.cat([lp[expert_idx] for lp in w3_weight], dim=0))

                        if md.linear_bias:
                            global_w1_bias.append(torch.cat([lp[expert_idx] for lp in w1_bias], dim=0))
                            global_w2_bias.append(torch.cat([lp[expert_idx] for lp in w2_bias], dim=1))
                            global_w3_bias.append(torch.cat([lp[expert_idx] for lp in w3_bias], dim=0))

                for global_expert_idx in range(margs.num_experts):
                    message[f"mlp {global_expert_idx} w1 weight"] = global_w1_weight[global_expert_idx]
                    message[f"mlp {global_expert_idx} w2 weight"] = global_w2_weight[global_expert_idx]
                    message[f"mlp {global_expert_idx} w3 weight"] = global_w3_weight[global_expert_idx]

                    if md.linear_bias:
                        message[f"mlp {global_expert_idx} w1 bias"] = global_w1_bias[global_expert_idx]
                        message[f"mlp {global_expert_idx} w2 bias"] = global_w2_bias[global_expert_idx]
                        message[f"mlp {global_expert_idx} w3 bias"] = global_w3_bias[global_expert_idx]

                queue_put(f"transformer layer {total_layer_num}", message)
                total_layer_num = total_layer_num + 1

    # Send final norm from tp_rank 0
    message = {
        "weight": models[0][0].language_model.encoder.final_norm.weight.data,
    }

    if norm_has_bias:
        message["bias"] = models[0][0].language_model.encoder.final_norm.bias.data
    queue_put("final norm", message)

    if md.output_layer:
        message = {
            "weight": torch.cat(
                [models[0][tp_rank].language_model.output_layer.weight.data for tp_rank in range(tp_size)],
                dim=0)
        }
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except Exception as e:
        queue.put("exit")
        raise e
