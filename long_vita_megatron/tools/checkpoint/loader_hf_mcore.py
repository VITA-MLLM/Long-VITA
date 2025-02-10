# coding=utf-8
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
import types
import torch
import transformers
from models import get_megatron_model
from models import get_huggingface_model


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama-2 HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                            'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    group.add_argument("--w-pack", type=bool,
                       help='True is w_pack weight for llm',
                       default=False)
    parser.add_argument('--add-qkv-bias', action='store_true',
                        help='Add bias for attention qkv',
                        default=False)
    parser.add_argument('--add-dense-bias', action='store_true',
                        help='Add bias for attention dense',
                        default=False)
    parser.add_argument('--params-dtype', type=str,
                        help='Set weight dtype',
                        default='fp16')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=1,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficiency reasons.')
    group.add_argument('--use-mcore-models', action='store_true',
                       help='Use the implementation from megatron core')
    group.add_argument('--model-type-hf', type=str,
                       help='huggingface model type e.g., llama2, qwen, ...')


def verify_transformers_version():
    major, minor, patch = map(int, transformers.__version__.split('.'))
    if major < 4 or minor < 31:
        raise ValueError("the version transformers should greater or equal 4.31")


def build_metadata(args, margs):
    # Metadata.
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    md.params_dtype = margs.params_dtype
    md.bert_binary_head = margs.bert_binary_head
    md.output_layer = margs.untie_embeddings_and_output_weights
    md.position_embedding_type = margs.position_embedding_type
    md.linear_bias = margs.add_bias_linear
    md.norm_has_bias = False
    md.swiglu = margs.swiglu
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = margs.vocab_size  # skips padding in saver
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.checkpoint_args = margs
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.embed_layernorm = margs.embed_layernorm

    return md


def get_message_preprocess(model, md):
    # Send embeddings.
    message = {
        "word embeddings": model.get_embedding_word_embeddings_weight()
    }

    # bloom
    if model.has_embedding_word_embeddings_norm():
        message["word embeddings norm_w"] = model.get_embedding_word_embeddings_norm_weight()
        message["word embeddings norm_b"] = model.get_embedding_word_embeddings_norm_bias()

    if md.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = model.get_embedding_position_embeddings_weight()
    else:
        if model.has_embedding_position_embeddings():
            raise ValueError("model should have position_embeddings")

    return message


def get_message_layer_norm(message, model, layer_idx, md):
    # Get non-parallel tensors from tp_rank 0.
    message["input norm weight"] = model.get_layers_input_layernorm_weight(layer_idx=layer_idx)
    message["post norm weight"] = model.get_layers_self_attention_pre_mlp_layernorm_weight(layer_idx=layer_idx)

    if md.norm_has_bias:
        message["input norm bias"] = model.get_layers_input_layernorm_bias(layer_idx=layer_idx)
        message["post norm bias"] = model.get_layers_self_attention_pre_mlp_layernorm_bias(layer_idx=layer_idx)

    return message


def get_message_layer_attn(message, model, layer_idx, md=None, args=None):
    # Grab all parallel tensors for this layer.
    qkv_weight = []
    qkv_bias = []
    dense_weight = []

    qkv_weight.append(model.get_layers_self_attention_linear_qkv_weight(layer_idx=layer_idx))
    dense_weight.append(model.get_layers_self_attention_linear_proj_weight(layer_idx=layer_idx))

    if args.add_qkv_bias:
        message["qkv bias"] = model.get_layers_self_attention_linear_qkv_bias(layer_idx=layer_idx)
    if args.add_dense_bias:
        message["dense bias"] = model.get_layers_self_attention_linear_proj_bias(layer_idx=layer_idx)

    if md.linear_bias:
        qkv_bias.append(model.get_layers_self_attention_linear_proj_bias(layer_idx=layer_idx))
        message["dense bias"] = model.get_layers_self_attention_linear_proj_bias(layer_idx=layer_idx)
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)

    # Simple concat of the rest.
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["dense weight"] = torch.cat(dense_weight, dim=1)

    return message


def get_message_layer_mlp(message, model, layer_idx, md=None, tp_size=1):
    # Grab all parallel tensors for this layer.
    mlp_l0_weight = []
    mlp_l0_bias = []
    mlp_l1_weight = []
    mlp_l0_weight.append(model.get_layers_mlp_linear_fc1_weight(layer_idx=layer_idx))
    mlp_l1_weight.append(model.get_layers_mlp_linear_fc2_weight(layer_idx=layer_idx))
    if md.linear_bias:
        mlp_l0_bias.append(model.get_layers_mlp_linear_fc1_bias(layer_idx=layer_idx))
    # Handle gated linear units.
    if md.swiglu:
        # Concat all the first halves ('W's) and all the second halves ('V's).
        for tp_rank in range(tp_size):
            mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
        message["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
        message["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
    else:
        message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

    # Simple concat of the rest.
    message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)
    if md.linear_bias:
        message["mlp l1 bias"] = model.get_layers_mlp_linear_fc2_bias(layer_idx=layer_idx)
        if md.swiglu:
            for tp_rank in range(tp_size):
                mlp_l0_bias[tp_rank] = torch.chunk(mlp_l0_bias[tp_rank], 2, dim=0)
            message["mlp l0 bias W"] = torch.cat([b[0] for b in mlp_l0_bias], dim=0)
            message["mlp l0 bias V"] = torch.cat([b[1] for b in mlp_l0_bias], dim=0)
        else:
            message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)

    return message


def get_message_postprocess(model, md):
    # Send final norm from tp_rank 0.
    message = {
        "weight": model.get_final_layernorm_weight(),
    }
    if md.norm_has_bias:
        message["bias"] = model.get_final_layernorm_bias()

    return message


def get_message_output_layer(model, md):
    # Send final norm from tp_rank 0.
    message = None
    if md.output_layer:
        message = {
            "weight": model.get_output_layer_weight()
        }

    return message


def _load_checkpoint(queue, args):
    # Llama-2 requires HF transformers >=4.31.0.
    verify_transformers_version()

    # Search in directory above this.
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    model_hf = get_huggingface_model(args)
    args_hf = model_hf.get_args()

    model_mg = get_megatron_model(args_cmd=args)
    model_mg.initialize_megatron_args(args_hf, queue)

    model_mg.set_tensor_model_parallel_world_size(model_mg.args.tensor_model_parallel_size)
    model_mg.set_pipeline_model_parallel_world_size(model_mg.args.pipeline_model_parallel_size)
    model_mg.set_virtual_pipeline_model_parallel_world_size(model_mg.args.virtual_pipeline_model_parallel_size)

    # Get first pipe stage.
    model_mg.set_tensor_model_parallel_rank(0)
    model_mg.set_pipeline_model_parallel_rank(0)

    margs = model_mg.get_args()
    md = build_metadata(args, margs)
    queue.put(md)

    model_hf.get_modules_from_pretrained()
    model_mg.get_modules_from_config()

    model_mg.update_module(model_hf)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    # Send embeddings.
    message = get_message_preprocess(model_mg, md)
    queue_put("embeddings", message)

    for layer_idx in range(margs.num_layers):
        # Grab all parallel tensors for this layer.
        message = {}
        message = get_message_layer_norm(message, model_mg, layer_idx, md)
        message = get_message_layer_attn(message, model_mg, layer_idx, md, args)
        message = get_message_layer_mlp(message, model_mg, layer_idx, md)

        queue_put(f"transformer layer {layer_idx}", message)

    # Send final norm from tp_rank 0.
    message = get_message_postprocess(model_mg, md)
    queue_put("final norm", message)

    message = get_message_output_layer(model_mg, md)
    if message is not None:
        queue_put("output layer", message)

    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
