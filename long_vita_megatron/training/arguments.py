# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
import argparse
from pathlib import Path
from functools import wraps
from long_vita_megatron.training.utils import print_rank0_by_args

cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent, "configs/finetune/templates.json")


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_fusion_op_args(parser)
    parser = _add_network_size_args(parser)
    parser = _add_lora_args(parser)
    parser = _add_data_args(parser)
    parser = _add_moe_args(parser)
    parser = _add_num_layer_allocation(parser)
    parser = _add_profile_args(parser)
    parser = _add_network_args(parser)
    parser = _add_training_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_algorithm_args(parser)
    parser = _add_alibi_args(parser)
    parser = _add_dataset_args(parser)
    parser = _add_high_availability_args(parser)
    parser = _add_cp_args(parser)
    parser = _add_mla_args(parser)
    parser = _add_yarn_args(parser)
    parser = _add_deepseek_moe_args(parser)
    parser = _add_rl_args(parser)

    return parser


def _add_mla_args(parser):
    group = parser.add_argument_group(title='multi-head latent attention')

    group.add_argument('--multi-head-latent-attention', action='store_true', default=False,
                       help='Use Multi-head Latent Attention(MLA)')
    group.add_argument('--q-lora-rank', type=int, default=None, help='The low rank of q')
    group.add_argument('--kv-lora-rank', type=int, default=None, help='The low rank of k and v')
    group.add_argument('--v-head-dim', type=int, default=None, help='The head dim of v')
    group.add_argument('--qk-rope-head-dim', type=int, default=None, help='The qk head dim for rope')
    group.add_argument('--qk-nope-head-dim', type=int, default=None, help='The qk head dim for only self-attn')

    return parser


def _add_yarn_args(parser):
    group = parser.add_argument_group(title='yarn')

    group.add_argument('--rope-scaling-beta-fast', type=int, default=32, help='Yarn rope: rope beta fast')
    group.add_argument('--rope-scaling-beta-slow', type=int, default=1, help='Yarn rope: rope beta slow')
    group.add_argument('--rope-scaling-factor', type=float, default=1.0, help='Yarn rope: rope factor')
    group.add_argument('--rope-scaling-mscale', type=float, default=1.0, help='Yarn rope: rope mscale')
    group.add_argument('--rope-scaling-mscale-all-dim', type=float, default=0.0, help='Yarn rope: rope mscale all dim')
    group.add_argument('--rope-scaling-original-max-position-embeddings', type=int, default=None,
                       help='Yarn rope: rope original max position embeddings')
    return parser


def _add_deepseek_moe_args(parser):
    group = parser.add_argument_group(title='deepseek moe')

    group.add_argument('--moe-intermediate-size', type=int, default=None, help='The ffn hidden size of MoE layer')
    group.add_argument('--n-shared-experts', type=int, default=None,
                       help='This value is the number of shared experts, which is equal to the intermediate_size '
                            'of the shared experts divided by the moe_intermediate_size.')
    group.add_argument('--topk-group', type=int, default=None,
                       help='Choose topK group experts in group_limited_greedy_topK method')
    group.add_argument('--routed-scaling-factor', type=float, default=None, help='The routed scaling factor')
    group.add_argument('--norm-topk-prob', action='store_true', default=False, help='Normalize the topk weight')
    group.add_argument('--seq-aux', action='store_true', default=False, help='Compute aux loss in seq_aux')
    group.add_argument('--first-k-dense-replace', type=int, default=None, help='Set first k layer as dense layer')
    group.add_argument('--moe-layer-freq', type=int, default=None, help='Set the occurrence frequency of the moe layer')
    group.add_argument('--moe-device-level-aux-loss-coeff', type=float, default=0.,
                       help='set the coeff for devicie-level balance loss in deepseek moe')
    group.add_argument('--moe-comm-aux-loss-coeff', type=float, default=0.,
                       help='set the coeff for communication balance loss in deepseek moe')

    return parser


def _add_profile_args(parser):
    group = parser.add_argument_group(title='profiler')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[-1],
                       help='Global ranks to profile.The default value of -1 means to profile all ranks')
    group.add_argument('--profile-level', type=str, default='level0',
                       choices=['level0', 'level1', 'level2'], help='profiling level0, level1, level2')
    group.add_argument('--profile-with-stack', action='store_true', help='profiling with stack info')
    group.add_argument('--profile-with-memory', action='store_true', help='profiling with memory info')
    group.add_argument('--profile-record-shapes', action='store_true', help='profiling with shape info')
    group.add_argument('--profile-with-cpu', action='store_true', help='profiling with cpu info')
    group.add_argument('--profile-save-path', type=str, default='./profile_dir',
                       help='path to save profiling files')

    return parser


def _add_cp_args(parser):
    group = parser.add_argument_group(title='cp parallel')
    group.add_argument('--context-parallel-algo', type=str, default='ulysses_cp_algo',
                       choices=['ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo', 'adaptive_cp_algo',
                                'hybrid_adaptive_cp_algo'], help='context parallel algorithm')
    group.add_argument('--ulysses-degree-in-cp', type=int, default=None)
    group.add_argument('--cp-attention-mask-type', type=str, default='causal',
                       choices=['causal', 'general'], help='context parallel attention mask type')
    group.add_argument('--use-cp-send-recv-overlap', action='store_true',
                       help='use it to enable cp send-recv-overlap.')
    group.add_argument('--cp-window-size', type=int, default=1,
                       help='inner window size of double ring attention')
    group.add_argument('--attention-mask-on-cpu', action='store_true',
                       help='store full attention mask on CPU instead of NPU')
    group.add_argument('--adaptive-cp-without-coarse', action='store_true',
                       help='does not coarse the attention mask in adaptive_cp feature, only recommended when full'
                            'sequence length is less than 8K and dynamic attention mask is not feasible')
    group.add_argument('--adaptive-cp-dynamic-attn-mask', action='store_true',
                       help='if the attention mask is dynamic across batches')
    group.add_argument('--adaptive-cp-only-reschedule', action='store_true',
                       help='not apply remapping but only rescheduling process in adaptive-cp feature')
    group.add_argument('--adaptive-cp-manually-set-mask-list', action='store_true',
                       help='manually set pre-cooked attention mask list')
    group.add_argument('--kv-head-repeat-before-uly-alltoall', action='store_true', default=True,
                       help='use it to expand key and value for ulysses when GQA/MQA is used.')
    return parser


def _validate_varlen_fa_args(args):
    # varlen FA layout must be TND
    if args.reset_position_ids:
        args.shape_order = 'TND'
        print_rank0_by_args(args, f"When reset_position_ids is enabled, shape_order should be TND.")


def _validate_cp_args(args):
    def _check_attention_head(args, uly_size):
        """
        check GQA & ulysses
        """
        head, remainder = divmod(args.num_attention_heads, uly_size * args.tensor_model_parallel_size)
        assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysses_size * tensor_model_parallel_size"
        if args.group_query_attention and args.num_query_groups >= 1:
            head_split_by_tp, remainder = divmod(args.num_query_groups, args.tensor_model_parallel_size)
            assert head_split_by_tp >= 1 and remainder == 0, f"num_query_groups must be divisible by tensor_model_parallel_size"

            if not args.kv_head_repeat_before_uly_alltoall:
                head_split_by_tp_cp, remainder = divmod(head_split_by_tp, uly_size)
                if not (head_split_by_tp_cp >= 1 and remainder == 0):
                    raise AssertionError(
                        'num_query_groups must be divisible by ulysses_size * tensor_model_parallel_size.\n'
                        'Solution 1. adjust the ulysses_size\n'
                        'Solution 2. You can enable --kv-head-repeat-before-uly-alltoall to roll on.\n'
                        'However, performance would be affected since it would increase communication volume \n'
                        'for ulysses alltoall as well as memory usage.')

    if args.context_parallel_size <= 1:
        if args.kv_head_repeat_before_uly_alltoall:
            args.kv_head_repeat_before_uly_alltoall = False
            print_rank0_by_args(args, f"When context_parallel is not activated, kv_head_repeat_before_uly_alltoall would be set to False for reducing memory usage.")
        if args.use_fused_ring_attention_update:
            raise AssertionError(f"fused_ring_attention_update only works when context parallel is activated.")
        return

    # In context parallel we use FA
    args.use_flash_attn = True
    print_rank0_by_args(args, f"[INFO] Setting args.use_flash_attn={args.use_flash_attn} since context parallel is enabled.")
    if not args.use_mcore_models:
        raise AssertionError(f"Context parallel is only supported in Mcore.")

    if args.context_parallel_algo == 'ulysses_cp_algo':
        assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size"
        _check_attention_head(args, args.context_parallel_size)

    if args.context_parallel_algo == 'megatron_cp_algo':
        assert args.seq_length % (
                    2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size"
        assert args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size, f'cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.'
        n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
        assert n_window >= 1 and remainder == 0, f'context parallel size must be divisible by cp_window_size when using double ring attention.'
        assert args.cp_window_size >= 1 and args.cp_window_size < args.context_parallel_size, f'cp_window_size should in range [1, context_parallel_size) when using double_ring_attention.'
        n_window, remainder = divmod(args.context_parallel_size, args.cp_window_size)
        assert n_window >= 1 and remainder == 0, f'context parallel size must be divisible by cp_window_size when using double ring attention.'
        if args.cp_attention_mask_type == 'general':
            assert args.micro_batch_size == 1, f'When cp_attention_mask_type is set to general, the value of mbs can only be 1.'

    if args.context_parallel_algo == 'hybrid_cp_algo':
        assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_cp_algo"
        ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
        assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be divisible by --context-parallel-size"
        assert args.seq_length % (
                    2 * args.context_parallel_size) == 0, f"sequence length must be divisible by 2 * context_parallel_size in hybrid cp"
        assert args.cp_window_size >= 1 and args.cp_window_size < ring_degree, f'cp_window_size should be in range [1, ring_degree) when using double ring attention with hybrid context parallelism.'
        n_window, remainder = divmod(ring_degree, args.cp_window_size)
        assert n_window >= 1 and remainder == 0, f'ring_degree should be divisible by cp_window_size when using double ring with hybrid context parallelism.'
        _check_attention_head(args, args.ulysses_degree_in_cp)
        if args.cp_attention_mask_type == 'general':
            assert args.micro_batch_size == 1, f'When cp_attention_mask_type is set to general, the value of mbs can only be 1.'

    if args.context_parallel_size > 1 and args.context_parallel_algo == 'adaptive_cp_algo':
        assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size"
        if args.cp_attention_mask_type == 'general':
            assert args.micro_batch_size == 1, f'When cp_attention_mask_type is set to general, the value of mbs can only be 1.'

    if args.context_parallel_size > 1 and args.context_parallel_algo == 'hybrid_adaptive_cp_algo':
        assert args.ulysses_degree_in_cp is not None, "--ulysses-degree-in-cp must be specified in hybrid_adaptive_cp_algo"
        ring_degree, remainder = divmod(args.context_parallel_size, args.ulysses_degree_in_cp)
        assert ring_degree > 1 and remainder == 0, "--ulysses-degree-in-cp must be devisible by --context-parallel-size"
        head, remainder = divmod(args.num_attention_heads, args.ulysses_degree_in_cp * args.tensor_model_parallel_size)
        assert head >= 1 and remainder == 0, f"num_attention_heads must be divisible by ulysse-degree-in-cp * tensor_model_parallel_size in hybrid cp"
        assert args.seq_length % args.context_parallel_size == 0, f"sequence length must be divisible by context_parallel_size in hybrid cp"
        if args.cp_attention_mask_type == 'general':
            assert args.micro_batch_size == 1, f'When cp_attention_mask_type is set to general, the value of mbs can only be 1.'

    if args.sliding_window:
        raise AssertionError("sliding window is not supported in context parallel.")


def _validate_tocken(args):
    """To avoid invalid tocken configration."""
    if args.pre_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] pre_tocken={args.pre_tocken} would be adjusted to {args.seq_length} for better performance.")
    if args.next_tocken > args.seq_length:
        print_rank0_by_args(args, f"[INFO] next_tocken={args.next_tocken} would be adjusted to {args.seq_length} for better performance.")


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                       help='Lora modules to save.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm'],
                       help='Lora register forward hook.')
    group.add_argument('--lora-fusion', action='store_true',
                       help='use fusion to accelerate lora.')
    return parser


def _add_moe_args(parser):
    group = parser.add_argument_group(title='moe')
    group.add_argument('--moe-router-topk', type=int, default=2,
                       help='Number of experts to route to for each token. The default is 2.')
    group.add_argument('--moe-router-load-balancing-type', type=str,
                       choices=['aux_loss', "group_limited_greedy", "softmax_topk"],
                       default='aux_loss',
                       help='Determines the load balancing strategy for the router. "aux_loss" corresponds '
                            'to the load balancing loss used in GShard and SwitchTransformer, "sinkhorn" corresponds '
                            'to the balancing algorithm used in S-BASE, "softmax_topk" implies no load balancing and '
                            'softmax before topk , "None" implies no load balancing, and "group_limited_greedy" corresponds '
                            'to the Device-Limited Routing method in DeepSeekV2.'
                            'The default is "aux_loss".')
    group.add_argument('--expert-interval', type=int, default=1,
                       help='Use experts in every "expert-interval" layers')
    group.add_argument('--moe-aux-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the aux loss: a starting value of 1e-2 is recommended.')
    group.add_argument('--moe-z-loss-coeff', type=float, default=0.0,
                       help='Scaling coefficient for the z-loss: a starting value of 1e-3 is recommended.')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time used in legacy moe layer called SwitchMLP.')
    group.add_argument("--use-fused-moe-token-permute-and-unpermute", action='store_true',
                       help="Use fused moe permute and unpermute.")

    # For megatron_moe drop
    group.add_argument('--moe-expert-capacity-factor', type=float, default=None,
                       help='The capacity factor for each expert, None means no token will be dropped.')
    group.add_argument('--moe-pad-expert-input-to-capacity', action='store_true',
                       help='Pads the input for each expert to match the expert capacity length, effective only after the --moe-expert-capacity-factor is set.')
    group.add_argument('--moe-token-drop-policy', type=str, default='probs', choices=['probs', 'position'],
                       help='The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.')
    group.add_argument('--moe-token-dispatcher-type', type=str, choices=['allgather', 'alltoall'], default='allgather',
                       help='The dispatcher type for moe token dispatching.')

    group.add_argument('--noisy-gate-policy', type=str, default=None,
                       help="noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.")
    group.add_argument('--enable-token-rearrange-opt', action='store_true',
                       help="Use this flag to enable token rearrange optimize")
    group.add_argument('--embedding-multiplier-scale', type=float, default=1.0,
                       help='add scale for embedding.')
    group.add_argument('--input-jitter', action='store_false', help='Add noise to the input tensor.')
    group.add_argument('--post-norm', action='store_true', help='post norm after attention or mlp.')
    group.add_argument('--output-multiplier-scale', type=float, default=None, help='Add scale for logits output.')
    group.add_argument("--moe-permutation-async-comm", action='store_true',
                       help="overlap moe permutation 3 all gather communications")
    group.add_argument("--shared-expert-gate", action='store_true',
                       help="moe model has shared expert gate")
    group.add_argument("--shared-expert-gate-output-dimension", type=int, default=1,
                       help="moe model shared expert gate output dimension for qwen2 moe, this parameter can only configured with"
                            "1 or hidden_state")
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data')
    group.add_argument('--is-instruction-dataset', action='store_true', help='use instruction dataset or not')
    group.add_argument('--full-shuffle-instruction-dataset', action='store_true',
                       help='full shuffle instruction dataset or not')
    group.add_argument('--variable-seq-lengths', action='store_true', help='Use variable seq lengths or not.')
    group.add_argument("--tokenizer-kwargs", type=str, nargs='+', default=None,
                       help="Kwargs of the huggingface tokenizer.")
    group.add_argument('--tokenizer-padding-side', type=str, default='right',
            help="tokenizer padding side")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'Llama2Tokenizer',
                                'PretrainedFromHF',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument("--input-layernorm-in-fp32", action='store_true',
                       help="Convert input-layernorm to fp32")
    group.add_argument("--no-shuffle", action='store_true',
                       help="Disable data shuffling, mainly for loss comparison.")
    return parser


def _add_num_layer_allocation(parser):
    group = parser.add_argument_group(title='num_layer_allocation')
    group.add_argument('--num-layer-list',
                       type=str, help='a list of number of layers, '
                                'seperated by comma; e.g., 4,4,4,4')
    return parser


def _add_fusion_op_args(parser):
    group = parser.add_argument_group(title='fusion_op_args')
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    group.add_argument("--use-fused-ring-attention-update", action='store_true',
                       help="Use fused ring attention update.")
    group.add_argument("--use-mc2", action='store_true',
                       help="Use mc2 for compute-comm overlap in tp.")
    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')
    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')
    group.add_argument('--embed-layernorm',
                       action='store_true',
                       default=False,
                       help='set padded vocab size'
                       )
    group.add_argument('--use-glm-rope',
                       action='store_true',
                       help='use custom partial rope in glm model.'
                       )

    group.add_argument('--sliding-window', type=int, default=None,
                       help='Window size when use sliding window attention.')
    group.add_argument('--output-layer-slice-num', type=int, default=1,
                       help='Set the number of slices for the weight of the output_layer')
    return parser


def _add_algorithm_args(parser):
    group = parser.add_argument_group(title='algorithm')
    group.add_argument('--rotary-base', type=float, help='rotary-base.')
    group.add_argument('--rope-scaling-type', type=str, default=None, choices=["llama3", "yarn"],
                       help='The sub-variant of RoPE to use, support type llama3 and yarn.')
    group.add_argument('--low-freq-factor', type=float,
                       help='rope low freq factory for rope scaling type.')
    group.add_argument('--high-freq-factor', type=float,
                       help='rope high freq factor for rope scaling type.')
    group.add_argument('--original-max-position-embeddings', type=float,
                       help='original max position embeddings for rope scaling type')

    group.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    group.add_argument('--reuse-fp32-param', action='store_true',
                       help='The distributed training optimizer frees up '
                            'param copies of FP32 to save memory.')
    group.add_argument('--recompute-activation-function', action='store_true',
                       help='Recompute the activation function in MLP layers.')
    group.add_argument('--recompute-activation-function-num-layers', type=int, default=None,
                       help='Can be used together with "--recompute-method block." '
                            'and "--recompute-num-layers". ')
    group.add_argument('--recompute-in-advance', action='store_true',
                       help='recompute early to reduce bubble and improve training.')
    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    group.add_argument("--add-qkv-bias", action="store_true", default=False,
                       help='Configuration for the qkv bias.')
    group.add_argument("--add-dense-bias", action="store_true", default=False,
                       help='Configuration for the dense bias.')
    group.add_argument("--skip-bias-add", action="store_false", default=True,
                       help='Configuration for the skip bias.')
    group.add_argument('--add-rmsnorm-offset', action='store_true', default=False,
                       help='RMSNorm unit offset.')
    group.add_argument('--geglu', action='store_true', default=False,
                       help='Geglu activate function.')
    group.add_argument('--input-embeds-norm', action='store_true', default=False,
                       help='input normalization.')
    group.add_argument('--gelu-tanh', action='store_true', default=False,
                       help='Tanh Geglu activate function.')
    group.add_argument('--output-logit-softcapping', type=float, help='output logit softcapping.')
    group.add_argument('--attn-logit-softcapping', type=float, help='attention logit softcapping.')
    group.add_argument('--query-pre-attn-scalar', type=int, help='attention scalar.')
    group.add_argument('--interleave-sliding-window', type=int,
                       help='Window size when use interleave sliding window attention.')
    group.add_argument(
        '--stage',
        default=None,
        choices=["sft", "dpo", "rm", "simpo"],
        help='Determine training mode'
    )

    return parser


def _add_rl_args(parser):
    group = parser.add_argument_group(title='reinforce learning in dpo')
    group.add_argument(
        '--dpo-beta',
        default=0.1,
        help='The beta parameter for the DPO loss.'
    )
    group.add_argument(
        '--simpo-beta',
        default=2.5,
        type=float,
        help='The beta parameter for the SimPO loss.'
    )
    group.add_argument(
        '--gamma-beta-ratio',
        default=1.4,
        type=float,
        help='The beta parameter for the SimPO loss.'
    )
    group.add_argument(
        '--dpo-loss-type',
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help='The type of DPO loss to use.'
    )
    group.add_argument(
        '--simpo-loss-type',
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
        help='The type of SimPO loss to use.'
    )
    group.add_argument(
        '--simpo-label-smoothing',
        default=0.0,
        help='The robust SimPO label smoothing parameter.'
    )
    group.add_argument(
        '--ref-model',
        default=None,
        help='Path to the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--refer-model-iter',
        type=int,
        default=1,
        help='iteration of the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--dpo-label-smoothing',
        default=0.0,
        help="The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5.",
    )
    group.add_argument(
        '--pref-ftx',
        default=0.0,
        help="The supervised fine-tuning loss coefficient in DPO training.",
    )
    group.add_argument(
        "--is-pairwise-dataset", action='store_true',
        help="Whether the dataset is pairwise format that has a chosen sequence and rejected "
             "sequence, which usually used in reinforce learning.")
    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    # transformer-impl保持local
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--enable-recompute-layers-per-pp-rank',
                       action='store_true', default=False,
                       help='If enabled, --recompute-num-layers will mean the number of '
                            'layers recomputed in each pp rank. Otherwise it means the number '
                            'of layers recomputed in each vpp rank.')
    group.add_argument('--pre-tockens', type=int, default=65536,
                       help='pre-tockens is used by Flash attention')
    group.add_argument('--next-tockens', type=int, default=0,
                       help='next-tockens is used by Flash attention')
    group.add_argument('--sparse-mode', type=int, default=0,
                       help='different modes of flash attention mask')
    group.add_argument('--shape-order', type=str, default='SBH',
                       choices=['SBH', 'BSH', 'BSND', 'BNSD'],
                       help='input shape order used by Flash attention')
    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='enable deterministic computing for npu')
    group.add_argument('--jit-compile', action='store_true', default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=['default', 'empty', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'chatml',
                                'chatml_de', 'qwen', 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
                                'deepseek2', 'deepseek2-lite', 'cpm', 'baichuan2', 'long_vita'],
                       help='Which template to use for constructing prompts in training/inference.'  'e.g., "qwen"')
    group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                       help='Path to the json file of templates.')
    group.add_argument('--pad-to-multiple-of', type=int, default=8,
                       help='Used for Padding multiple in finetune. The default is 8.')
    group.add_argument('--scale-emb', type=float, default=None,
                       help='scale embed tokens')
    group.add_argument('--dim-model-base', type=float, default=None,
                       help='dim-model-base')
    group.add_argument('--scale-depth', type=float, default=None,
                       help='scale-depth')
    group.add_argument('--swap-attention', action='store_true', default=False,
                       help='switch to open swap-attention feature.'
                            'The default is False.')
    group.add_argument('--swap-modules', type=str, default=None,
                       help='Swap modules for model. Should be used together with "--swap-attention."')
    group.add_argument('--load-checkpoint-loosely', action='store_true', default=False,
                       help='Enable loading checkpoint not strictly.')
    group.add_argument('--no-post-layer-norm', action='store_true', default=False,
                       help='Disable final layer norm.')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=45,
                       help='Timeout minutes for torch.distributed.')
    return parser


def add_parser_argument_choices_value(parser, argument_name, value):
    if parser._actions:
        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                add_parser_argument_choices_value(action, argument_name)
            elif isinstance(action, argparse.Action) and argument_name in action.option_strings:
                action.choices.append(value)


def _add_alibi_args(parser):
    add_parser_argument_choices_value(parser, "--position-embedding-type", 'alibi')

    group = parser.add_argument_group(title='alibi')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')

    return parser


def _add_high_availability_args(parser):
    group = parser.add_argument_group(title='high_availability')

    group.add_argument('--enable-high-availability',
                       action='store_true',
                       help='switch of the high availability feature')

    group.add_argument('--enable-optimizer-state-local-copy',
                       action='store_true',
                       help='high availability feature, enable parameter state local copy of distributed optimizer')

    group.add_argument('--enable-hbmfault-repair',
                       action='store_true',
                       help='high availability feature, enable hbmfault repair')

    return parser


def _add_dataset_args(parser):
    group = parser.add_argument_group(title='dataset_args')
    group.add_argument('--no-shared-storage',
                       action='store_true',
                       default=False,
                       help='if no shared storage, set it'
                       )
    return parser


def _validate_create_attention_mask_in_dataloader(args):
    args.create_attention_mask_in_dataloader = False
    reset_data = args.reset_attention_mask
    alibi_without_flash_attn = args.position_embedding_type == 'alibi' and not args.use_flash_attn
    if reset_data or alibi_without_flash_attn or args.tokenizer_padding_side == "left":
        args.create_attention_mask_in_dataloader = True
    print_rank0_by_args(args, f"[INFO] Setting args.create_attention_mask_in_dataloader to {args.create_attention_mask_in_dataloader} "
                 f"since reset_data={reset_data} or alibi_without_flash_attn={alibi_without_flash_attn} or "
                 f"args.tokenizer_padding_side={args.tokenizer_padding_side}")


def _validate_position_embedding(args):
    """
    validate position embedding arguments.
    """
    if args.use_glm_rope and args.use_fused_rotary_pos_emb:
        raise AssertionError('Fused rotary embedding is not supported in glm rope.')
    if args.position_embedding_type == 'alibi' and args.sliding_window is not None:
        raise AssertionError('Sliding Window Attention is forbidden when use alibi.')
    if args.tokenizer_padding_side == 'left' and args.position_embedding_type == 'alibi':
        raise AssertionError('Alibi is not support tokenizer-padding-side left now.')


def _validate_recompute_args(args):
    """
    validate re-computation arguments.
    """
    enable_pp_vpp = args.num_layers_per_virtual_pipeline_stage
    enable_vanilla_recomputation = args.recompute_granularity is not None and args.recompute_method == 'block'
    enable_swap = args.swap_attention
    enable_recompute_activation = args.recompute_activation_function
    enable_recomputation = enable_vanilla_recomputation or enable_swap or enable_recompute_activation
    if args.enable_recompute_layers_per_pp_rank and not (enable_pp_vpp and enable_recomputation):
        raise AssertionError("enable-recompute-layers-per-pp-rank should be works with pipeline and virtual pipeline, when enabling re-computation.")

    if args.recompute_activation_function:
        if args.recompute_method == "uniform":
            raise AssertionError('uniform recomputation is not compatible with activation function recomputation.')
        if args.recompute_granularity == "selective":
            raise AssertionError('--recompute-activation-function is not compatible with selective recomputation.')

    if args.swap_attention and args.swap_modules is None:
        if args.use_mcore_models:
            args.swap_modules = "input_layernorm,self_attention,pre_cross_attn_layernorm"
        else:
            args.swap_modules = "input_norm,self_attention,post_attention_norm"


def _validate_high_availability(args):

    if args.enable_high_availability:
        try:
            import mindio_ttp
        except ModuleNotFoundError as e:
            raise AssertionError(f"High availability feature requires the mindio_ttp package but is not installed.") from e
    if args.enable_optimizer_state_local_copy and not args.enable_high_availability:
        raise AssertionError('switch of the local copy is unsupported, please enable high availability feature first.')
    if args.enable_hbmfault_repair and not args.enable_high_availability:
        raise AssertionError(
            'switch of the enable hbmfault repair is unsupported, please enable high availability feature first.')
    if args.enable_high_availability and args.use_dist_ckpt:
        raise AssertionError('switch of the high availability feature is unsupported')


def _validate_instruction_finetune(args):
    if args.variable_seq_lengths:
        if args.context_parallel_size > 1:
            raise AssertionError('Context parallelism is forbidden when use variable seq lengths.')
        if args.num_experts is not None and args.moe_token_dispatcher_type == "allgather":
            raise AssertionError('moe_token_dispatcher_type "allgather" is forbidden when use variable seq lengths. you can choose "alltoall"')


def _validate_inference_args(args):
    if args.prompt_type is not None and hasattr(args, "hf_chat_template") and args.hf_chat_template:
        raise AssertionError('Prompt-type is forbidden when use huggingface chat template.')

    if hasattr(args, "history_turns") and args.history_turns < 0:
        raise AssertionError('History turns of chat must greater than 0.')


def _validate_evaluation_args(args):
    # five shot only supported on mmlu and ceval now
    if args.prompt_type is not None and hasattr(args, "task") and (args.task == "mmlu" or args.task == "ceval"):
        train_dir = os.path.dirname(args.task_data_path) + "/dev/"
        if not os.path.isdir(train_dir) or not os.path.isdir(args.task_data_path):
            raise ValueError(f"Test and dev directory must exists when specify prompt_type in evaluation")


def _validate_moe_args(args):
    if not args.use_mcore_models and args.num_experts and args.num_experts > 1:
        raise ValueError(f'MOE is not supported in legacy model. Please activate `--use-mcore-models` to enable moe features.')
    if args.moe_expert_capacity_factor is not None:
        if args.moe_token_dispatcher_type != "alltoall":
            raise ValueError(f'moe_expert_capacity_factor only works with alltoall token dispatcher')
        if args.moe_expert_capacity_factor < 0:
            args.moe_expert_capacity_factor = None
            print_rank0_by_args(f'When moe_expert_capacity_factor < 0, no token would be drop, so moe_expert_capacity_factor should be set to false.')
        if args.moe_router_load_balancing_type not in ["aux_loss", "none"]:
            raise ValueError(f'moe_expert_capacity_factor only works with aux_loss or none load balancing')
        if args.moe_expert_capacity_factor is None and args.moe_pad_expert_input_to_capacity:
            raise ValueError(f'moe_expert_capacity_factor must be set to use moe_pad_expert_input_to_capacity')
        if args.shared_expert_gate_output_dimension != 1 and args.shared_expert_gate_output_dimension != args.hidden_size:
            raise AssertionError('shared expert gate output dimension can only be configured with 1 or hidden_size')
        if hasattr(args, 'use_fused_moe_token_permute_and_unpermute') and args.use_fused_moe_token_permute_and_unpermute:
            raise AssertionError('moe_expert_capacity_factor mode does not support use_fused_moe_token_permute_and_unpermute')


def _validate_mla(args):
    if args.multi_head_latent_attention:
        if args.kv_lora_rank is None:
            raise AssertionError('The parameter kv-lora-rank should be set when use multi_head_latent_attention.')
        elif args.v_head_dim is None:
            raise AssertionError('The parameter v-head-dim should be set when use multi_head_latent_attention.')
        elif args.qk_rope_head_dim is None:
            raise AssertionError('The parameter qk-rope-head-dim should be set when use multi_head_latent_attention.')
        elif args.qk_nope_head_dim is None:
            raise AssertionError('The parameter qk-nope-head-dim should be set when use multi_head_latent_attention.')


def _validate_yarn(args):
    if args.rope_scaling_type == "yarn":
        if args.rope_scaling_original_max_position_embeddings is None:
            raise AssertionError('The parameter rope_scaling_original_max_position_embeddings should be set '
                                 'when use yarn.')


def _validate_transformer_block_build_layers(args):
    if args.num_experts is None:
        if args.first_k_dense_replace is not None or args.moe_layer_freq is not None:
            raise AssertionError('First-k-dense-replace and moe-layer-freq must be None when not using MoEs')
    else:
        if (args.first_k_dense_replace is None) != (args.moe_layer_freq is None):
            raise AssertionError('First-k-dense-replace and moe-layer-freq must be set together.')
        if args.first_k_dense_replace and args.num_layers <= args.first_k_dense_replace:
            raise AssertionError('Num-layer ({}) must be greater than first-k-dense-replace ({}) when first-k-dense-replace is set.'.format(args.num_layers,
            args.first_k_dense_replace))
    if args.num_experts is not None and args.use_mc2 and args.moe_grouped_gemm:
        raise AssertionError('Moe Grouped Gemm is not supported with mc2 in MOE model.')

    if args.num_layer_list:
        if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size:
            raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size")
        if not args.pipeline_model_parallel_size > 1:
            raise ValueError("Dynamic pipeline model should work with pipeline parallel.")
        if args.num_layers_per_virtual_pipeline_stage:
            raise ValueError("Dynamic pipeline model and virtual pipeline cannot be enabled at the same time.")


def _validate_group_limited_greedy(args):
    if args.moe_router_load_balancing_type == "group_limited_greedy":
        if args.topk_group is None:
            raise AssertionError('The parameter topk-group should be set when use group_limited_greedy.')
        elif args.routed_scaling_factor is None:
            raise AssertionError('The parameter routed_scaling_factor should be set when use multi_head_latent_attention.')
        elif args.topk_group >= args.expert_model_parallel_size:
            raise AssertionError('The topk group ({}) should be less than n-group(EP)({}).'.format(args.topk_group,
            args.expert_model_parallel_size))


def _validate_rl_training(args):
    return


def get_layer_offset(pp_size, num_layer_list):
    """
    Get layer number offset for pp stage. global_layer_number = local_layer_number + layer_number_offset
    For instance, num-layer-list=1,3,3,1,
    (1,123,123,1) + (0,1,4,7) = (1,234,567,8)
    For each pp_stage, we have layer_number_offset = prefix_sum[pp_stage + 1]
    """
    prefix_sum = [0] * (pp_size + 1)
    # take prefix_sum[0] as sentinel
    for index, num_layers in enumerate(num_layer_list):
        prefix_sum[index + 1] = prefix_sum[index] + num_layers
    return prefix_sum


def _validate_output_layer_slice_num(args):
    if args.output_layer_slice_num < 1:
        raise AssertionError('Output_layer_slice_num must be greater than 0.')
    elif args.output_layer_slice_num > 1:
        if args.tensor_model_parallel_size > 1:
            raise AssertionError('When output_layer_slice_num is greater than 1, only support TP size is 1.')
        if (args.padded_vocab_size is not None) and (args.padded_vocab_size % args.output_layer_slice_num != 0):
            raise AssertionError('Output_layer_slice_num needs to be divisible by padded_vocab_size.')
        elif (args.vocab_size is not None) and (args.vocab_size % args.output_layer_slice_num != 0):
            raise AssertionError('Output_layer_slice_num needs to be divisible by vocab_size.')

        if args.gradient_accumulation_fusion:
            args.gradient_accumulation_fusion = False
            print_rank0_by_args(f"gradient_accumulation_fusion would be set to {args.gradient_accumulation_fusion} "
                                f"since args.output_layer_slice_num > 1")


def core_transformer_config_from_args_wrapper(fn):
    @wraps(fn)
    def wrapper(args, config_class=None):
        config = fn(args, config_class)
        # Turn down batch_p2p_comm only when pp2vpp
        if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
            config.batch_p2p_comm = False

        if args.moe_expert_capacity_factor:
            # moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None.
            config.moe_expert_capacity_factor = args.moe_expert_capacity_factor
            # moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.
            config.moe_pad_expert_input_to_capacity = args.moe_pad_expert_input_to_capacity
            # The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
            config.moe_token_drop_policy = args.moe_token_drop_policy

        if args.num_layer_list:
            # For num layer list, we turn string into int list and store it in transformer config.
            config.num_layer_list = list(map(int, args.num_layer_list.split(',')))
            config.layer_offset = get_layer_offset(args.pipeline_model_parallel_size, config.num_layer_list)
            # validate num_layer_list
            if config.layer_offset[args.pipeline_model_parallel_size] != args.num_layers:
                raise ValueError(f"Incorrect num_layer_list config since its sum({config.layer_offset[args.pipeline_model_parallel_size]} is unequal to total num layers({args.num_layers}).")
        else:
            config.num_layer_list = None

        return config

    return wrapper


def _validate_optimizer(args):
    if args.reuse_fp32_param and not args.bf16:
        raise AssertionError('--reuse-fp32-param only support for `bf16`')


def _store_variables(args):
    """
    To bypass megatron validation, we store variables and restore them shortly afterward.
    """
    variable_dict = dict()
    variable_dict["variable_seq_lengths"] = args.variable_seq_lengths

    # Moe models require `--sequence-parallel` to be turned on before Megatron core_v0.7.0,
    # which conflicted with the behavior of turning it off by default during inference and evaluation.
    variable_dict["origin_sequence_parallel"] = args.sequence_parallel
    if args.num_experts is not None and hasattr(args, "temperature") and args.temperature is not None:
        args.sequence_parallel = True

    # to bypass megatron assertion of moe+spec
    variable_dict["spec"] = args.spec
    args.spec = None

    # Bypass megatron validation when pp == 2 and vpp is enabled.
    if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
        variable_dict["num_layers_per_virtual_pipeline_stage"] = args.num_layers_per_virtual_pipeline_stage
        variable_dict["overlap_p2p_comm"] = args.overlap_p2p_comm
        args.num_layers_per_virtual_pipeline_stage = None
        args.overlap_p2p_comm = None

    if not args.data_path:
        args.data_path = True

    args.mock_data = 0

    return variable_dict


def _restore_variables(args, variable_dict):
    args.variable_seq_lengths = variable_dict["variable_seq_lengths"]

    # Bypass megatron validation when pp == 2 and vpp is enabled.
    if variable_dict.get("num_layers_per_virtual_pipeline_stage") and args.pipeline_model_parallel_size == 2:
        args.num_layers_per_virtual_pipeline_stage = variable_dict["num_layers_per_virtual_pipeline_stage"]
        args.overlap_p2p_comm = variable_dict["overlap_p2p_comm"]

    # Moe models require `--sequence-parallel` to be turned on before Megatron core_v0.7.0,
    # which conflicted with the behavior of turning it off by default during inference and evaluation.
    if args.num_experts is not None and hasattr(args, "temperature") and args.temperature is not None:
        args.sequence_parallel = variable_dict["origin_sequence_parallel"]

    # to bypass megatron assertion of moe+spec
    args.spec = variable_dict["spec"]


def _add_dummy_args(args):
    """
    For arguments in mindspeed-core which is currently unsupported in mindspeed-llm.
    """
    # reduce_recompute_for_last_chunk would be registered if recompute-in-advance is supported.
    args.adaptive_recompute_device_swap = False
    args.adaptive_recompute_device_size = -1
    args.adaptive_recompute_profiling_step = 10
    args.moe_tp_extend_ep = False
    args.recompute_in_bubble = False
    args.use_nanopipe = False
    args.tp_2d = None
    args.tp_x = 1
    args.tp_y = 1
    args.use_nd_matmul = False
    args.moe_alltoall_overlap_comm = False
    args.moe_allgather_overlap_comm = False
    args.moe_without_activation = False
    args.disable_gloo_group = None
    args.ampipe_degree = 0


def _validate_noop_layer(args):
    if isinstance(args.noop_layers, str):
        noop_layers = set()
        for x in args.noop_layers.split(','):
            if int(x) >= args.num_layers or int(x) < 0:
                raise AssertionError(f'each element in args.noop_layers({args.noop_layers}) should bigger or equal '
                                     f'to 0 and smaller than args.num_layers({args.num_layers})')
            noop_layers.add(int(x))
        args.noop_layers = noop_layers
        if args.num_layer_list:
            print_rank0_by_args("num layer list would be disabled when noop-layer is activated.")
            args.num_layer_list = None


def _validate_vpp(args):
    """validate scenario that vpp is enabled when pp=2."""
    if args.pipeline_model_parallel_size != 2 or args.num_layers_per_virtual_pipeline_stage is None:
        return

    # VPP enabled when pp == 2, do check.
    num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
    assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
        'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'

    pp_stage_layers = args.num_layers / args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage and args.num_layers_per_virtual_pipeline_stage >= pp_stage_layers:
        raise ValueError("Num of layers in vpp stage should be less than pp stage, "
                         "please turn down args.num_layers_per_virtual_pipeline_stage.")

    args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                                                args.num_layers_per_virtual_pipeline_stage

    print_rank0_by_args(args, f'vpp_size would be {args.virtual_pipeline_model_parallel_size} since '
                        f'num_layers_per_virtual_pipeline_stage is {args.num_layers_per_virtual_pipeline_stage}')


def _validate_recompute_in_advance(args):
    args.reduce_recompute_for_last_chunk = False
    if args.recompute_in_advance:
        args.reduce_recompute_for_last_chunk = True
        if args.recompute_method == "uniform":
            raise AssertionError('recompute_in_advance does not support uniform recompute_method')
        if args.recompute_granularity == 'selective':
            raise AssertionError('recompute_in_advance does not support vanilla recompute_activations.')
        if not args.recompute_num_layers:
            raise AssertionError('recompute_num_layers must be configured when using recompute_in_advance')
        if args.pipeline_model_parallel_size <= 1 or args.num_layers_per_virtual_pipeline_stage != 1:
            raise AssertionError('recompute_in_advance only support pipelining with interleaving and vpp stage should be 1.')


def validate_args_decorator(megatron_validate_args):
    @wraps(megatron_validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        variable_dict = _store_variables(args)
        megatron_validate_args(args, defaults)
        _restore_variables(args, variable_dict)

        args.use_mc2 = False

        _validate_varlen_fa_args(args)
        _validate_cp_args(args)
        _validate_vpp(args)
        _validate_recompute_args(args)
        _validate_recompute_in_advance(args)
        _validate_create_attention_mask_in_dataloader(args)
        _validate_instruction_finetune(args)
        _validate_position_embedding(args)
        _validate_high_availability(args)
        _validate_inference_args(args)
        _validate_moe_args(args)
        _validate_mla(args)
        _validate_yarn(args)
        _validate_transformer_block_build_layers(args)
        _validate_group_limited_greedy(args)
        _validate_evaluation_args(args)
        _validate_output_layer_slice_num(args)
        _validate_optimizer(args)
        _validate_rl_training(args)
        _validate_noop_layer(args)
        _add_dummy_args(args)

        from long_vita_megatron.training.utils import print_args
        print_args('ModelLink Arguments', args)
        return args

    return wrapper
