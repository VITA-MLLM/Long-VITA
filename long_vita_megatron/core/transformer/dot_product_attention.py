# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
from torch import Tensor
from megatron.training import get_args
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import get_default_causal_mask

# from mindspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel
# from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
#                                            get_context_parallel_for_hybrid_ring_world_size,
#                                            get_context_parallel_for_hybrid_ring_rank,
#                                            get_context_parallel_for_hybrid_ring_global_ranks,
#                                            get_ring_ranks_for_intra_window,
#                                            get_ring_ranks_for_inter_window_kv,
#                                            get_ring_ranks_for_inter_window_dkv,
#                                            get_ring_group_for_intra_window,
#                                            get_ring_group_for_intra_window_send_recv_overlap)
# from mindspeed.core.context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
# from mindspeed.core.context_parallel.utils import get_scheduling_info
# from mindspeed.model.transformer import get_attention_mask
from long_vita_megatron.training.utils import get_actual_seq_len, get_position_ids

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


try:
    from einops import rearrange
except ImportError:
    rearrange = None


def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None):
    args = get_args()
    actual_seq_len = get_actual_seq_len()
    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.cp_attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        cp_para['cp_global_ranks'] = cp_global_ranks
        cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None
        cp_para['pse'] = pse
        cp_para['pse_type'] = pse_type

        cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
        cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
        cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
        cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
        cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

        output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p,
                                           actual_seq_len, actual_seq_len)
    else:
        cp_para['scheduling_info'] = get_scheduling_info()
        output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask,
                                                dropout_p)
    return output


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(self, *args, **kwargs)
        config.context_parallel_size = cp_size

        args = get_args()
        self.pse = None
        self.pse_type = None
        self.attn_logit_softcapping = args.attn_logit_softcapping
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf
        self.beta = 1.0
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number

        if args.position_embedding_type == 'alibi':
            get_alibi(self, args.seq_length)
            self.alibi_output_size = None
        else:
            self.alibi = None

        if args.query_pre_attn_scalar:
            self.norm_factor = args.query_pre_attn_scalar ** 0.5
            self.scale_mask_softmax.scale = 1.0
            self.softmax_scale = 1.0 / self.norm_factor

    return wrapper


def get_alibi(self, seq_length):
    args = get_args()
    self.alibi = Alibi()
    alibi = self.alibi._build_alibi_tensor(seq_length,
                                           args.num_attention_heads,
                                           args.square_alibi_mask,
                                           args.fill_neg_inf,
                                           ).to(torch.cuda.current_device())
    if args.params_dtype == torch.float16:
        alibi = alibi.to(torch.float16)
    elif args.params_dtype == torch.bfloat16:
        alibi = alibi.to(torch.bfloat16)
    self.alibi.alibi = alibi


def ulysses_context_parallel_forward_wrapper(fn):
    """
    Do repeat KV to support GQA+Ulysses. This wrapper would be remove if mindspeed-core support ulysses+GQA.
    """

    @wraps(fn)
    def wrapper(self, query: Tensor, key: Tensor, value: Tensor, *args, **kwargs):
        heads_per_gqa_group = self.local_attn.num_attention_heads_per_partition // self.local_attn.num_query_groups_per_partition
        global_args = get_args()
        should_kv_repeat_before_uly = global_args.use_flash_attn and global_args.kv_head_repeat_before_uly_alltoall

        if heads_per_gqa_group > 1 and should_kv_repeat_before_uly:
            key = key.repeat_interleave(heads_per_gqa_group, dim=2)
            value = value.repeat_interleave(heads_per_gqa_group, dim=2)

        return fn(self, query, key, value, *args, **kwargs)

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params):
        # if attention_mask is None:
        #     attention_mask = get_attention_mask()
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        args = get_args()
        heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        if not args.use_flash_attn:
            if heads_per_gqa_group > 1:
                key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                value = value.repeat_interleave(heads_per_gqa_group, dim=2)
        else:
            # Do repeat KV to support PFA
            should_kv_repeat_before_pfa = hasattr(args, 'use_kv_cache') and args.use_kv_cache
            if heads_per_gqa_group > 1 and should_kv_repeat_before_pfa:
                key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                value = value.repeat_interleave(heads_per_gqa_group, dim=2)

            return flash_attention_forward(self, query, key, value, attention_mask, attn_mask_type,
                                           packed_seq_params)

        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        if self.alibi is None:
            matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query.transpose(0, 1),  # [b * np, sq, hn]
                key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )
        else:
            if self.alibi.alibi_pse is None or self.alibi.output_size != output_size:
                self.alibi.output_size = output_size
                self.alibi.get_alibi_pse(attention_mask, output_size[0], output_size[2], output_size[3])

            q_trans = query.transpose(0, 1).contiguous()
            k_trans = key.transpose(0, 1).transpose(1, 2).contiguous()
            matmul_result = self.beta * self.alibi.alibi_pse + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        if self.attn_logit_softcapping is not None:
            matmul_result = matmul_result / self.attn_logit_softcapping
            matmul_result = torch.tanh(matmul_result)
            matmul_result = matmul_result * self.attn_logit_softcapping

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.square_alibi_mask:
            attention_scores = torch.max(
                attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
            )
            attention_probs = torch.nn.functional.softmax(attention_scores, -1)
        else:
            attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context

    return wrapper


def flash_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    if packed_seq_params is not None:
        raise AssertionError("packed_seq_params should be None.")

    args = get_args()

    seq_length, batch_size, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]

    # ViT
    # if query.size()[0] * args.context_parallel_size == args.vision_seq_length:
    if query.size()[0] == args.vision_seq_length and not args.vision_context_parallel:
        # print(f"query {query.size()}")
        query, key, value = [rearrange(x, 's b h d -> b s h d') for x in [query, key, value]]
        scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
            if self.scale_mask_softmax.scale is None else self.softmax_scale

        attn_output = flash_attn_func(
            query, key, value,
            dropout_p=self.attention_dropout.p,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        attn_output = rearrange(attn_output, 'b s h d -> s b (h d)', s=seq_length)
        # print(f"attn_output {attn_output.size()}")
        return attn_output

    # LLM
    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
        if self.scale_mask_softmax.scale is None else self.softmax_scale
    actual_seq_len = get_actual_seq_len()
    position_ids = get_position_ids()

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                                         'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        return do_ring_context_parallel(
            query, key, value, head_num=n_head, softmax_scale=scale, attn_mask=attention_mask, pse=self.pse,
            pse_type=self.pse_type)

    query, key, value = [rearrange(x, 's b h d -> b s h d') for x in [query, key, value]]
    # attention_mask = torch.triu(torch.ones([2048, 2048], dtype=bool, device=torch.cuda.current_device()), diagonal=1)

    if self.hidden_size_per_attention_head == 0:
        raise AssertionError("self.hidden_size_per_attention_head should not be ZERO.")
    if not hasattr(self, 'attention_mask') or \
            self.attention_mask is None or \
            self.attention_mask.shape[0] != seq_length:
        self.attention_mask = attention_mask

    use_sliding_windows = args.sliding_window is not None and seq_length > args.sliding_window

    if self.attn_mask_type == AttnMaskType.causal and attention_mask is None and seq_length > 1:
        attention_mask = get_default_causal_mask(seq_length)

    if attention_mask is not None:
        if attention_mask.dim() != 2 or attention_mask.size(0) != batch_size or attention_mask.size(1) != seq_length:
            attention_mask = None

        if actual_seq_len is not None:
            attention_mask = None

        if position_ids is not None and not (torch.diff(position_ids, dim=-1) >= 0).all():
            attention_mask = None

    # print(f"query {query.size()}")
    # print(f"key {key.size()}")
    # print(f"value {value.size()}")
    # if attention_mask is not None:
    #     print(f"attention_mask {attention_mask.size()}")
    output = _flash_attention_forward(
        query, key, value,
        attention_mask,
        seq_length,
        True,
        dropout=self.attention_dropout.p,
        position_ids=position_ids,
        softmax_scale=scale,
        sliding_window=None,
        use_top_left_mask=False,
        softcap=None,
        deterministic=None,
        # cu_seq_lens_q=actual_seq_len,
        # cu_seq_lens_k=actual_seq_len,
        # max_length_q=seq_length,
        # max_length_k=seq_length,
    )

    output = rearrange(output, 'b s h d -> s b (h d)', s=seq_length)

    return output
