# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import sys

import torch

try:
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    multi_tensor_applier = None

try:
    import amp_C
except ImportError:
    amp_C = None

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared

from mindspeed.utils import compute_actual_seq_len, set_actual_seq_len, set_position_ids


ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if mpu.get_expert_model_parallel_rank() > 0:
                if not getattr(param, 'allreduce', True) and is_not_tp_duplicate:
                    assert param_is_not_shared(param)
                    params_data.append(param.data.float() if args.bf16 else param.data)
            else:
                is_not_shared = param_is_not_shared(param)
                if is_not_shared and is_not_tp_duplicate:
                    params_data.append(param.data.float() if args.bf16 else param.data)

    # Check the availability of apex
    assert multi_tensor_applier is not None and amp_C is not None, \
        "apex is not available, please install it from https://github.com/NVIDIA/apex"

    # Calculate norm
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    if mpu.get_expert_model_parallel_world_size() == 1:
        # Sum across all model-parallel GPUs(tensor + pipeline).
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_model_parallel_group())
    else:
        # Sum across tensor, pipeline and expert model-parallel GPUs.
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_tensor_and_expert_parallel_group())
        torch.distributed.all_reduce(norm_2,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=mpu.get_pipeline_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    from megatron.training import get_args

    args = get_args()

    if args.reset_position_ids:
        position_ids = batch['position_ids']
        position_ids = position_ids.transpose(0, 1).contiguous()
        set_position_ids(position_ids)    

    cp_size = args.context_parallel_size
    if not cp_size > 1:
        return batch

    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.cp_attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp(batch):
    args = get_args()
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in list(batch.items()):

        # if "external_" in key and not args.vision_context_parallel:
        #     continue
        if key == "external_images":
            if "external_indices" in batch:
                calibration_index = torch.arange(args.seq_length, device='cuda').view(2 * cp_size, args.seq_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)
                indices_b, indices_s = batch["external_indices"].unbind(dim=0)
                mask = torch.isin(indices_s, calibration_index)
                if mask.any():
                    selected_i = torch.any(mask, dim=1)

                    val = val[selected_i, ...]
                    batch["external_images"] = val
            continue

        if key == "external_indices":
            # calibration_index = torch.arange(args.seq_length, device='cuda').view(2 * cp_size, args.seq_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)
            # indices_b, indices_s = val.unbind(dim=0)
            # mask = torch.isin(indices_s, calibration_index)
            # # print(f"_get_batch_on_this_cp_rank_in_megatron_cp mask {mask.size()}")
            # # print(f"_get_batch_on_this_cp_rank_in_megatron_cp val {val.size()}")
            # if mask.any():
            #     selected_i = torch.any(mask, dim=1)
            #     selected_i = selected_i.unsqueeze(1).repeat(1, indices_s.shape[1])
            #     # print(f"_get_batch_on_this_cp_rank_in_megatron_cp selected_i {selected_i.size()}")

            #     calibration_index_image = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(indices_s.shape[0], 1)[selected_i]
            #     # print(f"_get_batch_on_this_cp_rank_in_megatron_cp calibration_index_image {calibration_index_image.size()}")

            #     src_indices_b = torch.arange(indices_b.shape[0], device='cuda').unsqueeze(1).repeat(1, indices_b.shape[1])
            #     src_indices_s = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(indices_s.shape[0], 1)

            #     src_indices_b = src_indices_b[mask]
            #     src_indices_s = src_indices_s[mask]
            #     # print(f"_get_batch_on_this_cp_rank_in_megatron_cp src_indices_s {src_indices_s.size()}")
            #     src_indices_s = index_of_a_in_b(src_indices_s, calibration_index_image)


            #     tgt_indices_b = indices_b[mask]
            #     tgt_indices_s = indices_s[mask]
            #     # tgt_indices_s = (tgt_indices_s.view(-1, 1) == calibration_index).int().argmax(dim=1)
            #     tgt_indices_s = index_of_a_in_b(tgt_indices_s, calibration_index)

            #     batch['external_src_indices'] = torch.stack([src_indices_b, src_indices_s])
            #     batch['external_tgt_indices'] = torch.stack([tgt_indices_b, tgt_indices_s])

            calibration_index = torch.arange(args.seq_length, device='cuda').view(2 * cp_size, args.seq_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)
            indices_b, indices_s = val.unbind(dim=0)
            mask = torch.isin(indices_s, calibration_index)
            if mask.any():
                selected_i = torch.any(mask, dim=1)
                num_images = selected_i.sum()

                src_indices_b = torch.arange(num_images, device='cuda').unsqueeze(1).repeat(1, indices_b.shape[1])
                src_indices_s = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(num_images, 1)

                src_indices_b = src_indices_b[mask[selected_i]]
                src_indices_s = src_indices_s[mask[selected_i]]

                tgt_indices_b = indices_b[mask]
                tgt_indices_s = indices_s[mask]
                # tgt_indices_s = (tgt_indices_s.view(-1, 1) == calibration_index).int().argmax(dim=1)
                tgt_indices_s = index_of_a_in_b(tgt_indices_s, calibration_index)

                batch['external_src_indices'] = torch.stack([src_indices_b, src_indices_s])
                batch['external_tgt_indices'] = torch.stack([tgt_indices_b, tgt_indices_s])

                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} calibration_index {calibration_index} {calibration_index.size()}", flush=True)
                # print(f"cp_rank {cp_rank} mask {mask} {mask.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_b {indices_b} {indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} selected_i {selected_i} {selected_i.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_s {indices_s} {indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_b {src_indices_b} {src_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_s {src_indices_s} {src_indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_b {tgt_indices_b} {tgt_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_s {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
            # print(f"cp_rank {cp_rank} {key} {val} {val.size()}", flush=True)
            batch.pop(key)
            continue

        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    args = get_args()

    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in list(batch.items()):
        # if "external_" in key and not args.vision_context_parallel:
        #     continue
        if key == "external_images":
            continue
        if key == "external_indices":
            calibration_index = torch.arange(args.seq_length, device='cuda').chunk(cp_size, dim=0)[cp_rank].contiguous()
            indices_b, indices_s = val.unbind(dim=0) # how to deal with dynamic num_image
            mask = torch.isin(indices_s, calibration_index)
            if mask.any():
                src_indices_b = torch.arange(indices_b.shape[0], device='cuda').unsqueeze(1).repeat(1, indices_b.shape[1])
                src_indices_s = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(indices_s.shape[0], 1)
                src_indices_b = src_indices_b[mask]
                src_indices_s = src_indices_s[mask]
                tgt_indices_b = indices_b[mask]
                tgt_indices_s = indices_s[mask]
                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} tgt_indices_s_ori {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
                # tgt_indices_s = (tgt_indices_s.view(-1, 1) == calibration_index).int().argmax(dim=1)
                tgt_indices_s = index_of_a_in_b(tgt_indices_s, calibration_index)
                batch['external_src_indices'] = torch.stack([src_indices_b, src_indices_s])
                batch['external_tgt_indices'] = torch.stack([tgt_indices_b, tgt_indices_s])

                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} calibration_index {calibration_index} {calibration_index.size()}", flush=True)
                # print(f"cp_rank {cp_rank} mask {mask} {mask.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_b {indices_b} {indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_s {indices_s} {indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_b {src_indices_b} {src_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_s {src_indices_s} {src_indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_b {tgt_indices_b} {tgt_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_s {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
            # print(f"cp_rank {cp_rank} {key} {val} {val.size()}", flush=True)
            batch.pop(key)
            continue

    attention_mask = get_attention_mask()
    if attention_mask is not None:
        if len(attention_mask.shape) != 2:
            raise AssertionError("The fusion attention operator currently only support 2D attention mask.")
        seq_dim = 0
        mask_row = attention_mask.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
        from megatron.training import get_args
        if get_args().attention_mask_on_cpu:
            mask_list = [m.contiguous().npu(non_blocking=True) for m in mask_row.chunk(cp_size, dim=1)]
        else:
            mask_list = [m.contiguous() for m in mask_row.chunk(cp_size, dim=1)]
        batch['attention_mask'] = mask_list
        set_attention_mask(mask_list)

    for key, val in batch.items():
        if key != 'attention_mask' and val is not None:
            seq_dim = 1
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_ulysses_cp(batch):
    args = get_args()
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in list(batch.items()):

        if key == "external_images":
            continue
        if key == "external_indices":
            calibration_index = torch.arange(args.seq_length, device='cuda').chunk(cp_size, dim=0)[cp_rank].contiguous()
            indices_b, indices_s = val.unbind(dim=0) # how to deal with dynamic num_image
            mask = torch.isin(indices_s, calibration_index)
            if mask.any():
                src_indices_b = torch.arange(indices_b.shape[0], device='cuda').unsqueeze(1).repeat(1, indices_b.shape[1])
                src_indices_s = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(indices_s.shape[0], 1)
                src_indices_b = src_indices_b[mask]
                src_indices_s = src_indices_s[mask]
                tgt_indices_b = indices_b[mask]
                tgt_indices_s = indices_s[mask]
                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} tgt_indices_s_ori {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
                # tgt_indices_s = (tgt_indices_s.view(-1, 1) == calibration_index).int().argmax(dim=1)
                tgt_indices_s = index_of_a_in_b(tgt_indices_s, calibration_index)
                batch['external_src_indices'] = torch.stack([src_indices_b, src_indices_s])
                batch['external_tgt_indices'] = torch.stack([tgt_indices_b, tgt_indices_s])

                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} calibration_index {calibration_index} {calibration_index.size()}", flush=True)
                # print(f"cp_rank {cp_rank} mask {mask} {mask.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_b {indices_b} {indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} indices_s {indices_s} {indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_b {src_indices_b} {src_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_s {src_indices_s} {src_indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_b {tgt_indices_b} {tgt_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_s {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
            # print(f"cp_rank {cp_rank} {key} {val} {val.size()}", flush=True)
            batch.pop(key)
            continue

        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch


def _get_batch_on_this_cp_rank_in_hybrid_cp(batch):
    args = get_args()
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    u_size = get_context_parallel_for_hybrid_ulysses_world_size()
    r_size = get_context_parallel_for_hybrid_ring_world_size()

    u_rank = get_context_parallel_for_hybrid_ulysses_rank()
    r_rank = get_context_parallel_for_hybrid_ring_rank()

    for key, val in list(batch.items()):
        if key == "external_images":
            continue
        if key == "external_indices":
            calibration_index = torch.arange(args.seq_length, device='cuda').view(2 * cp_size, args.seq_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)
            indices_b, indices_s = val.unbind(dim=0) # how to deal with dynamic num_image
            mask = torch.isin(indices_s, calibration_index)
            if mask.any():
                src_indices_b = torch.arange(indices_b.shape[0], device='cuda').unsqueeze(1).repeat(1, indices_b.shape[1])
                src_indices_s = torch.arange(indices_s.shape[1], device='cuda').unsqueeze(0).repeat(indices_s.shape[0], 1)
                src_indices_b = src_indices_b[mask]
                src_indices_s = src_indices_s[mask]
                tgt_indices_b = indices_b[mask]
                tgt_indices_s = indices_s[mask]
                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} tgt_indices_s_ori {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
                # tgt_indices_s = (tgt_indices_s.view(-1, 1) == calibration_index).int().argmax(dim=1)
                tgt_indices_s = index_of_a_in_b(tgt_indices_s, calibration_index)
                batch['external_src_indices'] = torch.stack([src_indices_b, src_indices_s])
                batch['external_tgt_indices'] = torch.stack([tgt_indices_b, tgt_indices_s])

                # torch.set_printoptions(threshold=100_000)
                # print(f"cp_rank {cp_rank} src_indices_b {src_indices_b} {src_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} src_indices_s {src_indices_s} {src_indices_s.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_b {tgt_indices_b} {tgt_indices_b.size()}", flush=True)
                # print(f"cp_rank {cp_rank} tgt_indices_s {tgt_indices_s} {tgt_indices_s.size()}", flush=True)
            # print(f"cp_rank {cp_rank} {key} {val} {val.size()}", flush=True)
            batch.pop(key)
            continue

        if key == 'attention_mask':
            continue
        if val is not None:
            seq_dim = 1 if key != 'attention_mask' else 2
            val = val.view(
                *val.shape[0:seq_dim],
                2 * r_size,
                val.shape[seq_dim] // (2 * r_size),
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([r_rank, (2 * r_size - r_rank - 1)], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
            val = val.chunk(u_size, dim=seq_dim)[u_rank].contiguous()
            batch[key] = val

    return batch

def index_of_a_in_b(a, b):
    b_indices = torch.where(torch.isin(b, a))[0]
    b_values = b[b_indices]
    return b_indices[b_values.argsort()[a.argsort().argsort()]]


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()
    assert args.bf16

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    def broadcast_dynamic(item):
        if item is not None:
            item = item.npu()
            item_len = torch.tensor(item.numel(), device=torch.cuda.current_device())
            _broadcast(item_len)
            _broadcast(item)
        else:
            item_len = torch.empty((), dtype=torch.int64, device=torch.cuda.current_device())
            _broadcast(item_len)
            item = torch.empty([item_len.item()], dtype=torch.int64, device=torch.cuda.current_device())
            _broadcast(item)

        return item

    if mpu.get_tensor_model_parallel_rank() == 0:
       if data_iterator is not None:
           while True:
               data = next(data_iterator)
               if "tokens" in data:
                   break
       else:
           data = None

       batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True),
       }
       if "images" in data:
           batch['external_images'] = data["images"].cuda(non_blocking = True).bfloat16()
       else:
           batch['external_images'] = torch.ones([len(data["tokens"]), 3, args.image_size, args.image_size]).cuda(non_blocking = True).bfloat16()

       # batch["external_input_ids"] = torch.zeros((batch["external_images"].size(0), args.vision_seq_length), dtype=torch.int64).cuda(non_blocking = True)
       # batch["external_position_ids"] = torch.arange(1, args.vision_seq_length + 1, dtype=torch.int64).unsqueeze(0).repeat(batch["external_images"].size(0), 1).cuda(non_blocking = True)

       external_images_sizes = torch.tensor(batch["external_images"].size()).cuda(non_blocking = True)

       if "image_indices" in data:
           batch["external_indices"] = data["image_indices"].cuda(non_blocking = True).to(torch.int64)
           external_indices_sizes = torch.tensor(batch["external_indices"].size()).cuda(non_blocking = True)
       else:
           external_indices_sizes = torch.tensor([0, 0, 0]).cuda(non_blocking = True)

       # for k, v in batch.items():
       #     if v is not None:
       #         print("get_batch_on_this_tp_rank", k, v.size())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(batch['tokens'])
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

           _broadcast(external_images_sizes)
           _broadcast(external_indices_sizes)

           _broadcast(batch['external_images'])
           if external_indices_sizes.sum() > 0:
               _broadcast(batch['external_indices'])
           # _broadcast(batch['external_input_ids'])
           # _broadcast(batch['external_position_ids'])

       elif mpu.is_pipeline_first_stage():
           _broadcast(batch['tokens'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

           _broadcast(external_images_sizes)
           _broadcast(external_indices_sizes)

           _broadcast(batch['external_images'])
           if external_indices_sizes.sum() > 0:
               _broadcast(batch['external_indices'])
           # _broadcast(batch['external_input_ids'])
           # _broadcast(batch['external_position_ids'])

       elif mpu.is_pipeline_last_stage():
           _broadcast(batch['labels'])
           _broadcast(batch['loss_mask'])
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       else:
           _broadcast(batch['attention_mask'])
           _broadcast(batch['position_ids'])

       if args.reset_attention_mask:
           actual_seq_len = broadcast_dynamic(data['actual_seq_len'])
           set_actual_seq_len(actual_seq_len.tolist())  
           # print(f"actual_seq_len {actual_seq_len}")

    else:

       tokens=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           attention_mask=torch.empty(
                (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
       else:
           attention_mask=None
       position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())

       external_images_sizes = torch.empty((4), dtype = torch.int64 , device = torch.cuda.current_device())
       external_indices_sizes = torch.empty((3), dtype = torch.int64 , device = torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)

           _broadcast(external_images_sizes)
           _broadcast(external_indices_sizes)

           external_images = torch.empty(external_images_sizes.tolist(), dtype = torch.bfloat16 , device = torch.cuda.current_device())
           _broadcast(external_images)

           if external_indices_sizes.sum() > 0:
               external_indices = torch.empty(external_indices_sizes.tolist(), dtype = torch.int64 , device = torch.cuda.current_device())
               _broadcast(external_indices)
           else:
               external_indices = None

           # external_input_ids = torch.empty((external_images_sizes[0], args.vision_seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
           # _broadcast(external_input_ids)

           # external_position_ids = torch.empty((external_images_sizes[0], args.vision_seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
           # _broadcast(external_position_ids)
 
       elif mpu.is_pipeline_first_stage():
           labels=None
           loss_mask=None
   
           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

           _broadcast(external_images_sizes)
           _broadcast(external_indices_sizes)

           external_images = torch.empty(external_images_sizes.tolist(), dtype = torch.bfloat16 , device = torch.cuda.current_device())
           _broadcast(external_images)

           if external_indices_sizes.sum() > 0:
               external_indices = torch.empty(external_indices_sizes.tolist(), dtype = torch.int64 , device = torch.cuda.current_device())
               _broadcast(external_indices)
           else:
               external_indices = None

           # external_input_ids = torch.empty((external_images_sizes[0], args.vision_seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
           # _broadcast(external_input_ids)

           # external_position_ids = torch.empty((external_images_sizes[0], args.vision_seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
           # _broadcast(external_position_ids)

       elif mpu.is_pipeline_last_stage():
           tokens=None
           # position_ids=None
    
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)

           external_images = None
           external_indices = None
           # external_input_ids = None
           # external_position_ids = None

       else:
           _broadcast(attention_mask)
           _broadcast(position_ids)

           external_images = None
           external_indices = None
           # external_input_ids = None
           # external_position_ids = None

       if args.reset_attention_mask:
           actual_seq_len = broadcast_dynamic(None)
           set_actual_seq_len(actual_seq_len.tolist())
 
       batch = {
           'tokens': tokens,
           'labels': labels,
           'loss_mask': loss_mask,
           'attention_mask': attention_mask,
           'position_ids': position_ids,
       }

       if external_images is not None:
           batch["external_images"] = external_images
       if external_indices is not None:
           batch["external_indices"] = external_indices
       # if external_input_ids is not None:
       #     batch["external_input_ids"] = external_input_ids
       # if external_position_ids is not None:
       #     batch["external_position_ids"] = external_position_ids

    # print("batch", {k:v.size() if v is not None else None for k, v in batch.items()})
    return batch
