# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from logging import getLogger
from typing import Callable, Dict, List, Optional

import torch

from megatron.core import mpu

from megatron.core.transformer.module import MegatronModule

from megatron.training import get_args

logger = getLogger(__name__)


def _get_param_groups(
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Callable,
    scale_lr_cond: Callable,
    lr_mult: float,
    use_decoupled_learning_rate: bool,
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups based on weight decay condition (regularized vs
    non regularized), learning rate scale condition (lr vs lr_mult * lr),
    and whether it is expert parameters. scale_lr_cond is used during finetuning
    where head of the network requires a scaled version of the base learning rate.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        no_weight_decay_cond (func): function to determine whether a parameter
            should not perform weight decay.
        scale_lr_cond (func): function to determine whether a parameter
            should have a scaled learning rate.
        lr_mult (float): learning rate multiplier for parameters that
            satisfy scale_lr_cond.
        use_decoupled_learning_rate (bool): true if using decoupled learning rate.

    Returns:
        List of parameter groups.
    """
    args = get_args()

    # Map (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr) to params.
    params_map = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            is_expert_parallel = not getattr(param, 'allreduce', True)

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # Do not regularize biases and norm parameters.
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_mult, lr_mult = 1.0, 1.0
            elif not no_wd and scale_lr:
                wd_mult, lr_mult = 1.0, lr_mult
            elif no_wd and not scale_lr:
                wd_mult, lr_mult = 0.0, 1.0
            else:
                wd_mult, lr_mult = 0.0, lr_mult

            if ".vit." in name:
                lr_mult = args.vision_model_lr_mult
                lr_decay_rate = get_vit_lr_decay_rate(name)
                lr_mult = lr_mult * lr_decay_rate

                print(f"name {name} lr_decay_rate {lr_decay_rate}")
                logger.info(f"name {name} lr_decay_rate {lr_decay_rate}")

            is_decoupled_lr = False
            # For input/embedding and output layer: embedding.word_embeddings.weight / output_layer.weight.
            if use_decoupled_learning_rate and getattr(
                param, 'is_embedding_or_output_parameter', False
            ):
                is_decoupled_lr = True

            key = (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

            print(f"_get_param_groups name {name} key {key}")
            logger.info(f"_get_param_groups name {name} key {key}")

    param_groups = []
    for (wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr), params in params_map.items():
        assert len(params) > 0
        param_groups.append(
            {
                'params': params,
                'wd_mult': wd_mult,
                'lr_mult': lr_mult,
                'is_expert_parallel': is_expert_parallel,
                'is_decoupled_lr': is_decoupled_lr,
            }
        )

    # print(f"param_groups {param_groups}")
    return param_groups


def get_vit_lr_decay_rate(name):
    args = get_args()
    num_layers = args.vit_num_layers
    lr_decay_rate = args.vision_model_lr_decay_rate

    layer_id = num_layers + 1
    if ".vit." in name:
        if ".position_embeddings." in name or ".conv1." in name:
            layer_id = 0
        elif ".layers." in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
