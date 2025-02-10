# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial

from typing import Union

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

from long_vita import build_supervised_dataset_megatron
import logging
logger = logging.getLogger("__name__")


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
        print_rank_0("Building megatron mcore language model ...")

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
        model = GPTModel(
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
        )
    else:
        assert False
        assert (
            args.context_parallel_size == 1
        ), "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )

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
        tokens, labels, loss_mask, attention_mask, position_ids, _ = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

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
            shift_token=True,
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
    group.add_argument("--max-num-image", type=int, default=8, help='max_num_image')
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

