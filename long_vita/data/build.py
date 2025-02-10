import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset

from .data_collator import DataCollatorForSupervisedDataset, collate_fn_deepspeed
from .dataset_llama2 import Llama2Dataset
from .dataset_llama3 import Llama3Dataset
from .dataset_mistral import MistralDataset
from .dataset_qwen2 import Qwen2Dataset
from .dataset_vicuna import VicunaDataset

logger = logging.getLogger("__name__")


def build_supervised_dataset_deepspeed(
    model_config,
    model_args,
    data_args,
    training_args,
    tokenizer,
    create_position_ids=True,
    create_loss_mask=False,
    shift_token=False,
):
    logging.info("building dataset...")

    cfg_path = data_args.dataset_name
    model_max_length = model_args.model_max_length
    output_dir = training_args.output_dir

    # prompt_format = model_args.prompt_format

    create_attention_mask = data_args.create_attention_mask
    create_attention_mask_2d = data_args.create_attention_mask_2d

    image_size = model_args.image_size
    image_token_length = model_args.image_token_length

    max_num_frame = model_args.max_num_frame
    max_fps = model_args.max_fps

    reset_position_ids = data_args.reset_position_ids
    reset_attention_mask = data_args.reset_attention_mask
    variable_length = data_args.variable_length

    min_patch_grid = model_args.min_patch_grid
    max_patch_grid = model_args.max_patch_grid
    process_type = model_args.vision_process_type
    normalize_type = model_args.vision_normalize_type

    seed = training_args.seed
    cross_dataset_joint = data_args.cross_dataset_joint
    dataset_joint = data_args.dataset_joint

    if "qwen2" in getattr(model_config, "model_type", None):
        TrainDataset = Qwen2Dataset
    elif "qwen" in getattr(model_config, "model_type", None):
        raise NotImplementedError
        TrainDataset = Qwen2Dataset
    elif getattr(model_config, "model_type", None) == "llama":
        raise NotImplementedError
        TrainDataset = Llama2Dataset
    elif getattr(model_config, "model_type", None) == "mixtral":
        raise NotImplementedError
        TrainDataset = Llama2Dataset
    else:
        raise NotImplementedError

    train_dataset = TrainDataset(
        cfg_path,
        tokenizer,
        image_size=image_size,
        image_token_length=image_token_length,
        model_max_length=model_max_length,
        variable_length=variable_length,
        output_dir=output_dir,
        training_args=None,
        shift_token=shift_token,
        create_position_ids=create_position_ids,
        create_attention_mask=create_attention_mask,
        create_attention_mask_2d=create_attention_mask_2d,
        create_loss_mask=create_loss_mask,
        max_num_frame=max_num_frame,
        max_fps=max_fps,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        min_patch_grid=min_patch_grid,
        max_patch_grid=max_patch_grid,
        process_type=process_type,
        normalize_type=normalize_type,
        seed=seed,
        cross_dataset_joint=cross_dataset_joint,
        dataset_joint=dataset_joint,
        audio_tokenizer_type=audio_tokenizer_type,
        audio_tokenizer_path=audio_tokenizer_path,
        # audio_tokenizer=audio_tokenizer,
    )
    eval_dataset = None

    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = collate_fn_deepspeed

    return dict(train=train_dataset, validation=eval_dataset, data_collator=data_collator)


def build_supervised_dataset_megatron(
    args,
    tokenizer,
    create_position_ids=True,
    create_loss_mask=False,
    shift_token=False,
):
    logging.info("building dataset...")

    assert len(args.data_path) == 1
    cfg_path = args.data_path[0]
    model_max_length = args.data_seq_length
    output_dir = args.save

    prompt_format = args.prompt_format

    create_attention_mask = args.create_attention_mask_in_dataloader
    create_attention_mask_2d = args.create_attention_mask_in_dataloader
    # create_attention_mask=False
    # create_attention_mask_2d=True

    image_size = args.image_size
    image_token_length = args.image_token_length

    max_num_frame = args.max_num_frame
    max_fps = args.max_fps

    reset_position_ids = args.reset_position_ids
    reset_attention_mask = args.reset_attention_mask
    # reset_position_ids=True
    # reset_attention_mask=True

    min_patch_grid = args.min_patch_grid
    max_patch_grid = args.max_patch_grid
    process_type = args.vision_process_type
    normalize_type = args.vision_normalize_type

    seed = args.seed
    cross_dataset_joint = args.cross_dataset_joint

    if "qwen2" in prompt_format:
        TrainDataset = Qwen2Dataset
    elif prompt_format == "llama2":
        raise NotImplementedError
        TrainDataset = Llama2Dataset
    elif prompt_format == "mistral":
        raise NotImplementedError
        TrainDataset = MistralDataset
    elif prompt_format == "vicuna":
        raise NotImplementedError
        TrainDataset = VicunaDataset
    elif prompt_format == "llama3":
        raise NotImplementedError
        TrainDataset = Llama3Dataset
    else:
        raise NotImplementedError

    train_dataset = TrainDataset(
        cfg_path,
        tokenizer,
        image_size=image_size,
        image_token_length=image_token_length,
        model_max_length=model_max_length,
        variable_length=False,
        output_dir=output_dir,
        training_args=None,
        shift_token=shift_token,
        create_position_ids=create_position_ids,
        create_attention_mask=create_attention_mask,
        create_attention_mask_2d=create_attention_mask_2d,
        create_loss_mask=create_loss_mask,
        max_num_frame=max_num_frame,
        max_fps=max_fps,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        min_patch_grid=min_patch_grid,
        max_patch_grid=max_patch_grid,
        process_type=process_type,
        normalize_type=normalize_type,
        seed=seed,
        cross_dataset_joint=cross_dataset_joint,
    )
    eval_dataset = None

    return train_dataset, None, None

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
