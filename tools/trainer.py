# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)

if is_datasets_available():
    import datasets

from transformers import Trainer as HFTrainer

DATA_PRINT_ONCE = True
BATCH = None
def print_batch(batch, tokenizer, args):

    global DATA_PRINT_ONCE
    global BATCH

    if batch is not None:
        BATCH = batch
    else:
        batch = BATCH
        DATA_PRINT_ONCE = True

    if batch is None:
        return

    if DATA_PRINT_ONCE:

        global_rank = torch.distributed.get_rank()
        f = open(os.path.join(args.output_dir, f"print_batch_{global_rank}.log"), "a")

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

        if "input_ids" in batch and batch["input_ids"] is not None:
            tokens = batch["input_ids"]
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

            # labels__ = labels.cpu().clone().detach()
            # labels__[loss_mask.to(torch.int64)==0] = tokenizer("-", add_special_tokens=False).input_ids[0]
            # labels__ = tokenizer.batch_decode(labels__.tolist(), skip_special_tokens=False)
            # print(f"labels__ {labels__}", file=f)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} {v} {v.size()}", file=f)
            else:
                print(f"{k} {v}", file=f)

        f.close()

    DATA_PRINT_ONCE = False


class Trainer(HFTrainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "spawn",
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            if args.vision_model_lr_mult != 1.0 or args.vision_model_lr_decay_rate != 1.0:
                vision_parameters = [name for name, _ in opt_model.named_parameters() if "vision_model" in name]
            else:
                vision_parameters = []
            print("vision_parameters {vision_parameters}")

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in vision_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in vision_parameters)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if args.vision_model_lr_decay_rate != 1.0:
                for n, p in opt_model.named_parameters():
                    if p.requires_grad and n in vision_parameters:
                        pass
                    else:
                        continue

                    if n in decay_parameters:
                        weight_decay = self.args.weight_decay
                    else:
                        weight_decay = 0.0

                    lr = get_vit_lr_decay_rate(n, opt_model.config.visual.num_hidden_layers, self.args.vision_model_lr_decay_rate)

                    optimizer_grouped_parameters.append(
                        {
                            "params": [p],
                            "weight_decay": weight_decay,
                            "lr": lr,
                        }
                    )
                    print(f"create_optimizer name {n} weight_decay {weight_decay} lr {lr}")

            elif args.vision_model_lr_mult != 1.0:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in vision_parameters)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_model_lr_mult,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in vision_parameters)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_model_lr_mult,
                        },
                    ]
                )
                print(f"create_optimizer name {[n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in vision_parameters)]} weight_decay {self.args.weight_decay} lr {self.args.vision_model_lr_mult}")
                print(f"create_optimizer name {[n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in vision_parameters)]} weight_decay {0.0} lr {self.args.vision_model_lr_mult}")

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        print_batch(inputs, self.processing_class, self.args)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()

    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                while True:
                    batch_sample = next(epoch_iterator)
                    if "input_ids" in batch_sample:
                        break
                batch_samples += [batch_sample]
            except StopIteration:
                break

        # Keep default behavior the same
        if not self.model_accepts_loss_kwargs:
            return batch_samples, None
        return batch_samples, None

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if self.args.average_tokens_across_devices:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()
        return batch_samples, num_items_in_batch


def get_vit_lr_decay_rate(name, num_layers, lr_decay_rate):

    layer_id = num_layers + 1
    if "vision_model." in name:
        if ".position_embedding." in name or ".conv1." in name:
            layer_id = 0
        elif ".layers." in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
