# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Union

import torch

# from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import mpu


# Note: This is under development and is missing features like position embedding interpolation.
class CLIPViTModel(VisionModule):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        ln_pre_impl: Union[ModuleSpec, type] = TENorm,
        add_class_token: bool = True,
        class_token_len: int = 1,
        patch_dim: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        vision_context_parallel: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)

        # if has_config_logger_enabled(transformer_config):
        #     log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w

        assert self.img_h % self.patch_dim == 0
        assert self.img_w % self.patch_dim == 0
        self.num_patches_per_dim_h = self.img_h // self.patch_dim
        self.num_patches_per_dim_w = self.img_w // self.patch_dim
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w

        self.add_class_token = add_class_token
        self.class_token_len = class_token_len

        self.seq_length = self.num_patches + (self.class_token_len if self.add_class_token else 0)

        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=self.visual_hidden_size,
            kernel_size=self.patch_dim,
            stride=self.patch_dim,
            bias=False,
        )

        self.position_ids = torch.arange(self.seq_length).expand(1, -1).cuda()
        if not add_class_token:
            self.position_ids = torch.arange(1, self.seq_length + 1).expand(1, -1).cuda()

        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)
        if not add_class_token:
            self.position_embeddings = torch.nn.Embedding(self.seq_length + 1, self.visual_hidden_size)

        self.add_class_token = add_class_token
        if self.add_class_token:
            self.class_token = torch.nn.Parameter(
                torch.randn(1, self.class_token_len, self.visual_hidden_size)
            )
        if not self.add_class_token:
            self.class_token = torch.nn.Parameter(
                torch.randn(1, self.class_token_len, self.visual_hidden_size)
            )
            self.class_token.requires_grad = False

        self.ln_pre = build_module(
            ln_pre_impl,
            config=transformer_config,
            hidden_size=self.visual_hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        # TODO: Make pre_process and post_process configurable.
        # NOTE: a final layer norm and/or linear layer in some implementations are omitted here.
        # They can be added separately where needed.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
        )

        self.vision_context_parallel = vision_context_parallel

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of the CLIP ViT Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input data of shape [batch, img_h, img_w]
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        x = self.conv1(x)  # shape = [batch, hidden_size, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]

        if self.add_class_token:
            class_token = self.class_token.expand(
                x.shape[0], -1, -1
            )  # [batch, class_token_len, hidden_size]
            x = torch.cat(
                [class_token, x], dim=1
            )  # [batch, grid ** 2 + class_token_len, hidden_size]

        assert x.shape[1] == self.seq_length, f"{x.shape[1]} != {self.seq_length}"
        x = x + self.position_embeddings(self.position_ids)
        x = self.ln_pre(x)

        if mpu.get_context_parallel_world_size() != 1 and self.vision_context_parallel:
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
            val = x
            seq_dim = 1
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1) :],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], 
                                device="cpu", pin_memory=True).cuda(non_blocking=True)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
            x = val

        x = x.permute(1, 0, 2)  # [b, s, h] -> [s, b, h]
        # `permute` can make the tensor non-contiguous, breaking pipelining.
        x = x.contiguous()

        if attention_mask is None and False:
            attention_mask = torch.ones(
                1, 1, self.seq_length, self.seq_length
            ).cuda()  # [1, 1, s, s]
            attention_mask = attention_mask < 0.5  # to bool

        x = self.decoder(x, attention_mask)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        return x
