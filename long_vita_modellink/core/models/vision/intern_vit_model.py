
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

from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor, TransformerLayerSubmodules

from megatron.core.transformer.attention import Attention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim

class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.query_projection_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.kv_projection_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    def run_realtime_tests(self):
        """Performs a consistency check.

        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during data transmission).

        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably not
        required; transmitting hashes is sufficient."""

        if self.config.qk_layernorm:
            # check that all tensor parallel and data parallel ranks have the same
            # Q & K layernorm parameters.
            rank = get_data_parallel_rank()
            inputs = torch.stack(
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ]
            )
            dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
            dp_list[rank] = inputs
            torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())

            def _compare(srcs, tgts, names, parallelism):
                assert len(srcs) == len(tgts) == len(names)
                for src, tgt, name in zip(srcs, tgts, names):
                    assert torch.all(
                        src == tgt
                    ), f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. Diff: {torch.norm(src - tgt)}"

            for i, dp in enumerate(dp_list):
                q_w, q_b, k_w, k_b = torch.unbind(dp)
                _compare(
                    [q_w, q_b, k_w, k_b],
                    [
                        self.q_layernorm.weight.data,
                        self.q_layernorm.bias.data,
                        self.k_layernorm.weight.data,
                        self.k_layernorm.bias.data,
                    ],
                    ["q_w", "q_b", "k_w", "k_b"],
                    "DP",
                )

            rank = get_tensor_model_parallel_rank()
            tp_list = [
                torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())
            ]
            tp_list[rank] = inputs
            torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())

            for i, tp in enumerate(tp_list):
                q_w, q_b, k_w, k_b = torch.unbind(tp)
                _compare(
                    [q_w, q_b, k_w, k_b],
                    [
                        self.q_layernorm.weight.data,
                        self.q_layernorm.bias.data,
                        self.k_layernorm.weight.data,
                        self.k_layernorm.bias.data,
                    ],
                    ["q_w", "q_b", "k_w", "k_b"],
                    "TP",
                )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,)
        else:

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np * hn]
        query = query.reshape(query.size(0), query.size(1), -1)
        key = key.reshape(key.size(0), key.size(1), -1)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        key = key.reshape(key.size(0), key.size(1), -1, self.hidden_size_per_attention_head)

        return query, key, value


class InternViTTransformerLayer(TransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, hidden_dropout=hidden_dropout)

        self.ls1 = torch.nn.Parameter(0.01 * torch.ones(config.hidden_size))
        self.ls2 = torch.nn.Parameter(0.01 * torch.ones(config.hidden_size))

    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if attention_bias is not None:
            attention_output = attention_output + attention_bias

        # print(f"hidden_states {hidden_states.size()}")
        hidden_states = residual + attention_output * self.ls1

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output, mlp_bias = self.mlp(pre_mlp_layernorm_output)

        if mlp_bias is not None:
            mlp_output = mlp_output + mlp_bias

        hidden_states = residual + mlp_output * self.ls2

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context


# Note: This is under development and is missing features like position embedding interpolation.
class InternViTModel(VisionModule):
    """CLIP ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
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
            bias=True,
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
        # print("InternViTModel x", x.size())

        if attention_mask is None and False:
            attention_mask = torch.ones(
                1, 1, self.seq_length, self.seq_length
            ).cuda()  # [1, 1, s, s]
            attention_mask = attention_mask < 0.5  # to bool

        x = self.decoder(x, attention_mask)
        x = x.permute(1, 0, 2)  # [s, b, h] -> [b, s, h]
        x = x.contiguous()

        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        # print(f"eva_clip_model {torch.cuda.memory_summary()}")
        # torch.cuda.empty_cache()

        # print("EVA2ViTModel hidden_states", hidden_states.size())
        return x
