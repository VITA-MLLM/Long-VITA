# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules



from long_vita_modellink.core.models.vision.eva_vit_model import EVAViTTransformerLayer
from long_vita_modellink.core.models.vision.intern_vit_model import InternViTTransformerLayer
from long_vita_modellink.core.models.vision.siglip_vit_model import SigLIPViTTransformerLayer
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
)


def get_vit_layer_local_spec_for_siglip(use_te=True) -> ModuleSpec:
    return ModuleSpec(
        module=SigLIPViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=RowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            pre_mlp_layernorm=TENorm,
            input_layernorm=TENorm,
        ),
    )

def get_vit_layer_with_transformer_engine_spec_for_intern(use_te=True) -> ModuleSpec:
    return ModuleSpec(
        module=InternViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            pre_mlp_layernorm=IdentityOp,
            input_layernorm=IdentityOp,
        ),
    )

def get_vit_layer_local_spec_for_intern(use_te=True) -> ModuleSpec:
    return ModuleSpec(
        module=InternViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=RowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            pre_mlp_layernorm=TENorm,
            input_layernorm=TENorm,
        ),
    )

def get_vit_layer_with_transformer_engine_spec_for_eva(use_te=True) -> ModuleSpec:
    return ModuleSpec(
        module=EVAViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            pre_mlp_layernorm=TENorm,
            input_layernorm=TENorm,
        ),
    )

def get_vit_layer_local_spec_for_eva(use_te=True) -> ModuleSpec:
    return ModuleSpec(
        module=EVAViTTransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear if use_te else ColumnParallelLinear,
                    linear_fc2=RowParallelLinear if use_te else RowParallelLinear,
                ),
            ),
            pre_mlp_layernorm=TENorm,
            input_layernorm=TENorm,
        ),
    )

def get_vit_layer_spec(use_te=True) -> ModuleSpec:
    mlp = get_mlp_module_spec(use_te=False)

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TorchLayerNormWrapper,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TorchLayerNormWrapper,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )

def get_mlp_module_spec(use_te: bool = True) -> ModuleSpec:
    # Dense MLP w/ or w/o TE modules.
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
            linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
        ),
                
    )

def get_mlp_module_spec_te() -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TELayerNormColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )

class TorchLayerNormWrapper(torch.nn.LayerNorm):
    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps)
