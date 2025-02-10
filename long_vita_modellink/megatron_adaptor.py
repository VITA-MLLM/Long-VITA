import os
import sys
import argparse
from functools import wraps
import torch
from torch_npu.contrib import transfer_to_npu
import importlib.util
import megatron


def modellink_adaptation(aspm):
    from long_vita_modellink.core.transformer.dot_product_attention import flash_attention_forward
    aspm.register_patch('modellink.core.transformer.dot_product_attention.flash_attention_forward', flash_attention_forward)


def megatron_legacy_adaptation(aspm):
    from long_vita_modellink.legacy.data.data_samplers import build_pretraining_data_loader
    megatron.legacy.data.data_samplers.build_pretraining_data_loader = build_pretraining_data_loader
    megatron.training.training.build_pretraining_data_loader = build_pretraining_data_loader

    # from long_vita_modellink.legacy.model.transformer import _get_num_layers
    # aspm.register_patch('megatron.legacy.model.transformer._get_num_layers', _get_num_layers)
    # megatron.legacy.model.transformer._get_num_layers = _get_num_layers

    # from mindspeed.model.transformer import parallel_transformer_init_wrapper, parallel_transformer_forward_wrapper
    # from mindspeed.core.transformer.transformer import parallel_transformer_checkpointed_forward_wrapper
    # aspm.register_patch('long_vita_modellink.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init_wrapper)
    # aspm.register_patch('long_vita_modellink.legacy.model.transformer.ParallelTransformer.forward', parallel_transformer_forward_wrapper)
    # aspm.register_patch('long_vita_modellink.legacy.model.transformer.ParallelTransformer._checkpointed_forward', parallel_transformer_checkpointed_forward_wrapper)

    # from long_vita_modellink.legacy.model.transformer import ParallelTransformer
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__, force_patch=True)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer', ParallelTransformer)

    # spec = importlib.util.find_spec("mindspeed.model.language_model")
    # if spec is not None:
    #     from mindspeed.model.language_model import parallel_lm_logits, embedding_forward_wrapper
    #     aspm.register_patch('long_vita_modellink.legacy.model.vision_language_model.parallel_lm_logits', parallel_lm_logits)
    #     aspm.register_patch('long_vita_modellink.legacy.model.vision_language_model.Embedding.forward', embedding_forward_wrapper)

    # spec = importlib.util.find_spec("mindspeed.model.gpt_model")
    # if spec is not None:
    #     from mindspeed.model.gpt_model import post_language_model_processing_wrapper
    #     aspm.register_patch('long_vita_modellink.legacy.model.gpt_vl_model.post_language_model_processing', post_language_model_processing_wrapper)

    # spec = importlib.util.find_spec("mindspeed.core.transformer.custom_layers.transformer_engine")
    # if spec is not None:
    #     from mindspeed.core.transformer.custom_layers.transformer_engine import PTNorm
    #     aspm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TENorm', PTNorm)

    # from mindspeed.optimizer.optimizer import (mixed_precision_optimizer_step, \
    #                                   reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper)
    # from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper

    # aspm.register_patch('megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step',
    #                     mixed_precision_optimizer_step)
    # aspm.register_patch('megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__',
    #                     reuse_fp32_param_init_wrapper)
    # aspm.register_patch('megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__',
    #                     optimizer_config_init_wrapper)
    # aspm.register_patch('megatron.core.optimizer.distrib_optimizer.DistributedOptimizer.__init__',
    #                     reuse_fp32_param_distrib_optimizer_init_wrapper)


def megatron_core_adaptation(aspm):
    # from long_vita_modellink.core.models.vision.vit_layer_specs import get_vit_layer_local_spec_for_eva_clip
    # aspm.register_patch('megatron.core.models.vision.vit_layer_specs.get_vit_layer_with_transformer_engine_spec_for_eva_clip',
    #                     get_vit_layer_local_spec_for_eva_clip)

    # from llava_megatron_THUDM.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper
    # aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
    #                     get_gpt_layer_local_spec_wrapper)

    # from long_vita_modellink.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    # aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec', get_gpt_layer_local_spec, force_patch=True)
    # aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec', get_gpt_layer_local_spec, force_patch=True)

    from long_vita_modellink.core.transformer.transformer_block import get_num_layers_to_build
    aspm.register_patch('megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)

    from long_vita_modellink.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    aspm.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding', LanguageModelEmbedding)

    from long_vita_modellink.core.transformer.transformer_config import TransformerConfig
    aspm.register_patch('megatron.core.transformer.transformer_config.TransformerConfig', TransformerConfig)

    # from long_vita_modellink.core.transformer.transformer_layer import TransformerLayer
    # aspm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset', TransformerLayer._get_layer_offset)

    # from long_vita_modellink.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    # aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding', RotaryEmbedding)

def megatron_adaptation(aspm):
    from long_vita_modellink.training.checkpointing import ensure_directory_exists
    aspm.register_patch('megatron.training.checkpointing.ensure_directory_exists', ensure_directory_exists)

def optimizer_adaptation(aspm):
    # from long_vita_modellink.core.optimizer.__init__ import _get_param_groups
    # aspm.register_patch('megatron.core.optimizer.__init__._get_param_groups', _get_param_groups)
    # megatron.core.optimizer.__init__._get_param_groups = _get_param_groups
    from long_vita_modellink.core.optimizer import _get_param_groups
    megatron.core.optimizer._get_param_groups = _get_param_groups


def exe_adaptation():
    from mindspeed.patch_utils import MindSpeedPatchesManager as aspm
    megatron_core_adaptation(aspm)
    megatron_legacy_adaptation(aspm)

    megatron_adaptation(aspm)

    modellink_adaptation(aspm)

    optimizer_adaptation(aspm)

    aspm.apply_patches()

exe_adaptation()

# from modellink.patchs.megatron_mock import mock_megatron_dependencies, patch_npu_apex_torch
# mock_megatron_dependencies()
# patch_npu_apex_torch()
