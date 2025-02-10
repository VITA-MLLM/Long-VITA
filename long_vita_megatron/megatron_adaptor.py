import os
import sys
import argparse
from functools import wraps
import torch
import importlib.util
import megatron


def attention_adaptation(aspm):
    # from long_vita_megatron.core.transformer.dot_product_attention import flash_attention_forward
    # aspm.register_patch('modellink.core.transformer.dot_product_attention.flash_attention_forward', flash_attention_forward)
    # aspm.register_patch('megatron.core.transformer.dot_product_attention.flash_attention_forward', flash_attention_forward)

    # aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward', dot_product_attention_forward_wrapper)

    from long_vita_megatron.core.transformer.dot_product_attention import dot_product_attention_forward_wrapper, \
        dot_product_attention_init_wrapper
    # aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
    #                     dot_product_attention_init_wrapper)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                        dot_product_attention_forward_wrapper)


def megatron_legacy_adaptation(aspm):
    from long_vita_megatron.legacy.data.data_samplers import build_pretraining_data_loader
    megatron.legacy.data.data_samplers.build_pretraining_data_loader = build_pretraining_data_loader
    megatron.training.training.build_pretraining_data_loader = build_pretraining_data_loader

    # from long_vita_megatron.legacy.model.transformer import _get_num_layers
    # aspm.register_patch('megatron.legacy.model.transformer._get_num_layers', _get_num_layers)
    # megatron.legacy.model.transformer._get_num_layers = _get_num_layers

    # from mindspeed.model.transformer import parallel_transformer_init_wrapper, parallel_transformer_forward_wrapper
    # from mindspeed.core.transformer.transformer import parallel_transformer_checkpointed_forward_wrapper
    # aspm.register_patch('long_vita_megatron.legacy.model.transformer.ParallelTransformer.__init__', parallel_transformer_init_wrapper)
    # aspm.register_patch('long_vita_megatron.legacy.model.transformer.ParallelTransformer.forward', parallel_transformer_forward_wrapper)
    # aspm.register_patch('long_vita_megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward', parallel_transformer_checkpointed_forward_wrapper)

    # from long_vita_megatron.legacy.model.transformer import ParallelTransformer
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer.__init__', ParallelTransformer.__init__, force_patch=True)
    # aspm.register_patch('megatron.legacy.model.transformer.ParallelTransformer', ParallelTransformer)

    # spec = importlib.util.find_spec("mindspeed.model.language_model")
    # if spec is not None:
    #     from mindspeed.model.language_model import parallel_lm_logits, embedding_forward_wrapper
    #     aspm.register_patch('long_vita_megatron.legacy.model.vision_language_model.parallel_lm_logits', parallel_lm_logits)
    #     aspm.register_patch('long_vita_megatron.legacy.model.vision_language_model.Embedding.forward', embedding_forward_wrapper)

    # spec = importlib.util.find_spec("mindspeed.model.gpt_model")
    # if spec is not None:
    #     from mindspeed.model.gpt_model import post_language_model_processing_wrapper
    #     aspm.register_patch('long_vita_megatron.legacy.model.gpt_vl_model.post_language_model_processing', post_language_model_processing_wrapper)

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
    # from long_vita_megatron.core.models.vision.vit_layer_specs import get_vit_layer_local_spec_for_eva_clip
    # aspm.register_patch('megatron.core.models.vision.vit_layer_specs.get_vit_layer_with_transformer_engine_spec_for_eva_clip',
    #                     get_vit_layer_local_spec_for_eva_clip)

    from long_vita_megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec',
                        get_gpt_layer_local_spec)

    from long_vita_megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    aspm.register_patch(
        'megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec',
        get_gpt_layer_with_transformer_engine_spec)

    # from long_vita_megatron.core.transformer.transformer_block import get_num_layers_to_build
    # aspm.register_patch('megatron.core.transformer.transformer_block.get_num_layers_to_build', get_num_layers_to_build)

    from long_vita_megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    aspm.register_patch('megatron.core.models.common.embeddings.language_model_embedding.LanguageModelEmbedding', LanguageModelEmbedding)

    from long_vita_megatron.core.transformer.transformer_config import TransformerConfig
    aspm.register_patch('megatron.core.transformer.transformer_config.TransformerConfig', TransformerConfig)

    # from long_vita_megatron.core.transformer.transformer_layer import TransformerLayer
    # aspm.register_patch('megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset', TransformerLayer._get_layer_offset)

    # from long_vita_megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    # aspm.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding', RotaryEmbedding)

    from long_vita_megatron.core.tensor_parallel.layers import ColumnParallelLinear
    aspm.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear', ColumnParallelLinear)


def megatron_adaptation(aspm):
    from long_vita_megatron.training.checkpointing import ensure_directory_exists
    aspm.register_patch('megatron.training.checkpointing.ensure_directory_exists', ensure_directory_exists)

def optimizer_adaptation(aspm):
    # from long_vita_megatron.core.optimizer.__init__ import _get_param_groups
    # aspm.register_patch('megatron.core.optimizer.__init__._get_param_groups', _get_param_groups)
    # megatron.core.optimizer.__init__._get_param_groups = _get_param_groups
    from long_vita_megatron.core.optimizer import _get_param_groups
    megatron.core.optimizer._get_param_groups = _get_param_groups

def mcore_parallel_state_adaptation(aspm):
    import megatron.core
    from long_vita_megatron.core.parallel_state import initialize_model_parallel_wrapper, initialize_model_parallel
    from long_vita_megatron.core.parallel_state import destroy_model_parallel_wrapper
    from long_vita_megatron.core.parallel_state import get_context_parallel_group_for_send_recv_overlap
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.initialize_model_parallel',
                        initialize_model_parallel)
    aspm.register_patch('megatron.core.parallel_state.destroy_model_parallel',
                        destroy_model_parallel_wrapper)
    aspm.register_patch('megatron.core.parallel_state.get_context_parallel_group_for_send_recv_overlap',
                        get_context_parallel_group_for_send_recv_overlap)
    aspm.register_patch('megatron.core.mpu', megatron.core.parallel_state)


def patch_inference(aspm):
    from long_vita_megatron.inference.text_generation.tokenization import tokenize_prompts, _tokenize_prompts_and_batch
    from long_vita_megatron.inference.text_generation.forward_step import inference_forward_step_init_wrapper, _no_pipelining_forward_step, _with_pipelining_forward_step
    from long_vita_megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage, beam_search_and_return_on_first_stage

    aspm.register_patch('megatron.inference.text_generation.tokenization.tokenize_prompts', tokenize_prompts)
    aspm.register_patch('megatron.inference.text_generation.tokenization._tokenize_prompts_and_batch', _tokenize_prompts_and_batch)
    aspm.register_patch('megatron.inference.text_generation.generation.generate_tokens_probs_and_return_on_first_stage', generate_tokens_probs_and_return_on_first_stage)
    aspm.register_patch('megatron.inference.text_generation.generation.beam_search_and_return_on_first_stage', beam_search_and_return_on_first_stage)
    aspm.register_patch('megatron.inference.text_generation.forward_step.ForwardStep.__init__', inference_forward_step_init_wrapper)
    aspm.register_patch('megatron.inference.text_generation.forward_step._no_pipelining_forward_step', _no_pipelining_forward_step)
    aspm.register_patch('megatron.inference.text_generation.forward_step._with_pipelining_forward_step', _with_pipelining_forward_step)

def exe_adaptation():
    from long_vita_megatron.patch_utils import MindSpeedPatchesManager as aspm

    megatron_core_adaptation(aspm)

    megatron_legacy_adaptation(aspm)

    megatron_adaptation(aspm)

    # mcore_parallel_state_adaptation(aspm)

    attention_adaptation(aspm)

    optimizer_adaptation(aspm)

    patch_inference(aspm)

    from long_vita_megatron.training.arguments import parse_args_decorator
    aspm.register_patch('megatron.training.arguments.parse_args', parse_args_decorator)

    from long_vita_megatron.training.tokenizer import build_tokenizer
    aspm.register_patch('megatron.training.global_vars.build_tokenizer', build_tokenizer)

    aspm.apply_patches()

exe_adaptation()

