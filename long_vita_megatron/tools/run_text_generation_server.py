import os
import sys

from typing import Union

import long_vita_megatron.megatron_adaptor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, \
    get_gpt_layer_local_spec
from megatron.core.transformer.spec_utils import import_module

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.legacy.model import GPTModel
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from long_vita_megatron.inference.text_generation_server import MegatronServer
# from megatron.inference.text_generation import generate_and_post_process
# from megatron.inference.text_generation import beam_search_and_post_process
import torch

from long_vita_megatron.tasks.inference.infer_base import add_text_generate_args
from long_vita_megatron.tasks.inference.module import GPTVLModelInfer, MegatronModuleForCausalLM
# from long_vita_megatron.tasks.inference.text_generation.module import GPTVLModelInfer, MegatronModuleForCausalLM

from long_vita_megatron.pretrain_long_vita import vision_model_provider
# from long_vita_megatron.legacy.model.gpt_vl_model import GPTVLModel

def model_provider(pre_process=True, post_process=True) -> Union[GPTVLModelInfer, GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTVLModelInfer, GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    print(f"model_provider config {config}")
    if args.use_mcore_models:
        # from long_vita_megatron.core.models.multimodal.gpt_vl_model import GPTVLModel
        print("Building megatron mcore vision language model ...")

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
        print(f"model_provider transformer_layer_spec {transformer_layer_spec}")

        model = GPTVLModelInfer(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=False,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base,
            external_feature_model_provider=vision_model_provider,
            allow_missing_keys=['external_feature_model'],
        )
    else:
        from long_vita_megatron.legacy.model.gpt_vl_model import GPTVLModel
        if not args.context_parallel_size == 1:
            raise ValueError("Context parallelism is only supported with Megatron Core!")

        print("Building megatron legacy vision language model ...")
        model = GPTVLModel(
            config,
            parallel_output=False,
            pre_process=pre_process,
            post_process=post_process,
            external_feature_model_provider=vision_model_provider,
            allow_missing_keys=['external_feature_model'],
        )

    return model


def extra_args_provider(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')

    from long_vita_megatron.pretrain_long_vita import extra_args_provider
    parser = extra_args_provider(parser)

    parser = add_text_generate_args(parser)

    return parser


def main():
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    torch.backends.cudnn.benchmark = False

    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0",port=args.port)

        import socket
        ip_addr = socket.gethostbyname(socket.gethostname())
        print(f"ip_addr {ip_addr}")

    while True:
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice.item() == 0:
            try:
                # generate_and_post_process(model)
                model.generate()
            except ValueError as ve:
                pass
        elif choice.item() == 1:
            try:
                # beam_search_and_post_process(model)
                model.generate()
            except ValueError as ve:
                pass

if __name__ == "__main__":
    main()
