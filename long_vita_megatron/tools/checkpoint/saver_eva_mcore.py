# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys
import torch
from importlib.metadata import version
from pkg_resources import packaging

from setter import ModelSetter
from utils import get_mcore_transformer_block_key, print_memory_usage


class MCoreSetter(ModelSetter):

    transformer_block_key = None

    @classmethod
    def get_transformer_block(cls, model):
        return getattr(model, cls.transformer_block_key)

    @classmethod
    def has_position_embeddings(cls, model):
        return hasattr(model.embedding, "position_embeddings")

    @classmethod
    def set_embeddings(
        cls,
        model,
        word=None,
        pos=None,
        conv_weight=None,
        conv_bias=None,
    ):
        cls.set_tensor(model.embedding.word_embeddings.weight, word)
        if pos is not None:
            cls.set_tensor(model.embedding.position_embeddings.weight, pos)
        cls.set_tensor(model.conv1.weight, conv_weight)
        cls.set_tensor(model.conv1.bias, conv_bias)

class MCoreLocalSetter(MCoreSetter):

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):

        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention.
        cls.set_tensor(l.input_layernorm.weight, self_attn_norm_weight)
        if self_attn_norm_bias is not None:
            cls.set_tensor(l.input_layernorm.bias, self_attn_norm_bias)

        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)

        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        # MLP.
        cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
        if mlp_norm_bias is not None:
            cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)

        cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
        if mlp_fc1_bias is not None:
            cls.set_tensor(l.mlp.linear_fc1.bias, mlp_fc1_bias)

        cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        if mlp_fc2_bias is not None:
            cls.set_tensor(l.mlp.linear_fc2.bias, mlp_fc2_bias)


class MCoreTESetter(MCoreSetter):

    @classmethod
    def set_layer(
        cls,
        model,
        layer_idx,
        self_attn_norm_weight=None,
        self_attn_norm_bias=None,
        self_attn_qkv_weight=None,
        self_attn_qkv_bias=None,
        self_attn_proj_weight=None,
        self_attn_proj_bias=None,
        mlp_norm_weight=None,
        mlp_norm_bias=None,
        mlp_fc1_weight=None,
        mlp_fc1_bias=None,
        mlp_fc2_weight=None,
        mlp_fc2_bias=None,
    ):

        block = cls.get_transformer_block(model)
        l = block.layers[layer_idx]

        # Self attention.
        cls.set_tensor(l.input_layernorm.weight, self_attn_norm_weight)
        if self_attn_norm_bias is not None:
            cls.set_tensor(l.input_layernorm.bias, self_attn_norm_bias)

        cls.set_tensor(l.self_attention.linear_qkv.weight, self_attn_qkv_weight)
        if self_attn_qkv_bias is not None:
            cls.set_tensor(l.self_attention.linear_qkv.bias, self_attn_qkv_bias)

        cls.set_tensor(l.self_attention.linear_proj.weight, self_attn_proj_weight)
        if self_attn_proj_bias is not None:
            cls.set_tensor(l.self_attention.linear_proj.bias, self_attn_proj_bias)

        # MLP.
        cls.set_tensor(l.pre_mlp_layernorm.weight, mlp_norm_weight)
        if mlp_norm_bias is not None:
            cls.set_tensor(l.pre_mlp_layernorm.bias, mlp_norm_bias)

        cls.set_tensor(l.mlp.linear_fc1.weight, mlp_fc1_weight)
        if mlp_fc1_bias is not None:
            cls.set_tensor(l.mlp.linear_fc1.bias, mlp_fc1_bias)

        cls.set_tensor(l.mlp.linear_fc2.weight, mlp_fc2_weight)
        if mlp_fc2_bias is not None:
            cls.set_tensor(l.mlp.linear_fc2.bias, mlp_fc2_bias)


def get_model_setter(model_type, transformer_impl):
    setter = {
        "local" : MCoreLocalSetter,
        "transformer_engine" : MCoreTESetter,
    }[transformer_impl]
    setter.transformer_block_key = get_mcore_transformer_block_key(model_type)
    return setter


def add_arguments(parser):
    group = parser.add_argument_group(title='M-Core saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')

    group.add_argument('--target-image-size', type=int, default=224,)


def save_model_checkpoint(queue, args):

    # Transformer engine >= 0.12.0, for CPU initialization.
    te_version = packaging.version.Version(version("transformer-engine"))
    assert te_version >= packaging.version.Version("0.12.0"), \
        "transformer engine version: %s (>=0.12.0 required)." % te_version

    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir,
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.training.arguments import (parse_args, validate_args)
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.global_vars import set_global_variables, get_args
        from megatron.core.enums import ModelType
        from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron.legacy import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)


    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target-tensor-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target-pipeline-parallel-size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1


    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num-layers', str(md.num_layers),
                '--hidden-size', str(md.hidden_size),
                '--seq-length', str(md.seq_length),
                '--num-attention-heads', str(md.num_attention_heads),
                '--max-position-embeddings', str(md.max_position_embeddings),
                '--position-embedding-type', str(md.position_embedding_type),
                '--tokenizer-type', str(md.tokenizer_type),
                '--tensor-model-parallel-size', str(args.target_tensor_parallel_size),
                '--pipeline-model-parallel-size', str(args.target_pipeline_parallel_size),
                '--no-masked-softmax-fusion',
                '--no-bias-gelu-fusion',
                '--no-bias-dropout-fusion',
                '--no-async-tensor-model-parallel-allreduce',
                '--use-cpu-initialization',
                '--micro-batch-size', '1',
                '--no-load-optim',
                '--no-load-rng',
                '--no-save-optim',
                '--no-save-rng',
                '--no-initialization',
                '--save-interval', '1',
                '--save', args.save_dir
                ]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make-vocab-size-divisible-by', str(md.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    if md.output_layer:
        sys.argv.append('--untie-embeddings-and-output-weights')
    if not md.linear_bias:
        sys.argv.append('--disable-bias-linear')

    if md.model_type == 'BERT' and not md.bert_binary_head:
        sys.argv.append('--bert-no-binary-head')

    margs = parse_args()

    if hasattr (md, 'checkpoint_args'):
        # These are arguments that we are either changing, or cause problems for validation if they are set
        # Note that some of these deal with T5 so will need to be changed if we support T5.
        args_to_keep = ['tensor_model_parallel_size', 'pipeline_model_parallel_size', 'world_size', 'params_dtype',
                        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
                        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
                        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
                        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
                        'vocab_file', 'tokenizer_model',
                        'save_interval', 'save',
                        'perform_initialization', 'use_cpu_initialization',
                        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
                        'encoder_num_layers', 'encoder_seq_length',
                        'distribute_saved_activations',
                        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
                        'start_weight_decay', 'end_weight_decay']

        for arg, value in vars(md.checkpoint_args).items():
            if arg in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)

    # Explicitly copy sequence_parallel, apply_query_key_layer_scaling.
    margs.sequence_parallel = md.checkpoint_args.sequence_parallel
    margs.apply_query_key_layer_scaling = md.checkpoint_args.apply_query_key_layer_scaling

    validate_args(margs)

    # Use M-core models & unset loaded paths.
    margs.use_mcore_models = True
    margs.blendable_index_path = None
    margs.data_path = []
    margs.load = None
    margs.save = args.save_dir
    margs.tensorboard_dir = None
    margs.tokenizer_model = None
    margs.transformer_impl = args.saver_transformer_impl
    margs.image_size = args.target_image_size
    margs.img_h = args.target_image_size
    margs.img_w = args.target_image_size

    set_global_variables(margs, build_tokenizer=False)

    # Megatron args. (i.e., 'margs')
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'EVA':
        from megatron.training.arguments import core_transformer_config_from_args
        # from megatron.core.models.vision.eva_clip_model import EVA2ViTModel
        from long_vita_modellink.core.models.vision.eva_clip_model import EVA2ViTModel
        from megatron.core.transformer.transformer_config import VisionTransformerConfig
        from long_vita_modellink.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec_for_eva_clip,
        )
        def model_provider(pre_process=True):
            args = get_args()
            print('building EVA model ...')
            config = core_transformer_config_from_args(args, VisionTransformerConfig)
            transformer_layer_spec = get_vit_layer_with_transformer_engine_spec_for_eva_clip()
            model = EVA2ViTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.vocab_size,
                pre_process=pre_process,
            )
            return model
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # fake initializing distributed
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    #-----------
    embeddings_msg = queue_get("embeddings")

    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = embeddings_msg.pop("position embeddings")
    orig_word_embed = embeddings_msg.pop("word embeddings")
    conv_weight = embeddings_msg.pop("conv weight")
    conv_bias = embeddings_msg.pop("conv bias")
    check_message(embeddings_msg)

    if args.target_image_size != 224:
        print("saver_eva_mcore pos_embed", pos_embed.size())
        num_patches = int(224 / 14) ** 2
        new_num_patches = int(args.target_image_size / 14) ** 2
        num_extra_tokens = 1

        HW_pathes = int(224 / 14)
        new_HW_patches = int(args.target_image_size / 14)

        extra_tokens = pos_embed[:num_extra_tokens, :]
        pos_tokens = pos_embed[num_extra_tokens:, :]
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        pos_tokens = pos_tokens.reshape(HW_pathes, HW_pathes, -1).permute(2, 0, 1)
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        ori_dtype = pos_tokens.dtype
        pos_tokens = pos_tokens.to(torch.float32)
        pos_tokens = pos_tokens.unsqueeze(0)
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_HW_patches, new_HW_patches), mode='bicubic', align_corners=False)
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        pos_tokens = pos_tokens.squeeze(0)
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        pos_tokens = pos_tokens.to(ori_dtype)
        pos_tokens = pos_tokens.permute(1, 2, 0).flatten(0, 1)
        print("saver_eva_mcore pos_tokens", pos_tokens.size())
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        print("saver_eva_mcore new_pos_embed", new_pos_embed.size())
        pos_embed = new_pos_embed

    # Parameter setter class.
    setter = get_model_setter(md.model_type, margs.transformer_impl)

    # Get models.
    def get_models(count, dtype, pre_process):
        models = []
        for rank in range(count):
            models.append(model_provider(pre_process).to(dtype))
            print_memory_usage("saver", rank, count)
        return models

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    models = get_models(args.target_tensor_parallel_size, md.params_dtype, True)

    # Set embeddings.
    # --------------
    for tp_rank, model in enumerate(models):
        if pos_embed is None:
            assert not setter.has_position_embeddings(model)
        setter.set_embeddings(
            model,
            word=orig_word_embed,
            pos=pos_embed,
            conv_weight=conv_weight,
            conv_bias=conv_bias
        )

    # Transformer layers.
    # ------------------
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        if pp_rank > 0:
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            models = get_models(args.target_tensor_parallel_size, md.params_dtype, False)

        for layer in range(len(setter.get_transformer_block(models[0]).layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_norm_weight = msg.pop("input norm weight")
            if md.norm_has_bias:
                input_norm_bias = msg.pop("input norm bias")
            post_norm_weight = msg.pop("post norm weight")
            if md.norm_has_bias:
                post_norm_bias = msg.pop("post norm bias")
            if md.linear_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Special handling for swiglu
            if md.swiglu:
                mlp_l0_weight_W = torch.chunk(msg.pop("mlp l0 weight W"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(msg.pop("mlp l0 weight V"), args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
            else:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)

            if md.linear_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
                if md.swiglu:
                    mlp_l0_bias_W = torch.chunk(msg.pop("mlp l0 bias W"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias_V = torch.chunk(msg.pop("mlp l0 bias V"), args.target_tensor_parallel_size, dim=0)
                    mlp_l0_bias = [torch.cat(bias, dim=0) for bias in zip(mlp_l0_bias_W, mlp_l0_bias_V)]
                else:
                    mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                params_dict = {
                    "self_attn_norm_weight" : input_norm_weight,
                    "self_attn_qkv_weight" : qkv_weight[tp_rank],
                    "self_attn_proj_weight" : dense_weight[tp_rank],
                    "mlp_norm_weight" : post_norm_weight,
                    "mlp_fc1_weight" : mlp_l0_weight[tp_rank],
                    "mlp_fc2_weight" : mlp_l1_weight[tp_rank],
                }
                if md.norm_has_bias:
                    params_dict.update({
                        "self_attn_norm_bias" :
                        input_norm_bias if md.norm_has_bias else None,
                        "mlp_norm_bias" :
                        post_norm_bias if md.norm_has_bias else None,
                    })
                if md.linear_bias:
                    params_dict.update({
                        "self_attn_qkv_bias" : qkv_bias[tp_rank],
                        "self_attn_proj_bias" : dense_bias,
                        "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                        "mlp_fc2_bias" : mlp_l1_bias,
                    })
                setter.set_layer(models[tp_rank], layer, **params_dict)

            total_layer_num = total_layer_num + 1
            check_message(msg)

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None,
                            num_floating_point_operations_so_far=0)

    print("Done!")
