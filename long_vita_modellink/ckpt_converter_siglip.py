# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch


def convert(download_root, output_path, tensor_parallel_size, use_te):
    # device = "cuda"
    device = "cpu"

    import torch
    from PIL import Image
    from transformers import AutoModel, CLIPImageProcessor

    model = AutoModel.from_pretrained(
        # 'OpenGVLab/InternViT-300M-448px',
        download_root,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).cpu().eval()

    state_dict = model.state_dict()
    print("state_dict", state_dict.keys())
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 1152 // 16
    hidden_dim = 1152
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)


    for layer_idx in range(9999):
        if f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight" in state_dict:
            pass
        else:
            continue

        print(f"layer_idx {layer_idx}")

        linear_q_weight = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight")
        linear_k_weight = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight")
        linear_v_weight = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight")

        linear_qkv_weight = torch.cat([linear_q_weight, linear_k_weight, linear_v_weight], 0)
        print(f"linear_q_weight {linear_q_weight.size()}")
        print(f"linear_k_weight {linear_k_weight.size()}")
        print(f"linear_v_weight {linear_v_weight.size()}")
        print(f"linear_qkv_weight {linear_qkv_weight.size()}")

        state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.weight"] = linear_qkv_weight

        linear_q_bias = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias")
        linear_k_bias = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias")
        linear_v_bias = state_dict.pop(f"vision_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias")

        linear_qkv_bias = torch.cat([linear_q_bias, linear_k_bias, linear_v_bias], 0)
        print(f"linear_q_bias {linear_q_bias.size()}")
        print(f"linear_k_bias {linear_k_bias.size()}")
        print(f"linear_v_bias {linear_v_bias.size()}")
        print(f"linear_qkv_bias {linear_qkv_bias.size()}")

        state_dict[f"vision_model.encoder.layers.{layer_idx}.self_attn.qkv_proj.bias"] = linear_qkv_bias



    for name, tensor in state_dict.items():

        if name == "logit_scale":
            continue
        if name == "logit_bias":
            continue
        if "vision_model.head." in name:
            continue
        if "vision_model.post_layernorm." in name:
            continue
        if "text_model." in name:
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "embeddings.position_embedding" in name:
            new_name = "position_embeddings.weight"
            new_tensor = new_tensor.squeeze(0)
        elif "embeddings.patch_embedding.weight" in name:
            new_name = "conv1.weight"
        elif "embeddings.patch_embedding.bias" in name:
            new_name = "conv1.bias"
        # elif "ln_pre.weight" in name:
        #     new_name = "ln_pre.weight"
        # elif "ln_pre.bias" in name:
        #     new_name = "ln_pre.bias"
        elif "encoder.layers." in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"
            # base = f"transformer.layers.{layer_idx}"

            if ".self_attn.qkv_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.qkv_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.q_proj.weight" in name and False:
                new_name = f"{base}.self_attention.linear_q.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.q_proj.bias" in name and False:
                new_name = f"{base}.self_attention.linear_q.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.k_proj.weight" in name and False:
                new_name = f"{base}.self_attention.linear_k.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.k_proj.bias" in name and False:
                new_name = f"{base}.self_attention.linear_k.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.v_proj.weight" in name and False:
                new_name = f"{base}.self_attention.linear_v.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.v_proj.bias" in name and False:
                new_name = f"{base}.self_attention.linear_v.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".self_attn.out_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif ".self_attn.out_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif ".layer_norm1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif ".layer_norm1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif ".mlp.fc1.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif ".mlp.fc1.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif ".mlp.fc2.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif ".mlp.fc2.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif ".layer_norm2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif ".layer_norm2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

        assert new_name != "", f"unexpected layer name {name}"
        print(f"{new_name} {new_tensor.size()}")

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        print(f"{new_name} {[x.size() for x in new_tensors]}")
        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None

    for layer_idx in range(0):
        for i in range(tensor_parallel_size):
            if f"decoder.layers.{layer_idx}.self_attention.linear_q.weight" in new_state_dicts[i]["model"]:
                pass
            else:
                continue

            print(f"layer_idx {layer_idx}")

            linear_q_weight = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_q.weight"]
            linear_k_weight = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_k.weight"]
            linear_v_weight = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_v.weight"]
            linear_qkv_weight = torch.cat([linear_q_weight, linear_k_weight, linear_v_weight], 0)
            print(f"linear_q_weight {linear_q_weight.size()}")
            print(f"linear_k_weight {linear_k_weight.size()}")
            print(f"linear_v_weight {linear_v_weight.size()}")
            print(f"linear_qkv_weight {linear_v_weight.size()}")

            new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_v.weight"] = linear_qkv_weight

            linear_q_bias = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_q.bias"]
            linear_k_bias = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_k.bias"]
            linear_v_bias = new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_v.bias"]
            linear_qkv_bias = torch.cat([linear_q_bias, linear_k_bias, linear_v_bias], 0)
            print(f"linear_q_bias {linear_q_bias.size()}")
            print(f"linear_k_bias {linear_k_bias.size()}")
            print(f"linear_v_bias {linear_v_bias.size()}")
            print(f"linear_qkv_bias {linear_qkv_bias.size()}")

            new_state_dicts[i]["model"][f"decoder.layers.{layer_idx}.self_attention.linear_v.bias"] = linear_qkv_bias 


    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)

    output_path = os.path.join(output_path, "latest_checkpointed_iteration.txt")
    with open(output_path, 'w') as the_file:
        the_file.write('1')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert OpenAI CLIP VIT weights to megatron format.


Example usage:
python clip_converter.py --download-root /some/download/folder --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--download-root", type=str, required=True, help="Download folder for OpenAI CLIP weights"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")

    args = parser.parse_args()

    convert(args.download_root, args.output, args.tensor_parallel_size, args.use_te)

    print("done.")
