# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch

def convert_3200_to_4096(x):
    if x.size()[0] == 3200:
        pad = (x.dim() - 1) * [0, 0] + [0, 4096 - 3200]
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
    if x.dim() >= 2 and x.size()[1] == 3200:
        pad = (x.dim() - 2) * [0, 0] + [0, 4096 - 3200, 0, 0]
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
    if x.dim() >= 3 and x.size()[2] == 3200:
        pad = (x.dim() - 3) * [0, 0] + [0, 4096 - 3200, 0, 0, 0, 0]
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
    return x

def convert_9600_to_12288(x):
    if x.size()[0] == 9600:
        x = torch.cat([convert_3200_to_4096(xx) for xx in torch.chunk(x, 3, dim=0)], dim=0)
    if x.dim() >= 2 and x.size()[1] == 9600:
        x = torch.cat([convert_3200_to_4096(xx) for xx in torch.chunk(x, 3, dim=1)], dim=1)
    if x.dim() >= 3 and x.size()[2] == 9600:
        x = torch.cat([convert_3200_to_4096(xx) for xx in torch.chunk(x, 3, dim=2)], dim=2)
    return x

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
    kv_channels =128
    hidden_dim = 3200
    num_heads = 25

    kv_channels = 64
    hidden_dim = 1024
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():


        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        if "embeddings.class_embedding" in name:
            new_name = "class_token"
            # Our model uses class token that is expanded to input dimensions already.
            new_tensor = new_tensor.expand(1, 1, -1)
        elif "embeddings.position_embedding" in name:
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
            layer_idx = name.split(".")[2]
            base = f"decoder.layers.{layer_idx}"
            # base = f"transformer.layers.{layer_idx}"

            if ".attn.qkv.weight" in name:
                new_name = f"{base}.self_attention.linear_qkv.weight"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".attn.qkv.bias" in name:
                new_name = f"{base}.self_attention.linear_qkv.bias"
                new_tensor = new_tensor[indices]
                chunk_dim = 0
            elif ".attn.proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif ".attn.proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif ".attn.q_norm.weight" in name:
                new_name = f"{base}.self_attention.q_layernorm.weight"
                chunk_dim = 0
            elif ".attn.q_norm.bias" in name:
                new_name = f"{base}.self_attention.q_layernorm.bias"
                chunk_dim = 0
            elif ".attn.k_norm.weight" in name:
                new_name = f"{base}.self_attention.k_layernorm.weight"
                chunk_dim = 0
            elif ".attn.k_norm.bias" in name:
                new_name = f"{base}.self_attention.k_layernorm.bias"
                chunk_dim = 0
            elif ".norm1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif ".norm1.bias" in name:
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
            elif ".norm2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif ".norm2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"
            elif ".ls1" in name:
                new_name = f"{base}.ls1"
            elif ".ls2" in name:
                new_name = f"{base}.ls2"

        assert new_name != "", f"unexpected layer name {name}"
        print(f"{new_name} {new_tensor.size()}")

        new_tensor = convert_3200_to_4096(new_tensor)
        new_tensor = convert_9600_to_12288(new_tensor)

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
