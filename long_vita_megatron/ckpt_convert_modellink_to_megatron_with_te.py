
import argparse

import torch
import os


def convert(ckpt_load_dir, ckpt_save_dir):

    tracker_filename = os.path.join(ckpt_load_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
    iteration = int(metastring)

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    tracker_filename = os.path.join(ckpt_save_dir, 'latest_checkpointed_iteration.txt')
    with open(tracker_filename, 'w') as f:
        f.write(str(iteration))

    directory = 'iter_{:07d}'.format(iteration)
    ckpt_load_dir = os.path.join(ckpt_load_dir, directory)
    ckpt_save_dir = os.path.join(ckpt_save_dir, directory)


    for root, dirs, files in os.walk(ckpt_load_dir):
        for filename in files:
            if filename.endswith("model_optim_rng.pt"):
                filepath = os.path.join(root, filename)
                print(f"filepath {filepath}")
                state_dict = torch.load(filepath, map_location="cpu")

                for k in list(state_dict["model"].keys()):

                    new_k = None
                    if "external_feature_model" not in k:
                        if ".input_layernorm." in k:
                            new_k = k.replace(".input_layernorm.", ".self_attention.linear_qkv.layer_norm_")
                        if ".pre_mlp_layernorm." in k:
                            new_k = k.replace(".pre_mlp_layernorm.", ".mlp.linear_fc1.layer_norm_")

                    if new_k is not None:
                        print(f"{k} --> {new_k}")
                        v = state_dict["model"].pop(k)
                        state_dict["model"][new_k] = v

                new_filepath = filepath.replace(ckpt_load_dir, ckpt_save_dir)
                new_dir = os.path.dirname(new_filepath)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)

                torch.save(state_dict, new_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--ckpt_load_dir", type=str, required=True, help="modellink directory")
    parser.add_argument("--ckpt_save_dir", type=str, required=True, help="megatron directory")

    args = parser.parse_args()

    convert(args.ckpt_load_dir, args.ckpt_save_dir)

    print("done.")
