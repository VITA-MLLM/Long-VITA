

import torch
import os





CKPT_LOAD_DIR="/data_2/output/LM/scripts/modellink/llama31/finetune_llama31_70b_eva_4b_ptd_stage2.sh/20241207_194633/iter_0002000/"
LLM_SAVE_DIR="/data_2/output/LM/scripts/modellink/llama31/finetune_llama31_70b_eva_4b_ptd_stage2.sh/20241207_194633_llm/iter_0002000/"
VIT_SAVE_DIR="/data_2/output/LM/scripts/modellink/llama31/finetune_llama31_70b_eva_4b_ptd_stage2.sh/20241207_194633_vit/iter_0002000/"



for root, dirs, files in os.walk(CKPT_LOAD_DIR):
    for filename in files:
        if filename.endswith("model_optim_rng.pt"):
            filepath = os.path.join(root, filename)
            print(f"filepath {filepath}")

            # --------------------------------------------------------
            state_dict = torch.load(filepath, map_location="cpu")

            for k in list(state_dict["model"].keys()):
                if "unused" in k:
                    del state_dict["model"][k]
                elif "external_feature_model." in k:
                    del state_dict["model"][k]
                else:
                    pass

            llm_filepath = filepath.replace(CKPT_LOAD_DIR, LLM_SAVE_DIR)
            llm_dir = os.path.dirname(llm_filepath)
            if not os.path.exists(llm_dir):
                os.makedirs(llm_dir)

            assert llm_filepath != filepath
            print(f"llm_filepath {llm_filepath}")
            torch.save(state_dict, llm_filepath)


            # --------------------------------------------------------
            state_dict = torch.load(filepath, map_location="cpu")

            for k in list(state_dict["model"].keys()):
                if "unused" in k:
                    del state_dict["model"][k]
                elif "external_feature_model." in k:
                    pass
                else:
                    del state_dict["model"][k]
            for k in list(state_dict["model"].keys()):
                if "external_feature_model." in k:
                    new_k = k.split("external_feature_model.")[-1]
                    v = state_dict["model"].pop(k)
                    state_dict["model"][new_k] = v
                else:
                    pass

            if not state_dict["model"]:
                continue

            vit_filepath = filepath.replace(CKPT_LOAD_DIR, VIT_SAVE_DIR)

            vit_dir = os.path.dirname(vit_filepath)
            assert vit_dir.endswith("_000")
            vit_dir_new = vit_dir[:-4]
            vit_filepath = vit_filepath.replace(vit_dir, vit_dir_new)

            vit_dir = os.path.dirname(vit_filepath)
            if not os.path.exists(vit_dir):
                os.makedirs(vit_dir)

            assert vit_filepath != filepath
            print(f"vit_filepath {vit_filepath}")
            torch.save(state_dict, vit_filepath)
