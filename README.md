# Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy

<font size=7><div align='center' > [[📖 Long-VITA Paper](https://arxiv.org/abs/2502.05177)] [[🤗 Hugging Face](https://huggingface.co/VITA-MLLM)] </div></font>


## 🔥 News
* **`2025.02.09`** 🌟 We support training and inference on both Ascend NPU with Modellink and Nvidia GPU with Megatron.
* **`2025.02.05`** 🌟 The training code, deployment code, and model weights **have been released**. ~~We currently only support Ascend NPU and are working on adapting to Nvidia GPU~~.
* **`2024.02.05`** 🌟 We are proud to launch Long-VITA, a strong long-context visual language model supporting over one million tokens.


## Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Experimental Results](#-experimental-results)
- [Training, Inference and Evaluation](#-training-inference-and-evaluation)


## ✨ Highlights

- **Long Context**. Long-VITA can process more than **4K** frames or over **1M** visual tokens. It achieves state-of-the-art performance on Video-MME under 20B models.
- **Open Source**. Long-VITA is trained on **open-source data** only, consisting of a mix of 17M samples that are publicly available.
- **Strong Performance**. Long-VITA achieves competitive results on image and video understanding benchmarks among cutting-edge models under 20B parameters.
  

## 📈 Experimental Results
- **Comparison of image understanding**.

![image](https://github.com/user-attachments/assets/30f62f51-675e-4dac-9f18-f743c311f9be)



- **Comparison of video understanding**.

![image](https://github.com/user-attachments/assets/7f09662b-bd53-4504-927a-0e45214a049d)

![image](https://github.com/user-attachments/assets/87bd2f4d-baf5-4a63-8002-151e30f52147)







## ⭐ Training, Inference and Evaluation

We originally implemented Long-VITA on Ascend NPU and will adapt to Nvidia GPU.

- [DATA Preparation (Only for Training)](https://github.com/VITA-MLLM/Long-VITA/blob/main/DATA.md)
  
- [Ascend NPU with MindSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/NPU_MindSpeed.md)

- [Nvidia GPU with Megatron](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_Megatron.md)

- [Nvidia GPU with DeepSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_DeepSpeed.md)



