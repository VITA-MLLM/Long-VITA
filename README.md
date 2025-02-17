# Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy

<p align="center">
    <img src="https://github.com/user-attachments/assets/cc367b87-3e23-4f3d-bea7-936b05664f26" width="100%" height="100%">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.05177" target="_blank"><img src="https://img.shields.io/badge/Long%20VITA-Report-b5212f.svg?logo=arxiv" /></a>
    <a href="https://huggingface.co/VITA-MLLM" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-VITA%20MLLM-ffc107?color=ffc107&logoColor=white" /></a>
 </p>



## 🔥 News

* **`2025.02.17`** 🌟 We support training and inference on **Nvidia GPU with DeepSpeed** and **Huggingface Transformer**.
* **`2025.02.11`** 🌟 We add more [experiments](#-experimental-results) on logits-masked LM head. During inference, logits-masked LM head extends the max sequence length by $415$% and reduces time cost by $45$%.
* **`2025.02.09`** 🌟 We support training and inference on both **Ascend NPU with MindSpeed** and **Nvidia GPU with Megatron**.
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

![image](https://github.com/user-attachments/assets/235bdb0e-37e6-4a5f-b20b-21b0bb83278a)
![image](https://github.com/user-attachments/assets/72250c5b-7d33-4dba-98ab-0539bae08703)


- **Comparison of video understanding**.

![image](https://github.com/user-attachments/assets/7f09662b-bd53-4504-927a-0e45214a049d)

![image](https://github.com/user-attachments/assets/87bd2f4d-baf5-4a63-8002-151e30f52147)


- **Effectiveness of Logits-Masked LM Head**.

![image](https://github.com/user-attachments/assets/7a06b4dd-267c-470f-80f2-d26c87e23460)



## ⭐ Training, Inference and Evaluation

We originally implemented Long-VITA on Ascend NPU and will adapt to Nvidia GPU.

- [Data Preparation for Training](https://github.com/VITA-MLLM/Long-VITA/blob/main/DATA.md)
  
- [Ascend NPU with MindSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/NPU_MindSpeed.md)

- [Nvidia GPU with Megatron](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_Megatron.md)

- [Nvidia GPU with DeepSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_DeepSpeed.md)



