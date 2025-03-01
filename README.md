# Long-VITA: Scaling Large Multi-modal Models to 1 Million Tokens with Leading Short-Context Accuracy

<p align="center">
    <img src="https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/main/images/longvita.jpg" width="100%" height="100%">
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2502.05177" target="_blank"><img src="https://img.shields.io/badge/Long%20VITA-Report-b5212f.svg?logo=arxiv" /></a>
    <a href="https://huggingface.co/VITA-MLLM" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107?color=ffc107&logoColor=white" /></a>
    <a href="https://huggingface.co/spaces/shenyunhang/Long-VITA" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-ffc107?color=ffc107&logoColor=white" /></a>
 </p>



## :fire: News


* **`2025.02.27`** 🌟 We have an [Oneline Demo](https://huggingface.co/spaces/shenyunhang/Long-VITA) now.
* **`2025.02.27`** 🌟 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) of OpenCompass has supported our Long-VITA.
* **`2025.02.17`** 🌟 We support training on **Nvidia GPU with DeepSpeed** and inference on **Nvidia GPU with Transformer**.
* **`2025.02.09`** 🌟 We support training and inference on **Nvidia GPU with Megatron**.
* **`2025.02.05`** 🌟 We release training code, **training log**, deployment code, and model weights, which support **Ascend NPU with MindSpeed**.
* **`2024.02.05`** 🌟 We are proud to launch Long-VITA, a strong long-context visual language model supporting over one million tokens.


## Contents <!-- omit in toc -->


- [Highlights](#-highlights)
- [Experimental Results](#-experimental-results)
- [Models](#-models)
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

![image](https://github.com/user-attachments/assets/94389a9f-3134-4fd6-9531-62f626d38e39)





## 🐍 Models

 Model          | LLM Size | Training Context | Training Frames | MindSpeed Weights                               | Megatron Weights                                   | Huggingface Weights                                
---------------:|---------:|-----------------:|----------------:|------------------------------------------------:|---------------------------------------------------:|---------------------------------------------------:
 Long-VITA-16K  | 14B      | 16,384           | 64              | https://huggingface.co/VITA-MLLM/Long-VITA-16K  | https://huggingface.co/VITA-MLLM/Long-VITA-16K_MG  | https://huggingface.co/VITA-MLLM/Long-VITA-16K_HF  
 Long-VITA-128K | 14B      | 131,072          | 512             | https://huggingface.co/VITA-MLLM/Long-VITA-128K | https://huggingface.co/VITA-MLLM/Long-VITA-128K_MG | https://huggingface.co/VITA-MLLM/Long-VITA-128K_HF 
 Long-VITA-1M   | 14B      | 1,048,576        | 4,096           | https://huggingface.co/VITA-MLLM/Long-VITA-1M   | https://huggingface.co/VITA-MLLM/Long-VITA-1M_MG   | https://huggingface.co/VITA-MLLM/Long-VITA-1M_HF   









## ⭐ Training, Inference and Evaluation

We originally implemented Long-VITA on Ascend NPU and will adapt to Nvidia GPU.

- [Data Preparation for Training](https://github.com/VITA-MLLM/Long-VITA/blob/main/DATA.md)
  
- [Ascend NPU with MindSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/NPU_MindSpeed.md)

- [Nvidia GPU with Megatron](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_Megatron.md)

- [Nvidia GPU with DeepSpeed](https://github.com/VITA-MLLM/Long-VITA/blob/main/GPU_DeepSpeed.md)



