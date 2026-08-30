# Sinhala ASR Call Centre Text Cleaner (SinLlama)

## Overview
This repository contains the end-to-end machine learning pipeline for correcting noisy Sinhala Automatic Speech Recognition (ASR) transcripts. Developed for the telecommunications sector (specifically SLT Mobitel), this project utilizes a highly optimized Llama-3-8B architecture (SinLlama) to correct grammatical errors and domain-specific jargon (e.g., billing queries, package upgrades) while strictly preserving contextual data like phone numbers and account states.

The project explores two distinct NLP approaches to achieve this:
1. **Few-Shot Prompting (Baseline):** Utilizing the base model with in-context examples.
2. **QLoRA Fine-Tuning (Production):** A highly efficient, custom-trained LoRA adapter deployed for localized, zero-cost inference[cite: 3, 4].

---

## Repository Structure

```text
├── Few-Shot Prompting/
│   ├── Few Shot Documentation.docx
│   ├── Few Shot Documentation.pdf
│   ├── SinLlama_Llama_3_8B_Merged_cleaner_few-shot.ipynb
│   └── local_cleaner_few-shot.py
├── QLoRA fine-tuning on SinLlama/
│   ├── Dataset/
│   │   ├── prepare_dataset.py
│   │   ├── qlora_dataset.json
│   │   └── qlora_dataset.jsonl
│   ├── SinLlama Inference/
│   │   ├── Fine_Tuned_SinLlama_Inference.ipynb
│   │   ├── Local Inference For Fine-Tuned SinLlama.pdf
│   │   ├── local_inference_cpu.py
│   │   └── local_inference_gpu.py
│   ├── SinLlama Training/
│   │   ├── SinLlama-QLoRA Test Outputs.txt
│   │   ├── SinLlama_QLoRA.ipynb
│   │   ├── SinLlama_QLoRA_v2.ipynb
│   │   └── SinLlama_Training_Mode_Documentation.pdf
│   └── lora_backup/
│       ├── adapter_config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── .gitignore
```
(Note: Large weight files like adapter_model.safetensors and .gguf files are excluded from this repository via .gitignore due to file size limits.)

---

## 1. QLoRA Fine-Tuning Pipeline (Production)
To create a consistent, reliable correction engine without relying on expensive API calls, the model was fine-tuned using the QLoRA (Quantized Low-Rank Adaptation) methodology.  

- Base Model: SAWithanage/SinLlama-Llama-3-8B-Merged.  
- Dataset: 126 custom JSONL records of SLT Mobitel Call Centre transcripts (113 training samples, 13 evaluation samples) formatted into Alpaca-style instruction prompts.
- Frameworks: Unsloth, TRL SFTTrainer, PEFT, and Hugging Face Transformers.

### Overfitting Mitigation
Initial training on the small dataset resulted in catastrophic repetition loops. The final, stable model was trained using heavily regularized hyperparameters:
- Epochs: Reduced to 1.5 to prevent memorization of the limited dataset.
- Learning Rate: Lowered to 4e-5 with a cosine scheduler for gradual weight updates.
- Weight Decay: Increased to 0.1 for stronger regularization.

---

## 2. Local Inference Modes
The fine-tuned LoRA adapters (stored in the lora_backup directory) can be executed locally in two modes depending on available hardware.  

### GPU Inference Mode (Recommended)
This mode leverages 4-bit NormalFloat (NF4) quantization via bitsandbytes to compress the 16 GB base model down to approximately 5 GB, running efficiently on consumer-grade NVIDIA GPUs.  
- GPU VRAM: Minimum 8 GB required (e.g., RTX 3060, RTX 4060, Tesla T4); 12-16 GB recommended.
- System RAM: 16 GB.
- Storage: 20 GB free space (SSD preferred).
- Execution Script: local_inference_gpu.py.

### CPU Inference Mode (Fallback)
This mode runs the unquantized model in full precision (float32), heavily relying on standard system memory and CPU cores.  - System RAM: 32 GB minimum required to load the uncompressed 16 GB model and handle memory spikes during text generation.
- CPU: Modern multi-core processor (Intel Core i7/i9 or AMD Ryzen 7/9).
- Execution Script: local_inference_cpu.py.

---

## 3. Few-Shot Prompting (Baseline Prototype)
Before committing to fine-tuning, the base model was tested using a Few-Shot Prompting technique.  
- Methodology: Five handcrafted Noisy-to-Clean Sinhala text pairs were embedded directly into the prompt context to steer the unmodified base model.
- Quantization: 4-bit (NF4, double quantization, float16 compute).
- Limitations: While useful for fast experimentation without a training pipeline, the model lacked a fine-tuned stopping behavior. In testing, it accurately corrected early sentences but occasionally degenerated into endless repetition loops unless forcefully truncated at the first newline character.

---

## 4. Setup & Installation
### For GPU Inference (Windows/Linux)
Ensure the NVIDIA CUDA Toolkit is installed.

```text
# 1. Install PyTorch configured for CUDA 12.1 (adjust URL if needed)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 2. Install Hugging Face ecosystem and quantization libraries
pip install transformers peft accelerate bitsandbytes
```
### For CPU Inference

```text
pip install torch transformers peft
```

---

## 5. Usage
To run the localized ASR cleaner:
1. Ensure your terminal is in the directory containing the inference script and the lora_backup folder.
2. Execute the desired Python script:

### For GPU:

```text
python local_inference_gpu.py
```

### For CPU:

```text
python local_inference_cpu.py
```
Upon execution, the script will automatically wrap the noisy transcript in the Alpaca prompt template, load the weights, and output the clean Sinhala text.

---
