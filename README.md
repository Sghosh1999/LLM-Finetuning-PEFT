# Fine-Tuning Llama 3.1/3.2 1B with QLoRA & Unsloth: Professional Guide

## Links

- **Kaggle Notebook:** [Finetuning Llama3-1.1B SFT QLoRA Unsloth](https://www.kaggle.com/code/sghosh99/finetuning-llama3-1-1b-sft-qlora-unsloth/notebook?scriptVersionId=245754632)
- **Hugging Face Model:** [SGHOSH1999/FineLlama3.1-1B-Instruct](https://huggingface.co/SGHOSH1999/FineLlama3.1-1B-Instruct)

## Overview

This repository provides a comprehensive notebook for fine-tuning Llama 3.1/3.2 1B models using QLoRA and the Unsloth library. The workflow is designed for efficient, parameter-efficient fine-tuning (PEFT) on conversational datasets, leveraging advanced techniques such as 4-bit quantization, LoRA adapters, and gradient checkpointing. The notebook is optimized for both research and production, supporting rapid experimentation and deployment.

---

## Features

- **Supports Llama 3.1/3.2, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermes, and more**
- **4-bit QLoRA and 16-bit LoRA** for memory and compute efficiency
- **Automatic RoPE Scaling** for arbitrary sequence lengths
- **Rank-Stabilized LoRA (rsLoRA)** for improved learning stability
- **Gradient Checkpointing** to minimize VRAM usage
- **Flexible Chat Template Support** (ChatML, Llama3, Mistral, etc.)
- **Easy dataset conversion** from ShareGPT to HuggingFace format
- **Integration with Hugging Face Hub** for model sharing
- **Export to GGUF for llama.cpp and GPT4All compatibility**

---

## Quick Start

### 1. Environment Setup

- Recommended: Google Colab (Tesla T4) for free GPU access
- Install dependencies:
  ```python
  !pip install pip3-autoremove
  !pip-autoremove torch torchvision torchaudio -y
  !pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu121
  !pip install unsloth
  ```

### 2. Model Loading

- Load Llama 3.2 1B Instruct (or any supported model) with Unsloth’s `FastLanguageModel`.
- Supports 4-bit quantization for fast, memory-efficient training.

### 3. LoRA Adapter Configuration

- Configure LoRA with recommended parameters (e.g., `r=16`, `alpha=16`, all linear modules targeted).
- Optionally enable rsLoRA for stability.
- Use Unsloth’s gradient checkpointing for large context lengths.

### 4. Data Preparation

- Use [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) (ShareGPT format).
- Convert to HuggingFace multiturn format using `standardize_sharegpt`.
- Apply chat templates for conversational fine-tuning.

### 5. Training

- Use Hugging Face TRL’s `SFTTrainer` with Unsloth enhancements.
- Recommended settings:
  - Batch size: 2
  - Gradient accumulation: 4
  - Learning rate: 2e-4
  - Max steps: 60 (for demo; increase for full training)
  - Optimizer: `adamw_8bit`
- Optionally, train only on assistant responses with `train_on_responses_only`.

### 6. Inference

- Enable 2x faster inference with Unsloth.
- Use chat templates for prompt formatting.
- Supports streaming output with `TextStreamer`.

### 7. Saving & Exporting

- Save LoRA adapters locally or push to Hugging Face Hub.
- Export to GGUF for use with llama.cpp or GPT4All.

---

## Example Usage

See the notebook for step-by-step code and explanations, including:

- Model loading and quantization
- LoRA/rsLoRA setup
- Dataset conversion and formatting
- Training with SFTTrainer
- Inference and streaming
- Saving and exporting models

---

## Best Practices

- Use 4-bit quantization for large models to avoid OOM errors
- Always convert datasets to a consistent chat format
- Monitor GPU memory usage with provided utilities
- For production, increase training steps and dataset size
- Use gradient checkpointing for long context lengths

---

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [FineTome-100k Dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- [Hugging Face TRL](https://huggingface.co/docs/trl/sft_trainer)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPT4All](https://gpt4all.io/index.html)

---

## License

This project is provided for research and educational purposes. Please check the licenses of individual models and datasets before commercial use.

---

## Acknowledgements

- Unsloth team for their high-performance PEFT library
- Maxime Labonne for the FineTome-100k dataset
- Hugging Face for open-source model and training tools

---

## Contact

For questions or support, join the [Unsloth Discord](https://discord.gg/u54VK8m8tk) or open an issue on the [GitHub repository](https://github.com/unslothai/unsloth).
