# origin-codegen-model

The goal here is to fine-tune an LLM (Qwen2.5-Coder-3B / Qwen3-1.7B) with LoRA on ~20k instruction pairs (`origin_instruction_tuning_dataset_v3.json`) so it generates Origin language code from plain English instructions. Talk to it in the `origin-codegen.py` REPL, and it writes `.or` files.

## How it works

- `train.py` (now fixed) — LoRA fine-tune setup with `peft` + `transformers`
- `origin-codegen.py` — REPL for prompting the trained model
- `test_model.py` — Evals generated code by actually executing it via `origin i origintest.or` (execution-based eval, not vibes)
- `fororigin.py` / `otherfororigin.py` — data helpers

## Status: Ready to train ✅

The training script has been fixed and is ready to run. Key fixes:

- Removed duplicate code blocks that caused training errors
- Added missing `TrainingArguments` and `Trainer` imports from transformers
- Cleaned up duplicate dataset shuffling and print statements
- LoRA config is now consistent (r=16, alpha=32, all target modules)

## Requirements

- NVIDIA GPU with CUDA support (required for reasonable training time)
- Python packages: `torch`, `transformers`, `peft`, `datasets`, `accelerate`
- 20,500 instruction tuning pairs in `origin_instruction_tuning_dataset_v3.json`

## Stack

Python, PyTorch, Hugging Face `transformers` + `peft` (LoRA), Qwen coder models.

## Quick Start

```bash
# If repo already cloned:
cd C:\Users\smile\OneDrive\Documents\GitHub\origin-codegen-model && git pull && python train.py

# Or download ZIP from GitHub:
# 1. Go to https://github.com/boblio-max/origin-codegen-model
# 2. Click Code → Download ZIP
# 3. Extract and run: python train.py
```
