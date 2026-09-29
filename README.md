# origin-codegen-model

the goal here is sick: fine-tune an LLM (Qwen2.5-Coder-3B / Qwen3-1.7B) with LoRA on ~20k instruction pairs (`origin_instruction_tuning_dataset_v3.json`) so it generates Origin language code from plain English instructions. talk to it in the `origin-codegen.py` REPL, it writes `.or` files.

## how it's supposed to work

- `train.py` (594 lines) — LoRA fine-tune setup with `peft` + `transformers`
- `origin-codegen.py` — REPL for prompting the trained model
- `test_model.py` — evals generated code by actually executing it via `origin i origintest.or` (execution-based eval, not vibes)
- `fororigin.py` / `otherfororigin.py` — data helpers

## honest status: WIP / broken

not gonna lie, `train.py` doesn't run right now — duplicated blocks, syntax errors, undefined `epochs`/`train_loader`, and two conflicting LoRA configs (r8 vs r16) fighting each other. the git log literally says "STILL NOT WORKING". the dataset and the eval idea are solid though; the training script needs surgery. the sister repo `origin-codegen-model-inorigin` ports this whole task into Origin scripts and is in better shape.

## stack

Python, PyTorch, Hugging Face `transformers` + `peft` (LoRA), Qwen coder models.
