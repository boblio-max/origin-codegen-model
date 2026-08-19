import json
import random
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "origin_instruction_tuning_dataset_v3.json"
MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./origin_codegen_model"

MAX_SEQ_LENGTH = 512

TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2

NUM_EPOCHS = 3
LEARNING_RATE = 1e-4

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ============================================================
# FORMAT DATASET
# ============================================================

def format_example(example):
    instruction = example["instruction"].strip()
    output = example["output"].strip()

    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Response\n"
        f"{output}"
    )


def format_prompt(example):
    instruction = example["instruction"].strip()

    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Response\n"
    )


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("=" * 60)
print("DATASET")
print("=" * 60)

print(f"Total examples loaded: {len(dataset)}")

if len(dataset) != 10499:
    print(
        f"WARNING: Expected 10499 examples, "
        f"but found {len(dataset)}."
    )

random.shuffle(dataset)

train_data = dataset[:10000]
eval_data = dataset[10000:]

print(f"Training examples: {len(train_data)}")
print(f"Evaluation examples: {len(eval_data)}")
print()


# ============================================================
# TOKENIZER
# ============================================================

print("=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")
print()


# ============================================================
# DATASET CLASS
# ============================================================

class OriginDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        example = self.data[index]

        full_text = format_example(example)
        prompt_text = format_prompt(example)

        # ----------------------------------------------------
        # Tokenize complete example
        # ----------------------------------------------------

        full_tokens = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # ----------------------------------------------------
        # Create labels
        # ----------------------------------------------------

        labels = input_ids.clone()

        # ----------------------------------------------------
        # Tokenize prompt separately
        # ----------------------------------------------------

        prompt_tokens = self.tokenizer(
            prompt_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            add_special_tokens=True,
        )

        prompt_len = prompt_tokens["input_ids"].size(1)

        # ----------------------------------------------------
        # Mask prompt tokens
        # ----------------------------------------------------

        labels[:prompt_len] = -100

        # Mask padding
        labels[attention_mask == 0] = -100

        # ----------------------------------------------------
        # Safety:
        # If truncation removed the entire response,
        # don't allow the example to contribute invalid loss.
        # ----------------------------------------------------

        if (labels != -100).sum() == 0:
            labels[-1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# CREATE DATASETS
# ============================================================

training_dataset = OriginDataset(
    train_data,
    tokenizer,
    MAX_SEQ_LENGTH,
)

eval_dataset = OriginDataset(
    eval_data,
    tokenizer,
    MAX_SEQ_LENGTH,
)


# ============================================================
# MODEL
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

if torch.cuda.is_available():
    model_dtype = torch.float16
else:
    model_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype,
    trust_remote_code=True,
)

# Required for training
model.config.use_cache = False

# Helps reduce VRAM usage
model.gradient_checkpointing_enable()

# Important when using gradient checkpointing
model.enable_input_require_grads()


# ============================================================
# LoRA
# ============================================================

print("=" * 60)
print("CONFIGURING LoRA")
print("=" * 60)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()

print()


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    per_device_eval_batch_size=TRAIN_BATCH_SIZE,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05,

    max_grad_norm=1.0,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=250,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=250,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=10,

    logging_first_step=True,

    report_to="none",

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    fp16=use_fp16,

    bf16=False,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED,

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    remove_unused_columns=False,

    # --------------------------------------------------------
    # Memory optimization
    # --------------------------------------------------------

    gradient_checkpointing=True,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="adamw_torch",

    # --------------------------------------------------------
    # Dataloader
    # --------------------------------------------------------

    dataloader_num_workers=2,

    dataloader_pin_memory=True,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=training_dataset,

    eval_dataset=eval_dataset,

    processing_class=tokenizer,
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

effective_batch_size = (
    TRAIN_BATCH_SIZE *
    GRADIENT_ACCUMULATION_STEPS
)

steps_per_epoch = (
    len(train_data) //
    effective_batch_size
)

total_steps = steps_per_epoch * NUM_EPOCHS


print("=" * 60)
print("STARTING ORIGIN CODEGEN TRAINING")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Training examples: {len(train_data)}")
print(f"Evaluation examples: {len(eval_data)}")
print(f"Sequence length: {MAX_SEQ_LENGTH}")
print(f"Batch size: {TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {effective_batch_size}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Estimated total optimizer steps: {total_steps}")
print(f"FP16: {use_fp16}")
print()
print("Starting training...")
print()


# ============================================================
# TRAIN
# ============================================================

train_result = trainer.train()


# ============================================================
# TRAINING SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(f"Training loss: {train_result.training_loss}")

print()


# ============================================================
# FINAL EVALUATION
# ============================================================

print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

eval_results = trainer.evaluate()

print(f"Final evaluation loss: {eval_results['eval_loss']}")

if "eval_runtime" in eval_results:
    print(f"Evaluation runtime: {eval_results['eval_runtime']:.2f}s")

print()


# ============================================================
# SAVE
# ============================================================

print("=" * 60)
print("SAVING MODEL")
print("=" * 60)

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")

print()
print("=" * 60)
print("ORIGIN CODEGEN TRAINING COMPLETE")
print("=" * 60)