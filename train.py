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


DATASET_PATH = "origin_instruction_tuning_dataset_v3.json"
MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./origin_codegen_model"

MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
SEED = 42


random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def format_example(example):
    instruction = example["instruction"]
    output = example["output"]

    return (
        f"### Instruction\n"
        f"{instruction}\n\n"
        f"### Response\n"
        f"{output}"
    )


with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total examples loaded: {len(dataset)}")

if len(dataset) != 10499:
    print(
        f"WARNING: Expected 10499 examples, "
        f"but found {len(dataset)}."
    )

random.shuffle(dataset)

train_data = dataset[:10000]
test_data = dataset[10000:]

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


class OriginDataset(Dataset):

    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]

        formatted_text = format_example(example)

        full_tokens = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        instruction_text = (
            f"### Instruction\n"
            f"{example['instruction']}\n\n"
            f"### Response\n"
        )

        instruction_tokens = self.tokenizer(
            instruction_text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        instruction_len = instruction_tokens["input_ids"].size(1)

        labels[:instruction_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


training_dataset = OriginDataset(
    train_data,
    tokenizer,
    MAX_SEQ_LENGTH,
)

test_dataset = OriginDataset(
    test_data,
    tokenizer,
    MAX_SEQ_LENGTH,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

model.config.use_cache = False


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


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=True,
    seed=SEED,
    data_seed=SEED,
    remove_unused_columns=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)


print("=" * 60)
print("STARTING ORIGIN CODEGEN TRAINING")
print("=" * 60)

print(f"Model: {MODEL_NAME}")
print(f"Training examples: {len(train_data)}")
print(f"Evaluation examples: {len(test_data)}")
print(f"Sequence length: {MAX_SEQ_LENGTH}")
print(f"Batch size: {TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(
    f"Effective batch size: "
    f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print()

trainer.train()


print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

eval_results = trainer.evaluate()

print(f"Final evaluation loss: {eval_results['eval_loss']}")


print("Saving model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Model saved to: {OUTPUT_DIR}")