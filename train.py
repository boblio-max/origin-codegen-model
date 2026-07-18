import json
import random
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Configuration
DATASET_PATH = "origin_instruction_tuning_dataset_v3.json"
MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./origin_codegen_model"
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 2e-4

# 1. Data Loading and Preprocessing
def format_example(example):
    instruction = example["instruction"]
    output = example["output"]
    return (
        f"### Instruction\n{instruction}\n\n### Response\n{output}"
    )

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total examples loaded: {len(dataset)}")

random.shuffle(dataset)

# Split dataset (using 10499 examples, so 10000 train, 499 test)
train_data = dataset[:10000]
test_data = dataset[10000:]

print(f"Training examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

# 2. Model and Tokenizer Initialization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Custom Dataset with Label Masking
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

        # Tokenize the full text
        full_tokens = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # Create labels and mask the instruction part
        labels = input_ids.clone()

        # Find the start of the response to mask the instruction
        instruction_text = f"### Instruction\n{example['instruction']}\n\n### Response\n"
        instruction_tokens = self.tokenizer(
            instruction_text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False # Important: do not add special tokens here
        )
        # Get the length of the instruction part, including the prompt for the response
        instruction_len = instruction_tokens["input_ids"].size(1)

        # Mask the instruction part by setting labels to -100
        labels[:instruction_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Initialize datasets
training_dataset = OriginDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
test_dataset = OriginDataset(test_data, tokenizer, MAX_SEQ_LENGTH)

# 4. Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    load_best_model_at_end=True,


    report_to="none", # Disable reporting to services like W&B
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=test_dataset, # Add evaluation dataset
    tokenizer=tokenizer,
)

# 5. Start Training
print("Starting training...")
trainer.train()

# 6. Save Model and Logs
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Trainer automatically saves logs, but we can also access them if needed
# For simplicity, we'll rely on Trainer's logging to output_dir/runs
print(f"Training complete. Model saved to {OUTPUT_DIR}")
