import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

logs = {}

INSTRUCTION_HEADER = "### Instruction\n"
RESPONSE_HEADER = "\n\n### Response\n"


def format_example(example):
    prompt = INSTRUCTION_HEADER + example["instruction"] + RESPONSE_HEADER
    full_text = prompt + example["output"]
    return prompt, full_text


with open("origin_instruction_tuning_dataset_v3.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
print(len(dataset))
print(dataset[0])

random.shuffle(dataset)
train_set = dataset[: int(len(dataset) * 0.9)]
test_set = dataset[int(len(dataset) * 0.9):]
print(len(train_set))
print(len(test_set))

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# bf16 is far more stable for training than raw fp16 without a GradScaler.
# Falls back to fp32 if bf16 isn't supported by the GPU.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model_dtype = torch.bfloat16 if use_bf16 else torch.float32
print(f"Loading model in {model_dtype}")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=model_dtype)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(model.device)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


class OriginDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt, full_text = format_example(self.data[index])

        tokens = tokenizer(
            full_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Number of tokens belonging to the prompt (instruction + header),
        # used to mask them out of the loss so we only train on the response.
        prompt_len = len(
            tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
        )

        labels = input_ids.clone()
        # Mask padding tokens
        labels[attention_mask == 0] = -100
        # Mask prompt tokens (instruction + header) so loss is response-only
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / max(len(loader), 1)


training_data = OriginDataset(train_set)
train_loader = DataLoader(training_data, batch_size=2, shuffle=True)

eval_data = OriginDataset(test_set)
eval_loader = DataLoader(eval_data, batch_size=2, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

GRAD_ACCUM_STEPS = 8  # effective batch size = 2 * 8 = 16
epochs = 3

model.train()
for epoch in range(epochs):
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

    # Flush any remaining accumulated gradients at epoch end
    if (step + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    train_loss = total_loss / len(train_loader)
    eval_loss = evaluate(model, eval_loader)
    print(f"Epoch {epoch+1} train loss: {train_loss:.4f} | eval loss: {eval_loss:.4f}")
    logs[epoch + 1] = {"train_loss": train_loss, "eval_loss": eval_loss}

model.save_pretrained("origin_codegen_model")
tokenizer.save_pretrained("origin_codegen_model")
with open("training_logs.json", "w") as f:
    json.dump(logs, f, indent=2)