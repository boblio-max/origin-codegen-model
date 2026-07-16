import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

logs = {}
def format_example(example):

    return (
        "### Instruction\n"
        + example["instruction"]
        + "\n\n### Response\n"
        + example["output"]
    )
    
with open("origin_instruction_tuning_dataset_v3.json","r",encoding="utf-8") as f:
    dataset = json.load(f)

print(len(dataset))
print(dataset[0])

random.shuffle(dataset)

train_set = dataset[:1800]
test_set = dataset[1800:]

print(len(train_set))
print(len(test_set))

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained( MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained( MODEL_NAME,dtype=torch.float16)
device = "cuda"
model.to(device)
print(model.device)

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

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

class OriginDataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        text=format_example(self.data[index])

        tokens=tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids":tokens["input_ids"].squeeze(0),
            "attention_mask":tokens["attention_mask"].squeeze(0),
            "labels":tokens["input_ids"].squeeze(0)
        }

training_data = OriginDataset(train_set)

train_loader = DataLoader(
    training_data,
    batch_size=2,
    shuffle=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4
)

epochs = 3
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(
        f"Epoch {epoch+1} loss:",
        total_loss / len(train_loader)
    )
    logs.append({
        "epoch": epoch + 1,
        "loss": total_loss / len(train_loader)
    })

model.save_pretrained("origin_codegen_model")
tokenizer.save_pretrained("origin_codegen_model")
with open("training_logs.json", "w") as f:
    json.dump(logs)
    