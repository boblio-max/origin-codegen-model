import json
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import torch.nn.functional as F
import subprocess
# ==========================
# Configuration
# ==========================

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_PATH = "origin_instruction_dataset.json"

MAX_LENGTH = 512
BATCH_SIZE = 1


# ==========================
# Device
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Tokenizer
# ==========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def logits_to_text(logits):
    predicted_token_ids = torch.argmax(
        logits,
        dim=-1
    )
    generated_text = tokenizer.decode(
        predicted_token_ids[0],
        skip_special_tokens=True
    )
    return generated_text

def text_to_vector(text):
    vector = []
    for char in text:
        vector.append(ord(char))
    return torch.tensor(vector, dtype=torch.float32)


def compare_outputs(generated, expected):

    gen_vec = text_to_vector(generated)
    exp_vec = text_to_vector(expected)
    max_len = max(
        len(gen_vec),
        len(exp_vec)
    )
    gen_vec = F.pad(
        gen_vec,
        (0, max_len - len(gen_vec))
    )
    exp_vec = F.pad(
        exp_vec,
        (0, max_len - len(exp_vec))
    )
    similarity = F.cosine_similarity(
        gen_vec,
        exp_vec,
        dim=0
    )
    return similarity.item()

def execute_loss(logits, ids, expected_code):
    gentext = logits_to_text(logits)
    loss_function = nn.CrossEntropyLoss()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    loss = loss_function(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loss_val = loss.item()
    with open("gen_code.or", "w") as f:
        f.write(gentext)
    with open("exp_code.or", "w") as f1:
        f1.write(expected_code)
    gen_result = subprocess.run(["origin", "gen_code.or"], capture_output=True, text=True).stdout
    exp_result = subprocess.run(["origin", "exp_code.or"], capture_output=True, text=True).stdout
    return loss_val * compare_outputs(gen_result, exp_result)
    
# ==========================
# Dataset
# ==========================

class OriginDataset(Dataset):

    def __init__(self, path):

        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        example = self.data[idx]

        instruction = example["instruction"]
        expected_code = example["output"]


        prompt = (
            "### Instruction\n"
            + instruction
            + "\n\n### Output\n"
        )


        tokens = tokenizer(
            prompt,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )


        expected_tokens = tokenizer(
            expected_code,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )


        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "instruction": instruction,
            "expected_code": expected_code,
            "expected_tokens": expected_tokens["input_ids"].squeeze(0)
        }



dataset = OriginDataset(DATASET_PATH)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================
# Model
# ==========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


# LoRA setup (same as training)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)


model = get_peft_model(
    model,
    lora_config
)


model.eval()


# ==========================
# Debug Loop
# ==========================

for batch in loader:

    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)


    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


    logits = outputs.logits


    print("\n==============================")
    print("Instruction:")
    print(batch["instruction"][0])


    print("\nExpected Code:")
    print(batch["expected_code"][0])


    print("\nExpected Token IDs:")
    print(
        batch["expected_tokens"][0].tolist()
    )


    print("\nLogits Shape:")
    print(logits.shape)


    print("\nFirst Token Logits:")
    print(
        logits[0][0]
    )


    execute_loss(logits, batch["expected_tokens"].to(model.device), batch["expected_code"][0])