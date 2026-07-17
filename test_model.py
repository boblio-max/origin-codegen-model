import json
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import random

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "origin_codegen_model"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

correct_count = 0

with open("test_set.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
sum_prob = 0.0
num = len(dataset)
for i in range(len(dataset)):
    dataset = random.shuffle(dataset)
    for example in tqdm(dataset, desc="Testing"):
        prompt = f"""
        ### Instruction
        {example["instruction"]}

        ### Response
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = result.split("### Response\n")[1].strip()

        with open("origintest.or", "w", encoding="utf-8") as f:
            f.write(generated_code)

        proc = subprocess.run(
            ["origin", "i", "origintest.or"],
            capture_output=True,
            text=True
        )

        actual_output = proc.stdout.strip()
        expected_output = example["expected_output"].strip()

        if actual_output == expected_output:
            correct_count += 1

    probability = (correct_count / len(dataset)) * 100
    print(f"Accuracy{i}: {probability:.2f}% ({correct_count}/{len(dataset)})")
    sum_prob += probability

print(f"Final Accuracy(Sum of Probabilities): {sum_prob/num:.2f}%")