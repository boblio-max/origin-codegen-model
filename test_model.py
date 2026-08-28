import json
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import random

MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "origin_codegen_model"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

with open("test_set.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

num_iterations = 10
sum_prob = 0.0
for i in range(num_iterations):
    correct_count = 0
    random.shuffle(dataset)
    for example in tqdm(dataset, desc=f"Iteration {i+1}"):
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
        parts = result.split("### Response\n")
        generated_code = parts[1].strip() if len(parts) > 1 else result.strip()

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
    print(f"Iteration {i+1} Accuracy: {probability:.2f}% ({correct_count}/{len(dataset)})")
    sum_prob += probability

print(f"Average Accuracy: {sum_prob/num_iterations:.2f}%")
