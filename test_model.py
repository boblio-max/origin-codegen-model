import json
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

correct_count = 0

def format_example(example):

    return (
        "### Instruction\n"
        + example["instruction"]
        + "\n\n### Output\n"
        + example["expected_output"]
    )
    
with open("test_set.json","r",encoding="utf-8") as f:
    dataset = json.load(f)
    
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "origin_codegen_model"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)
model.to(device)
model.eval()

for example in tqdm(dataset):
    prompt = format_example(example)
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(result.split("### Output\n")[1].strip())
    
    # print("Instruction:", example["instruction"])
    # print("Expected Output:", example["expected_output"])
    # print("Model Output:", result.split("### Output\n")[1].strip())
    # print("="*50)
    
    # with open("origintests.or", "w") as f:
    #     f.write(result.split("### Output\n")[1].strip())
    
    # # pip install origin-or
    # result_from_origin = subprocess.run(
    #     ["origin", "i","origintests.or"],
    #     capture_output=True,
    #     text=True
    # )
    # print(result_from_origin.stdout)
    # if example["expected_output"] == result_from_origin.stdout.strip():
    #     correct_count += 1

probability = (correct_count / len(dataset)) * 100
print(f"Correctness probability: {probability:.2f}% ({correct_count}/{len(dataset)})")
