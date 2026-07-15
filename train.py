import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "origin_lora"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

def generate(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

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

    return result.split("### Response\n")[1].strip()

print("Welcome to the Origin-Codegen-Model! Type 'exit' to quit.")

while True:
    user_input = input("> ")
    if user_input.lower() == "exit":
        break
    if user_input.lower() == "help":
        print("This is the Origin Codegen Model, trained on Origin code.")
        print("Its base model is the Qwen2.5-1.5B model, and it has been fine-tuned using LoRA.")
        print("You can ask it to generate code or answer questions related to programming,")
        print("and it will provide you a response in origin code")
    response = generate(user_input)
    print(f"Model: {response}")