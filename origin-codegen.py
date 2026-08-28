import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-1.7B"
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
run = True
while run:
    user = input("> ")
    if user.lower() == "exit":
        run = False
        break
    prompt = f"""
    ### Instruction
    {user}

    ### Response
    """
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

    # print(result)
    print(result.split("### Response\n")[1].strip())