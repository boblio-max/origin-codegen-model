import json
from transformers import AutoTokenizer

DATASET_PATH = "origin_instruction_tuning_dataset_v3.json"
MODEL_NAME = "Qwen/Qwen3-1.7B"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lengths = []

for i, example in enumerate(dataset):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        formatted_text = (
            f"### Instruction\n{instruction}\n\n"
            f"### Input\n{input_text}\n\n"
            f"### Response\n{output}"
        )
    else:
        formatted_text = (
            f"### Instruction\n{instruction}\n\n"
            f"### Response\n{output}"
        )

    tokens = tokenizer(
        formatted_text,
        add_special_tokens=True,
        truncation=False,
    )

    token_count = len(tokens["input_ids"])
    lengths.append(token_count)

max_length = max(lengths)
min_length = min(lengths)
average_length = sum(lengths) / len(lengths)

sorted_lengths = sorted(lengths)

def percentile(values, percentile):
    index = int(len(values) * percentile / 100)
    index = min(index, len(values) - 1)
    return values[index]

print("=" * 60)
print("ORIGIN DATASET TOKEN LENGTH ANALYSIS")
print("=" * 60)

print(f"Total examples: {len(dataset)}")
print(f"Shortest example: {min_length} tokens")
print(f"Average length: {average_length:.2f} tokens")
print(f"Median length: {percentile(sorted_lengths, 50)} tokens")
print(f"90th percentile: {percentile(sorted_lengths, 90)} tokens")
print(f"95th percentile: {percentile(sorted_lengths, 95)} tokens")
print(f"99th percentile: {percentile(sorted_lengths, 99)} tokens")
print(f"Longest example: {max_length} tokens")

print()
print("TRUNCATION ANALYSIS")
print("-" * 60)

for limit in [256, 384, 512, 768, 1024, 1536, 2048]:
    count = sum(length > limit for length in lengths)
    percentage = (count / len(lengths)) * 100

    print(
        f">{limit:4} tokens: "
        f"{count:5} examples "
        f"({percentage:.2f}%)"
    )

longest_index = lengths.index(max_length)
longest_example = dataset[longest_index]

print()
print("=" * 60)
print("LONGEST EXAMPLE")
print("=" * 60)

print(f"Dataset index: {longest_index}")
print(f"Token count: {max_length}")
print()
print("Instruction:")
print(longest_example["instruction"])
print()
print("Input:")
print(longest_example.get("input", ""))
print()
print("Output:")
print(longest_example["output"])