print("=== SCRIPT STARTED ===")

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATA_PATH = r"D:\Disk D\Study Material\AI & ML\PolicyCortex\llm\data\train.jsonl"
OUTPUT_DIR = r"D:\Disk D\Study Material\AI & ML\PolicyCortex\llm\lora_adapters"

# -----------------------------
# Load Dataset
# -----------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def format_example(example):
    prompt = (
        f"<s>[INST]\n"
        f"{example['instruction']}\n\n"
        f"{example['input']}\n"
        f"[/INST]\n"
    )
    response = example["output"]
    return {"text": prompt + response}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -----------------------------
# Quantization (4-bit QLoRA)
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA Config
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# Training Args
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    report_to="none"
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

print("=== ABOUT TO START TRAINING ===")
trainer.train()
print("=== TRAINING FINISHED ===")

# -----------------------------
# Save Adapters
# -----------------------------
print("=== SAVING LORA ADAPTERS ===")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("=== SCRIPT COMPLETED SUCCESSFULLY ===")

print("[SUCCESS] LoRA adapters saved to", OUTPUT_DIR)