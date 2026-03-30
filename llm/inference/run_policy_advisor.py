import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
# Absolute path so the script works regardless of the working directory
LORA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lora_adapters"))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model (4-bit quantized, CPU-safe)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="cuda:0",
    trust_remote_code=True,
    attn_implementation="eager"
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    is_trainable=False
)

model.eval()

print("Model loaded successfully [OK]")

prompt = (
    "You are a cybersecurity advisor for startups.\n\n"
    "Draft a professional cybersecurity policy for:\n"
    "- Control: Multi-Factor Authentication\n"
    "- Company size: Small startup\n"
    "- Infrastructure: Cloud\n"
    "- Data sensitivity: High\n\n"
    "Explain why it matters and provide implementation steps.\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating response...\n")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

print("\n=== GENERATED POLICY ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))