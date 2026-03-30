from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("[INFO] Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print("[SUCCESS] Model loaded on device:", model.device)