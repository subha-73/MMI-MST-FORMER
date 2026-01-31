# check_available_models.py
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("="*60)
print("AVAILABLE GEMINI MODELS")
print("="*60)

print("\n📝 Text Generation Models:")
print("-"*60)
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"[OK] {model.name}")

print("\n🔢 Embedding Models:")
print("-"*60)
for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        print(f"[OK] {model.name}")

print("\n" + "="*60)
print("Copy the exact model names shown above to use in your code")
print("="*60)