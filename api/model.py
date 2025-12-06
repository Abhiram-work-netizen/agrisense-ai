import google.generativeai as genai
import os

# Your key
GENAI_KEY = "AIzaSyDtvpP9plv3rfUKm2fPmzAyj_UQhSScotw"
genai.configure(api_key=GENAI_KEY)

print("Checking available models for your key...\n")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")