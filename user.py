# Install required libraries
#Use the finetuned llm
!pip install transformers torch -q

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import files
import zipfile

# Load the fine-tuned model
print("Please upload your 'financial_model.zip' file.")
uploaded = files.upload()
zip_file = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("./financial_model_extracted")
model_path = "./financial_model_extracted"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded successfully.")

# Generate response from the model
def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Interactive prompt loop
print("\nAsk a question about financial performance (or type 'exit' to stop):")
while True:
    prompt = input("Enter your prompt: ").strip()
    if prompt.lower() == 'exit':
        print("Goodbye!")
        break
    if not prompt:
        print("Please enter a valid prompt.")
        continue
    
    response = generate_response(prompt)
    print("\n=== Response ===")
    print(response)
    print()
