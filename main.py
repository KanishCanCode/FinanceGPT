# Install required libraries
!pip install pdfplumber transformers torch datasets timeout-decorator -q

import pdfplumber
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch
from google.colab import files
from google.colab import drive
import re
import os
import timeout_decorator

# Step 1: File Input
print("Please upload your 3 PDF files (or click Cancel for Google Drive)")
try:
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No files uploaded")
    pdf_files = list(uploaded.keys())
except ValueError:
    print("Switching to Google Drive...")
    drive.mount('/content/drive')
    pdf_dir = '/content/drive/My Drive/FinancialPDFs'
    if not os.path.exists(pdf_dir):
        print(f"Error: Directory '{pdf_dir}' not found.")
        raise FileNotFoundError
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if len(pdf_files) < 3:
        print(f"Error: Found {len(pdf_files)} PDFs. Need at least 3.")
        raise FileNotFoundError
print(f"Processing {len(pdf_files)} PDF files: {pdf_files}")

# Step 2: Extract Key Financial Data (100 pages)
@timeout_decorator.timeout(120, timeout_exception=TimeoutError)
def extract_financial_data(pdf_file, max_pages=100):
    text = ""
    patterns = {
        "revenue": r"revenue[s]?\s*[:\-\s]*(?:₹|\$)?\s*([\d,\.]+)\s*(lakh|crore)?",
        "profit": r"(net income|profit)\s*[:\-\s]*(?:₹|\$)?\s*([\d,\.]+)\s*(lakh|crore)?",
        "debt": r"(total debt|liabilities)\s*[:\-\s]*(?:₹|\$)?\s*([\d,\.]+)\s*(lakh|crore)?",
        "cash": r"cash (flow|balance)\s*[:\-\s]*(?:₹|\$)?\s*([\d,\.]+)\s*(lakh|crore)?"
    }
    indicators = {k: {"value": None, "unit": None} for k in patterns.keys()}

    try:
        input_file = pdf_file if isinstance(pdf_file, str) else list(uploaded.keys())[list(uploaded.values()).index(pdf_file)]
        with pdfplumber.open(input_file) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text() or ""
                text += page_text
                print(f"Processed page {i+1}/{min(max_pages, len(pdf.pages))} of {os.path.basename(pdf_file)}")

                for key, pattern in patterns.items():
                    match = re.search(pattern, page_text, re.IGNORECASE)
                    if match and not indicators[key]["value"]:
                        numeric_part = match.group(2) if len(match.groups()) >= 2 else None
                        if numeric_part is None:
                            print(f"Warning: No numeric value found for '{key}' in {os.path.basename(pdf_file)}")
                            continue
                        try:
                            value = float(numeric_part.replace(",", ""))
                            unit = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
                            if unit == "lakh":
                                value *= 100000
                            elif unit == "crore":
                                value *= 10000000
                            indicators[key] = {"value": value, "unit": unit}
                        except (ValueError, AttributeError) as e:
                            print(f"Warning: Failed to parse '{key}' in {os.path.basename(pdf_file)}: {e}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
    return text, indicators

pdf_data = []
indicators_list = []
for pdf_file in pdf_files:
    try:
        text, ind = extract_financial_data(pdf_file)
        if text:
            pdf_data.append(text)
            indicators_list.append(ind)
        else:
            print(f"Skipping {os.path.basename(pdf_file)}: No data extracted")
    except TimeoutError:
        print(f"Timeout: Skipped {os.path.basename(pdf_file)} after 120 seconds")
    except Exception as e:
        print(f"Failed to process {os.path.basename(pdf_file)}: {e}")
if not pdf_data:
    raise ValueError("No data extracted from any PDF.")

# Step 3: Preprocessing for Fine-Tuning
def format_currency(value):
    """Convert value to lakh or crore if large enough."""
    if value is None:
        return "N/A"
    if value >= 10000000:
        return f"{value / 10000000:.2f} crore"
    elif value >= 100000:
        return f"{value / 100000:.2f} lakh"
    return f"{value:.2f}"

def process_financial_text(text, indicators):
    lines = text.split('\n')
    training_texts = []
    for line in lines:
        line = line.strip()
        if line and any(char.isdigit() for char in line):
            if "revenue" in line.lower() and indicators["revenue"]["value"]:
                value = format_currency(indicators["revenue"]["value"])
                training_texts.append(f"Revenue is ₹{value} – total sales.")
            elif "profit" in line.lower() and indicators["profit"]["value"]:
                value = format_currency(indicators["profit"]["value"])
                training_texts.append(f"Profit is ₹{value} – earnings after costs.")
            elif "debt" in line.lower() and indicators["debt"]["value"]:
                value = format_currency(indicators["debt"]["value"])
                training_texts.append(f"Debt is ₹{value} – borrowed funds.")
            elif "cash" in line.lower() and indicators["cash"]["value"]:
                value = format_currency(indicators["cash"]["value"])
                training_texts.append(f"Cash flow is ₹{value} – money movement.")
            else:
                training_texts.append(line)
    return training_texts if training_texts else ["No financial data found."]

processed_data = []
for i, data in enumerate(pdf_data):
    processed_data.extend(process_financial_text(data, indicators_list[i]))

df = pd.DataFrame(processed_data, columns=['financial_text'])
print("Sample of processed data for fine-tuning:")
print(df.head())
dataset = Dataset.from_pandas(df)

# Step 4: Fine-Tuning
#Time is very less so i have decresed the fine-tune quality.
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["financial_text"], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["financial_text"])

training_args = TrainingArguments(
    output_dir="./financial_model",
    num_train_epochs=1,
    per_device_train_batch_size=5,
    save_steps=500,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=1e-5,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting fine-tuning (2 epochs, batch size 2)...")
trainer.train()
model.save_pretrained("./financial_model")
tokenizer.save_pretrained("./financial_model")
print("Model fine-tuning completed!")

# Step 5: Simplified Storytelling with Key Info
def generate_insight(prompt, indicators, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = f"{generated}\n"
    if indicators["revenue"]["value"]:
        value = format_currency(indicators["revenue"]["value"])
        story += f"Revenue: ₹{value} – total sales.\n"
    if indicators["profit"]["value"]:
        value = format_currency(indicators["profit"]["value"])
        story += f"Profit: ₹{value} – what they kept.\n"
    if indicators["debt"]["value"]:
        value = format_currency(indicators["debt"]["value"])
        story += f"Debt: ₹{value} – borrowed funds.\n"
    if indicators["cash"]["value"]:
        value = format_currency(indicators["cash"]["value"])
        story += f"Cash Flow: ₹{value} – money movement.\n"
    story += "Tip: Revenue is income, profit is savings!"
    return story

# Step 6: Direct Text Output (Key Info Only)
print("\nKey Financial Insights:")
for company_idx, pdf_file in enumerate(pdf_files):
    company_name = os.path.basename(pdf_file)
    print(f"\n=== {company_name} ===")
    prompt = f"Tell me about {company_name}'s financial performance."
    insight = generate_insight(prompt, indicators_list[company_idx])
    print(insight)

!zip -r financial_model.zip ./financial_model
files.download("financial_model.zip")
#click on cancel to use the code without uplading any file.
#it will continue with already uploaded files to my drive


