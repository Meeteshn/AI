import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load and clean dataset
df = pd.read_csv("poems.csv")  # Ensure this CSV is extracted from the Kaggle dataset
df = df[df['Poem'].notna()]    # Use correct column name
df['text'] = df['Poem']

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text']])

# Load tokenizer and model from local directory
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token  # For padding
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to(device)

# Tokenize dataset
def tokenize(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-poetry",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    prediction_loss_only=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("gpt2-poetry")
tokenizer.save_pretrained("gpt2-poetry")
