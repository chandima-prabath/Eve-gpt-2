import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Function to read Eve-casual.txt
def read_eve_casual(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

# Update the directory and model output path
directory = "/path/to/dataset/directory/Eve-casual.txt"  # Replace with the path to your "Eve-casual.txt" dataset
model_output_path = "/path/to/output/directory/Eve-chatbot"  # Replace with the path to your desired output directory

# Read Eve-casual.txt
combined_text = read_eve_casual(directory)
combined_text = re.sub(r'\n+', '\n', combined_text).strip()  # Remove excess newline characters

# Set up the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Use the GPT-2 variant of your choice
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Use the GPT-2 variant of your choice

# Prepare the dataset
train_dataset = TextDataset(tokenizer=tokenizer, file_path=directory, block_size=128)  # Use Eve-casual.txt as the dataset
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir=model_output_path,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model(model_output_path)

# Save the tokenizer
tokenizer.save_pretrained(model_output_path)

# Rest of your script remains the same

# Test the chatbot
prompt = "What would you like to chat about?"  # Replace with your desired prompt
response = generate_response(model, tokenizer, prompt)
print("Generated response:", response)
