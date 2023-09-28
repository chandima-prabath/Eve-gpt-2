#This code is an implementation of a chatbot using the GPT-2 language model from Hugging Face's Transformers library.

!pip install transformers
!pip install torch
!pip install -U PyPDF2
!pip install python-docx

import os
import re
from PyPDF2 import PdfReader
import docx
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

#Required functions to read text from various files located in a directory. Files can be a mix of pdf, docx, or txt.
# Functions to read different file types
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = ""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            combined_text += read_pdf(file_path)
        elif filename.endswith(".docx"):
            combined_text += read_word(file_path)
        elif filename.endswith(".txt"):
            combined_text += read_txt(file_path)
    return combined_text

#The train_chatbot function uses the combined text data to train a GPT-2 model using the provided training arguments. The resulting trained model and tokenizer are then saved to a specified output directory.

def train_chatbot(directory, model_output_path, train_fraction=0.8):
    # Read documents from the directory
    combined_text = read_documents_from_directory(directory)
    combined_text = re.sub(r'\n+', '\n', combined_text).strip()  # Remove excess newline characters

    # Split the text into training and validation sets
    split_index = int(train_fraction * len(combined_text))
    train_text = combined_text[:split_index]
    val_text = combined_text[split_index:]

    # Save the training and validation data as text files
    with open("train.txt", "w") as f:
        f.write(train_text)
    with open("val.txt", "w") as f:
        f.write(val_text)

    # Set up the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")  #also try gpt2, gpt2-large and gpt2-medium, also gpt2-xl
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")  #also try gpt2, gpt2-large and gpt2-medium, also gpt2-xl

    # Prepare the dataset
    train_dataset = TextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=128)
    val_dataset = TextDataset(tokenizer=tokenizer, file_path="val.txt", block_size=128)
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
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_output_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(model_output_path)

#The generate_response function takes a trained model, tokenizer, and a prompt string as input and generates a response using the GPT-2 model.
def generate_response(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Create the attention mask and pad token id
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

#The main function is the entry point for the program. It specifies the path to the directory containing the training data and the path to the output directory for the trained model and tokenizer. It then trains the chatbot using the train_chatbot function and generates a response to a specified prompt using the generate_response function.
def main():
    directory = "/content/drive/MyDrive/ColabNotebooks/data/chatbot_docs/my_publications/"  # Replace with the path to your directory containing the files
    model_output_path = "/content/drive/MyDrive/ColabNotebooks/models/chat_models/my_thesis/"

    # Train the chatbot
    train_chatbot(directory, model_output_path)

    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_output_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_output_path)

    # Test the chatbot
    prompt = "What is bulk metallic glass?"  # Replace with your desired prompt
    response = generate_response(model, tokenizer, prompt)
    print("Generated response:", response)

if __name__ == "__main__":
    main()
