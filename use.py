#Now, let us test the model.
#Use the following code if you are only performing inference (generating text). This can be placed in a separate notebook. 

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_response(model, tokenizer, prompt, max_length=250):
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

model_path = "/content/drive/MyDrive/ColabNotebooks/models/chat_models/my_thesis/"
# Load the fine-tuned model and tokenizer
my_chat_model = GPT2LMHeadModel.from_pretrained(model_path)
my_chat_tokenizer = GPT2Tokenizer.from_pretrained(model_path)

#In the case of the GPT-2 tokenizer, the model uses a byte-pair encoding (BPE) algorithm, which tokenizes text into subword units. As a result, one word might be represented by multiple tokens.
#For example, if you set max_length to 50, the generated response will be limited to 50 tokens, which could be fewer than 50 words, depending on the text.

prompt = "Summarize Bhattiprolu's thesis"  # Replace with your desired prompt
#prompt = "What is the most promising future technology?"
response = generate_response(my_chat_model, my_chat_tokenizer, prompt, max_length=100)  #
print("Generated response:", response)
