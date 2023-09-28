import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Function to generate a response
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

def main():
    # Load the fine-tuned model and tokenizer
    model_output_path = "/path/to/output/directory/Eve-chatbot"  # Replace with the actual path to your fine-tuned model
    tokenizer = GPT2Tokenizer.from_pretrained(model_output_path)
    model = GPT2LMHeadModel.from_pretrained(model_output_path)

    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print("Chatbot: Goodbye!")
            break

        # Generate a response
        response = generate_response(model, tokenizer, user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
