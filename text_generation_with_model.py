# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "microsoft/DialoGPT-medium"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# question=""
# answer=""
# prompt = "Tell me a joke about programming."

# inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=50,
#     temperature=0.8,
#     do_sample=True
# )

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)

# First ensure you have the library installed:
# pip install transformers torch

from transformers import pipeline

# Initialize the text-generation pipeline with a pre-trained model (e.g., GPT-2)
generator = pipeline('text-generation', model='gpt2')

# Define your starting prompt
prompt = "Artificial Intelligence will change the future of programming by"

# Generate text
results = generator(
    prompt, 
    max_new_tokens=50,       # Number of new tokens to generate
    num_return_sequences=1,   # Number of variations to return
    do_sample=True,          # Enable sampling for creative outputs
    temperature=0.7,         # Control creativity (lower is more focused)
    top_k=50                 # Limits pool to top 50 token choices
)

# Print the final generated text
print(results[0]['generated_text'])
