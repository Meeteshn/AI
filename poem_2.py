from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("gpt2-poetry")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-poetry")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

prompt = "In the shadow of the stars"
result = generator(prompt, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)

print(result[0]['generated_text'])
