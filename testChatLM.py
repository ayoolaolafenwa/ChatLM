from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "ayoolaolafenwa/ChatLM"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

prompt = "<user>: Compose an email for a person asking for loan to fund their business from a bank. <chatbot>: "
prompt2 = "<user>: Write a birthday message for my sister. <chatbot>: "
prompt3 = "<user>: Give me a financial advise on investing in stocks. <chatbot>: "
prompt4 = "<user>: I am confused on how to start my morning every day, what should I do? <chatbot>: " 
prompt5 = "<user>: I am always tired and it reduces my day to day working capacity. What should I do to feel better? <chatbot>: "

sequences = pipeline(
   prompt,
    max_length=500,
    do_sample=True,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
