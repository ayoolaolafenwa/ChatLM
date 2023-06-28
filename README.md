# ChatLM

It is a chat Large Language model finetuned with pretrained [Falcon-1B model](https://huggingface.co/tiiuae/falcon-rw-1b)
and trained on [chat-bot-instructions prompts dataset](https://huggingface.co/datasets/ayoolaolafenwa/sft-data).
ChatLM was trained on a dataset containing normal day to day human conversations, due to limited data used in training
it is not suitable for tasks like coding and current affairs. 

## Load Model in bfloat16
``` python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ayoolaolafenwa/ChatLM"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True,
torch_dtype=torch.bfloat16)

prompt = "<user>: Give me a financial advise on investing in stocks. <chatbot>: "

tokens = tokenizer(prompt, return_tensors="pt")

token_ids = tokens.input_ids
attention_mask=tokens.attention_mask

token_ids = token_ids.to(model.device)
attention_mask=attention_mask.to(model.device)

outputs = model.generate(input_ids=token_ids, attention_mask = attention_mask,  max_length=2048,do_sample=True,
num_return_sequences=1,top_k = 10, temperature = 0.7, eos_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(outputs[0])
output_text = output_text.replace("<|endoftext|>", "")

print(output_text)
```

## Load Model in bfloat16 and int8
``` python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "ayoolaolafenwa/ChatLM"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True,
torch_dtype=torch.bfloat16, load_in_8bit=True)

prompt = "<user>: Give me a financial advise on investing in stocks. <chatbot>: "

tokens = tokenizer(prompt, return_tensors="pt")

token_ids = tokens.input_ids
attention_mask=tokens.attention_mask

token_ids = token_ids.to(model.device)
attention_mask=attention_mask.to(model.device)

outputs = model.generate(input_ids=token_ids, attention_mask = attention_mask,  max_length=2048,do_sample=True,
num_return_sequences=1,top_k = 10, temperature = 0.7, eos_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(outputs[0])
output_text = output_text.replace("<|endoftext|>", "")

print(output_text)
```
## Training procedure for Supervised Finetuning

Chatbot Instructions prompts dataset from https://huggingface.co/datasets/alespalla/chatbot_instruction_prompts/viewer/alespalla--chatbot_instruction_prompts
was processed into a supervised finetuning for training a user prompt and corresponding response.

##### Download Data
``` python
from datasets import load_dataset

dataset = load_dataset("alespalla/chatbot_instruction_prompts", split = "train")
dataset.save_to_disk('ChatBotInsP')
dataset.to_csv('CIPtrain.csv')
```

##### Code to process dataset into Supervised finetuning format
``` python
# Import pandas library
import pandas as pd

# Read the text dataset from csv file
text_data = pd.read_csv("CIPtrain.csv")

# Create empty lists for prompts and responses
prompts = []
responses = []

# Loop through the text data
for i in range(len(text_data)):
    # Get the sender, message, and timestamp of the current row
    prompt = text_data["prompt"][i]
    prompt = str(prompt)

    response = text_data["response"][i]
    response = str(response)
    
    # Add the message to the prompts list with <user> tag
    prompts.append("<user>: " + prompt)
    #elif sender == "bot":
    # Add the message to the responses list with <chatbot> tag
    responses.append("<chatbot>: " + response)

# Create a new dataframe with prompts and responses columns
new_data = pd.DataFrame({"prompt": prompts, "response": responses})

#alespalla/chatbot_instruction_prompts
# Write the new dataframe to a csv file
new_data.to_csv("MyData/chatbot_instruction_prompts_train.csv", index=False)
```
I appended the user's prompts in the dataset with the tag <user> and the response with the tag <chatbot>.
Check the the modified dataset https://huggingface.co/datasets/ayoolaolafenwa/sft-data .

ChatLM was trained with preatrained [Falcon-1B model](https://huggingface.co/tiiuae/falcon-rw-1b) and finetuned on the prepared supervised 
dataset on a single H100 GPU. Check the full code for training [here](https://github.com/ayoolaolafenwa/ChatLM/blob/main/trainSFT.py). 
Check the training config [here](https://github.com/ayoolaolafenwa/ChatLM/blob/main/trainConf.conf)
