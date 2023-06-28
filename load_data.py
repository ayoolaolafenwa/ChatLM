from datasets import load_dataset

dataset = load_dataset("alespalla/chatbot_instruction_prompts", split = "train")
dataset.save_to_disk('ChatBotInsP')
dataset.to_csv('CIPtrain.csv')

