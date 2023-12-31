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

