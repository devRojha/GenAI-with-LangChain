from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True : 
    user_input = input("You: ")
    chat_history.append({"role": "user", "content": user_input})
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat...")
        break
    result = model.invoke(chat_history)
    print(f"Bot: {result.content}")
    chat_history.append({"role": "assistant", "content": result.content})

print(f"This is the entire chat history : ")
print(chat_history)