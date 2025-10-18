from langchain_openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model='gpt-3.5-turbo-instruct', openai_api_key=openai_key)

result = llm.invoke("what is the capital of India?")

print(result)