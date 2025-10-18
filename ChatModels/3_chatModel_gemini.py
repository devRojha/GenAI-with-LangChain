from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro', 
    temperature=0, 
    max_completion_tokens=500,
    google_api_key=google_api_key
)

result = model.invoke("Explain the theory of relativity in simple terms.")

print(result)