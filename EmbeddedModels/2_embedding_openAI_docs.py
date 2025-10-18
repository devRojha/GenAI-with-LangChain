from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
openai_key = os.getenv("OPEN_API_KEY")

embedding = OpenAIEmbeddings(
    model='text-embedding-3-large',
    dimensions=32,
    openai_api_key=openai_key
)

documents  = [
    "Delhi is the capital of India.",
    "kolkata is the cultural capital of India.",
    "Mumbai is the financial capital of India.",
    "Peris is the capital of France."
]

result = embedding.embed_documents(documents)
 
print(str(result))