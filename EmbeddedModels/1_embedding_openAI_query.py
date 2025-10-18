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

result = embedding.embed_query("what is langchain?")

print(result)