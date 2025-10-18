from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()  

# huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") => not needed

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
text ="my name is dev"
documents = [
    "Delhi is the capital of India.",
    "kolkata is the cultural capital of India.",
    "Mumbai is the financial capital of India.",
    "Peris is the capital of France."
]

vector = embedding.embed_documents(documents)

print(str(vector))