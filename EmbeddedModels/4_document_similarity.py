from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embbeding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "do you know bumrah"

docsEmbeddings = embbeding.embed_documents(documents)
queryEmbedding = embbeding.embed_query(query)


scores = cosine_similarity([queryEmbedding], docsEmbeddings)[0]

index, score =  sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"\nMost similar document: {documents[index]}")
print(f"Similarity score: {score}")