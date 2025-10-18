from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFacePipeline.from_model_id(
    model_id='mistralai/Mistral-7B-Instruct-v0.2',
    task='text-generation',
    pipeline_kwargs= dict(
        temperature=0.7,
        max_new_tokens=256,
    ),
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Explain the theory of relativity in simple terms.")

print(result.content)