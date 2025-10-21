from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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


# 1 detailed explanation
template1 = PromptTemplate(
    template='Write a detailed explanation on the topic: {topic}',
    input_variables=['topic']
)

prompt1 = template1.invoke({'topic' : 'black hole'})

report = model.invoke(prompt1).content

# 2 Five line summery
template2 = PromptTemplate(
    template='Write a 5 line summary on the report: {report}',
    input_variables=['report']
)
 
prompt2 = template2.invoke({'report' : report})

summary = model.invoke(prompt2).content

print(f"{report}")
print(f"\nNow 5 line summery is below : \n")
print(f"{summary}")