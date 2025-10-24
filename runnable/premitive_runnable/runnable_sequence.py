
# R1 -> R2 -> R3 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

from dotenv import load_dotenv
import os

load_dotenv()

hf_token =  os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

prompt_template1 = PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template="Expain the below joke \n {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

chain = RunnableSequence(prompt_template1, model, parser, prompt_template2, model, parser)

response = chain.invoke("cricket")

print(response)


