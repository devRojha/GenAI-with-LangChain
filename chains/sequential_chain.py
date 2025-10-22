#topic => LLM (Detailed report) => Report => LLM (5 line summary) => Summary

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)


prompt_template1 = PromptTemplate(
    template='Write a detailed explanation on the topic: {topic}',
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template='Write a 5 line summary on the report: {report}',
    input_variables=['report']
)

parser = StrOutputParser()

chain = prompt_template1 | model | parser | prompt_template2 | model | parser

result = chain.invoke({'topic' : 'black hole'})

print(result)

# show the chain graph
chain.get_graph().print_ascii()