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

prompt_template = PromptTemplate(
    template='Give me a 5 impotatnt facts about {topic} ',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt_template | model | parser

result = chain.invoke({'topic' : 'black hole'})

chain.get_graph().print_ascii()