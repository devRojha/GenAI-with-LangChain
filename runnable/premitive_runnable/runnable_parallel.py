
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

prompt_template1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template="Generate a post content about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

chain1 = RunnableSequence(prompt_template1, model, parser)
chain2 = RunnableSequence(prompt_template2, model, parser)

parallel_chain = RunnableParallel({
    'tweet': chain1,
    'linkedIn': chain2
})

response = parallel_chain.invoke({'topic' : 'Prisma'})

print(response['tweet'])
print(response['linkedIn'])


