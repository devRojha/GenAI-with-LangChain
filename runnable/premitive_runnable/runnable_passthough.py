
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

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

passthrough = RunnablePassthrough() #-> give the same output as input


joke_generator = RunnableSequence(prompt_template1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation' : RunnableSequence(prompt_template2, model, parser)
})

final_chain = RunnableSequence(joke_generator, parallel_chain)

response = final_chain.invoke({'topic' : 'Rat'})

print (response)

