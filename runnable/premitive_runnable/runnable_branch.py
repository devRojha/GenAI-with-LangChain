#                                       > 500 words -> llm -> parse -> summarise
#   topic -> prompt -> llm -> parse -> 
#                                       < 500 words   -> As it is

#                               passthrought -> joke
# prompt -> llm -> parser ->                                    -> output
#                               lambda -> count words 




from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv
import os

load_dotenv()

hf_token =  os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

prompt_template1 = PromptTemplate(
    template="write a detail report on {topic}",
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template="Summarise the below report in less than 500 words \n {content}",
    input_variables=['content']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)


content_generator = RunnableSequence(prompt_template1, model, parser)

branch_chain = RunnableBranch(
    # (condition, runnable),
    (lambda x: len(x.split()) > 200 , RunnableSequence(prompt_template2, model, parser )),
    RunnablePassthrough()
)

final_chain = RunnableSequence(content_generator, branch_chain)

response = final_chain.invoke({'topic' : 'Russia Vs Ukraine'})

print(response)
