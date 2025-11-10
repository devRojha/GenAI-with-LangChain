from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser =  StrOutputParser()

model = ChatHuggingFace(llm = llm)



loader = TextLoader('./document_loader/cricket.txt', encoding='utf-8')

# page_content, metadata
docs = loader.load() 

# print(type(docs))

# print (docs[0])


chain = prompt | model | parser

result = chain.invoke({'poem' : docs[0].page_content})
print (result)