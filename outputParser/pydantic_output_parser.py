from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=1.0,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel) :
    name: str = Field(description="name of the person")
    age : int = Field(gt=18, description="age of the person")
    city : str = Field(description="name of the city where the person belong to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} of a male persont \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({'place' : 'Russia'})

# print(prompt)

# result = model.invoke(prompt)

# actual_result = parser.parse(result.content)

chain = template | model | parser

actual_result = chain.invoke({'place' : 'India'})

print(actual_result)