from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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

model = ChatHuggingFace(llm=llm)


schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

# prompt = template.invoke({'topic' : 'black hole'})

# result  = model.invoke(prompt)

# actual_result = parser.parse(result.content)

chain = template | model | parser

actual_result = chain.invoke({'topic' : 'black hole'})

print(actual_result)


# disadvantage

# Can't do data validation => {name : str , age : int} => have to do manualy