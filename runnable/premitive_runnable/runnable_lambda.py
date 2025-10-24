#                               passthrought -> joke
# prompt -> llm -> parser ->                                    -> output
#                               lambda -> count words 




from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough

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

prompt_template = PromptTemplate(
    template="write a joke on {topic}",
    input_variables=['topic']
)

def word_counter(text):
    return len(text.split())

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)


joke_generator = RunnableSequence(prompt_template, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
    # 'word_count' : RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(joke_generator, parallel_chain)

result = final_chain.invoke({'topic' : 'AI'})

final_result = '''{} \n word count - {}'''.format(result['joke'], result['word_count'])

print (final_result)

