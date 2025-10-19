from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_template = ChatPromptTemplate([
    ('system', 'You are a helpfull {domain} expert'),
    ('human', 'You are a helpfull {domain} expert'),
])

prompt = chat_template.invoke({'domain' : 'cricket', 'topic' : 'LBW'})

print (f"{prompt}")