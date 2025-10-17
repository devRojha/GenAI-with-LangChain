from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("Explain the theory of relativity in simple terms.")

print(result)

## for only content
print(result.content)
