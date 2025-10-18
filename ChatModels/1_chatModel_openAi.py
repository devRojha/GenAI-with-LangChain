from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPEN_API_KEY= os.getenv("OPEN_API_KEY")


# model initialization, 
# temperature (0 to 2) => code, predictive near to 0, creative near to 2
# max_completion_tokens => limit the response length
model = ChatOpenAI(
    model='gpt-4', 
    temperature=0, 
    max_completion_tokens=500,
    openai_api_key=OPEN_API_KEY
)


result = model.invoke("Explain the theory of relativity in simple terms.")

print(result)

## for only content
print(result.content)
