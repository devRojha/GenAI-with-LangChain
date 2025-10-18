from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

model = ChatAnthropic(
    model='claude-3-5-sonnet-20241022',
    temperature=0,
    max_completion_tokens=500,
    anthropic_api_key=ANTHROPIC_API_KEY
)

result = model.invoke("Explain the theory of relativity in simple terms.")

print(result)