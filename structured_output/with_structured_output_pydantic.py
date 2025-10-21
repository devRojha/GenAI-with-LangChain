# hugging face dosn't support structured output currently, so using OpenAI for this example
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
load_dotenv()

OPEN_API_KEY= os.getenv("OPEN_API_KEY")

model = ChatOpenAI(
    model='gpt-4', 
    temperature=0, 
    max_completion_tokens=500,
    openai_api_key=OPEN_API_KEY
)


review = '''
The hardware 15 great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the Ul looks outdated compared to other brands. Hoping for a software update to fix this.

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
'''

class ReviewAnalysisOutput(BaseModel):
    key_themes : list[str] = Field(description="A list of key themes mentioned in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="The overall sentiment of the review, e.g., positive, negative, neutral")
#    sentiment: Literal["pos", "neg"] = Field(description="The overall sentiment of the review, e.g., positive, negative, neutral") # give etither pos or neg
    pros : Optional[list[str]] = Field(description="A list of pros mentioned in the review")
    cons : Optional[list[str]] = Field(description="A list of cons mentioned in the review")
    name : Optional[str] = Field(description="Name of the product being reviewed")  

structured_model = model.with_structured_output(ReviewAnalysisOutput)

result = structured_model.invoke(review)
# result is a pydantic object so geting name => result.name not result['name']
print(result)