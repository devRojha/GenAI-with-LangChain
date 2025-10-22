#
#           |=> Positive => model => Thank you for your feedback!
# Feedback =|
#           |=> Negative => model => We're sorry to hear that. How can we improve?
#

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


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

class FeedbackClassification(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="The sentiment of the feedback")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedbackClassification)

prompt1_template = PromptTemplate(
    template='Classify the sentiment of the following feedback text into Posititve or Negetive. \n feedback -> {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2_template_positive = PromptTemplate(
    template='Generate a polite response to the positive feedback: {feedback}',
    input_variables=['feedback']
)

prompt3_template_negative = PromptTemplate(
    template='Generate an apologetic response to the negative feedback: {feedback}',
    input_variables=['feedback']
)


classifier_chain = prompt1_template | model | parser2

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2_template_positive | model | parser1),
    (lambda x: x.sentiment == "Negative", prompt3_template_negative | model | parser1),
    RunnableLambda(lambda x: "Unable to classify feedback sentiment.")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : "This product quality is very bad, i don't recomand this"})

print(result)

chain.get_graph().print_ascii()