from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7,
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

st.header("Research Paper Summarizer")

paper_input = st.selectbox("Select Research Paper Name", ["Select...", "Attention Is All You Need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

language_input = st.selectbox("Select Language", ["English", "Hindi", "Urdu"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# load the prompt template

template = load_prompt('template.json')

# here we are inoking two times 

# prompt = template.invoke({
#     'paper_input': paper_input,
#     'language_input': language_input,
#     'style_input': style_input,
#     'length_input': length_input
# })

# if st.button('Summarize'):
#     result = model.invoke(prompt)
#     st.subheader("Summary:")
#     st.write(result.content)



# prompt = template.invoke({}) => result = modle.invoke(prompt)

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'language_input': language_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)