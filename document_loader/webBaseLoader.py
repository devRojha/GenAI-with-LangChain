from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

prompt = PromptTemplate(
    template='Answer the following quesiton - \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser =  StrOutputParser()

model = ChatHuggingFace(llm = llm)


url = 'https://www.flipkart.com/canon-eos-r50-v-mirrorless-camera-body-withrf-s14-30mm-f4-6-3is-stmpz-lens/p/itmcf209cfd1dab3?pid=DLLHBG2MC6C4GSZS&lid=LSTDLLHBG2MC6C4GSZSGV8HKG&marketplace=FLIPKART&store=jek%2Fp31%2Ftrv&srno=b_1_1&otracker=browse&fm=organic&iid=en_sLBYd3QcWkMc5pAdvBg2oFKiKlSUR6zo2_TEuW5eApS55OdmPAIw-RsCv2wCMSUG-1yLKpFmZYBqZFVADa3imPUFjCTyOHoHZs-Z5_PS_w0%3D&ppt=hp&ppn=homepage&ssid=wny8umf1kg0000001762797874705'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser


result = chain.invoke({
    'question' : 'what is the price',
    'text' : docs[0].page_content
})

print(result)