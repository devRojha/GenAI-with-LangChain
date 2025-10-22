#            |=> Notes (model1) |
# Documents =|                  |=> (model3 => merge) shows to user
#            |=> quiz (model2)  |

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

llm3 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2-Exp",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)


model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)
model3 = ChatHuggingFace(llm=llm3)

prompt_template1 = PromptTemplate(
    template='Generate short and simple notes from the followin content : \n {content}',
    input_variables=['content']
)

prompt_template2 = PromptTemplate(
    template='Generate 5 quiz with 4 option and answer from the followin content : \n {content}',
    input_variables=['content']
)

prompt_template3 = PromptTemplate(
    template='Merge the provided notes on the top then following by the quiz and then last shows the correct answer of the quiz \n notes ->  {notes} \n quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
        'notes' : prompt_template1 | model1 | parser,
        'quiz' : prompt_template2 | model2 | parser
    }
)

mergeChain = prompt_template3 | model3 | parser

chain = parallel_chain | mergeChain

content = '''
Black holes are among the most mysterious cosmic objects, much studied but not fully understood. These objects aren't really holes. They're huge concentrations of matter packed into very tiny spaces. A black hole is so dense that gravity just beneath its surface, the event horizon, is strong enough that nothing - not even light - can escape. The event horizon isn't a surface like Earth's or even the Sun's. It's a boundary that contains all the matter that makes up the black hole.

There is much we don't know about black holes, like what matter looks like inside their event horizons. However, there is a lot that scientists do know about black holes.

Black holes don't emit or reflect light, making them effectively invisible to telescopes. Scientists primarily detect and study them based on how they affect their surroundings:

Black holes can be surrounded by rings of gas and dust, called accretion disks, that emit light across many wavelengths, including X-rays.
A supermassive black hole's intense gravity can cause stars to orbit around it in a particular way. Astronomers tracked the orbits of several stars near the center of the Milky Way to prove it houses a supermassive black hole, a discovery that won the 2020 Nobel Prize.
When very massive objects accelerate through space, they create ripples in the fabric of space-time called gravitational waves. Scientists can detect some of these by the ripples' effect on detectors.
Massive objects like black holes can bend and distort light from more distant objects. This effect, called gravitational lensing, can be used to find isolated black holes that are otherwise invisible.

Wormholes. They don't provide shortcuts between different points in space, or portals to other dimensions or universes.
Cosmic vacuum cleaners. Black holes don't suck in other matter. From far enough away, their gravitational effects are just like those of other objects of the same mass.
'''

result = chain.invoke ({'content' : content})

print(result)

chain.get_graph().print_ascii()