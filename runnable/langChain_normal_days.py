import random

# fake LLM implementation
class FalseLLM:
    def __init__(self):
        print("Initialized false LLM")

    def predict(self, prompt):

        response_list = [
            "Fact 1: Black holes can emit radiation due to quantum effects near the event horizon, known as Hawking radiation.",
            "Fact 2: The first image of a black hole was captured in 2019 by the Event Horizon Telescope.",
            "india is the capital of india.",
            "Ai stands for Artificial Intelligence.",
        ]

        return random.choice(response_list)

llm = FalseLLM()
prompt = "give 3 facts about the black hole"
response = llm.predict(prompt)
print(response)


# prompt template implementation
class FalsePromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    

template = FalsePromptTemplate(
    template='write a {length} poem about {topic} ',
    input_variables=['length','topic']
)

prompt = template.format({'length':'short','topic' : 'black hole'})

response = llm.predict(prompt)

print(response)

# chain implementation

class FalseLLMChain:
    def __init__(self, llm, Prompt_template):
        self.llm = llm
        self.prompt = Prompt_template

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        print(final_prompt)
        result = self.llm.predict(final_prompt)
        return result

llm = FalseLLM()

template = FalsePromptTemplate(
    template='write a {length} poem about {topic} ',
    input_variables=['length','topic']
)

chain = FalseLLMChain(llm, template)

response = chain.run({'length' : 'long', 'topic' : 'Cricket'})

print(response)
