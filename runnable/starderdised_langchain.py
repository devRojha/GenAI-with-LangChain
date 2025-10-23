from abc import ABC, abstractmethod

class Runnable (ABC) :

    @abstractmethod
    def invoke(input_data):
        pass


import random

# fake LLM implementation
class DummyLLM(Runnable) :
    def __init__(self):
        print("Initialized Dummy LLM")

    def invoke(self, prompt):

        response_list = [
            "Fact 1: Black holes can emit radiation due to quantum effects near the event horizon, known as Hawking radiation.",
            "Fact 2: The first image of a black hole was captured in 2019 by the Event Horizon Telescope.",
            "india is the capital of india.",
            "Ai stands for Artificial Intelligence.",
        ]

        return {'response' : random.choice(response_list)}


# prompt template implementation
class DummyPromptTemplate(Runnable) :
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    


# Runnablechain implementation

class RunnableConnector(Runnable) : 
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)  #-> input_data is updated such as output of previous is input of current
        
        return input_data

# stirng parser

class DummyStrOutputParser(Runnable) : 
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data['response']



llm = DummyLLM()

template = DummyPromptTemplate(
    template='write a {length} poem about {topic}',
    input_variables=['length','topic']
)

parser = DummyStrOutputParser()

chain = RunnableConnector([template, llm, parser])

response = chain.invoke({'length' : 'long', 'topic' : 'Cricket'})

print(response)
