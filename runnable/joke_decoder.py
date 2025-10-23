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
            "Delhi is the capital of india.",
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



template1 = DummyPromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

template2 = DummyPromptTemplate(
    template='explain the following joke. \n joke ->  {response}',
    input_variables=['response']
)

llm = DummyLLM()

parser = DummyStrOutputParser()

chain1 = RunnableConnector([template1, llm])

chain2 = RunnableConnector([template2, llm, parser])

final_chain = RunnableConnector([chain1, chain2])

response = final_chain.invoke({'topic' : 'AI'})

print(response)





 