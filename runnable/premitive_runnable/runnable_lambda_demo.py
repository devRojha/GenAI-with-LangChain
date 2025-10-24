from langchain.schema.runnable import RunnableLambda


def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)

response = runnable_word_counter.invoke("hii there how are you?")

print(response)