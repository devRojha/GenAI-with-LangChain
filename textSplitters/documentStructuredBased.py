# making chunks on => para (\n\n) -> line (\n) -> words -> character


from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

spiltter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=400,
    chunk_overlap=0,
   
)

text = '''
class Student:
    def __init__(self, name, age, grade):
        # Constructor initializes the object attributes
        self.name = name
        self.age = age
        self.grade = grade

    def display_info(self):
        # Function to display student information
        print(f"Name: {self.name}, Age: {self.age}, Grade: {self.grade}")

# Creating an object of the class
student1 = Student("Devraj", 20, "A")

# Calling the function
student1.display_info()

'''
chunks = spiltter.split_text(text)
print(len(chunks))
print (chunks[0])