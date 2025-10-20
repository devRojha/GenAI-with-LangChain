# hugging face dosn't support structured output currently, so using OpenAI for this example
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

OPEN_API_KEY= os.getenv("OPEN_API_KEY")

model = ChatOpenAI(
    model='gpt-4', 
    temperature=0, 
    max_completion_tokens=500,
    openai_api_key=OPEN_API_KEY
)

#schema
json_schema = {
    "title" : "student",
    "description": "Schema for about student",
    "type": "object",
    "properties": {
        "name" : {
            "type": "string",
            "description": "Name of the student"
        },
        "age" : {
            "type": "integer",
            "description": "Age of the student",
            "minimum": 0,
            "maximum": 70,
            "default": 18
        },
        "email" : {
            "type": "array",
            "items": {
                "type": "string",
                "format": "email"
            },
            "description": "Email addresses of the student"
        },
        "phone": {
            "type": ["array", "null"],
            "items": {
                "type": "string",
                "pattern": "^[0-9]{10}$"
            },
            "description": "Phone numbers of the student"
        },
        "streem": {
            "type": "string",
            "enum": ["science", "commerce", "arts"],
            "description": "Category of the student"
        }
    },
    "required": ["name", "email"]
}


student_data = '''
The student, Devraj Kumar, is currently 21 years old and belongs to the science stream. They can be contacted via their email address(es): ["devraj@example.com", "dkumar@gmail.com"], and their phone number(s) are: ["9876543210", "9123456789"]. This information provides a complete overview of the student's personal details, academic category, and preferred contact methods.
'''

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(student_data)

# the tyep of result is like dict so getting name => result['name']

print(result)