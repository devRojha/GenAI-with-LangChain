from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = 'Devraj' # default value
    age: Optional[int] = None # optional field
    email : EmailStr # email validation
    cgpa : float = Field(gt=0.0, lt=10, default=5, description="Cumulative GPA") # cgpa should be between 0.0 and 10.0


new_student1 = {'name' : 'Prince', 'age' : 21, 'email' : 'abc@gmail.com', 'cgpa' : 8.5}
new_student2 = {'age' : '22', 'email' : 'abc@gamil.com', 'cgpa' : 8.1} #type coercion will happen here for age field change to integer

student1 = Student(**new_student1)
student2 = Student(**new_student2)

print(student1)
print(student2)

#dict formate
# student1_dict = dict(student1)
student1_dict = student1.model_dump()
print(type(student1_dict))
# json formate
student1_json = student1.model_dump_json()
print(student1_json)