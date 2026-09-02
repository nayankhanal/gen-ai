from pydantic import BaseModel, Field, EmailStr, HttpUrl
from typing import List, Optional

class Student(BaseModel):
    name: str = Field(..., description="The name of the student.")
    age: Optional[int] = Field(default=None, description="The age of the student.", optional=True, strict=True)
    grade: str = Field(..., description="The grade of the student."),
    email: Optional[EmailStr] = Field(default=None, description="The email address of the student.", optional=True)
    cgpa: float = Field(..., description="The CGPA of the student.", gt=0.0, lt=4.0)


new_student = {"name": "Alice", "age": 32, "grade": "A", "email": "alice@example.com", "cgpa": 3.8}

student = Student(**new_student)

print(student)