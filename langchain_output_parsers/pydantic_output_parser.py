from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

class Person(BaseModel):
    name: str = Field(..., description="The person's full name.")
    age: int = Field(..., description="The person's age in years.")
    city: str = Field(..., description="The person's city of residence.")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give the details about a fictional {place} person.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

chain_result = chain.invoke({"place": "indian"})

print(chain_result)