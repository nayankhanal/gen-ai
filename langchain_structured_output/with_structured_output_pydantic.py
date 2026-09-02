from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from typing import Literal
from pydantic import BaseModel, Field

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

class Review(BaseModel):
    reviewer: str = Field(..., description="The name of the person providing the review.")
    rating: int = Field(..., description="The rating given by the reviewer, on a scale of 1 to 5.")
    comment: str = Field(..., description="The review comment provided by the user.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the review, which can be positive, negative, or neutral.")

structured_model = model.with_structured_output(Review)

# print(structured_model)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.""")

print(result)
print(type(result))

result_dict = dict(result)
print(result_dict)

# python with_structured_output_pydantic.py