from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from typing import Literal
from pydantic import BaseModel, Field

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

# json
review_schema = {
    "title": "Review",
    "description": "Schema for a product review",
    "type": "object",
    "properties": {
        "reviewer": {
            "type": "string",
            "description": "The name of the person providing the review."
        },
        "rating": {
            "type": "integer",
            "description": "The rating given by the reviewer, on a scale of 1 to 5."
        },
        "comment": {
            "type": "string",
            "description": "The review comment provided by the user."
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative"],
            "description": "The sentiment of the review, which can be positive, negative, or neutral."
        }
    },
    "required": ["reviewer", "rating", "comment", "sentiment"]
}

structured_model = model.with_structured_output(review_schema)

# print(structured_model)

result = structured_model.invoke("""The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.""")

print(result)
print(type(result))

result_dict = dict(result)
print(type(result_dict))
