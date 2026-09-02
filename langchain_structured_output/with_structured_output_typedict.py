from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import TypedDict, Annotated, Literal

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

class Review(TypedDict):
    reviewer: Annotated[str, "The name of the person providing the review."]
    rating: Annotated[int, "The rating given by the reviewer, on a scale of 1 to 5."]
    comment: Annotated[str, "The review comment provided by the user."]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "The sentiment of the review, which can be positive, negative, or neutral."]

structured_model = model.with_structured_output(Review)

print(structured_model)

# result = structured_model.invoke("""The hardware is great, but the software feels bloated.
# There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
# compared to other brands. Hoping for a software update to fix this.""")

# print(result)


# python with_structured_output_typedict.py