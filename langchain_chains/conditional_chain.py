from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import Literal

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)
parser = StrOutputParser()

class ReviewResponse(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the review, either 'positive' or 'negative'.")

prompt1 = PromptTemplate(
    template="Analys the given review and do the sentiment analysis ,classify either that is positive or negative: \n {feedback}",
    input_variables=["feedback"]
)

structured_model = model.with_structured_output(ReviewResponse)

classifier_chain = prompt1 | structured_model

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback: \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Sentiment cannot be found."),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product is amazing! I love it."})

print(result)  # Output: positive

chain.get_graph().print_ascii()