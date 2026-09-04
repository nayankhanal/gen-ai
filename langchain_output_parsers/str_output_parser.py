from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

template1= PromptTemplate(
    template="Write a detailed long report about {topic}",
    input_variables=["topic"]
)
# the current state of AI in healthcare.


template2 = PromptTemplate(
    template="Summarize this report in 3 sentences: \n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke(
    {"topic": "the current state of AI in healthcare"}
)

report = model.invoke(prompt1)

prompt2 = template2.invoke(
    {"text": report.content}
)

summary = model.invoke(prompt2)

print(summary)
