from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)
parser = StrOutputParser()

template1= PromptTemplate(
    template="Write a detailed long report about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarize this report in 3 sentences: \n {text}",
    input_variables=["text"]
)

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "the current state of AI in healthcare"})

print(result)