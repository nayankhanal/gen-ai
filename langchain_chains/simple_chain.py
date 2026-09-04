from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

prompt = PromptTemplate(
    template="Write 5 interesting facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()