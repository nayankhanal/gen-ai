from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed long report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize this report in 3 sentences: \n {text}",
    input_variables=["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser

chain_result = chain.invoke({"topic": "glacial melting and its impact on sea levels"})

print(chain_result)

chain.get_graph().print_ascii()