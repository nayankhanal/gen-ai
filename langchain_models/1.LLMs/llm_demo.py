from langchain_openai import OpenAI
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

llm = ChatGroq(model="openai/gpt-oss-120b")

result = llm.invoke("Write a poem about the beauty of nature.")

print(result)

