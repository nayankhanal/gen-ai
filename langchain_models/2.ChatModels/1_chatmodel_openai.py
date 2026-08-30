from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# model = ChatOpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=150)

model = ChatGroq(model="openai/gpt-oss-120b", temperature=1.5, max_tokens=500)

result = model.invoke("Write a poem about the beauty of nature.")

print(result.content)