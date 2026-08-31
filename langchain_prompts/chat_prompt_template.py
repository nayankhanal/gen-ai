from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7,)

template = ChatPromptTemplate(
    [
        ('system', "You are a helpful {field} assistant."),
        ('human', "{input}"),
    ]
)

user_input = {"field": "medical", "input": "What are the symptoms of blood cancer?"}

chain = template | model

# print(template.invoke(user_input))
result = chain.invoke(user_input)
print(result.content[0]['text'])
