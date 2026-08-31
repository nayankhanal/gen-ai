from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.7)

template = ChatPromptTemplate(
    [
        ('system', "You are a helpful {field} assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ('human', "{input}"),
    ]
)

chat_history = []

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())


# print(chat_history)

prompt = template.invoke({"chat_history": chat_history, "field": "customer support", "input": "Where is my refund?"})

result = model.invoke(prompt)
print(result.content[0]['text'])