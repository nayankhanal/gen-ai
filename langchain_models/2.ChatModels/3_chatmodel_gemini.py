from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-1.5", temperature=0.7, max_tokens=150)

result = model.invoke("Write a poem about the beauty of nature.")

print(result.content)