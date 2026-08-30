from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7, max_tokens=150)

result = model.invoke("Write a poem about the beauty of nature.")

print(result.content)