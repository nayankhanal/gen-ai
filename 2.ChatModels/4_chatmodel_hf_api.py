from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_tokens=500)

result = model.invoke("what is the capital of Nepal?")

# print(result)
print("=================================")
print(result.content)