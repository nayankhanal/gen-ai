from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_vector = embeddings.embed_query("Kathmandu is the capital of Nepal.")

print(embedding_vector)
print("================================")
print(f"Embedding vector length: {len(embedding_vector)}")
print("================================")
print(f"Embedding vector type: {type(embedding_vector)}")
print("================================")
print(str(embedding_vector))