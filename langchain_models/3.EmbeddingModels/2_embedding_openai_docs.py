from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "Kathmandu is the capital of Nepal.",
    "Kathmandu is the largest city in Nepal.",
    "Kathmandu is the cultural and economic hub of Nepal.",
    "Kathmandu is known for its rich history and heritage.",
    "Kathmandu is a popular tourist destination in Nepal.",
]

embedding_vectors = embeddings.embed_documents(documents)

print(embedding_vectors)