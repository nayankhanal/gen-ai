from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# embedding_vector = embeddings.embed_query("Kathmandu is the capital of Nepal.")

documents = [
    "Kathmandu is the capital of Nepal.",
    "Kathmandu is the largest city in Nepal.",
    "Kathmandu is the cultural and economic hub of Nepal.",
    "Kathmandu is known for its rich history and heritage.",
    "Kathmandu is a popular tourist destination in Nepal.",
]

embedding_vector = embeddings.embed_documents(documents)

print(embedding_vector)