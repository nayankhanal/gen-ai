from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# query = "Who is the best Indian cricketer?"
query = "Who is called the God of Cricket?"
# query = "tell me about virat kohli"

embedding_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_vector], embedding_vectors)[0]

index, score = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1])[-1]

print(query)
print("================================")
print(documents[index])
print("================================")
print(f"Similarity score: {score}")

