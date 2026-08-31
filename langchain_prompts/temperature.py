from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=500,
#         temperature=0.7
#     )
# )

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    provider="featherless-ai",
)

model = ChatHuggingFace(llm=llm, temperature=0.7, max_tokens=500)

result = model.invoke("who is the prime minister of Nepal?")

print(result.content)
