# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     provider="featherless-ai",
# )

# model = ChatHuggingFace(llm=llm, temperature=0.7, max_tokens=500)

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7, max_tokens=500)

chat_history = []

while True:
    user_input = input("You: ")
    if user_input == "exit":
        print(chat_history)
        break
    chat_history.append(user_input)
    result = model.invoke(chat_history)
    chat_history.append(result.content[0]['text'])
    print(f"Model: {result.content[0]['text']}")
