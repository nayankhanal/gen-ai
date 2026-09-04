from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)
parser = JsonOutputParser()

template1= PromptTemplate(
    template="Give me name, age, and email of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt = template1.invoke({})

def get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)

result = model.invoke(prompt)

final_result = parser.parse(get_text(result.content))
print(final_result)

print("===============Using Chain=================")

chain = template1 | model | parser

chain_result = chain.invoke({})

print(chain_result)
