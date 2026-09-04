from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # Load environment variables from .env file

from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.7)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 of the response."),
    ResponseSchema(name="fact_2", description="Fact 2 of the response."),
    ResponseSchema(name="fact_3", description="Fact 3 of the response."),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Provide 3 interesting facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "the current state of AI in healthcare"})

print(result)

