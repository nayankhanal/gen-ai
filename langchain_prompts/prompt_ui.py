from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import streamlit as st


llm = HuggingFaceEndpoint(
        repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        provider="featherless-ai",
    )

model = ChatHuggingFace(llm=llm, temperature=0.7,)


st.header("Research tool")

# research paper select box
research_paper = st.selectbox(
    "Select a research paper:",
    [
        "Paper 1: A Comprehensive Study on Machine Learning Algorithms",
        "Paper 2: Advances in Natural Language Processing",
        "Paper 3: Deep Learning for Computer Vision",
    ],
)

# short, medium or long
length_input = st.radio(
    "Select the length of the answer:",
    ("Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (6+ paragraphs)"),
)

# explaination style
style_input = st.radio(
    "Select the explanation style:",
    ("Beginner Friendly", "Code Oriented", "Mathematical", "Technical"),
)

# template for the prompt
template = load_prompt("template.json")

if st.button("Submit"):

    chain = template | model

    result = chain.invoke({
        "paper_input": research_paper,
        "length_input": length_input,
        "style_input": style_input
    })

    st.write(result.content)