import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# load .env all variable
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') 
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGSMITH_TRACING'] = 'true'


# setup prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI assistant. Answer the following question based only on the context below."),
    ("user", "{input}")
])

llm = ChatGroq(model='llama3-70b-8192')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# Streamlit setup
st.title("Simple GenAI Project With Langchain and Groq LLM")
text_input = st.text_input("What question you have in mind?")


if text_input:
   st.write(chain.invoke({"input": text_input}))

