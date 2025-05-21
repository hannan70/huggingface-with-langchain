import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# LOAD ENV FILE VARIABLE
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY') 
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Q&A Chat With Groq"

# setup prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert AI assistant. Answer the following question based only on the context belo"),
        ("user", "{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    output_parser = StrOutputParser()
    llm = ChatGroq(model=llm,  temperature=temperature, max_tokens=max_tokens)
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Streamlit 
st.title("Q&A Chatbot With Groq API")

# sidebar
st.sidebar.title("settings")
llm = st.sidebar.selectbox('Choose Model', ['llama3-70b-8192', 'llama3-8b-8192', 'llama-3.3-70b-versatile'])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=150)

# Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("Write here:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")



