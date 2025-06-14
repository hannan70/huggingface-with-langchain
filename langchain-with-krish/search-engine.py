
import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()

# load groq api from env file
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# import all necessary modules
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Arxiv tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# wikipedia tools
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper) 

tavily_tool  = TavilySearchResults(tavily_api_key=tavily_api_key)

# setup all tools
tools = [wikipedia, arxiv, tavily_tool]

llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# initialize agent
search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    early_stopping_method="generate"
    )

st.subheader("🔍 LangChain Chatbot with Real-Time Web Search")

if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Hi there! I’m your virtual assistant. Feel free to ask me anything — I’m here to help!."
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

question = st.chat_input("What is generative AI?")

if question is not None:
    if question.strip():
        # append user message before processing
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Searching..."):
            streamlit_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(question, callbacks=[streamlit_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.success(response)