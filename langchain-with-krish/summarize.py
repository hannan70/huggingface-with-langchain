import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# import all necessary modules
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import streamlit as st
import validators

# setup prompt template
prompt = PromptTemplate.from_template(
    """
    Summarize the following content in approximately 300 words. The summary should be written entirely in {language}.

    Content:
    {text}
    """
)

# Streamlit UI
st.subheader("📝🌐 AI-Powered Text & Web Summarizer")
st.sidebar.title("Settings")

# Model and language selection
model_name = st.sidebar.selectbox('Choose Model', ['llama-3.3-70b-versatile', 'llama3-70b-8192', 'llama3-8b-8192'])
translate_language = st.sidebar.selectbox("Translate Language", ['English', 'Bangla'])

# LLM initialization
llm = ChatGroq(model=model_name, api_key=groq_api_key)

# input box
input_content = st.chat_input("Please provide input (text or URL) here...")

# Session
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "👋 Hi there! I’m your virtual summarization assistant.\n\nYou can provide a **website URL** or **plain text**, and I’ll summarize it for you in your preferred language. Just drop the content below!."
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])



if not input_content:
    st.info("Please provide input (text or URL) here...)")
else:
    try:
        # check if input is a valid url
        if validators.url(input_content):
            try: 
                loader = WebBaseLoader(web_path=input_content) 
                docs = loader.load()

                if not docs:
                    raise ValueError("❌ Unable to extract content from the URL.")
                
            except Exception as e:
                st.error(f"❌ Failed to fetch URL: {e}")
                st.stop()

        else:
            #  Summarize for plain text
            try: 
                if len(input_content.strip()) < 30:
                    raise ValueError("✍️ Please provide more detailed text for summarization.")
                docs = [Document(page_content=input_content)]
                
            except Exception as e:
                st.error(f"❌ Failed to Summarize plain text: {e}")
                st.stop()
        
        # append user message before processing
        st.session_state.messages.append({"role": "user", "content": input_content})
        st.chat_message("user").write(input_content)
        

        with st.spinner("Summarizing content... This might take a moment."):
            # Run summarization chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain.run({"input_documents": docs, "language": translate_language})
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        if "Request too large" in str(e) or "code: 413" in str(e):
            st.warning("⚠️ The content is too large for the selected model or plan. Try reducing input size.")
        else:
            st.error(f"❌ An unexpected error occurred: {e}")

 