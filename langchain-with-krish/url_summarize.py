import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import validators
import re

# setup enviroment variable
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY') 

# import langchain 
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi

# setup prompt template
prompt = PromptTemplate.from_template(
    """
        Summarize the following content in approximately 300 words. The summary should be written entirely in {language}.
        Content:
        {text} 
    """
)

# setup streamlit
st.set_page_config(page_title="Summarize Text from YouTube or Website")
st.title("Summarize Text from YouTube or Website") 

# Sidebar 
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox('Choose Models', ['llama3-70b-8192', 'llama3-8b-8192', 'llama-3.3-70b-versatile'])
llm = ChatGroq(model=model_name)


# Translate language
translate_language = st.sidebar.selectbox("Translate Language", ['English', 'Bangla'])

# url
input_url = st.text_input("URL", label_visibility="collapsed")


if st.button("Summarize"):
    # validation
    if not input_url:
        st.error("Please Provide The information")
    elif not validators.url(input_url):
        st.error("Please Provide Valid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # check video url language 
                if "youtube.com" in input_url: 
                    lang_code = None
                    lang_name = None

                    # get video id
                    video_id_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", input_url)
                    video_id = video_id_match.group(1)
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                    for transcript in transcript_list:
                        lang_code = transcript.language_code
                        lang_name = transcript.language
                
                    try:
                        loader = YoutubeLoader.from_youtube_url(input_url, add_video_info=False, language=lang_code)  
                    except Exception as e:  
                        st.error("Please Select Valid Video Language")
                else:
                    loader = UnstructuredURLLoader(urls=[input_url], headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                    }) 
                docs = loader.load() 
                # Summarization Chain
                chain = load_summarize_chain(llm,  chain_type="stuff", prompt=prompt)
                summary = chain.run({"input_documents": docs, "language": translate_language})
                st.success(summary) 

        except Exception as e:
            st.error(e)