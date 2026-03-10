from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

tone_selection = ["Formal", "Concise", "Strategic"]


st.header("Executive Strategic Article Generator")
user_query = st.text_input("Please enter the topic for the article. To stop, simply leave the input empty and press Enter.")
tone = st.selectbox("Please select a tone for the article: ", tone_selection)

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates a complete executive-level article in a {selected_tone} tone. The article should be clean, well-structured, engaging, and in a professional format."),
    ("human", "Write a complete, professional article about: {article_topic}")
])

if st.button("Generate Article"):
    if user_query:
        st.write("Generating article...")
    else:
        st.write("Generation stopped, exiting...")