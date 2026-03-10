from langchain_core.prompts import ChatPromptTemplate 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv() 
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates a complete executive-level article in a {selected_tone} tone. The article should be clean, well-structured, engaging, and in a professional format."),
    ("human", "Write a complete, professional article about: {article_topic}")
])

tone_selection = ["Formal", "Concise", "Strategic"]

st.header("Executive Strategic Article Generator")
user_query = st.text_input("Please enter the topic for the article...")
tone = st.pills("Please select tone for the article", tone_selection, default="Formal")

if st.button("Generate Article"):
    if user_query: 
        formatted_prompt = prompt.format_messages(article_topic=user_query, selected_tone=tone)

        st.subheader("Generated Article:") 
        def streamResponse():
            for chunk in llm.stream(formatted_prompt):
                yield chunk 
        st.write_stream(streamResponse())

        if st.button("Generate Another Article"):
            st.rerun()
    else:
        st.warning("Please enter a topic to generate the article.")
