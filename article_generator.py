# I did this assignment late, so I included some of the latest topics discussed in class.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
parser = StrOutputParser()

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
        with st.spinner("Generating article..."):    
            chain = prompt | llm | parser
            article = chain.invoke({
                "selected_tone": tone,
                "article_topic": user_query
            })

            st.subheader("Generated Article:")
            st.markdown(article)

            if st.button("Generate Another Article"):
                st.rerun()
    else:
        st.warning("Please enter a topic to generate the article.")
