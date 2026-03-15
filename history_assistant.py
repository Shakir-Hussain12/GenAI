from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import streamlit as st

load_dotenv()
retriever = WikipediaRetriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
"""
    You are a history assistant for the general public.

    Answer using ONLY the provided Wikipedia passages.

    User question: {user_question}

    Wikipedia passages:
    {retrieved_passages}
"""
)

rag_chain = (
    {
        "retrieved_passages": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "user_question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


st.title("Wikipedia History Assistant")
question = st.text_input("Ask a history question: ")

if st.button("Search for the answer"):
    if question:
        def streaming_response():
            for token in rag_chain.stream(question):
                yield token

        try:
            st.subheader(f"Information retrieved for: '{question}'")
            st.write_stream(streaming_response())
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
