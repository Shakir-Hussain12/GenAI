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

    Here is a history of our previous discussions:
    {previous_discussions}

    Answer using ONLY the provided Wikipedia passages.
    User question: {user_question}

    Wikipedia passages:
    {retrieved_passages}
"""
)

rag_chain = (
    {
        "retrieved_passages": lambda docs: "\n\n".join(doc.page_content for doc in retriever.invoke(docs["user_question"])),
        "user_question": lambda x: x["user_question"],
        "previous_discussions": lambda x: x["previous_discussions"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

if "previous_discussions" not in st.session_state:
    st.session_state.previous_discussions = []

st.title("Wikipedia History Assistant")
question = st.text_input("Ask a history question: ")

if st.button("Search for the answer"):
    if question:
        discussion_history = "\n\n".join(st.session_state.previous_discussions)
        response_content = []
        def streaming_response():
            inputs = {
                "user_question": question,
                "previous_discussions": discussion_history
            }

            for token in rag_chain.stream(inputs):
                response_content.append(token)
                yield token

        try:
            st.subheader(f"Information retrieved for: '{question}'")
            st.write_stream(streaming_response())
        except Exception as e:
            st.error(f"An error occurred: {e}")

        st.session_state.previous_discussions.append(f"User: {question}")
        st.session_state.previous_discussions.append(f"AI: {''.join(response_content)}")
    else:
        st.warning("Please enter a question.")
