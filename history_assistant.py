# I haven't used the ChatBot style agent to maintain history, rather I have implemented manual history tracking across single independent requests.

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

summary_prompt = ChatPromptTemplate.from_template(
"""
    Summarize the following passage in a concise manner keeping track of the user and assistant's previous discussions:
    {chat_history}
"""
)

prompt = ChatPromptTemplate.from_template(
"""
    You are a history assistant for the general public.

    Here is a history of our previous discussions:
    {previous_discussions}

    Answer using ONLY the provided Wikipedia passages and the information from our previous discussions. If you don't know the answer, say you don't know. Do not make up an answer.
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

def concat_discussions(discussions):
    return "\n\n".join(discussions)

if st.button("Search for the answer"):
    if question:
        response_content = []
        def streaming_response():
            inputs = {
                "user_question": question,
                "previous_discussions": concat_discussions(st.session_state.previous_discussions)
            }

            for token in rag_chain.stream(inputs):
                response_content.append(token)
                yield token

        try:
            st.subheader(f"Information retrieved for: '{question}'")
            st.write_stream(streaming_response())
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # appending new discussion to the history of discussions
        st.session_state.previous_discussions.append(f"User: {question}")
        st.session_state.previous_discussions.append(f"AI: {''.join(response_content)}")

        with st.expander("Previous Discussions"):
            for discussion in st.session_state.previous_discussions:
                st.write(discussion)

        
        # summarizing the history of discussions to maintain a concise context for the LLM
        summary_chain = (   
            {
                "chat_history": RunnablePassthrough()
            }
            | summary_prompt
            | llm
            | StrOutputParser()
        )

        # setting a threshold of max 10 messages to store in history 
        if len(st.session_state.previous_discussions) > 10:
            prev_summary = summary_chain.invoke(concat_discussions(st.session_state.previous_discussions))
            st.session_state.previous_discussions = [f"Summary of previous discussions: {prev_summary}"]
        
        if st.button("Clear History"):
            st.session_state.previous_discussions = []
        
        if st.button("Ask a new question"):
            st.rerun()
    else:
        st.warning("Please enter a question.")

