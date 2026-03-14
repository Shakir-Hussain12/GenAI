from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel  
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st
import PyPDF2

load_dotenv() 
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


prompt = PromptTemplate.from_template(
            """
                You are an expert drug information analyzer. Carefully analyze and extract requested details from the following drug information: {drug_info} 
            """ 
        )

 
summary_prompt = PromptTemplate.from_template(
    """
        Create a comprehensive summary of the provided document containing drug information: {drug_info}
    """
)

class DrugReport(BaseModel):
    drug_name: str = Field(..., description="Name of the drug") 
    active_ingredients: str = Field(..., description="Active ingredients in the drug") 
    indications: str = Field(..., description="Indications for use") 
    dosage: str = Field(..., description="Recommended dosage") 
    side_effects: str = Field(..., description="Possible side effects") 
    contraindications: str = Field(..., description="Contraindications for use")
    summary: str = Field(..., description="A comprehensive summary of the drug information")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text_content = ""
    text_content += "\n".join(page.extract_text() for page in pdf_reader.pages) 
    return text_content    
      
st.title("Drug Analyzer")
uploaded_file = st.file_uploader("Upload a drug information file", type=["pdf"])

if st.button("Analyze"):
    if uploaded_file is not None:
        drug_info_text = extract_text_from_pdf(uploaded_file) 
        structured_llm = llm.with_structured_output(DrugReport)

        # using chains to make sure input is sterilized properly
        structured_chain = prompt | structured_llm
        # additional chain to generate summary of the given document
        summary_chain = summary_prompt | llm
        chain = RunnableParallel({
            "drug_report": structured_chain,
            "summary": summary_chain
        })    
        with st.spinner("Analyzing drug information..."): 
            try:
                response = chain.invoke({"drug_info": drug_info_text}) 
                st.success("Analysis complete!") 
                with st.expander("Structured Drug Analysis Summary"): 
                    st.write(f"**Drug Name:** {response['drug_report'].drug_name}")
                    st.write(f"**Active Ingredients:** {response['drug_report'].active_ingredients}")
                    st.write(f"**Indications:** {response['drug_report'].indications}")
                    st.write(f"**Dosage:** {response['drug_report'].dosage}")
                    st.write(f"**Side Effects:** {response['drug_report'].side_effects}")
                    st.write(f"**Contraindications:** {response['drug_report'].contraindications}")
                with st.expander("Brief Drug Information Summary"):
                    st.write(response['drug_report'].summary)
                with st.expander("Document Summary"):
                    st.markdown(response['summary'].content)     
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")          

    else:
        st.warning("Please upload a drug information file to analyze.")