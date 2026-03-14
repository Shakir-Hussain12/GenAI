from langchain_core.prompts import PromptTemplate 
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
                You are a drug information analyzer. Analyze the following drug information and generate short summary: {drug_info}
                Provide the analysis in the following format:
                Drug Name: <drug_name>
                Active Ingredients: <active_ingredients>
                Indications: <indications>
                Dosage: <dosage>
                Side Effects: <side_effects>
                Contraindications: <contraindications>
            """
        )

class DrugReport(BaseModel):
    drug_name: str = Field(..., description="Name of the drug"),
    active_ingredients: str = Field(..., description="Active ingredients in the drug"),
    indications: str = Field(..., description="Indications for use"),
    dosage: str = Field(..., description="Recommended dosage"),
    side_effects: str = Field(..., description="Possible side effects"),
    contraindications: str = Field(..., description="Contraindications for use"),

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
        with st.spinner("Analyzing drug information..."):
            response = structured_llm.invoke(input={"drug_info": drug_info_text}) 
    else:
        st.warning("Please upload a drug information file to analyze.")