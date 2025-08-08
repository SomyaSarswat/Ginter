import streamlit as st
from utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_email,
    chunk_text,
    get_embeddings,
    validate_text_extraction
)
from build_index import SemanticSearchIndex
from llm_reasoning import parse_query, get_query_embedding, reason_over_clauses
from config import GROQ_API_KEY  # Import from config file

# Verify API key is available at startup
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please configure it in config.py")
    st.stop()

st.set_page_config(page_title="Insurance Policy Query System", layout="wide")
st.title("Insurance Policy Query System using LLMs")

def main():
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, EML)", 
        type=["pdf", "docx", "eml"]
    )

    if not uploaded_file:
        return

    try:
        process_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

def process_uploaded_file(uploaded_file):
    """Handle file processing pipeline"""
    st.info("Extracting text from document...")
    
    file_type = uploaded_file.type
    text = ""
    
    try:
        if file_type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "message/rfc822":
            text = extract_text_from_email(uploaded_file)
        else:
            raise ValueError("Unsupported file type")
            
        validate_text_extraction(text)
        
        st.success("Text extraction complete. Building semantic index...")
        process_text_content(text)
        
    except Exception as e:
        st.error(f"Failed to process document: {str(e)}")
        raise

def process_text_content(text):
    """Process extracted text through the pipeline"""
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    index = SemanticSearchIndex()
    index.build(embeddings, chunks)
    st.success(f"Indexed {len(chunks)} chunks of text.")

    query = st.text_input(
        "Enter your query (e.g. '46M, knee surgery, Pune, 3-month policy')",
        placeholder="Describe the insurance claim scenario"
    )

    if not query:
        return

    try:
        st.info("Parsing query and retrieving relevant clauses...")
        query_struct = parse_query(query)
        query_embedding = get_query_embedding(query)
        retrieved_clauses = index.search(query_embedding, top_k=5)

        display_retrieved_clauses(retrieved_clauses)
        
        st.info("Reasoning over retrieved clauses to produce decision...")
        decision = reason_over_clauses(query_struct, retrieved_clauses)
        
        display_decision(decision)
        
    except Exception as e:
        st.error(f"Query processing failed: {str(e)}")
        raise

def display_retrieved_clauses(clauses):
    """Display retrieved clauses in an organized way"""
    st.markdown("### Retrieved Policy Clauses")
    with st.expander("View Relevant Policy Clauses"):
        for i, clause in enumerate(clauses, 1):
            st.markdown(f"**Clause {i}:**")
            st.write(clause)
            st.divider()

def display_decision(decision):
    """Display the final decision"""
    st.markdown("### Claim Decision")
    
    if decision.get("Decision", "").lower() == "approved":
        st.success("✅ Claim Approved")
    else:
        st.error("❌ Claim Rejected")
    
    st.json(decision)

if __name__ == "__main__":
    main()