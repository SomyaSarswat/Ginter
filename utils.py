import PyPDF2
import docx
from email import policy
from email.parser import BytesParser
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Global model cache
_model = None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPDF2.PdfReadError as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")

def extract_text_from_docx(docx_file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_file)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        raise ValueError(f"DOCX extraction failed: {str(e)}")

def extract_text_from_email(email_file) -> str:
    """Extract text from email file"""
    try:
        msg = BytesParser(policy=policy.default).parsebytes(email_file.read())
        if msg.is_multipart():
            for part in msg.iter_parts():
                if part.get_content_type() == "text/plain":
                    return part.get_content()
        elif msg.get_content_type() == "text/plain":
            return msg.get_content()
        return ""
    except Exception as e:
        raise ValueError(f"Email extraction failed: {str(e)}")

def validate_text_extraction(text: str) -> None:
    """Validate extracted text"""
    if not text.strip():
        raise ValueError("No text content found in document")
    if len(text.strip()) < 50:
        raise ValueError("Extracted text seems too short")

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 100
) -> List[str]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks"""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(texts).tolist()