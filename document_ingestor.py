import fitz  # PyMuPDF
from docx import Document

def extract_text_from_pdf(uploaded_file):
    # Use stream + filetype for Streamlit in-memory uploads
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text() for page in doc])
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])
