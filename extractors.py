import io
import email
import extract_msg

def extract_text_from_pdf(file):
    import fitz
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    from docx import Document
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_eml(file):
    msg = email.message_from_bytes(file.read())
    return "\n".join(
        part.get_payload(decode=True).decode(errors="ignore")
        for part in msg.walk()
        if part.get_content_type() == "text/plain"
    )

def extract_text_from_msg(filepath):
    msg_file = extract_msg.Message(filepath)
    return msg_file.body or ""
