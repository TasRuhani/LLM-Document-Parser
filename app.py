import streamlit as st
import json
import re

from document_ingestor import extract_text_from_pdf, extract_text_from_docx
from vector_store import chunk_text, embed_chunks, build_faiss_index, embedder
from query_parser import parse_query_with_llm
from semantic_search import semantic_search
from decision_engine import get_decision_llm
from clause_classifier import ClauseClassifier

# Page setup
st.set_page_config(page_title="LLM Clause Reasoner", layout="wide")
st.title("LLM-Powered Policy Clause Reasoner")

# Upload multiple files
uploaded_files = st.file_uploader("Upload policy documents (PDF, DOCX, or TXT email)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
query = st.text_area("Enter your query", height=100)
run = st.button("🔍 Process Query")

# Initialize clause classifier
classifier = ClauseClassifier(model_path="./model/legal-bert-finetuned", threshold=0.3)

def read_uploaded_file(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# Process pipeline
if uploaded_files and query and run:
    with st.spinner("Reading documents..."):
        combined_text = ""
        for file in uploaded_files:
            combined_text += read_uploaded_file(file) + "\n"

    chunks = chunk_text(combined_text)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)

    with st.spinner("Parsing query..."):
        structured_query = parse_query_with_llm(query)

    with st.spinner("Retrieving relevant clauses..."):
        retrieved = semantic_search(query, chunks, index, embedder)

    with st.spinner("Getting decision from LLM..."):
        decision_json = get_decision_llm(structured_query, "\n".join(retrieved))

        try:
            decision_data = json.loads(decision_json)
            justification = decision_data.get("justification", "No justification found.")
        except json.JSONDecodeError:
            justification_match = re.search(r'"justification"\s*:\s*"([^"]+)"', decision_json)
            justification = justification_match.group(1) if justification_match else "Could not parse justification."

        st.subheader("Justification")
        st.write(justification)

    # Display retrieved clauses + predicted clause types
    st.subheader("Retrieved Clauses and Predicted Clause Types")
    for i, clause in enumerate(retrieved):
        st.markdown(f"**Clause {i+1}:** {clause}")

        predicted_types = classifier.predict(clause)
        # if predicted_types:
        #     st.markdown(
        #         f"**Predicted Clause Types:** {', '.join([f'{label} ({score:.2f})' for label, score in predicted_types])}"
        #     )
        # else:
        #     st.markdown("**Predicted Clause Types:** _None with high enough confidence_")
