import streamlit as st
import time
import json
import re
import os

# Import modules
from env_setup import get_openrouter_keys
from extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_eml,
    extract_text_from_msg
)
from faiss_utils import build_faiss_index, semantic_search
from processing import chunk_text, embed_with_cache
from classifier import ClauseClassifier
from llm_utils import parse_query_with_llm, get_decision_llm

import redis
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# --- Load environment variables and secrets ---
OPENROUTER_API_KEY, OPENROUTER_MODEL = get_openrouter_keys()

if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
    st.error("OpenRouter API key or model not found. Please set them in your .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

redis_client = redis.Redis(host="localhost", port=6379, db=0)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Clause Reasoner", layout="wide")
st.title("LLM-Powered Clause Reasoner")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "classified_chunks" not in st.session_state:
    st.session_state.classified_chunks = []

classifier = ClauseClassifier(model_path="./model/legal-bert-finetuned", device=device)

with st.expander("Upload Documents"):
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, EML, or MSG",
        type=["pdf", "docx", "eml", "msg"],
        accept_multiple_files=True,
    )
    upload_button = st.button("Process Documents")

    if upload_button and uploaded_files:
        start_time_docs = time.time()
        with st.spinner("Processing documents..."):
            all_text = ""
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    all_text += extract_text_from_pdf(file) + "\n"
                elif file.name.endswith(".docx"):
                    all_text += extract_text_from_docx(file) + "\n"
                elif file.name.endswith(".eml"):
                    all_text += extract_text_from_eml(file) + "\n"
                elif file.name.endswith(".msg"):
                    path = f"/tmp/{file.name}"
                    with open(path, "wb") as f:
                        f.write(file.read())
                    all_text += extract_text_from_msg(path) + "\n"
                    os.remove(path)

            chunks = chunk_text(all_text)
            with st.spinner("Generating embeddings..."):
                embeddings = embed_with_cache(chunks, embedder, redis_client)
            index = build_faiss_index(embeddings)
            with st.spinner("Classifying chunks..."):
                classified_predictions = classifier.predict(chunks)
                classified = list(zip(chunks, classified_predictions))

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.classified_chunks = classified

        end_time_docs = time.time()
        st.success(f"Documents processed and classified in {end_time_docs - start_time_docs:.2f} seconds.")

query = st.text_area("Enter your query")
run = st.button("Run Query")

if run and query and st.session_state.index:
    start_time_query = time.time()
    with st.spinner("Parsing query..."):
        structured = parse_query_with_llm(client, OPENROUTER_MODEL, query)

    with st.spinner("Searching relevant clauses..."):
        retrieved = semantic_search(query, st.session_state.chunks, st.session_state.index, embedder)

    with st.spinner("Filtering classified clauses for query..."):
        relevant_clauses = []
        for clause, preds in st.session_state.classified_chunks:
            if any(label.lower() in query.lower() for label, _ in preds):
                relevant_clauses.append(clause)
        if not relevant_clauses:
            relevant_clauses = retrieved

    with st.spinner("Getting decision from LLM..."):
        decision_json = get_decision_llm(client, OPENROUTER_MODEL, structured, "\n".join(relevant_clauses))
    try:
        data = json.loads(decision_json)
    except Exception:
        match = re.search(r'"justification"\s*:\s*"([^"]+)"', decision_json)
        data = {"decision": "?", "amount": "?", "justification": match.group(1) if match else decision_json}

    end_time_query = time.time()

    st.subheader("Decision")
    st.write(f"**Decision**: {data.get('decision', '?')}")
    st.write(f"**Amount**: {data.get('amount', '?')}")
    st.subheader("Justification")
    st.write(data.get("justification", "No justification."))
    st.subheader("Retrieved Clauses")
    for i, clause in enumerate(retrieved):
        st.markdown(f"**Clause {i+1}:** {clause}")

    st.info(f"Query processed in {end_time_query - start_time_query:.2f} seconds.")
