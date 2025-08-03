import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ✅ Load a lightweight, fast embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ✅ Improved paragraph-aware chunking
def chunk_text(text, max_chunk_len=500):
    # Split text by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= max_chunk_len:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_numpy=True)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # CPU index
    index.add(embeddings)
    return index
