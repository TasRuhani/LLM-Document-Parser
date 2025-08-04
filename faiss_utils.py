import numpy as np
import faiss

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def semantic_search(query, chunks, index, embedder, top_k=5):
    qvec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(qvec), top_k)
    return [chunks[i] for i in I[0]]
