import numpy as np

def semantic_search(query, chunks, index, embedder, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_vec), top_k)
    return [chunks[i] for i in I[0]]
