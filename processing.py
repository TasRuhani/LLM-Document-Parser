import re
import hashlib
import json
import numpy as np

def chunk_text(text, max_chunk_len=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if len(current) + len(para) <= max_chunk_len:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

def embed_with_cache(chunks, embedder, redis_client):
    vectors = []
    to_embed = []
    chunk_hashes = [hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks]
    cached_vectors = redis_client.mget(chunk_hashes)
    for i, cached in enumerate(cached_vectors):
        if cached:
            vectors.append(json.loads(cached))
        else:
            to_embed.append(chunks[i])
    if to_embed:
        new_embeddings = embedder.encode(to_embed, convert_to_numpy=True).tolist()
        cache_dict = {
            hashlib.sha256(chunk.encode()).hexdigest(): json.dumps(vec)
            for chunk, vec in zip(to_embed, new_embeddings)
        }
        if cache_dict:
            redis_client.mset(cache_dict)
        vectors.extend(new_embeddings)
    return np.array(vectors)
