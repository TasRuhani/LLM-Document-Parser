import re
import hashlib
import json
import numpy as np

# Define a TTL for cache entries (e.g., 24 hours in seconds)
CACHE_TTL_SECONDS = 24 * 60 * 60

def chunk_text(text, max_chunk_len=500):
    """Chunks text into smaller paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) <= max_chunk_len:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

async def embed_with_cache_async(chunks, embedder, redis_client):
    """
    Asynchronously creates embeddings for text chunks, using Redis for caching.
    """
    vectors = []
    to_embed_indices = []
    chunks_to_embed = []
    
    chunk_hashes = [hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks]
    
    # Fetch cached vectors from Redis
    cached_vectors = await redis_client.mget(chunk_hashes)
    
    # Process cache results
    all_vectors = [None] * len(chunks)
    for i, cached in enumerate(cached_vectors):
        if cached:
            all_vectors[i] = json.loads(cached)
        else:
            to_embed_indices.append(i)
            chunks_to_embed.append(chunks[i])

    # Embed chunks that were not found in the cache
    if chunks_to_embed:
        new_embeddings = embedder.encode(chunks_to_embed, convert_to_numpy=True).tolist()
        
        # Use a Redis pipeline to set all new cache entries at once
        pipe = redis_client.pipeline()
        for i, vec in enumerate(new_embeddings):
            original_index = to_embed_indices[i]
            all_vectors[original_index] = vec
            chunk_hash = chunk_hashes[original_index]
            pipe.set(chunk_hash, json.dumps(vec), ex=CACHE_TTL_SECONDS)
        
        await pipe.execute()

    return np.array(all_vectors)