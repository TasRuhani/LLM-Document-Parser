from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel
import os
import json
import re
import torch
import requests
import tempfile
import io
import asyncio
from urllib.parse import urlparse

# Async-ready libraries
import redis.asyncio as aioredis
from openai import AsyncOpenAI

from sentence_transformers import SentenceTransformer

# Import modules from your project
from env_setup import get_openrouter_keys
from extractors import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_eml,
    extract_text_from_msg
)
from faiss_utils import build_faiss_index, semantic_search
from processing import chunk_text, embed_with_cache_async
from classifier import ClauseClassifier
from llm_utils import get_direct_answer_async

app = FastAPI()

# ---------------- Load environment variables and secrets globally ----------------
OPENROUTER_API_KEY, OPENROUTER_MODEL = get_openrouter_keys()

if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
    raise ValueError("OpenRouter API key or model not found in .env file.")

# Use AsyncOpenAI client
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Use async redis client
redis_client = aioredis.from_url("redis://localhost:6379", db=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load models globally for efficiency ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
classifier = ClauseClassifier(model_path="./model/legal-bert-finetuned", device=device)

# ---------------- Define the data model for the webhook payload ----------------
class ClaimRequest(BaseModel):
    documents: str
    questions: list[str]

# ---------------- Authentication ----------------
security = HTTPBearer()
EXPECTED_TOKEN = "98e1dde7d1f85ec2a9339f39b927d66f8766461fa06a087cea0c9c8daf5c43d4"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# ---------------- Main API Endpoint ----------------
@app.post("/api/v1/hackrx/run", dependencies=[Depends(verify_token)])
async def run_parser(claim_request: ClaimRequest):
    try:
        # --- 1. Download and Extract Text ---
        doc_url = claim_request.documents
        try:
            response = await asyncio.to_thread(requests.get, doc_url)
            response.raise_for_status()
            file_content = response.content
        except requests.exceptions.RequestException as req_e:
            return JSONResponse(status_code=500, content={"error": f"Failed to download document: {req_e}"})

        parsed_url = urlparse(doc_url)
        path = parsed_url.path
        
        if path.lower().endswith(".pdf"):
            all_text = await asyncio.to_thread(extract_text_from_pdf, io.BytesIO(file_content))
        elif path.lower().endswith(".docx"):
            all_text = await asyncio.to_thread(extract_text_from_docx, io.BytesIO(file_content))
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported document type."})

        if not all_text.strip():
            return JSONResponse(status_code=400, content={"error": "No text extracted from document."})

        # --- 2. Process Text into Embeddings and Index ---
        chunks = await asyncio.to_thread(chunk_text, all_text)
        embeddings = await embed_with_cache_async(chunks, embedder, redis_client)
        index = await asyncio.to_thread(build_faiss_index, embeddings)
        
        # --- 3. Generate Answers for All Questions in Parallel ---
        async def generate_answer(question: str):
            # Find relevant clauses using semantic search
            retrieved_clauses = await asyncio.to_thread(
                semantic_search, question, chunks, index, embedder, top_k=5
            )
            
            # Call the LLM function for direct Q&A
            answer = await get_direct_answer_async(
                client, OPENROUTER_MODEL, question, "\n".join(retrieved_clauses)
            )

            # PATCH: Added a 1-second delay to avoid free-tier rate limits.
            await asyncio.sleep(1)

            return answer

        # Create and run all question-answering tasks concurrently
        tasks = [generate_answer(q) for q in claim_request.questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        processed_answers = []
        for ans in answers:
            if isinstance(ans, Exception):
                processed_answers.append(f"Error generating answer: {str(ans)}")
            else:
                processed_answers.append(ans)
        
        return JSONResponse(content={"answers": processed_answers})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"An unexpected server error occurred: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=8000, reload=True)