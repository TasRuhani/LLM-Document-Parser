from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import os
import json
import re
import redis
import torch
import multiprocessing as mp
from openai import OpenAI
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
from processing import chunk_text, embed_with_cache
from classifier import ClauseClassifier
from llm_utils import parse_query_with_llm, get_decision_llm

app = FastAPI()

# ---------------- Load environment variables and secrets globally ----------------
OPENROUTER_API_KEY, OPENROUTER_MODEL = get_openrouter_keys()

if not OPENROUTER_API_KEY or not OPENROUTER_MODEL:
    raise ValueError("OpenRouter API key or model not found. Please set them in your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

redis_client = redis.Redis(host="localhost", port=6379, db=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load models globally for efficiency ----------------
# This ensures models are only loaded once when the server starts
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
classifier = ClauseClassifier(model_path="./model/legal-bert-finetuned", device=device)


# ---------------- Define the data model for the webhook payload ----------------
# This is a good practice for validating incoming data
class ClaimRequest(BaseModel):
    query: str
    documents: list  # Assuming documents are file-like objects or paths


@app.post("/api/v1/hackrx/run")
async def run_parser(request: Request):
    try:
        data = await request.json()
        claim_request = ClaimRequest(**data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid request format: {e}"})

    try:
        # --- 1. Process documents to get chunks, embeddings, and classified data ---
        all_text = ""
        for doc in claim_request.documents:
            all_text += doc + "\n"

        chunks = chunk_text(all_text)
        embeddings = embed_with_cache(chunks, embedder, redis_client)
        index = build_faiss_index(embeddings)

        classified_predictions = classifier.predict(chunks)
        classified_chunks = list(zip(chunks, classified_predictions))

        # --- 2. Perform semantic search and LLM calls ---
        retrieved = semantic_search(claim_request.query, chunks, index, embedder)

        relevant_clauses = []
        for clause, preds in classified_chunks:
            if any(label.lower() in claim_request.query.lower() for label, _ in preds):
                relevant_clauses.append(clause)
        if not relevant_clauses:
            relevant_clauses = retrieved

        structured = parse_query_with_llm(client, OPENROUTER_MODEL, claim_request.query)
        decision_json = get_decision_llm(client, OPENROUTER_MODEL, structured, "\n".join(relevant_clauses))

        try:
            data = json.loads(decision_json)
        except Exception:
            match = re.search(r'"justification"\s*:\s*"([^"]+)"', decision_json)
            data = {"decision": "N/A", "justification": match.group(1) if match else decision_json}

        # --- 3. Return the final decision ---
        return JSONResponse(content={
            # "decision": data.get("decision", "?"),
            "justification": data.get("justification", "No justification."),
        })

    except Exception as e:
        # Return a 500 Internal Server Error if something goes wrong
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)