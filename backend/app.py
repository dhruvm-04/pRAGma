"""
pRAGma - minimal RAG backend.

Pipeline:  PDF -> text -> chunks -> embeddings -> (retrieval) -> LLM -> answer

  1. INGEST     POST /ingest        upload PDF, split into chunks, embed chunks
  2. RETRIEVE   (inside /query)     embed question, cosine-rank chunks
  3. GENERATE   (inside /query)     stuff top k chunks into a prompt, LLM call
"""

import asyncio
import io
from multiprocessing import context
import os
import time
import uuid

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

# CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"    # smal, fast, free, local
CHUNK_SIZE   = 1200
CHUNK_OVERLAP = 150                        # overlap to preserve context
TOP_K        = 4

# price per 1M tokens (Llama 3.3-70B)
PRICE_IN = 0.59
PRICE_OUT = 0.79

# "VECTOR STORE"
CHUNKS = []                     # list[str]  - the text chunks
EMBEDS = None                   # np.ndarray - one embedding row per chunk
SOURCE = None                   # str | None - filename of the ingested PDF
JOBS = {}                       # job_id -> {status, message, done, total, source}

# EMBEDDING MODEL
# First execution loads model, after that calls are quick
_embedder = None
_model_ready = False

def get_embedder():
    global _embedder, _model_ready
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
        _model_ready = True
    return _embedder


app = FastAPI(title="RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for a local dev demo
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str


# Chunking - helper
def chunk_text(text: str) -> list[str]:
    """Cut normalized text into fixed-size chunks with a small overlap."""
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Embedding - helper
def embed(texts: list[str], on_batch=None) -> np.ndarray:
    """Embed a list of strings in batches; rows are L2-normalized (unit length).

    on_batch(done, total) is called after every batch so a long ingest can
    report live progress to the UI. Batching keeps large docs fast instead of
    embedding one chunk at a time.
    """
    batch_size = 16
    vecs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        vecs.append(np.array(list(get_embedder().embed(batch)), dtype=np.float32))
        if on_batch:
            on_batch(i + len(batch), total)
    out = np.concatenate(vecs)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


# Retrieval - helper
def cosine_top_k(query_vec: np.ndarray, k: int):
    """Return the (index, score) of the k most similar chunks.

    Because every vector is unit length, dot product == cosine similarity.
    """
    scores = EMBEDS @ query_vec
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


# LLM call - helper
def call_llm(prompt: str):
    """Send the prompt to a free Groq model and return (answer, in_tok, out_tok)."""
    if not GROQ_API_KEY:
        return ("API key not set in .env.", 0, 0)
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=60,
    )
    try:
        data = resp.json()
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
    except (KeyError, ValueError):
        return (f"LLM error ({resp.status_code}): {resp.text[:300]}", 0, 0)


# ROUTING
@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS), "source": SOURCE}


def run_ingest(job_id: str, data: bytes):
    """Background worker: parse the PDF, chunk it, embed it, updating the job as it goes."""
    global CHUNKS, EMBEDS, SOURCE
    job = JOBS[job_id]
    try:
        job.update(status="running", message="parsing pdf…", done=0, total=0)

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No text found in the PDF - is it a scanned image?")

        if not _model_ready:
            job.update(message="loading embed model (first run, ~40s)…")
        job.update(total=len(chunks), message="embedding chunks…")

        EMBEDS = embed(chunks, on_batch=lambda done, total: job.update(done=done, total=total))
        CHUNKS, SOURCE = chunks, job.get("source")

        job.update(status="done", message="done", done=len(chunks))
    except Exception as e:
        job.update(status="error", message=str(e))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "message": "queued", "done": 0, "total": 0, "source": file.filename}
    data = await file.read()
    asyncio.get_running_loop().run_in_executor(None, run_ingest, job_id, data)
    return {"job_id": job_id}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str):
    # ingestion progress status
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: job.get(k) for k in ("status", "message", "done", "total", "source")}


@app.post("/query")
def query(q: Query):
    """Retrieve the most relevant chunks, generate an answer, log latency/cost."""
    if not CHUNKS:
        raise HTTPException(400, "Ingest a PDF first.")

    t0 = time.time()

    # question embedding
    q_vec = embed([q.question])[0]

    # top-k chunks retrieval
    hits = cosine_top_k(q_vec, TOP_K)
    context = "\n\n".join(f"[chunk {i}] {CHUNKS[i]}" for i, _ in hits)

    # grounding
    prompt = f"You are an expert retrieval-augmented question answering assistant. Answer using only the provided context, treating it as the sole source of truth. Do not use prior knowledge, assumptions, or external information. First identify relevant evidence, then produce a concise, accurate response fully supported by the context. If the context is insufficient, ambiguous, or lacks the requested information, reply exactly: 'The provided context does not contain sufficient information to answer this question.' Never guess, invent facts, infer unsupported conclusions, or explain your reasoning. Output only the final answer.\n\nContext:\n{context}\n\nQuestion: {q.question}\n\nAnswer:"
    t_llm = time.time()
    answer, in_tok, out_tok = call_llm(prompt)
    llm_ms = int((time.time() - t_llm) * 1000)

    return {
        "answer": answer,
        "chunks": [
            {
                "index": i,
                "score": round(s, 3),
                "snippet": CHUNKS[i][:300] + ("..." if len(CHUNKS[i]) > 300 else ""),
            }
            for i, s in hits
        ],
        "latency": {
            "retrieval_ms": int((t_llm - t0) * 1000),
            "llm_ms": llm_ms,
            "total_ms": int((time.time() - t0) * 1000),
        },
        "tokens": {"input": in_tok, "output": out_tok},
        "cost_usd": round(in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT, 6),
    }
