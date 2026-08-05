"""
pRAGma - minimal RAG backend with query translation.

Pipeline:  PDF -> text -> chunks -> embeddings -> (retrieval) -> LLM -> answer

  1. INGEST     POST /ingest        upload PDF, split into chunks, embed chunks
  2. RETRIEVE   (inside /query)     embed question, cosine-rank chunks
  3. GENERATE   (inside /query)     stuff top k chunks into a prompt, LLM call

Query Translation Methods:
  1. Multi Query - generate 3 different versions of the question
  2. RAG Fusion  - multiple queries, ranked by cosine similarity
  3. Decomposition - break into incremental sub-questions (least to most)
  4. Step Back   - step back to a broader question, then answer
  5. HyDE        - generate hypothetical document, embed that
"""

import asyncio
import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastembed import TextEmbedding
from pydantic import BaseModel
from pypdf import PdfReader

load_dotenv()

# CONFIG
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE   = 1200
CHUNK_OVERLAP = 150
TOP_K        = 4

# "VECTOR STORE"
CHUNKS = []                     # list[str]  - the text chunks
EMBEDS = None                   # np.ndarray - one embedding row per chunk
SOURCE = None                   # str | None - filename of the ingested PDF
JOBS = {}                       # job_id -> {status, message, done, total, source}

# Query Translation Methods
TRANSLATION_METHODS = {
    "multi_query": "Multi Query",
    "rag_fusion": "RAG Fusion",
    "decomposition": "Decomposition",
    "step_back": "Step Back Prompting",
    "hyde": "HyDE",
}

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


# Query Translation Helpers
def translate_multi_query(question: str) -> list[str]:
    """Generate 3 different versions of the question."""
    prompt = f"""You are a helpful assistant that generates alternative versions of a question.
Generate 3 different but semantically equivalent versions of the following question.
Each version should be on a new line, numbered 1-3.

Question: {question}

Versions:
1."""
    try:
        answer, _, _ = call_llm(prompt)
        versions = [line.strip() for line in answer.split('\n') if line.strip() and line[0].isdigit()]
        # Clean up numbering
        versions = [v.split('.', 1)[1].strip() if '.' in v else v for v in versions]
        # Ensure we have the original question too
        return [question] + versions[:3]
    except Exception:
        return [question]


def translate_rag_fusion(question: str) -> list[str]:
    """Generate multiple queries for RAG Fusion - returns original + variations."""
    # For RAG Fusion, we generate multiple queries and then rank results by cosine similarity
    return translate_multi_query(question)


def translate_decomposition(question: str) -> list[str]:
    """Break question into incremental sub-questions (least to most)."""
    prompt = f"""You are a helpful assistant that breaks down complex questions into simpler sub-questions.
Break the following question into 2-3 simpler sub-questions that build on each other (least to most complex).
Each sub-question should be on a new line, numbered 1-3.

Question: {question}

Sub-questions:
1."""
    try:
        answer, _, _ = call_llm(prompt)
        sub_questions = [line.strip() for line in answer.split('\n') if line.strip() and line[0].isdigit()]
        sub_questions = [sq.split('.', 1)[1].strip() if '.' in sq else sq for sq in sub_questions]
        return sub_questions[:3]
    except Exception:
        return [question]


def translate_step_back(question: str) -> list[str]:
    """Step back to a broader question, then answer."""
    prompt = f"""You are a helpful assistant that generates a broader, more general version of a question.
Generate ONE broader "step-back" question that encompasses the original question.
The step-back question should be more general and help retrieve relevant context.

Original Question: {question}

Step-back Question:"""
    try:
        answer, _, _ = call_llm(prompt)
        step_back = answer.strip().split('\n')[0].strip()
        return [step_back, question]
    except Exception:
        return [question]


def translate_hyde(question: str) -> list[str]:
    """Generate a hypothetical document (HyDE) that would answer the question."""
    prompt = f"""You are a helpful assistant that generates a hypothetical document passage that would answer a question.
Write a concise, realistic document passage (2-3 sentences) that would contain the answer to the following question.
Write as if you are the source document.

Question: {question}

Hypothetical Document:"""
    try:
        answer, _, _ = call_llm(prompt)
        hyde_doc = answer.strip().split('\n')[0].strip()
        return [hyde_doc]
    except Exception:
        return [question]


def translate_query(question: str, method: str) -> list[str]:
    """Route to the appropriate translation method."""
    if method == "multi_query":
        return translate_multi_query(question)
    elif method == "rag_fusion":
        return translate_rag_fusion(question)
    elif method == "decomposition":
        return translate_decomposition(question)
    elif method == "step_back":
        return translate_step_back(question)
    elif method == "hyde":
        return translate_hyde(question)
    else:
        return [question]


app = FastAPI(title="RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for a local dev demo
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    method: str = "none"  # translation method: none, multi_query, rag_fusion, decomposition, step_back, hyde


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
    """Retrieve the most relevant chunks, generate an answer, log latency."""
    if not CHUNKS:
        raise HTTPException(400, "Ingest a PDF first.")

    t0 = time.time()
    translation_steps = []

    # Step 1: Translate the query based on method
    if q.method != "none" and q.method in TRANSLATION_METHODS:
        translated_queries = translate_query(q.question, q.method)
        translation_steps.append({
            "method": TRANSLATION_METHODS[q.method],
            "queries": translated_queries
        })
    else:
        translated_queries = [q.question]

    # Step 2: Retrieve for each translated query
    all_hits = {}  # chunk_index -> max_score
    retrieval_details = []

    for tq in translated_queries:
        q_vec = embed([tq])[0]
        hits = cosine_top_k(q_vec, TOP_K)
        
        # Store retrieval details for UI
        retrieval_details.append({
            "query": tq,
            "hits": [{"index": i, "score": round(s, 3)} for i, s in hits]
        })

        # Aggregate scores (for RAG Fusion, use max; for others, could use different strategies)
        for i, s in hits:
            if i not in all_hits or s > all_hits[i]:
                all_hits[i] = s

    # Sort by score descending and take top K
    sorted_hits = sorted(all_hits.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    
    context = "\n\n".join(f"[chunk {i}] {CHUNKS[i]}" for i, _ in sorted_hits)

    # Step 3: Generate answer
    # For decomposition, we might want to answer sub-questions incrementally
    if q.method == "decomposition" and len(translated_queries) > 1:
        # Answer each sub-question and combine
        sub_answers = []
        for sub_q in translated_queries:
            sub_prompt = f"""Answer the following question using only the provided context. Be concise.

Context:
{context}

Question: {sub_q}

Answer:"""
            sub_answer, _, _ = call_llm(sub_prompt)
            sub_answers.append(f"Q: {sub_q}\nA: {sub_answer}")
        
        # Final synthesis
        synthesis_prompt = f"""Synthesize a final answer from these sub-question answers:

{chr(10).join(sub_answers)}

Original Question: {q.question}

Final Answer:"""
        prompt = synthesis_prompt
    else:
        prompt = f"""You are an expert retrieval-augmented question answering assistant. Answer using only the provided context, treating it as the sole source of truth. Do not use prior knowledge, assumptions, or external information. First identify relevant evidence, then produce a concise, accurate response fully supported by the context. If the context is insufficient, ambiguous, or lacks the requested information, reply exactly: 'The provided context does not contain sufficient information to answer this question.' Never guess, invent facts, infer unsupported conclusions, or explain your reasoning. Output only the final answer.

Context:
{context}

Question: {q.question}

Answer:"""

    t_llm = time.time()
    answer, in_tok, out_tok = call_llm(prompt)
    llm_ms = int((time.time() - t_llm) * 1000)

    return {
        "answer": answer,
        "chunks": [
            {
                "index": i,
                "score": round(all_hits[i], 3),
                "snippet": CHUNKS[i][:300] + ("..." if len(CHUNKS[i]) > 300 else ""),
            }
            for i, _ in sorted_hits
        ],
        "translation_steps": translation_steps,
        "retrieval_details": retrieval_details,
        "latency": {
            "retrieval_ms": int((t_llm - t0) * 1000),
            "llm_ms": llm_ms,
            "total_ms": int((time.time() - t0) * 1000),
        },
        "tokens": {"input": in_tok, "output": out_tok},
    }


# Serve the production React build from the same FastAPI server.  Keep this
# mount last so FastAPI's API endpoints and built-in /docs route take priority.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
