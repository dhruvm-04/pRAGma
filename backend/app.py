import asyncio
import io
import os
import time
import uuid
from pathlib import Path

import chromadb
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 4

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

app = FastAPI(title="RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str
    method: str = "none"


chunks = []
source = None
jobs = {}

embedder = None
collection = None


def get_embedder():
    global embedder
    if embedder is None:
        embedder = TextEmbedding(model_name=EMBED_MODEL)
    return embedder


def get_collection():
    global collection
    if collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(name="rag_docs")
    return collection


def chunk_text(text):
    text = " ".join(text.split())
    result = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP
    while start < len(text):
        result.append(text[start : start + CHUNK_SIZE])
        start += step
    return result


def embed(text_list, on_batch=None):
    batch_size = 16
    vectors = []
    total = len(text_list)

    for start in range(0, total, batch_size):
        batch = text_list[start : start + batch_size]
        vectors.append(np.array(list(get_embedder().embed(batch)), dtype=np.float32))
        if on_batch:
            on_batch(min(start + len(batch), total), total)

    vectors = np.concatenate(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return vectors


def call_llm(prompt):
    if not GROQ_API_KEY:
        return "API key not set in .env.", 0, 0

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
        return f"LLM error ({resp.status_code}): {resp.text[:300]}", 0, 0


def translate_query(question, method):
    prompts = {
        "multi_query": (
            "Generate 3 semantically similar versions of this question, one per line:\n\n"
            f"{question}"
        ),
        "decomposition": (
            "Break this question into 2 or 3 simpler sub-questions, one per line:\n\n"
            f"{question}"
        ),
        "step_back": (
            "Write one broader step-back question for this:\n\n"
            f"{question}"
        ),
        "hyde": (
            "Write a short hypothetical document passage that would answer this question:\n\n"
            f"{question}"
        ),
    }

    if method == "rag_fusion":
        method = "multi_query"

    prompt = prompts.get(method)
    if not prompt:
        return [question]

    answer, _, _ = call_llm(prompt)
    lines = [line.strip() for line in answer.splitlines() if line.strip()]

    cleaned = []
    for line in lines:
        if "." in line[:3]:
            line = line.split(".", 1)[1].strip()
        cleaned.append(line)

    if method == "step_back" and cleaned:
        return [cleaned[0], question]
    if method == "hyde" and cleaned:
        return [cleaned[0]]
    if method == "decomposition" and cleaned:
        return cleaned[:3]
    if cleaned:
        return [question] + cleaned[:3]
    return [question]


def run_ingest(job_id, data):
    global chunks, source
    job = jobs[job_id]

    try:
        job.update(status="running", message="parsing pdf...", done=0, total=0)

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        new_chunks = chunk_text(text)

        if not new_chunks:
            raise ValueError("No text found in the PDF.")

        job.update(status="running", message="embedding chunks...", total=len(new_chunks))

        vectors = embed(new_chunks, on_batch=lambda done, total: job.update(done=done, total=total))
        coll = get_collection()
        coll.delete(where={"source": job.get("source")})
        coll.add(
            ids=[f"{job_id}-{i}" for i in range(len(new_chunks))],
            documents=new_chunks,
            embeddings=vectors.tolist(),
            metadatas=[{"chunk_index": i, "source": job.get("source")} for i in range(len(new_chunks))],
        )

        chunks = new_chunks
        source = job.get("source")
        job.update(status="done", message="done", done=len(new_chunks))
    except Exception as exc:
        job.update(status="error", message=str(exc))


@app.get("/health")
def health():
    try:
        count = get_collection().count()
    except Exception:
        count = len(chunks)
    return {"status": "ok", "chunks": count, "source": source}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        "status": "queued",
        "message": "queued",
        "done": 0,
        "total": 0,
        "source": file.filename,
    }
    data = await file.read()
    asyncio.get_running_loop().run_in_executor(None, run_ingest, job_id, data)
    return {"job_id": job_id}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: job.get(k) for k in ("status", "message", "done", "total", "source")}


@app.post("/query")
def query(q: Query):
    if not get_collection().count():
        raise HTTPException(400, "Ingest a PDF first.")

    start_time = time.time()
    translation_steps = []

    if q.method != "none":
        translated_queries = translate_query(q.question, q.method)
        translation_steps.append({"method": q.method, "queries": translated_queries})
    else:
        translated_queries = [q.question]

    coll = get_collection()
    scores = {}
    docs = {}
    retrieval_details = []

    for query_text in translated_queries:
        q_vec = embed([query_text])[0].tolist()
        results = coll.query(
            query_embeddings=[q_vec],
            n_results=TOP_K,
            include=["distances", "metadatas", "documents"],
        )

        hits = []
        for i, distance in enumerate(results["distances"][0]):
            meta = results["metadatas"][0][i] or {}
            index = int(meta.get("chunk_index", i))
            score = 1.0 - float(distance)
            hits.append((index, score))
            docs[index] = results["documents"][0][i]
            scores[index] = max(scores.get(index, score), score)

        retrieval_details.append(
            {"query": query_text, "hits": [{"index": i, "score": round(s, 3)} for i, s in hits]}
        )

    top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:TOP_K]
    context = "\n\n".join(f"[chunk {i}] {docs.get(i, '')}" for i, _ in top_hits)

    if q.method == "decomposition" and len(translated_queries) > 1:
        answers = []
        for sub_question in translated_queries:
            prompt = f"""Answer using only the context.

Context:
{context}

Question: {sub_question}

Answer:"""
            sub_answer, _, _ = call_llm(prompt)
            answers.append(f"Q: {sub_question}\nA: {sub_answer}")

        prompt = f"""Combine these answers into one final response:

{chr(10).join(answers)}

Original question: {q.question}

Final answer:"""
    else:
        prompt = f"""Answer only from the context. If the answer is not there, say:
The provided context does not contain sufficient information to answer this question.

Context:
{context}

Question: {q.question}

Answer:"""

    llm_start = time.time()
    answer, in_tok, out_tok = call_llm(prompt)

    return {
        "answer": answer,
        "chunks": [
            {
                "index": i,
                "score": round(scores[i], 3),
                "snippet": docs.get(i, "")[:300] + ("..." if len(docs.get(i, "")) > 300 else ""),
            }
            for i, _ in top_hits
        ],
        "translation_steps": translation_steps,
        "retrieval_details": retrieval_details,
        "latency": {
            "retrieval_ms": int((llm_start - start_time) * 1000),
            "llm_ms": int((time.time() - llm_start) * 1000),
            "total_ms": int((time.time() - start_time) * 1000),
        },
        "tokens": {"input": in_tok, "output": out_tok},
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
