# pRAGma

A minimal **retrieval-augmented generation (RAG)** app: upload any document and chat with it.
Every answer shows **exactly which chunks it read** and a **latency + cost log**
making grounding and reliability visible instead of assumed.

Built small as PoC with a single React component

## What it does

| Step | What happens | Where |
|------|--------------|-------|
| **Ingest** | PDF → text → fixed-size chunks → local embeddings → in-memory store | `POST /ingest` |
| **Retrieve** | question embedded, chunks ranked by cosine similarity, top-k pulled | `cosine_top_k` in `app.py` |
| **Generate** | top-k chunks stuffed into a grounding prompt, answered by a free LLM | `call_llm` in `app.py` |

The UI then shows, for each answer:

- the answer text
- the **retrieved chunks** (index, similarity score, snippet) - the evidence
- a **latency/cost log** (retrieval ms · LLM ms · total ms · tokens · est. $)

## Tech stack

- **Backend:** Python, FastAPI, FastEmbed (github/qdrant/fastembed)
  (`bge-small-en-v1.5`, local, no key), PyPDF, Llama 3.3 from Groq.
- **Frontend:** React (Vite), plain CSS

## Project structure

```
.
├── backend/
│   ├── app.py            # the entire RAG pipeline
│   ├── requirements.txt
│   └── .env.example      # add Groq API key
├── frontend/
│   ├── index.html
│   ├── vite.config.js    # /ingest /query → localhost:8000
│   └── src/
│       ├── App.jsx       # upload + chat + evidence/latency UI
│       ├── index.css
│       └── main.jsx
└── .gitignore
```

## Getting started

Prerequisites: **Python 3.11+** and **Node 18+**.

### Run everything on port 8000

Install the backend and frontend dependencies as described below, then build the
frontend once:

```bash
cd frontend
npm install
npm run build
```

Start FastAPI:

```bash
cd backend
.\.venv\Scripts\python -m uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** for the UI and
**http://localhost:8000/docs** for the FastAPI documentation. Run `npm run build`
again whenever you change the frontend.

### Backend setup

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Paste API key from Groq, then:

```bash
Copy-Item .env.example .env          # PowerShell
# edit .env -> GROQ_API_KEY=API_key
```

Run it:

```bash
.\.venv\Scripts\python -m uvicorn app:app --reload --port 8000
```

API docs will be at http://localhost:8000/docs.

### Frontend development (optional)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, upload a PDF and start asking questions.


## Usage

1. **Ingest** - pick a PDF. A live progress bar shows the pipeline in action
   (`parsing pdf… → embedding chunks… → 343/343 chunks (100%)`). The status
   line shows how many chunks were embedded and how long it took. Ingestion
   runs as a background job, so the UI stays responsive.
2. **Ask** - type a question and press Enter. The answer appears with the
   `// retrieved chunks` panel beneath it: each chunk shows its index,
   similarity score, and the text the model actually read.
3. **Read the log** - every response carries `retr {n}ms · llm {n}ms · total
   {n}ms · {in}/{out} tok · ${cost}`, so you can watch latency/cost behavior
   as you ask harder questions.

The chat stays in memory - reload the page to reset.

## API reference

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/health` | - | `{ status, chunks, source }` |
| `POST` | `/ingest` | multipart `file` (PDF) | `{ job_id }` - starts a background job |
| `GET` | `/ingest/status/{job_id}` | - | `{ status, message, done, total, source }` - poll this for live progress |
| `POST` | `/query` | `{ "question": "..." }` | `{ answer, chunks[], latency, tokens, cost_usd }` |

## This is RAG, not just LLM calls

The difference:
1. **`chunk_text`** - the PDF is cut into 1200 char chunks so retrieval works
   on passages, not a wall of text.
2. **`cosine_top_k`** - the question is embedded and the top-4 chunks are
   picked by cosine similarity. The UI surfaces these exact chunks as
   evidence, so you can audit why an answer says what it says.
3. **The grounding prompt** - the LLM is told to answer *only* from the
   retrieved context and to say so when the answer isn't in it. That is the
   anti-hallucination core of RAG.

## Limitations

- The "vector store" is two in-memory Python lists and restarting the backend
  clears it. A real system would use a vector DB.
- Chunking is character-length based; production RAG uses structure-aware
  chunking (headings, sections, tables).
- Similarity search is brute-force numpy; large corpora need an ANN index.
- Works on **text-based** PDFs. Scanned/image PDFs need OCR first.
