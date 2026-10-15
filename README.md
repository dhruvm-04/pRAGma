# pRAGma

A small app for asking questions about PDFs and seeing where the answers came from. Upload a document, pick a retrieval method, and ask a question. Alongside the answer, you can look through the retrieved chunks, the queries used to find them, and how they were ranked.

The backend is FastAPI with Chroma for storage, FastEmbed for local embeddings, and Groq for the LLM calls. The frontend is plain HTML, CSS, and JavaScript. FastAPI serves it directly, so there's no frontend build to run.

## Getting started

You'll need Python 3.11+ and a Groq API key. These commands are for PowerShell, starting from the folder containing this README:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Create a `.env` file in `backend`:

```dotenv
GROQ_API_KEY=your-key
# Optional: choose a model available to your Groq account.
# GROQ_MODEL=your-model-id
```

Then start the app from that same folder:

```powershell
.\.venv\Scripts\python -m uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000**, upload a PDF, and ask something about it. The first upload takes longer because FastEmbed needs to download the embedding model, `BAAI/bge-small-en-v1.5`.

If you change the HTML, CSS, or JavaScript, just refresh the browser.

## What happens to a document?

The app extracts the PDF's text and splits it into chunks of 1,000 characters, with 200 characters of overlap. It embeds those chunks locally and saves them in `backend/chroma_db`, so your documents remain indexed after a restart.

Questions search across all uploaded PDFs. Uploading another file with the same filename replaces its earlier chunks. The conversation itself isn't saved; refreshing the page clears it.

## Retrieval methods

You can switch methods from the question box:

| Method | What it does |
| --- | --- |
| Direct | Searches with your question as written. |
| Multi Query | Adds up to three rewrites and keeps each chunk's best similarity score. |
| RAG Fusion | Searches with those question variations, then combines their result rankings using RRF. |
| Decomposition | Breaks the question into smaller questions, answers them, and combines the answers. |
| Step Back | Searches for broader background as well as your specific question. |
| HyDE | Writes a hypothetical answer to use as a search query. The final answer still uses the retrieved document text. |

### How RAG Fusion works here

The original question and up to three distinct rewrites each retrieve ten candidate chunks, or fewer if the library is smaller. Reciprocal rank fusion gives each chunk a score based on its position in each result list:

```text
RRF score = sum(1 / (60 + rank))
```

Ranks start at 1. A chunk gets a contribution from each list it appears in, so a chunk found by several queries can outrank one that appeared first in only one search. The top four chunks after fusion go into the answer prompt.

The function is in `backend/app.py`. It uses unique Chroma IDs to keep chunks from different PDFs separate, and follows the [original RRF method](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).

You can expand the evidence panels to see the search queries, their rankings, and each contribution to the final score. An RRF score is a ranking score, not a similarity percentage. The other retrieval modes show cosine similarity instead.

## Finding your way around

```text
backend/
  app.py             # PDF ingestion, retrieval, RRF, LLM calls, and API routes
  test_app.py        # ranking and API tests
  requirements.txt
frontend/
  index.html
  assets/
    app.js           # upload, chat, and evidence display
    styles.css
    theme.js
    favicon.svg
```

The API docs are at **http://localhost:8000/docs** while the app is running. The main endpoints are:

| Endpoint | Use |
| --- | --- |
| `GET /health` | Check the indexed chunk count. |
| `POST /ingest` | Upload a PDF as a multipart `file`; get back a job ID. |
| `GET /ingest/status/{job_id}` | Check upload and indexing progress. |
| `POST /query` | Ask a question and get an answer with its evidence. |

For example, the query body can be:

```json
{
  "question": "What are the main ideas in this document?",
  "method": "rag_fusion"
}
```

Leaving out `method` uses Direct retrieval. The browser starts with RAG Fusion selected.

Responses also include timings and token counts. Timings separate query rewriting, retrieval, and answer generation; token counts cover all LLM calls made for the question.

## Running the tests

From the folder containing this README:

```powershell
.\backend\.venv\Scripts\python -m pip install httpx
.\backend\.venv\Scripts\python -m unittest discover -s backend -v
```

The tests use synthetic embeddings and mock LLM responses, so they don't need an API key or a model download. They cover RRF scoring, query handling, API errors, and retrieval through an in-memory Chroma collection for all six methods.

## A few limitations

This is a local, single-user app. All sessions share the same document collection, and upload jobs live in memory, so run one server worker.

PDFs need selectable text; there's no OCR for scans. Chunking is based on character counts, which can split sentences or tables. Answers are prompted to stay within the retrieved context, but they can still be wrong. The evidence is there so you can check them.
