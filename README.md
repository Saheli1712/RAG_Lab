# RAG Lab

An educational, black-background PDF chatbot that exposes each genuine RAG stage: extraction, overlapping chunks, provider embeddings, cosine retrieval, and evidence-based answers. Python/FastAPI serves the UI and model calls. JavaScript holds the document chunks and vectors **in the current browser tab** and runs cosine ranking; no database is required. This is a teaching demo, not a confidential-document service.

## Quick start

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api.index:app --reload
```

Open <http://127.0.0.1:8000>. Enter your OpenAI API key in the page for this session. Alternatively, set `OPENAI_API_KEY` in your environment; use `.env.example` as a reference. Python does **not** automatically load `.env.example`. Never commit a real key. A key supplied through the page remains in the password field until reset/refresh and is transmitted to the Python backend with each model request. If you use a shared server key on a public deployment, add authentication, quotas, and abuse controls first.

The default embedding model is `text-embedding-3-small` with 256 dimensions; the default answer model is `gpt-4o-mini`. Provider calls can incur charges. Override `EMBED_MODEL` only with an OpenAI embeddings model supporting the `dimensions` parameter; set `CHAT_MODEL` to a compatible chat completions model. Model access and billing must be enabled in your provider account.

## What happens

1. **Ingestion:** Python validates a PDF and extracts selectable text page by page using pypdf. This is unrelated to *prompt injection*, an attack where document text attempts to issue instructions.
2. **Chunking:** Each page becomes chunks of at most 900 characters, overlapping by about 150 characters. Chunks retain source page numbers.
3. **Embeddings:** The API turns each chunk into a real 256-dimensional vector, in batches of 20. The response returns vectors and chunks to the browser. The original PDF and extracted text are not saved on the server.
4. **Retrieval:** A question receives its own embedding. The browser computes cosine similarity against the document vectors and selects the top four chunks. Their real scores and full text are visible in the chat.
5. **Generation:** Python sends only those excerpts plus a small amount of dialogue context to the chat model. The model is instructed to use the excerpts as evidence, cite pages as `[p. N]`, and admit when they do not establish the answer. Click page citations to open the PDF at that page (browser PDF support varies).

Text and vectors remain in memory in this tab and disappear on refresh or reset. They are sent to the model provider for embeddings and answers, so use non-sensitive documents unless your organisation has approved that processing. The model can still make mistakes; inspect the cited passages. The answer API accepts passages supplied by the browser; it is not a trusted document verification service. There is no OCR, authentication, durable chat history, or multi-document index.

## GitHub → Vercel

1. Commit the contents of this folder as a GitHub repository. Keep `.env` and `.env.local` out of Git.
2. In Vercel, **Add New → Project**, import the GitHub repository, choose the repository root as the project root, and deploy. The supported root entry point `app.py` exports the FastAPI app from `api/index.py`; the same app serves `/` and `/static/*`.
3. Prefer users entering their own provider key in the interface for a public educational demo. For a private deployment, optionally add `OPENAI_API_KEY`, `EMBED_MODEL`, and `CHAT_MODEL` under Vercel project environment variables, then redeploy. Never put a key in frontend code or `vercel.json`.
4. Open the deployment and check `/api/health`, then upload a small text PDF and ask a question.

Vercel Functions impose request/response payload and execution limits. This demo limits PDF uploads to **2 MB**, **40 pages**, **72,000 extracted characters**, and **100 chunks** to leave headroom under the platform's request/response size limits. A PDF close to the text/chunk limit or a slow provider can still time out; use a shorter PDF. Serverless instances do not retain documents between requests, which is why the browser holds session state. A production version for larger documents needs direct-to-object-storage upload, a persistent vector store, background ingestion, authentication, and quotas.

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests cover chunk provenance and overlap, vector ranking, invalid/blank PDFs, app routes, missing keys, and mocked provider embeddings. A live provider call and a Vercel deployment require your credentials/account and are not part of the offline tests.

### Try these questions

- Upload a report containing a named revenue figure and ask, “What revenue does the report state?” Inspect the cited page.
- Ask about a fact absent from the PDF. The answer should say it could not find it in the document.
- Upload a scanned page. The app should explain that OCR is required.
- Upload a PDF over 2 MB or a non-PDF. The app should show a specific rejection.

## Structure

```text
api/index.py       FastAPI routes, provider calls, and grounding prompt
api/rag.py         PDF extraction, chunking, and ranking primitives
app.py             Vercel FastAPI entry point
static/            Accessible single-page UI, CSS, browser-side retrieval
tests/             Focused offline tests
requirements.txt  Pinned Python dependencies
vercel.json        Function duration configuration
.python-version    Python version
.env.example       Optional server settings
```
