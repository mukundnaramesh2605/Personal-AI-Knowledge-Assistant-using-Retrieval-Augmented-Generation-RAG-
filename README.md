# 📚 Personal AI Knowledge Assistant (RAG)

A privacy-friendly, document-grounded question-answering app. Upload your own
files, and ask questions in natural language — the assistant answers **only**
from the content you provide, cites its sources, and refuses to guess when the
documents don't contain the answer.

Built with [Streamlit](https://streamlit.io/), [FAISS](https://github.com/facebookresearch/faiss),
[Sentence Transformers](https://www.sbert.net/), and a pluggable LLM backend
([Groq](https://groq.com/) by default, with a Hugging Face fallback).

---

## Table of contents

- [What it does](#what-it-does)
- [Key features](#key-features)
- [How it works (architecture)](#how-it-works-architecture)
- [Project structure](#project-structure)
- [Supported file types](#supported-file-types)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Configuration reference](#configuration-reference)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Module reference](#module-reference)
- [Troubleshooting](#troubleshooting)
- [Tech stack](#tech-stack)

---

## What it does

Retrieval-Augmented Generation (RAG) combines a search engine with a language
model. Instead of relying on what the model memorized during training, the app:

1. Indexes **your** documents into a searchable vector database.
2. For each question, retrieves the most relevant passages.
3. Feeds only those passages to the LLM as context.
4. Returns an answer that is grounded in — and cites — your documents.

The result is an assistant that answers questions about *your* knowledge base
(notes, reports, manuals, spreadsheets, slides…) without hallucinating facts
from outside it.

---

## Key features

- **Answers grounded in your documents.** A strict system prompt forbids the
  model from using outside knowledge and tells it to say so when the answer
  isn't in the provided context.
- **Confidence guard.** If the best retrieval match scores below a threshold,
  the app refuses to answer rather than guessing — the primary defense against
  hallucination.
- **Source citations.** Every answer lists which files (and page / slide /
  sheet) it drew from.
- **Retrieval transparency.** Expand any answer to see the exact chunks and
  similarity scores behind it.
- **Live per-answer metrics.** Latency, top similarity score, number of chunks
  retrieved, and token usage are shown for each response.
- **Wide format support.** 13 file types including PDF, Word, PowerPoint,
  Excel/CSV, HTML, Markdown, RTF, JSON, and plain text.
- **Pluggable LLM provider.** Groq (fast, free) by default; Hugging Face
  Inference as a fallback. Swapping providers is a one-line change.
- **Built-in evaluation harness.** Measure retrieval accuracy, latency, guard
  behavior, and answer quality against a set of test questions.

---

## How it works (architecture)

```
          ┌─────────────────────── INGEST (once per upload) ───────────────────────┐
          │                                                                         │
 Upload   │   document_loader        text_processor          vector_store          │
 files ───┼──▶ load_documents()  ──▶  chunk_records()  ──▶  VectorStore.build()     │
          │   extract text +         split into 800-char    embed (MiniLM) +        │
          │   metadata per file      overlapping chunks     index in FAISS          │
          │                                                                         │
          └─────────────────────────────────────────────────────────────────────┘
                                                                     │
                                                                     ▼  (FAISS index
                                                                         in session)
          ┌─────────────────────── QUERY (per question) ──────────────────────────┐
          │                                                                        │
 Question │   vector_store            rag_pipeline               LLM provider      │
 ─────────┼──▶ search(query, k)  ──▶  build grounded prompt  ──▶  Groq / HF   ──┐   │
          │   top-k cosine-sim        + confidence guard                       │   │
          │   chunks                                                           ▼   │
          │                                                          answer + sources│
          └────────────────────────────────────────────────────────────────────┘
```

**Pipeline stages:**

1. **Load** — `document_loader` routes each file to a format-specific extractor
   and produces `{"text", "metadata"}` records. Metadata carries the source
   filename (plus page / slide / sheet) so answers can be cited.
2. **Chunk** — `text_processor` splits records into ~800-character overlapping
   chunks (150-char overlap) using a recursive splitter, preserving context
   across boundaries.
3. **Embed & index** — `vector_store` encodes chunks with the
   `all-MiniLM-L6-v2` Sentence Transformer (384-dim), L2-normalizes them, and
   indexes them in a FAISS inner-product index so scores are cosine similarity.
4. **Retrieve** — at query time, the question is embedded and the top-*k* most
   similar chunks are returned with scores.
5. **Guard & generate** — `rag_pipeline` checks the top score against the
   confidence threshold; if it passes, it builds a grounded prompt (system
   rules + recent chat history + context + question) and calls the LLM.
6. **Respond** — the answer is returned with source labels, the supporting
   chunks, grounded/confidence flags, and timing/token metrics.

---

## Project structure

```
.
├── app.py                 # Streamlit UI: upload, build index, chat loop
├── evaluate.py            # Offline evaluation harness (CLI)
├── requirements.txt       # Pinned Python dependencies
├── test_questions.json    # Sample evaluation questions
├── sample_docs/           # Example documents + questions for evaluation
│   └── test_questions.json
├── eval_results/          # Output of evaluate.py
│   ├── results.csv        # Per-question results
│   └── summary.json       # Aggregate metrics
└── utils/                 # Core pipeline package
    ├── __init__.py
    ├── document_loader.py # Extract text from 13 file formats
    ├── text_processor.py  # Split text into overlapping chunks
    ├── vector_store.py    # Embed + FAISS index + similarity search
    └── rag_pipeline.py    # Retrieve → prompt → generate, with guard & metrics
```

---

## Supported file types

| Category      | Extensions                          |
| ------------- | ----------------------------------- |
| Documents     | `.pdf`, `.docx`, `.rtf`, `.txt`     |
| Presentations | `.pptx`                             |
| Spreadsheets  | `.xlsx`, `.xls`, `.csv`, `.tsv`     |
| Web / markup  | `.html`, `.htm`, `.md`              |
| Data          | `.json`                             |

> Images and the legacy binary `.doc` format are **not** supported. Files that
> can't be read are skipped and reported in the UI rather than failing the whole
> upload.

---

## Getting started

### Prerequisites

- **Python 3.11** (the project is developed and tested against 3.11)
- A **Groq API key** — free at [console.groq.com](https://console.groq.com/)
  (or a Hugging Face token if you use the fallback provider)

### Installation

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd "Personal AI Knowledge Assistant using Retrieval-Augmented Generation (RAG)"

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from
> Hugging Face. This happens once and is then cached locally.

### Configuration

The app reads its LLM credentials from environment variables, loaded from a
local `.env` file if present.

Create a `.env` file in the project root:

```dotenv
# Primary provider (default)
GROQ_API_KEY=your_groq_api_key_here

# Optional fallback provider
HF_TOKEN=your_huggingface_token_here
```

> `.env`, `env.example`, and other local/secret files are gitignored and must
> not be committed.

### Running the app

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (default: http://localhost:8501).

---

## Using the app

1. **Add documents** — in the sidebar, upload one or more supported files.
2. **Build knowledge base** — click the button. The app loads, chunks, embeds,
   and indexes your files, then reports how many chunks/sources were indexed.
3. **Adjust settings** (optional) — pick a model, set how many chunks to
   retrieve (`top-k`), and tune the confidence threshold.
4. **Ask questions** — type in the chat box. Each answer shows:
   - the response, grounded in your documents;
   - a metrics row (latency, top score, chunks, tokens);
   - an expandable **Sources** list;
   - an expandable **Retrieved chunks** view with scores.
5. **Clear chat** — resets the conversation without rebuilding the index.

If retrieval is too weak for a question, you'll see a
*"Low confidence: not grounded in your documents"* warning instead of a
fabricated answer.

---

## Configuration reference

These can be changed in the sidebar at runtime, or via the defaults in code.

| Setting               | Where                         | Default            | Notes |
| --------------------- | ----------------------------- | ------------------ | ----- |
| Model                 | sidebar / `app.py`            | `openai/gpt-oss-120b` | Options listed in `CHAT_MODELS` |
| Chunks to retrieve    | sidebar (`top-k`)             | `4`                | 1–8 |
| Confidence threshold  | sidebar (`min_score`)         | `0.2`              | 0.0–0.6; below this the app refuses to answer |
| Chunk size            | `text_processor.py`           | `800` chars        | |
| Chunk overlap         | `text_processor.py`           | `150` chars        | |
| Embedding model       | `vector_store.py`             | `all-MiniLM-L6-v2` | 384-dim |
| LLM provider          | `rag_pipeline.py`             | `groq`             | `groq` or `huggingface` |
| Temperature           | `rag_pipeline.py`             | `0.2`              | |
| Max tokens            | `rag_pipeline.py`             | `512`              | |

---

## Evaluation

`evaluate.py` runs a set of test questions against a document folder and reports
the metrics in the project rubric.

```bash
# Using defaults (sample_docs/ + test_questions.json)
python evaluate.py

# Custom document set and questions
python evaluate.py --docs ./sample_docs --questions test_questions.json \
    --top_k 4 --min_score 0.2
```

**Metrics reported:**

- **Retrieval accuracy** — how often the expected source appears in the top-*k*.
- **Latency** — average and p95 response time.
- **Guard behavior** — in-scope questions answered vs. out-of-scope questions
  correctly refused.
- **Response quality** — a keyword-match sanity check on answers.

Results print to the console and are saved to `eval_results/` as
`results.csv` (per-question) and `summary.json` (aggregate).

**Test-question format** (`test_questions.json`):

```json
[
  {
    "question": "What is the main topic of the document?",
    "type": "in_scope",
    "expected_source": "notes.txt",
    "expected_keywords": ["hello"]
  },
  {
    "question": "What is the capital of France?",
    "type": "out_of_scope"
  }
]
```

| Field               | Required | Description |
| ------------------- | -------- | ----------- |
| `question`          | yes      | The question to ask |
| `type`              | yes      | `in_scope` or `out_of_scope` |
| `expected_source`   | no       | File that should be retrieved (in-scope) |
| `expected_keywords` | no       | Keywords expected in the answer |

---

## Deployment

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) with no
code changes:

1. Push the repository to GitHub.
2. Create a new Streamlit app pointing at `app.py`.
3. Add your `GROQ_API_KEY` (and optional `HF_TOKEN`) under **Settings →
   Secrets**. The app reads them as environment variables, so the local `.env`
   path and the cloud Secrets path are identical.

---

## Module reference

| Module                     | Responsibility |
| -------------------------- | -------------- |
| `app.py`                   | Streamlit UI — upload handling, index building, chat loop, metrics/source rendering, session state. |
| `utils/document_loader.py` | Per-format text extraction into `{text, metadata}` records; graceful skipping of unreadable files. |
| `utils/text_processor.py`  | Recursive overlapping chunking; `chunk_stats` summary for the UI. |
| `utils/vector_store.py`    | Sentence-Transformer embeddings, FAISS inner-product index, cosine similarity search, save/load persistence. |
| `utils/rag_pipeline.py`    | Prompt construction, confidence guard, LLM generation (Groq/HF), and per-answer metrics. |
| `evaluate.py`              | Offline evaluation harness and metrics reporting. |

---

## Troubleshooting

| Symptom | Likely cause / fix |
| ------- | ------------------ |
| `GROQ_API_KEY not set` banner | Add the key to `.env` (local) or Secrets (cloud). |
| First run is slow | The embedding model is downloading (~90 MB); subsequent runs use the cache. |
| "No readable text found" | The file may be image-only (e.g. a scanned PDF) or an unsupported type. |
| Answers say "could not find it" | Retrieval scored below the confidence threshold — lower `min_score` or add more relevant documents. |
| Model errors in an answer | Check provider status and that your API key is valid and has quota. |

---

## Tech stack

- **UI:** Streamlit
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector search:** FAISS (CPU)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **LLM:** Groq (default) / Hugging Face Inference (fallback)
- **Document parsing:** pypdf, python-docx, python-pptx, pandas + openpyxl/xlrd,
  striprtf, BeautifulSoup + lxml, markdown
- **Config:** python-dotenv

See [`requirements.txt`](requirements.txt) for exact pinned versions.
