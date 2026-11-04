"""
Personal AI Knowledge Assistant (RAG) - Streamlit app.

Ties the pipeline together:
  upload documents -> load -> chunk -> embed + index (FAISS)
  -> chat: retrieve -> prompt -> LLM answer with source references.

Standout features:
  - Live metrics per answer (latency, top score, chunks, tokens)
  - Confidence guard (refuses to answer on weak retrieval)
  - Retrieval transparency (shows the chunks behind each answer)

Run:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from utils.document_loader import load_documents, SUPPORTED_EXTENSIONS
from utils.text_processor import chunk_records, chunk_stats
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline


# ---------- Page setup ----------

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Personal AI Knowledge Assistant")
st.caption("Upload your documents and ask questions. Answers come only from what you upload.")

CHAT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []      # [{role, content, sources, hits, metrics, grounded}]
if "store" not in st.session_state:
    st.session_state.store = None
if "stats" not in st.session_state:
    st.session_state.stats = None


# ---------- Helpers ----------

def save_uploads(files):
    """Write Streamlit uploads to a temp folder and return their paths."""
    tmp_dir = tempfile.mkdtemp(prefix="rag_uploads_")
    paths = []
    for f in files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths


def build_knowledge_base(files):
    """Full ingest: load -> chunk -> embed -> index. Stores result in session."""
    paths = save_uploads(files)

    with st.spinner("Reading documents..."):
        records, errors = load_documents(paths)

    if not records:
        st.error("No readable text found in the uploaded files.")
        if errors:
            st.write("Skipped:", errors)
        return

    with st.spinner("Splitting into chunks..."):
        chunks = chunk_records(records)

    with st.spinner("Building the knowledge base (embeddings + FAISS)..."):
        store = VectorStore().build(chunks)

    st.session_state.store = store
    st.session_state.stats = chunk_stats(chunks)
    st.session_state.messages = []

    st.success(f"Ready. Indexed {st.session_state.stats['chunks']} chunks "
               f"from {st.session_state.stats['sources']} file(s).")
    if errors:
        st.warning(f"Skipped {len(errors)} file(s): "
                   + ", ".join(e["source"] for e in errors))


def render_extras(msg):
    """Metrics row, sources, and retrieved chunks for one assistant message."""
    m = msg.get("metrics")
    if m:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latency", f"{m['total_time']}s")
        c2.metric("Top score", m["top_score"])
        c3.metric("Chunks", m["chunks_retrieved"])
        c4.metric("Tokens", m["total_tokens"])

    if msg.get("sources"):
        with st.expander("Sources"):
            for src in msg["sources"]:
                st.markdown(f"- {src}")

    if msg.get("hits"):
        with st.expander("Retrieved chunks (what the answer is based on)"):
            for i, h in enumerate(msg["hits"], start=1):
                label = h.get("metadata", {}).get("source", "unknown")
                score = round(float(h["score"]), 3)
                snippet = h["text"][:400] + ("..." if len(h["text"]) > 400 else "")
                st.markdown(f"**[{i}] {label}** — score {score}")
                st.caption(snippet)


# ---------- Sidebar ----------

with st.sidebar:
    st.header("1. Add documents")

    allowed = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploaded = st.file_uploader("Upload one or more files", type=allowed,
                                accept_multiple_files=True)
    st.caption("Supported: " + ", ".join(allowed))

    if st.button("Build knowledge base", type="primary", disabled=not uploaded):
        build_knowledge_base(uploaded)

    st.divider()
    st.header("2. Settings")
    model = st.selectbox("Model", CHAT_MODELS, index=0)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
    min_score = st.slider("Confidence threshold", 0.0, 0.6, 0.2, 0.05,
                          help="If the best match scores below this, the app "
                               "refuses to answer instead of guessing.")

    if st.session_state.stats:
        st.divider()
        s = st.session_state.stats
        st.metric("Indexed chunks", s["chunks"])
        st.metric("Sources", s["sources"])
        st.metric("Avg chunk size", f"{s['avg_chars']} chars")

    if st.button("Clear chat"):
        st.session_state.messages = []

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Add it to .env (local) or Secrets (cloud).")


# ---------- Chat ----------

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("grounded") is False:
                st.warning("Low confidence: not grounded in your documents.")
            render_extras(msg)

# Input
if st.session_state.store is None:
    st.info("Upload documents and build the knowledge base to start chatting.")
else:
    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag = RAGPipeline(st.session_state.store, model=model)
                result = rag.answer(query, chat_history=history,
                                    k=top_k, min_score=min_score)

            st.markdown(result["answer"])
            if not result["grounded"]:
                st.warning("Low confidence: not grounded in your documents.")
            render_extras(result)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "hits": result["hits"],
            "metrics": result["metrics"],
            "grounded": result["grounded"],
        })
