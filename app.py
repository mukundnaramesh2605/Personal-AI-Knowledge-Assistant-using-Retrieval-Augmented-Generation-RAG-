"""
Personal AI Knowledge Assistant (RAG) - Streamlit app.

Ties the pipeline together:
  upload documents -> load -> chunk -> embed + index (FAISS)
  -> chat: retrieve -> prompt -> LLM answer with source references.

Run:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from utils.document_loader import load_documents, SUPPORTED_EXTENSIONS
from utils.text_processor import chunk_records, chunk_stats
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline, DEFAULT_MODELS


# ---------- Page setup ----------

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 Personal AI Knowledge Assistant")
st.caption("Upload your documents and ask questions. Answers come only from what you upload.")

# Chat models available on the Groq account used here.
CHAT_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []      # [{role, content, sources}]
if "store" not in st.session_state:
    st.session_state.store = None       # VectorStore
if "stats" not in st.session_state:
    st.session_state.stats = None       # chunk stats dict


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
    st.session_state.messages = []      # fresh chat for a new document set

    st.success(f"Ready. Indexed {st.session_state.stats['chunks']} chunks "
               f"from {st.session_state.stats['sources']} file(s).")
    if errors:
        st.warning(f"Skipped {len(errors)} file(s): "
                   + ", ".join(e["source"] for e in errors))


# ---------- Sidebar: upload and settings ----------

with st.sidebar:
    st.header("1. Add documents")

    allowed = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploaded = st.file_uploader(
        "Upload one or more files",
        type=allowed,
        accept_multiple_files=True,
    )
    st.caption("Supported: " + ", ".join(allowed))

    if st.button("Build knowledge base", type="primary", disabled=not uploaded):
        build_knowledge_base(uploaded)

    st.divider()
    st.header("2. Settings")
    model = st.selectbox("Model", CHAT_MODELS, index=0)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)

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


# ---------- Main: chat ----------

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Input
if st.session_state.store is None:
    st.info("Upload documents and build the knowledge base to start chatting.")
else:
    query = st.chat_input("Ask a question about your documents...")
    if query:
        # Show and store the user turn
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Build chat history (role/content only) for context
        history = [{"role": m["role"], "content": m["content"]}
                   for m in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                rag = RAGPipeline(st.session_state.store, model=model)
                result = rag.answer(query, chat_history=history, k=top_k)

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for src in result["sources"]:
                        st.markdown(f"- {src}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
