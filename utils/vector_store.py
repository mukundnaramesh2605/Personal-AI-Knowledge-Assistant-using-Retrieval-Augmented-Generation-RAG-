"""
Vector store for the RAG knowledge assistant.

Embeds text chunks with a Sentence Transformer, indexes them in FAISS,
and answers similarity searches. Embeddings are L2-normalized and the
index uses inner product, so scores are cosine similarity in [-1, 1].

The index and its parallel metadata list are saved together so the
knowledge base persists between runs (and across Streamlit reruns).
"""

import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"   # small, fast, 384-dim, good general quality


class VectorStore:
    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []   # parallel to index rows: {"text", "metadata"}

    # ---------- Build ----------

    def _embed(self, texts):
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # makes inner product == cosine
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def build(self, chunks):
        """
        Build the index from chunk records.

        Args:
            chunks: List of {"text", "metadata"} from text_processor.
        """
        if not chunks:
            raise ValueError("No chunks to index.")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vectors = self._embed(texts)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # exact search, cosine via IP
        self.index.add(vectors)
        return self

    # ---------- Search ----------

    def search(self, query, k=4):
        """
        Return the top-k most similar chunks to the query.

        Args:
            query: User question text.
            k: Number of chunks to return.

        Returns:
            List of {"text", "metadata", "score"}, best first.
        """
        if self.index is None:
            raise ValueError("Index is empty. Build or load it first.")

        q_vec = self._embed([query])
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            hit = dict(self.chunks[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results

    # ---------- Persistence ----------

    def save(self, directory="vector_db"):
        """Save the FAISS index and metadata to a folder."""
        if self.index is None:
            raise ValueError("Nothing to save. Build the index first.")
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "store.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, directory="vector_db"):
        """Load a saved index and metadata back into a VectorStore."""
        index_path = os.path.join(directory, "index.faiss")
        store_path = os.path.join(directory, "store.pkl")
        if not (os.path.exists(index_path) and os.path.exists(store_path)):
            raise FileNotFoundError(f"No saved index found in '{directory}'.")

        with open(store_path, "rb") as f:
            data = pickle.load(f)

        store = cls(model_name=data["model_name"])
        store.index = faiss.read_index(index_path)
        store.chunks = data["chunks"]
        return store

    def __len__(self):
        return len(self.chunks)
