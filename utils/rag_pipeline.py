"""
RAG pipeline for the knowledge assistant.

Ties retrieval, prompt construction, and generation together:
  1. Retrieve the top-k chunks for the user's question (Phase 5).
  2. Build a grounded prompt from context + chat history + question (Phase 6).
  3. Call the LLM and return the answer with its source references.

The prompt constrains the model to answer only from the retrieved context
and to say so when the answer is not present. This is the main guard
against hallucination.

LLM provider is pluggable:
  - "groq"        (default) free, fast. Token from GROQ_API_KEY.
  - "huggingface" fallback. Token from HF_TOKEN.
Set the token locally in .env and on Streamlit Cloud in Secrets.
"""

import os
import time

# Load variables from a local .env file if present. On Streamlit Cloud the
# same keys come from Secrets as environment variables, so this is a no-op
# there and the code path stays identical.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_PROVIDER = "groq"
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
}
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.2   # below this top similarity, treat retrieval as too weak to answer

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the "
    "provided context from the user's documents. Follow these rules:\n"
    "1. Answer only from the context below. Do not use outside knowledge.\n"
    "2. If the answer is not in the context, say you could not find it in "
    "the documents. Do not guess.\n"
    "3. Be clear and concise. Quote figures and names exactly as written.\n"
    "4. When useful, mention which source the answer came from."
)


def format_context(hits):
    """
    Turn retrieved chunks into a labelled context block.

    Returns:
        (context_text, sources)
          context_text: numbered chunks with source tags for the prompt
          sources: deduped list of readable source labels for the UI
    """
    blocks, sources, seen = [], [], set()
    for i, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        label = meta.get("source", "unknown")
        if "page" in meta:
            label += f" (page {meta['page']})"
        elif "slide" in meta:
            label += f" (slide {meta['slide']})"
        elif "sheet" in meta:
            label += f" (sheet {meta['sheet']})"

        blocks.append(f"[{i}] Source: {label}\n{hit['text']}")
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return "\n\n".join(blocks), sources


def build_messages(context, chat_history, query):
    """
    Assemble chat-format messages: system rules, recent history, then the
    context-grounded question.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep the last few turns so follow-up questions have continuity.
    for turn in (chat_history or [])[-6:]:
        messages.append(turn)

    user_content = (
        f"Context from documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )
    messages.append({"role": "user", "content": user_content})
    return messages


class RAGPipeline:
    def __init__(self, vector_store, provider=DEFAULT_PROVIDER, model=None,
                 token=None, temperature=0.2, max_tokens=512):
        self.store = vector_store
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = self._make_client(token)

    def _make_client(self, token):
        """Build the LLM client for the chosen provider. Imports lazily."""
        if self.provider == "groq":
            from groq import Groq
            key = token or os.getenv("GROQ_API_KEY")
            return Groq(api_key=key)

        if self.provider == "huggingface":
            from huggingface_hub import InferenceClient
            key = token or os.getenv("HF_TOKEN")
            return InferenceClient(provider="auto", api_key=key)

        raise ValueError(f"Unknown provider: {self.provider}")

    def _generate(self, messages):
        """
        Call the LLM. Both Groq and HuggingFace return an OpenAI-style
        response, so the access path is the same.

        Returns (text, token_usage_dict).
        """
        if self.provider == "groq":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:  # huggingface
            resp = self.client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        text = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        tokens = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        return text, tokens

    def answer(self, query, chat_history=None, k=DEFAULT_TOP_K, min_score=DEFAULT_MIN_SCORE):
        """
        Full RAG cycle for one question, with metrics and a confidence guard.

        Returns:
            {
              "answer": str,
              "sources": [str, ...],
              "hits": [ {text, metadata, score}, ... ],
              "grounded": bool,          # False when retrieval was too weak
              "confidence": float,       # top similarity score
              "metrics": { ... }         # timing, chunk count, token usage
            }
        """
        t0 = time.perf_counter()
        hits = self.store.search(query, k=k)
        retrieval_time = time.perf_counter() - t0

        top_score = float(hits[0]["score"]) if hits else 0.0
        metrics = {
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": 0.0,
            "total_time": round(retrieval_time, 3),
            "chunks_retrieved": len(hits),
            "top_score": round(top_score, 3),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Confidence guard: if the best match is weak, do not let the model
        # guess. This is the main defence against hallucination.
        if not hits or top_score < min_score:
            return {
                "answer": ("I could not find a strong enough match in your "
                           "documents to answer this confidently."),
                "sources": [],
                "hits": hits,
                "grounded": False,
                "confidence": round(top_score, 3),
                "metrics": metrics,
            }

        context, sources = format_context(hits)
        messages = build_messages(context, chat_history, query)

        t1 = time.perf_counter()
        try:
            answer, tokens = self._generate(messages)
        except Exception as e:
            answer = f"The model could not generate a response. ({e})"
            tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        generation_time = time.perf_counter() - t1

        metrics.update({
            "generation_time": round(generation_time, 3),
            "total_time": round(retrieval_time + generation_time, 3),
            **tokens,
        })

        return {
            "answer": answer,
            "sources": sources,
            "hits": hits,
            "grounded": True,
            "confidence": round(top_score, 3),
            "metrics": metrics,
        }
