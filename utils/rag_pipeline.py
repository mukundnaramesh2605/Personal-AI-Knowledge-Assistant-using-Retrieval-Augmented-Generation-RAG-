"""
RAG pipeline for the knowledge assistant.

Ties retrieval, prompt construction, and generation together:
  1. Retrieve the top-k chunks for the user's question (Phase 5).
  2. Build a grounded prompt from context + chat history + question (Phase 6).
  3. Call the LLM and return the answer with its source references.

The prompt constrains the model to answer only from the retrieved context
and to say so when the answer is not present. This is the main guard
against hallucination.

LLM: HuggingFace serverless inference by default. Set the token in an
env var HF_TOKEN (locally in .env, on Streamlit Cloud in Secrets).
Swap providers by replacing _generate() only; the rest stays the same.
"""

import os

from huggingface_hub import InferenceClient


DEFAULT_LLM = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_PROVIDER = "auto"   # let the router pick a provider that serves the model
DEFAULT_TOP_K = 4

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
    def __init__(self, vector_store, model=DEFAULT_LLM, token=None,
                 provider=DEFAULT_PROVIDER, temperature=0.2, max_tokens=512):
        self.store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token = token or os.getenv("HF_TOKEN")
        # provider routing goes through router.huggingface.co (the old
        # api-inference.huggingface.co host is decommissioned).
        self.client = InferenceClient(provider=provider, api_key=self.token)

    def _generate(self, messages):
        """Single place that talks to the LLM. Replace to switch providers."""
        resp = self.client.chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def answer(self, query, chat_history=None, k=DEFAULT_TOP_K):
        """
        Full RAG cycle for one question.

        Returns:
            {
              "answer": str,
              "sources": [str, ...],   # readable labels for the UI
              "hits": [ {text, metadata, score}, ... ]
            }
        """
        hits = self.store.search(query, k=k)

        if not hits:
            return {
                "answer": "I could not find anything relevant in the documents.",
                "sources": [],
                "hits": [],
            }

        context, sources = format_context(hits)
        messages = build_messages(context, chat_history, query)

        try:
            answer = self._generate(messages)
        except Exception as e:
            answer = f"The model could not generate a response. ({e})"

        return {"answer": answer, "sources": sources, "hits": hits}
