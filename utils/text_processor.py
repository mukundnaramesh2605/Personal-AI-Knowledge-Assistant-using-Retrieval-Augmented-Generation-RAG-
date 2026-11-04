"""
Text processor for the RAG knowledge assistant.

Takes the {"text", "metadata"} records from document_loader and splits
them into smaller overlapping chunks. Overlap preserves context across
chunk boundaries so a sentence split in two is still retrievable.

Each output chunk keeps its parent's metadata (source, page, slide) and
gains a chunk_id for traceability.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter


# Sensible defaults. Tune these in Phase 8.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def _make_splitter(chunk_size, chunk_overlap):
    # Splits on paragraphs first, then lines, sentences, words, as needed.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_records(records, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Split loader records into overlapping chunks.

    Args:
        records: List of {"text": str, "metadata": dict} from document_loader.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters shared between neighbouring chunks.

    Returns:
        List of {"text": str, "metadata": dict} chunks. Each metadata dict
        carries the parent fields plus a "chunk_id" (index within its source).
    """
    splitter = _make_splitter(chunk_size, chunk_overlap)
    chunks = []

    for record in records:
        text = record.get("text", "")
        base_meta = record.get("metadata", {})
        if not text.strip():
            continue

        for i, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if not piece:
                continue
            meta = dict(base_meta)        # copy so records stay independent
            meta["chunk_id"] = i
            chunks.append({"text": piece, "metadata": meta})

    return chunks


def chunk_stats(chunks):
    """Quick summary for the UI or logs."""
    if not chunks:
        return {"chunks": 0, "sources": 0, "avg_chars": 0}
    sources = {c["metadata"].get("source") for c in chunks}
    avg = sum(len(c["text"]) for c in chunks) // len(chunks)
    return {"chunks": len(chunks), "sources": len(sources), "avg_chars": avg}
