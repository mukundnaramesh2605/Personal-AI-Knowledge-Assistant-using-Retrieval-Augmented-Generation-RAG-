"""
Evaluation script for the RAG knowledge assistant (Phase 8).

Runs a set of test questions against a document set and reports the
metrics in the project rubric:

  - Retrieval Accuracy : how often the expected source is in the top-k
  - Latency            : average and p95 response time
  - Hallucination/Guard: whether out-of-scope questions are refused and
                         in-scope questions are answered
  - Response Quality   : rough keyword-match check on the answer

Results print to the console and save to eval_results/ as CSV + summary.

Usage:
  python evaluate.py --docs ./sample_docs --questions test_questions.json
  python evaluate.py                      # uses defaults below

Test-question format (test_questions.json): a list of objects
  {
    "question": "...",
    "type": "in_scope" | "out_of_scope",
    "expected_source": "file.pdf",      # in_scope only, optional
    "expected_keywords": ["a", "b"]     # optional, for quality check
  }
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

from utils.document_loader import load_documents, SUPPORTED_EXTENSIONS
from utils.text_processor import chunk_records
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline


def collect_paths(docs_dir):
    """Find every supported file in the docs folder."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(docs_dir, f"*{ext}")))
    return sorted(paths)


def keyword_match(answer, keywords):
    """True if every expected keyword appears in the answer (case-insensitive)."""
    if not keywords:
        return None   # not checked
    low = answer.lower()
    return all(k.lower() in low for k in keywords)


def run(docs_dir, questions_file, top_k, min_score, model):
    # --- Build the knowledge base once ---
    paths = collect_paths(docs_dir)
    if not paths:
        raise SystemExit(f"No supported documents found in '{docs_dir}'.")

    print(f"Indexing {len(paths)} file(s) from '{docs_dir}'...")
    records, errors = load_documents(paths)
    chunks = chunk_records(records)
    store = VectorStore().build(chunks)
    rag = RAGPipeline(store, model=model)
    print(f"Indexed {len(chunks)} chunks. Skipped {len(errors)} file(s).\n")

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for q in questions:
        res = rag.answer(q["question"], chat_history=[], k=top_k, min_score=min_score)
        retrieved = [h.get("metadata", {}).get("source") for h in res["hits"]]

        expected = q.get("expected_source")
        retrieval_hit = (expected in retrieved) if expected else None
        kw = keyword_match(res["answer"], q.get("expected_keywords"))

        rows.append({
            "question": q["question"],
            "type": q.get("type", "in_scope"),
            "expected_source": expected or "",
            "retrieved_sources": ", ".join(s for s in retrieved if s),
            "retrieval_hit": retrieval_hit,
            "grounded": res["grounded"],
            "top_score": res["metrics"]["top_score"],
            "latency_s": res["metrics"]["total_time"],
            "tokens": res["metrics"]["total_tokens"],
            "keyword_match": kw,
            "answer": res["answer"].replace("\n", " ")[:200],
        })

    return rows


def summarize(rows):
    in_scope = [r for r in rows if r["type"] == "in_scope"]
    out_scope = [r for r in rows if r["type"] == "out_of_scope"]

    # Retrieval accuracy over in-scope questions that declared an expected source
    checked = [r for r in in_scope if r["retrieval_hit"] is not None]
    hits = sum(1 for r in checked if r["retrieval_hit"])
    retrieval_acc = (hits / len(checked)) if checked else None

    # Latency
    lat = [r["latency_s"] for r in rows]
    avg_lat = round(float(np.mean(lat)), 3) if lat else 0
    p95_lat = round(float(np.percentile(lat, 95)), 3) if lat else 0

    # Guard behaviour
    in_answered = sum(1 for r in in_scope if r["grounded"])
    out_refused = sum(1 for r in out_scope if not r["grounded"])

    # Quality (keyword match where provided)
    quality = [r for r in rows if r["keyword_match"] is not None]
    quality_pass = sum(1 for r in quality if r["keyword_match"])

    return {
        "total_questions": len(rows),
        "retrieval_accuracy": retrieval_acc,
        "retrieval_checked": len(checked),
        "avg_latency_s": avg_lat,
        "p95_latency_s": p95_lat,
        "in_scope_answered": f"{in_answered}/{len(in_scope)}",
        "out_of_scope_refused": f"{out_refused}/{len(out_scope)}",
        "quality_pass": f"{quality_pass}/{len(quality)}" if quality else "n/a",
    }


def save_results(rows, summary, out_dir="eval_results"):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return csv_path, summary_path


def print_summary(summary):
    acc = summary["retrieval_accuracy"]
    acc_str = f"{acc * 100:.0f}%" if acc is not None else "n/a"
    print("=" * 44)
    print("EVALUATION SUMMARY")
    print("=" * 44)
    print(f"Questions tested      : {summary['total_questions']}")
    print(f"Retrieval accuracy    : {acc_str} ({summary['retrieval_checked']} checked)")
    print(f"Avg latency           : {summary['avg_latency_s']} s")
    print(f"P95 latency           : {summary['p95_latency_s']} s")
    print(f"In-scope answered     : {summary['in_scope_answered']}")
    print(f"Out-of-scope refused  : {summary['out_of_scope_refused']}")
    print(f"Response quality pass : {summary['quality_pass']}")
    print("=" * 44)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate the RAG assistant.")
    ap.add_argument("--docs", default="sample_docs", help="Folder of documents to index")
    ap.add_argument("--questions", default="test_questions.json", help="Test questions JSON")
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--min_score", type=float, default=0.2)
    ap.add_argument("--model", default=None, help="Override the default model")
    args = ap.parse_args()

    rows = run(args.docs, args.questions, args.top_k, args.min_score, args.model)
    summary = summarize(rows)
    print_summary(summary)
    csv_path, summary_path = save_results(rows, summary)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {summary_path}")
