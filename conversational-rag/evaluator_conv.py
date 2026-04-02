"""step5_evaluate.py — Retrieval (Recall@k, MRR) + generation (ROUGE) metrics."""

import json
import time
import numpy as np
from rouge_score import rouge_scorer
from retrieve import Retriever
from conv_rag import Generator

EVAL_SET = [
    ("How does backpropagation work?","Backpropagation","Backpropagation computes gradients via the chain rule from output to input."),
    ("What is dropout and why is it used?","Dropout","Dropout deactivates random neurons during training to prevent overfitting."),
    ("Explain the attention mechanism.","Attention","Attention computes weighted sums of values guided by query-key similarity."),
    ("What is batch normalization?","Batch normalization","Batch normalization normalizes layer inputs across the mini-batch."),
    ("What is a large language model?","Large language model","An LLM is a neural network trained on massive text to generate language."),
    ("What is retrieval-augmented generation?", "Retrieval-augmented generation", "RAG retrieves relevant documents and feeds them to a generator."),
    ("What is gradient descent?","Gradient descent","Gradient descent updates weights by following the negative gradient of the loss."),
]


def chunk_matches(chunk, gold_title):
    """Return True if the gold title appears in the chunk's title or topic."""
    gold = gold_title.lower()
    return gold in chunk.title.lower() or gold in chunk.topic.lower()


def recall_at_k(retriever, k):
    hits = 0
    for query, gold, _ in EVAL_SET:
        chunks = retriever.retrieve(query, top_k=k)
        if any(chunk_matches(c, gold) for c in chunks):
            hits += 1
    return hits


def mean_reciprocal_rank(retriever):
    rr_scores = []
    for query, gold, _ in EVAL_SET:
        chunks = retriever.retrieve(query, top_k=10)
        score = 0
        for rank, chunk in enumerate(chunks, start=1):
            if chunk_matches(chunk, gold):
                score = 1 / rank
                break
        rr_scores.append(score)
    return np.mean(rr_scores)


def evaluate_generation(generator, scorer):
    rouge1_scores, rouge2_scores, rougeL_scores, latencies = [], [], [], []
    per_query = []

    for i, (query, _, reference) in enumerate(EVAL_SET, start=1):
        print(f"  [{i}/{len(EVAL_SET)}] {query[:55]}…")

        result = generator.answer(query)
        scores = scorer.score(reference, result.answer)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
        latencies.append(result.latency_s)

        per_query.append({
            "query":      query,
            "answer":     result.answer,
            "latency_s":  result.latency_s,
            "rouge1":     round(scores["rouge1"].fmeasure, 4),
        })

    return rouge1_scores, rouge2_scores, rougeL_scores, latencies, per_query


def main():
    retriever = Retriever()
    generator = Generator(retriever)
    scorer    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # ── Retrieval ──────────────────────────────────────────────────────────
    print("\n=== RETRIEVAL ===")
    for k in [1, 3, 5]:
        hits = recall_at_k(retriever, k)
        print(f"  Recall@{k}: {hits}/{len(EVAL_SET)}")

    mrr = mean_reciprocal_rank(retriever)
    print(f"  MRR: {mrr:.4f}")

    # ── Generation ─────────────────────────────────────────────────────────
    print(f"\n=== GENERATION (expected ~{len(EVAL_SET) * 25}s on CPU) ===")
    r1, r2, rL, lats, per_query = evaluate_generation(generator, scorer)

    print(f"\n  ROUGE-1: {np.mean(r1):.4f}  ROUGE-2: {np.mean(r2):.4f}  ROUGE-L: {np.mean(rL):.4f}")
    print(f"  Avg latency: {np.mean(lats):.1f}s  (min {min(lats):.1f}s / max {max(lats):.1f}s)")

    report = {
        "retrieval":  {"mrr": round(float(np.mean(mrr)), 4)},
        "generation": {
            "rouge1":       round(float(np.mean(r1)), 4),
            "rouge2":       round(float(np.mean(r2)), 4),
            "rougeL":       round(float(np.mean(rL)), 4),
            "avg_latency_s": round(float(np.mean(lats)), 2),
        },
        "per_query": per_query,
    }

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("eval_report.json saved")


if __name__ == "__main__":
    main()