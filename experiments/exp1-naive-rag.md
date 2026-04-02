


# Experiment {{1}} — {{Naive RAG}}

## Setup

| Item             | Value               |
| ---------------- | ------------------- |
| Model            | {{qwen2.5:1.5b}}      |
| Inference Engine | {{ollama}}    |
| Machine          | {{AMD Athlon 3050U, 8GB RAM}} |
| Hardware         | {{CPU only}}       |

---

## Prompt Used

```text
{{"""
Answer using ONLY the context. If not found, say "I don't know."

Context:
{context}

User: {query}
Assistant:"""}}
```

---

## Retrieved Context (if RAG)

```text
{{Top-5 chunks retrieved per query using FAISS cosine similarity 
over all-MiniLM-L6-v2 embeddings. 70 vectors in index covering 
topics: backpropagation, dropout, attention, batch normalization,
LLMs, RAG, gradient descent.}}
```

---

## Model Output

```text
{{

=== RETRIEVAL ===
  Recall@1: 7/7
  Recall@3: 7/7
  Recall@5: 7/7
  MRR: 1.0000

=== GENERATION (expected ~175s on CPU) ===
  [1/7] How does backpropagation work?…
  [2/7] What is dropout and why is it used?…
  [3/7] Explain the attention mechanism.…
  [4/7] What is batch normalization?…
  [5/7] What is a large language model?…
  [6/7] What is retrieval-augmented generation?…
  [7/7] What is gradient descent?…
  ROUGE-1: 0.1305  ROUGE-2: 0.0403  ROUGE-L: 0.0961

}}
```

---

## Metrics

| Metric                 | Value  |
| ---------------------- | ------ |
| Latency                | 57.0s avg (min 45.0s / max 86.6s) |
| Prompt tokens (approx) | 4524   |
| Answer tokens (approx) | 1369   |
| Total tokens (approx)  | 5893   |
| ROUGE-1                | 0.1305 |
| ROUGE-2                | 0.0403 |
| ROUGE-L                | 0.0961 |

---

## Observations

1. {{Retrieval is perfect — all 7 queries hit the correct chunk at rank 1, giving MRR of 1.0. The bottleneck is entirely in generation.}}
2. {{ROUGE scores are low (0.13 / 0.04 / 0.10), indicating the model paraphrases heavily rather than reproducing reference phrasing. The 1.5B model struggles to synthesize concise, on-point answers from retrieved context.}}
3. {{Latency is high at 57s average on CPU-only inference. The max of 86.6s suggests some queries produce significantly longer outputs before truncation.}}

---

## Key Insight

{{Retrieval quality is not the bottleneck in naive RAG on a small corpus — generation quality and model size are, with a 1.5B model producing low ROUGE despite perfect chunk recall.}}

---

## Reproducibility Notes

Steps to reproduce this experiment:

1. {{Build the FAISS index: `python indexer.py` — encodes chunks with `all-MiniLM-L6-v2` and saves `faiss.index` + `chunks.json`.}}
2. {{Start Ollama and pull the model: `ollama serve` then `ollama pull qwen2.5:1.5b`.}}
3. {{Run the evaluator: `python evaluator.py` — outputs metrics to console and saves `eval_report.json`.}}

---

## Next Experiment Ideas

* {{Swap `qwen2.5:1.5b` for `qwen2.5:3b` or `llama3.2:3b` and compare ROUGE scores at the same retrieval setup.}}
* {{Add query rewriting (conversational RAG) and measure whether standalone queries improve ROUGE on follow-up questions.}}