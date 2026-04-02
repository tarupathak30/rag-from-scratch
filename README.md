# RAG from Scratch

Building retrieval-augmented generation from scratch, no LangChain, no LlamaIndex, just Python + FAISS + Ollama.

This project explores how well small local LLMs perform in a RAG setup, focusing on retrieval quality vs generation limitations under CPU-only constraints.

## Project Structure
```
rag-from-scratch/
├── naive-rag/
│   ├── wiki.py              # fetch Wikipedia articles as knowledge base
│   ├── chunking.py          # split articles into chunks
│   ├── embedding.py         # embed chunks with all-MiniLM-L6-v2
│   ├── retrieve.py          # FAISS vector search
│   ├── generator.py         # Ollama LLM generation
│   └── evaluator.py         # Recall@k, MRR, ROUGE metrics
│
├── conversational-rag/
│   ├── conv_rag.py          # query rewriting + chat history
│   ├── retriever.py         # FAISS retriever
│   └── evaluator_conv.py    # eval for conversational RAG
│
├── experiments/
│   ├── exp0-sanity.md       # sanity check
│   ├── exp1-naive-rag.md    # baseline naive RAG
│   ├── exp2-model-comparison.md   # gemma3:1b vs qwen2.5:1.5b vs qwen2.5:3b
│   └── exp3-eval-stress-test.md   # stress testing with hard queries
│
├── wiki.py                  # knowledge base builder
├── requirements.txt
└── README.md
```

## Stack

| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS |
| LLM | `qwen2.5:1.5b` via Ollama |
| Evaluation | ROUGE + Recall@k + MRR |
| Hardware | CPU only (AMD Athlon 3050U, 8GB RAM) |

## Quickstart

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Pull the model:**
```bash
ollama pull qwen2.5:1.5b
```

**3. Build the knowledge base:**
```bash
python wiki.py          # fetch Wikipedia articles
python chunking.py      # chunk them
python embedding.py     # embed + build FAISS index
```

**4. Run naive RAG:**
```bash
python naive-rag/generator.py
```

**5. Run evaluation:**
```bash
python naive-rag/evaluator.py
```

**6. Run conversational RAG:**
```bash
python conversational-rag/conv_rag.py
```

## Key Results

| Experiment | Recall@5 | MRR | ROUGE-1 |
|---|---|---|---|
| Naive RAG (gemma3:1b) | 7/7 | 1.000 | 0.123 |
| Naive RAG (qwen2.5:1.5b) | 7/7 | 1.000 | 0.201 |
| Naive RAG (qwen2.5:3b) | 7/7 | 1.000 | 0.110 |
| Stress test (11 queries) | 9/11 | 0.784 | 0.124 |

## Key Learnings

- Retrieval and generation are independent problems, perfect Recall@k does not guarantee good answers
- Small models (1.5B) ignore prompt instructions and over-generate, hurting ROUGE
- Semantic embeddings handle paraphrased queries well, "dilution" matched "dropout" correctly
- Stress testing with out-of-corpus and vague queries reveals weaknesses that clean eval sets hide


## Limitations
- High latency due to CPU-only inference
- Small LLM struggles with instruction following
- ROUGE may not fully capture answer quality
- Retrieval works well only on small corpus
