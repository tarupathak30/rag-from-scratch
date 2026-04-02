"""
step3_retrieve.py  —  Query the FAISS index, return top-k relevant chunks.

Can be used as a library (retrieve()) or run directly for a quick demo.

Input  : faiss_index.bin, chunk_meta.json
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


# ── Config 
INDEX_FILE  = "faiss_index.bin"
META_FILE   = "chunk_meta.json"
EMBED_MODEL = "all-MiniLM-L6-v2"   # must match step2
TOP_K       = 3



@dataclass
class RetrievedChunk:
    chunk_id: int
    topic:    str
    title:    str
    text:     str
    score:    float   # cosine similarity (higher = more relevant)


class Retriever:
    """Load-once, query-many FAISS retriever."""

    def __init__(
        self,
        index_path: str  = INDEX_FILE,
        meta_path:  str  = META_FILE,
        model_name: str  = EMBED_MODEL,
    ) -> None:
        print("Loading FAISS index …")
        self.index = faiss.read_index(index_path)

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        print(f"Loading encoder '{model_name}' …")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Retriever ready  ({self.index.ntotal} vectors)")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the top-k chunks most similar to *query*."""
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:       # FAISS returns -1 when index has < top_k items
                continue
            m = self.meta[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=m["chunk_id"],
                    topic=m["topic"],
                    title=m["title"],
                    text=m["text"],
                    score=float(score),
                )
            )
        return results


# ── Quick demo 
if __name__ == "__main__":
    retriever = Retriever()

    demo_queries = [
        "How does backpropagation work?",
        "What is attention mechanism in transformers?",
        "Explain overfitting and how to prevent it",
    ]

    for query in demo_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print("─" * 60)
        for chunk in retriever.retrieve(query, top_k=3):
            print(f"  [{chunk.score:.4f}] ({chunk.title})  {chunk.text[:120]}…")