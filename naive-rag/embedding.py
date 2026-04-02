"""
step2_embed.py  —  Encode chunks with sentence-transformers, build FAISS index.

Input  : chunks.json
Output : faiss_index.bin   (FAISS flat-L2 index)
         chunk_meta.json   (parallel list of chunk metadata, same order as index)
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── Config 
CHUNKS_FILE  = "chunks.json"
INDEX_FILE   = "faiss_index.bin"
META_FILE    = "chunk_meta.json"

EMBED_MODEL  = "all-MiniLM-L6-v2"   # fast, 384-dim, good quality
BATCH_SIZE   = 64
# 


def main() -> None:
    # ── Load chunks 
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(texts)} chunks.")

    # ── Encode 
    print(f"Encoding with '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
        batch = texts[i : i + BATCH_SIZE]
        emb   = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"Embedding matrix: {embeddings.shape}")

    # ── Build FAISS index 
    # IndexFlatIP = inner product on L2-normalised vectors == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"✓ FAISS index saved  →  {INDEX_FILE}  ({index.ntotal} vectors)")

    # Save metadata (strip heavy text to keep file small) 
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "topic":    c["topic"],
            "title":    c["title"],
            "text":     c["text"],
        }
        for c in chunks
    ]
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved     →  {META_FILE}")


if __name__ == "__main__":
    main()