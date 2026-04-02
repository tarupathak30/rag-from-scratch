"""
step1_chunk.py  —  Split raw Wikipedia docs into overlapping text chunks.

Input  : wiki_knowledge_base.json   (list of {"topic", "title", "content"})
Output : chunks.json                (list of {"chunk_id", "topic", "title", "text"})
"""

import json
import re


# ── Config 
INPUT_FILE  = "wiki_knowledge_base.json"
OUTPUT_FILE = "chunks.json"

CHUNK_SIZE    = 300   # words per chunk
CHUNK_OVERLAP = 50    # words that bleed into the next chunk



def clean_text(text: str) -> str:
    """Remove Wikipedia section markers and collapse whitespace."""
    text = re.sub(r"==+[^=]+=+", " ", text)   # == Heading == artifacts
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Slide a fixed-size window with overlap over a word list."""
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(words[start : start + size])
        start += size - overlap
    return chunks


def main() -> None:
    with open(INPUT_FILE, "r") as f:
        docs = json.load(f)

    all_chunks = []
    chunk_id = 0

    for doc in docs:
        topic   = doc.get("topic", "")
        title   = doc.get("title", "")
        content = clean_text(doc.get("content", ""))

        words = content.split()
        if not words:
            continue

        for word_group in chunk_words(words, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "topic":    topic,
                    "title":    title,
                    "text":     " ".join(word_group),
                }
            )
            chunk_id += 1

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"{len(docs)} docs  →  {len(all_chunks)} chunks  →  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()