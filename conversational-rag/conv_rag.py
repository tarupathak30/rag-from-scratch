"""Conversational RAG — chat history + query rewriting + Generator for eval."""
import json, textwrap, urllib.request, time
from dataclasses import dataclass, field
from retrieve import Retriever

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_URL   = "http://localhost:11434/api/generate"
TOP_K        = 5

history = []


def ollama(prompt: str, max_tokens: int = 512) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def rewrite(query: str) -> str:
    if not history:
        return query
    recent = "\n".join(f"{t['role']}: {t['content']}" for t in history[-4:])
    prompt = (f"Rewrite the question as a standalone search query. "
              f"Output ONLY the rewritten query.\n\n"
              f"History:\n{recent}\n\nQuestion: {query}\nRewritten:")
    result = ollama(prompt, max_tokens=64)
    return result if len(result) < 200 else query


def chat(user_message: str) -> str:
    search_query = rewrite(user_message)
    if search_query != user_message:
        print(f"  [→ {search_query}]")

    chunks = retriever.retrieve(search_query, top_k=TOP_K)
    context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))[:3000]

    hist_str = "\n".join(f"{t['role'].title()}: {t['content']}" for t in history[-6:])
    prompt = textwrap.dedent(f"""
        Answer using ONLY the context. If not found, say "I don't know."

        Context:
        {context}

        {hist_str}
        User: {user_message}
        Assistant:""")

    answer = ollama(prompt)
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return answer


# ── Evaluator-compatible wrapper ──────────────────────────────────────────────

@dataclass
class RAGAnswer:
    query:     str
    answer:    str
    sources:   list = field(default_factory=list)
    latency_s: float = 0.0

class Generator:
    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()

    def answer(self, query: str) -> RAGAnswer:
        t0     = time.perf_counter()
        chunks = self.retriever.retrieve(query, top_k=TOP_K)
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))[:3000]
        prompt = textwrap.dedent(f"""
            Answer using ONLY the context. If not found, say "I don't know."

            Context:
            {context}

            User: {query}
            Assistant:""")
        ans = ollama(prompt)
        return RAGAnswer(query, ans, chunks, round(time.perf_counter() - t0, 2))


retriever = Retriever()

if __name__ == "__main__":
    print("Conversational RAG — type 'reset' or 'quit'\n")
    while True:
        q = input("You: ").strip()
        if not q:        continue
        if q == "quit":  break
        if q == "reset": history.clear(); print("  [cleared]\n"); continue
        print(f"Bot: {chat(q)}\n")