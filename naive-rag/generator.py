"""step4_generate.py — Retrieve chunks, call Ollama, return answer + metrics."""
import json, time, textwrap, urllib.request
from dataclasses import dataclass, field
from retrieve import Retriever

OLLAMA_MODEL  = "qwen2.5:3b"
OLLAMA_URL    = "http://localhost:11434/api/generate"
TOP_K         = 5
MAX_CTX_CHARS = 3000


@dataclass
class RAGAnswer:
    query:         str
    answer:        str
    sources:       list  = field(default_factory=list)
    latency_s:     float = 0.0
    prompt_tokens: int   = 0
    answer_tokens: int   = 0
    total_tokens:  int   = 0


class Generator:
    def __init__(self, retriever=None):
        self.retriever = retriever or Retriever()

    def answer(self, query: str) -> RAGAnswer:
        t0      = time.perf_counter()
        chunks  = self.retriever.retrieve(query, top_k=TOP_K)
        context = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))[:MAX_CTX_CHARS]
        prompt  = textwrap.dedent(f"""
            Answer using ONLY the context. If not found, say "I don't know."

            Context:
            {context}

            User: {query}
            Assistant:""")

        payload = json.dumps({
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }).encode()
        req = urllib.request.Request(OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())

        pt = d.get("prompt_eval_count", 0)
        at = d.get("eval_count", 0)

        return RAGAnswer(
            query         = query,
            answer        = d["response"].strip(),
            sources       = chunks,
            latency_s     = round(time.perf_counter() - t0, 2),
            prompt_tokens = pt,
            answer_tokens = at,
            total_tokens  = pt + at,
        )


if __name__ == "__main__":
    gen = Generator()
    for q in ["How does backpropagation work?",
              "What is dropout?",
              "Explain the attention mechanism."]:
        r = gen.answer(q)
        print(f"\nQ: {r.query}\nA: {r.answer}")
        print(f"   {r.latency_s}s | {r.prompt_tokens}pt + {r.answer_tokens}at = {r.total_tokens} total tokens")
        print(f"   Sources: {[s.title for s in r.sources]}")