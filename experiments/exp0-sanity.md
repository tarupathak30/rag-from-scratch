# Experiment 0 — Local LLM Sanity Check 

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
"""
Answer ONLY from this context.

Context:
Diffusion models generate images by reversing a gradual noise process.

Question: How do diffusion models generate images?
"""
```


---

## Model Output

```text
Diffusion models generate images by simulating the reverse of a gradually adding noise process to start with clean, pixelated images and gradually introducing noise until it reaches an image that resembles the original clear, detailed picture.
```

---

## Metrics

| Metric                 | Value             |
| ---------------------- | ----------------- |
| Latency                | {{22.81 seconds}}       |
| Prompt tokens (approx) | {{23}} |
| Answer tokens (approx) | {{36}} |
| Total tokens (approx)  | {{59}}  |

---

## Observations

1. model obeys context restriction.
2. generation is slow on CPU
3. confirms this setup is usable for RAG but latency is a concern.

---

## Key Insight

small local LLMs can follow strict context instructions, but may still distort core concepts under constrained prompts.



---

## Next Experiment Ideas

* implement naive/simple RAG : single-document retrieval using embeddings + cosine similarity to ground LLM Responses.
* build conversational RAG : extend naive RAG by maintaining chat history and retrieving context across turns to handle follow up questions
* design modular RAG pipeline : seperate retrieval, ranking and generation components to experiment with chunking strategies, re-ranking and prompt templates.
