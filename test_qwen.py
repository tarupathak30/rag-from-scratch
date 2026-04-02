import requests
import time 


start = time.time()


PROMPT = """
Answer ONLY from this context.

Context:
Diffusion models generate images by reversing a gradual noise process.

Question: How do diffusion models generate images?
"""


response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5:1.5b",
        "prompt": PROMPT,
        "stream": False
    }
)

end = time.time()
latency = end - start

data = response.json()
answer = data["response"]

# --- Metrics ---
prompt_tokens_est = len(PROMPT.split())
answer_tokens_est = len(answer.split())

print("\n--- ANSWER ---\n")
print(answer)

print("\n--- METRICS ---")
print(f"Latency: {latency:.2f} seconds")
print(f"Prompt tokens (approx): {prompt_tokens_est}")
print(f"Answer tokens (approx): {answer_tokens_est}")
print(f"Total tokens (approx): {prompt_tokens_est + answer_tokens_est}")