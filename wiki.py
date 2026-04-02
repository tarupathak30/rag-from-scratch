import wikipedia
import json
import time

topics = [
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Backpropagation",
    "Gradient descent",
    "Loss function",
    "Activation function",
    "Overfitting",
    "Regularization (machine learning)",
    "Dropout (neural networks)",
    "Batch normalization",

    "Convolutional neural network",
    "Recurrent neural network",
    "Transformer (machine learning)",
    "Attention mechanism",
    "Self-attention",

    "Natural language processing",
    "Language model",
    "Large language model",
    "Prompt engineering",
    "Tokenization",
    "Word embedding",
    "Sentence embedding",

    "BERT (language model)",
    "GPT (language model)",
    "T5 (language model)",

    "Retrieval-augmented generation",
    "Vector database",
    "Semantic search",
    "Cosine similarity",
    "Embedding",
    "Information retrieval",

    "Fine-tuning (machine learning)",
    "Transfer learning",
    "Hyperparameter optimization",

    "Diffusion model",
    "Reinforcement learning",
    "Self-supervised learning",
]

docs = []
seen_titles = set()  # FIX: prevent duplicate pages

for topic in topics:
    try:
        results = wikipedia.search(topic)

        if not results:
            print(f"No result: {topic}")
            continue

        page = wikipedia.page(results[0], auto_suggest=False)

        # FIX: skip if we've already collected this page
        if page.title in seen_titles:
            print(f"Duplicate skipped: {topic} → {page.title}")
            continue

        seen_titles.add(page.title)

        text = page.content[:2000]

        # FIX: store topic name alongside content (was storing bare strings)
        docs.append({"topic": topic, "title": page.title, "content": text})

        print(f"Collected: {topic} → {page.title}")

        time.sleep(1)

    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation: {topic} → trying first option: {e.options[0]}")
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            if page.title not in seen_titles:
                seen_titles.add(page.title)
                docs.append({"topic": topic, "title": page.title, "content": page.content[:2000]})
                print(f"  Collected via disambiguation: {page.title}")
        except Exception as inner_e:
            # FIX: was silently passing — now logs the actual error
            print(f"  Skipped {topic} after disambiguation | Error: {inner_e}")

    except Exception as e:
        print(f"Skipped: {topic} | Error: {e}")

# FIX: was printing `topic` (last loop value) instead of `topics` (the list)
print(f"\nCollected {len(docs)} documents for {len(topics)} topics.")

with open("wiki_knowledge_base.json", "w") as f:
    json.dump(docs, f, indent=2)

print("Saved to wiki_knowledge_base.json")