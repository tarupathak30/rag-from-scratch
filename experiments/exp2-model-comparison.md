
# Experiment 2 — Naive-RAG(Swap `qwen2.5:1.5b` for `qwen2.5:3b`)

## Setup

| Item             | Value               |
| ---------------- | ------------------- |
| Model            | {{qwen2.5:3b}}      |
| Inference Engine | {{ollama}}    |
| Machine          | {{AMD Athlon 3050U, 8GB RAM}} |
| Hardware         | {{CPU only}}       |


---

## Retrieved Context (if RAG)

```text
python generator.py
(venv) PS D:\agentic_ai\rag-from-scratch\naive-rag> python generator.py
Loading FAISS index …
Loading encoder 'all-MiniLM-L6-v2' …
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 1694.34it/s]
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
✓ Retriever ready  (70 vectors)

Q: How does backpropagation work?
A: Backpropagation works by computing the gradient of a loss function with respect to the weights of a neural network. It is an efficient method that computes these gradients layer by layer, moving backward from the output layer towards the input layer. This approach avoids redundant calculations and can be derived through dynamic programming.

The process involves:
1. Forward pass: Propagating inputs through the network to compute outputs.
2. Computing loss: Comparing the computed outputs with the actual target values to determine a loss value.
3. Backward pass (or backward propagation): Computing gradients of the loss function with respect to the weights using the chain rule, starting from the output layer and moving towards the input layer.

During this process, shared weight optimization is used in convolutional neural networks (CNNs) to reduce the number of parameters and connections, thereby mitigating issues like vanishing or exploding gradients. This method helps in training deep neural networks more effectively.
   122.32s | 674pt + 188at = 862 total tokens
   Sources: ['Backpropagation', 'Convolutional neural network', 'Dilution (neural networks)', 'Convolutional neural network', 'Gradient descent']

Q: What is dropout?
A: Dropout is a regularization technique used in artificial neural networks to reduce overfitting during training. It involves randomly setting a fraction of input units to 0 at each update during training time—both during inference and training time. This technique helps prevent complex co-adaptations on the training data by forcing the network to learn more robust features that are useful for generalization.
   78.42s | 651pt + 75at = 726 total tokens
   Sources: ['Dilution (neural networks)', 'Dilution (neural networks)', 'Loss function', 'Batch normalization', 'Diffusion model']

Q: Explain the attention mechanism.
A: The attention mechanism in machine learning and natural language processing (NLP) is designed to determine the importance of each component in a sequence relative to other components within that sequence. In NLP, this importance is represented by "soft" weights assigned to each word in a sentence. These soft weights change with every step of the input, unlike "hard" weights which are computed during the backwards training pass.

The attention mechanism addresses weaknesses found in recurrent neural networks (RNNs), particularly their tendency to favor more recent information at the end of a sequence over earlier parts of the sequence. By allowing each token equal access to any part of the sentence through an attention scheme, it captures global dependencies and allows for better understanding of context.

The attention mechanism was inspired by human cognitive processes where humans can focus on different aspects or details within a larger context. In machine learning models like transformers, this is implemented as parallel attention schemes rather than serial recurrent neural networks. This change in implementation speeds up the model's processing time while maintaining its effectiveness.

Self-attention, which involves each element in the input sequence attending to all others, was central to the Transformer architecture and enabled capturing global dependencies within a sequence. This idea has been applied not only to natural language processing but also to other areas such as computer vision, where diffusion models have combined with text encoders and cross-attention modules for text-conditioned generation.

In summary, the attention mechanism enhances model performance by allowing each token in a sequence to weigh its importance based on context, thereby improving understanding of sequences and facilitating more accurate predictions or classifications.
   123.98s | 630pt + 316at = 946 total tokens
   Sources: ['Attention (machine learning)', 'Attention (machine learning)', 'Diffusion model', 'BERT (language model)', 'Generative pre-trained transformer']

```

---

## Model Output

```text

Retriever ready  (70 vectors)

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
  ROUGE-1: 0.1101  ROUGE-2: 0.0309  ROUGE-L: 0.0911
  Avg latency: 107.7s  (min 91.0s / max 139.7s)
  Tokens — prompt: 4524  answer: 1269  total: 5793
eval_report.json saved

```

---

## Metrics

| Metric                 | Value  |
| ---------------------- | ------ |
| Latency                | 107.7s avg|
| Prompt tokens (approx) | 4524   |
| Answer tokens (approx) | 1269   |
| Total tokens (approx)  | 5793   |
| ROUGE-1                | 0.1101 |
| ROUGE-2                | 0.0309 |
| ROUGE-L                | 0.0911 |


---

## Observations

1. Q: What is dropout?
    A: Dropout is a regularization technique used in artificial neural networks to reduce overfitting during training. It involves randomly setting a fraction of input units to 0 at each update during training time—both during inference and training time. This technique helps prevent complex co-adaptations on the training data by forcing the network to learn more robust features that are useful for generalization.
   78.42s | 651pt + 75at = 726 total tokens
   Sources: ['Dilution (neural networks)', 'Dilution (neural networks)', 'Loss function', 'Batch normalization', 'Diffusion model']
   
   
   In here, the retriever is pulling the sources Dilution(neural networks) for dropout despite being the different title, a thing I observed here.

---

## Key Insight

One line learning is that the bigger model doesn't mean the speed and quality can't be painfully traded.


---

## Next Experiment Ideas

* Appending the EVAL_SET in evaluator.py with vague, complex and out of the corpus queries.

