







# Experiment {{3}} — {{Tweaked EVAL_SET for evaluation being honest}}

## Setup

| Item             | Value               |
| ---------------- | ------------------- |
| Model            | {{qwen2.5:3b}}      |
| Inference Engine | {{ollama}}    |
| Machine          | {{AMD Athlon 3050U, 8GB RAM}} |
| Hardware         | {{CPU only}}       |




## Model Output

```text
{{  Recall@1: 8/11
  Recall@3: 9/11
  Recall@5: 9/11
  MRR: 0.7841
}}
```

---

## Metrics

| Metric                 | Value  |
| ---------------------- | ------ |
| Latency                | 102.8s avg|
| Prompt tokens (approx) | 7052   |
| Answer tokens (approx) | 1693   |
| Total tokens (approx)  | 8745   |
| ROUGE-1                | 0.1240 |
| ROUGE-2                | 0.0338 |
| ROUGE-L                | 0.1032 |

---

## Observations

1. {{In this query, the model straight away denied giving any output, the case where the query is not present in the corpus, {
      "query": "What is quantum computing?",
      "answer": "I don't know. The provided context does not contain any information about quantum computing.",
      "latency_s": 72.27,
      "rouge1": 0.1818,
      "prompt_tokens": 602,
      "answer_tokens": 18,
      "total_tokens": 620
    }}}
2. {{This {
      "query": "What technique prevents overfitting by randomly disabling neurons?",
      "answer": "Dropout is a regularization technique for reducing overfitting in artificial neural networks by preventing complex co-adaptations on training data. It randomly sets a fraction of input units to 0 during each update of the model, which helps prevent the network from relying too heavily on any single neuron or feature.",
      "latency_s": 96.18,
      "rouge1": 0.2333,
      "prompt_tokens": 668,
      "answer_tokens": 61,
      "total_tokens": 729
    } and {
      "query": "What is dropout and why is it used?",
      "answer": "Dropout is a regularization technique used in artificial neural networks to prevent overfitting during the training process. It works by randomly setting a fraction of input units to 0 at each update during training time; this technique causes a drop in output activations from those units, which helps to reduce co-adaptations between neurons. This method forces the network to learn more robust features that are not overly dependent on any single input unit.\n\nDropout is used because it can help improve the generalization of neural networks by reducing their reliance on specific inputs or hidden units during training. By randomly dropping out a fraction of these units, dropout encourages the network to become less sensitive to any particular feature and more robust in its learning process. This technique helps prevent overfitting, which is when a model learns the noise in the training data instead of the underlying pattern, leading to poor performance on new, unseen data.",
      "latency_s": 101.15,
      "rouge1": 0.0843,
      "prompt_tokens": 661,
      "answer_tokens": 180,
      "total_tokens": 841
    } rouge score seems to be really different, when one query is direct(with low rouge score) and the other is paraphrased(with high rouge score)}}
3. {{In this query, How does gradient descent use backpropagation?, the score turns out to be 0.067. The query confuses the model.}}
4. {{In this the model said everything it knew, instead of being concise, instead of following prompt instructions. {
      "query": "Explain the attention mechanism.",
      "answer": "The attention mechanism in machine learning and natural language processing (NLP) is designed to determine the importance of each component in a sequence relative to other components within that sequence. In NLP, this importance is represented by \"soft\" weights assigned to each word in a sentence. These soft weights are computed during the forward pass through the neural network and can change with every step of the input.\n\nThe attention mechanism addresses some weaknesses found in recurrent neural networks (RNNs), particularly their tendency to favor more recent information at the end of a sequence over earlier parts, which can lead to loss of context. Attention allows each token equal access to any part of the sentence directly, rather than only through previous states. This is achieved by encoding vectors called token embeddings across a fixed-width sequence that can range from tens to millions of tokens in size.\n\nPreviously, attention was implemented within a serial recurrent neural network (RNN) language translation system. However, more recent designs, such as the transformer architecture, have removed this slower sequential RNN and relied more heavily on faster parallel attention schemes. The transformer architecture replaced recurrence with attention mechanisms, enabling models like BERT, T5, and GPT to capture global dependencies by allowing each element in the input sequence to attend to all others.\n\nThe modern era of machine attention was revitalized by grafting an attention mechanism into an Encoder-Decoder framework. This integration allowed for more efficient processing and improved performance compared to traditional RNN-based models.",
      "latency_s": 122.08,
      "rouge1": 0.038,
      "prompt_tokens": 630,
      "answer_tokens": 295,
      "total_tokens": 925
    },}}
---

## Key Insight

{{Adding just 4 stress test queries dropped MRR from 1.0 to 0.784. 
A good eval set should include some queries designed to break the system so that we can know the real limits.}}


---

## Next Experiment Ideas

* {{idea_1}}
* {{idea_2}}
* {{idea_3}}
