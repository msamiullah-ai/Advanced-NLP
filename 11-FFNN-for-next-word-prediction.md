# FFNN Language Models In-Depth & the Path to RNNs

---

## Do We Train the Word2Vec Embeddings Further During Gradient Descent?

When using pre-trained Word2Vec embeddings in a downstream neural network, we have to decide whether to **update** the embedding matrix during gradient descent, or keep it **fixed**. We have two choices:

<img width="1384" height="774" alt="image" src="https://github.com/user-attachments/assets/3d08a5b1-74c0-4ff3-874e-c9ec03468201" />


### Option A: Frozen Embeddings

- Lock the embedding matrix and don't backpropagate into it.
- The embeddings stay **exactly** as Word2Vec learned them.
- **Good when**: your downstream training data is small (avoids overfitting) or you want fast training (fewer parameters to update).
- **Trainable params**: `W1, b1, W2, b2` only.

If we freeze the embeddings, the Word2Vec weights remain unchanged — preserving the general linguistic knowledge learned from a massive corpus. Useful when the downstream dataset is small, since it reduces overfitting and speeds up training. The downside: the embeddings cannot adapt to the specific task.

### Option B: Fine-tuned Embeddings

- Allow gradients to flow back into the embedding matrix and update it during training.
- This lets the embeddings **adapt to your specific task**.
- **Good when**: you have a large downstream dataset, or there's domain shift from the original Word2Vec training corpus.
- **Trainable params**: `W, W1, b1, W2, b2` (all of them).

If we fine-tune the embeddings, gradients are allowed to flow back into the embedding matrix, updating word vectors so they can learn task-specific meanings. This can improve performance with enough training data — but with a small dataset, updates may **overwrite** the valuable general knowledge Word2Vec captured, causing the model to lose useful language information. This problem is known as **catastrophic forgetting**.

> 💡 **Why this tradeoff exists**: pre-trained embeddings contain valuable linguistic knowledge learned from huge datasets. When fine-tuning on a small dataset, gradient descent can modify those embeddings too aggressively, causing the model to forget useful general language information. That's precisely why we often freeze embeddings, or carefully control (e.g. rate-limit) their updates.

### Gradient Flow During Training (Backpropagation)

```
W (Embeddings)  ←dL/dW?   W₁, b₁         ←dL/dW1   W₂, b₂         ←dL/dW2   Cross-Entropy Loss
YOUR CHOICE                Always updated              Always updated              L = -log P(w_t | context)
```

- `W1, b1` and `W2, b2` are **always** updated — the choice only applies to whether gradients also flow all the way back into the embedding matrix `W`.

---

## The Complete FFNN Architecture (Bengio-style): 3 Words In, 1 Word Out

**Setup**: `P(w_t | w_{t-3}, w_{t-2}, w_{t-1})` with `|V| = 10,000`, `d = 300`, `h = 512`, concatenated input `3d = 900`.

<img width="1179" height="781" alt="image" src="https://github.com/user-attachments/assets/43b1b3ac-1883-4812-a97b-49de4eebcaa5" />


**Layer by layer:**

1. **Input Layer** — One-hot, `|V| × 1` (×3 words). Example: `w_{t-3}` = "I", `w_{t-2}` = "like".

2. **Embedding Layer** — Shared `W_emb`, size `d × |V|` **each** (used once per input position, but the *same* matrix). No activation.

3. **Concatenate** — `[e_{t-3} ; e_{t-2} ; e_{t-1}]`, giving a `3d × 1 = 900 × 1` vector.

4. **Hidden Layer** — `tanh`/ReLU, size `h × 3d` (with `h = 512`).

5. **Output Layer** — Softmax, `|V| × h`.

> 💡 **Reading the diagram**: `W_emb` is **shared across all 3 inputs** — meaning the exact same embedding matrix is used to look up `e_{t-3}`, `e_{t-2}`, and `e_{t-1}`. Each word's one-hot vector (`x_1...x_10,000`) is multiplied against this same shared matrix to select its corresponding embedding row.

### The Math, Formally

<img width="1128" height="305" alt="image" src="https://github.com/user-attachments/assets/b53342f6-e7fa-44c1-a475-38d6b44511ee" />


```
W_emb (shared): d × |V|     (row lookup, no activation)
e_{t-k} = W_emb[w_{t-k}]    for k = 1, 2, 3

W1: h × 3d = 512 × 900
W2: |V| × h = 10,000 × 512

h = tanh(W1 · x + b1)
ŷ = softmax(W2 · h + b2)
```

**Concatenation:**
```
x = [e_{t-3} ; e_{t-2} ; e_{t-1}]  →  3d × 1 = 900 × 1
```

**Full Model:**
```
P(w_t | w_{t-3}, w_{t-2}, w_{t-1}) = softmax(W2 · tanh(W1 · [e_{t-3};e_{t-2};e_{t-1}] + b1) + b2)
```

> **Concatenation (NOT averaging) preserves word position. No recurrence — this is a fixed 3-word window.**

---

## Why the Embedding Layer Doesn't Count as a "Hidden Layer"

The reason we do **not** count the embedding layer as a hidden layer is because its function is fundamentally different from a neural network hidden layer.

- The embedding layer is technically a **learnable** layer, but it is not considered a hidden layer in the usual FFNN architecture.
- The embedding matrix `E` contains the learned word representations. Its job is only:
  ```
  Word ID → Dense word vector
  ```
- A true hidden layer, by contrast, performs a **computation** (a matrix multiply followed by a non-linear activation) that mixes and transforms information — the embedding layer is a pure lookup, with no such transformation happening.

---

## Position-Specific Weights: What "Shared" Really Means Here

FFNN language models **share the word embedding matrix**, but they do **not** share the parameters that process words according to their position in the context window.

- Each position has its **own weights**, so learning from a word appearing in one position does **not** directly improve how the model processes that word in another position.

> 💡 **In simple words**: the same embedding matrix looks up "dog" the same way whether it's the 1st, 2nd, or 3rd context word — but once concatenated, the hidden layer weights `W1` treat "dog in position 1" completely differently from "dog in position 3," since those are different slices of the 900-dimensional input vector. This is one of the structural weaknesses that later motivates RNNs (see below), which reuse the *same* weights at every time step.

*Reference: [Worked example on shared vs. position-specific weights](https://chatgpt.com/s/t_6a5afeb584b08191a26d2d6cd3e9dadf)*

---

## Choosing the Right Activation Function

> **We use Softmax in the case of probability-making.**
> **We use tanh in the case of scaling things.**

- **Softmax** — used at the output layer, to convert raw scores into a valid probability distribution over the vocabulary.
- **tanh** (or ReLU) — used at hidden layers, to introduce non-linearity and keep values in a bounded, well-scaled range.

---

## Predicting One Word Using One-Word Context

The simplest possible version of this architecture: instead of 3 context words, predict the next word from just **one** context word.

*Reference: [Complete flow for one-word-context prediction](https://chatgpt.com/s/t_6a5b003c32b481919996aaa422c22f9b)*

---

## Limitations of FFNNs & Why We Need RNNs

*Reference: [ChatGPT conversation on FFNN limitations and the motivation for RNNs](https://chatgpt.com/s/t_6a5b06ff6e788191b262d09a532c371c)*

A Feedforward Neural Network is limited for language because it treats text as a **fixed-size** collection of words:

1. It cannot naturally handle sentences of **different lengths** (the 3-word window above is rigid — always exactly 3 words).

2. It does **not share parameters** across different positions in the input window (as discussed above — position 1 and position 3 are learned independently).

3. It has **no memory** of previous words once they're processed — each prediction only sees its fixed window, with no running context beyond it.

### The Path Forward

To model sequential data properly, we need a network that:
- Reads words **one by one**.
- Maintains a **running summary** of previous information.
- Uses the **same parameters** at every time step.

This idea leads directly to **Recurrent Neural Networks (RNNs)**, which introduce a **hidden state** that acts as memory, while sharing weights across the entire sequence — directly solving all three FFNN limitations above.
