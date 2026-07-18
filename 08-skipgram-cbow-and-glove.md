# Word2Vec Training Mechanics & GloVe

---

## Training Data Generation: The Sliding Window

To train Word2Vec, we first create a training dataset from the corpus using a **sliding window**.

- We choose a window size `c`, which determines how many neighboring words on the left and right are considered the context of a word.

- The total window size is therefore `2c + 1`: `c` words to the left, 1 center word, and `c` words to the right.

- As this window moves across the corpus **one word at a time**, each word becomes the center word once, and its surrounding words become the context words.

- For every center–context pair, we create **one training example**.

> 💡 **"The window continues sliding until every word has been used as the center word"** simply means: the window moves one word at a time across the entire corpus, so every word gets a turn generating training pairs with its surrounding context words.

The sliding window converts raw text into thousands or millions of `(center word, context word)` training pairs, which are then used to train the Word2Vec neural network.

**Skip-gram vs. CBOW — how the window is used:**

| Model | Direction | Training examples per window |
|---|---|---|
| **Skip-gram** | Center → Each Context | Many training pairs (one per context word) |
| **CBOW** | All Context → Center | One training example per window |

---

## Subsampling of Frequent Words

*Reference: [ChatGPT conversation on subsampling](https://chatgpt.com/s/t_6a5875f295b4819187af15b93edf7fe8)*

**Subsampling** speeds up Word2Vec and improves embedding quality by **randomly removing** many occurrences of very frequent words (like "the," "is," "of") while keeping rare, informative words.

- This prevents common words from dominating the training process and gives more learning emphasis to meaningful words.

> 💡 A **stop word** is simply a very common function word that appears extremely frequently in a language but carries little lexical (content) meaning by itself — these are exactly the kind of words subsampling is designed to downweight.

---

## What Do We Keep After Training?

The neural network used in Word2Vec is only a **tool** for learning word embeddings — it is not the end product.

**❌ We throw away:**
- The prediction task itself.
- The output layer.
- The softmax classifier.
- The neural network architecture used for prediction.

**✅ We keep:**
- The learned **embedding matrix** (the weights connecting the input layer to the hidden layer).

**During training:**
```
         One-hot word
               │
               ▼
      Embedding Matrix (W)
               │
        Hidden Layer (Embedding)
               │
               ▼
        Output Layer (W′)
               │
               ▼
      Predict context word
```

**After training:**
```
         One-hot word
               │
               ▼
      Embedding Matrix (W)
               │
               ▼
      Word Embedding Vector
```

Everything below the embedding layer is discarded.

> 💡 **Why can we discard the rest?** The neural network's only job was to adjust the embedding matrix so it became good at predicting context words. Once that job is done, we no longer care about predicting context words — we only want the learned vectors.
>
> **Analogy**: It's like training a student. The exercises and exams (the prediction task) are just for learning. After graduation, you don't keep the exams — you keep the knowledge the student gained.

---

## Skip-gram (Full Softmax Version)

The original Skip-gram model is a single-hidden-layer neural network trained using **full Softmax** to predict a context word from a center word.

### Why There's No Activation Function

Unlike traditional neural networks, the hidden layer has **no activation function** — because it isn't performing feature extraction.

- The input is a one-hot vector. Multiplying it by the embedding matrix `W` simply **selects the corresponding row** of `W` — the embedding of the input word.
- Thus, the hidden layer acts as a **lookup table**, not a computational layer.

> 💡 Applying an activation function like ReLU, Sigmoid, or Tanh would unnecessarily **distort** the learned embedding by restricting or modifying its values. Skip-gram leaves the embedding unchanged so the optimizer can freely learn both positive and negative values, letting words be placed anywhere in the embedding space.

**The deeper reason**: Word2Vec aims to learn a **linear** embedding space, in which semantic relationships can be represented through simple vector operations — addition, subtraction, dot products, cosine similarity. This is *why* analogies like `king − man + woman ≈ queen` work: the vectors preserve meaningful directions and distances.

- Introducing a non-linear activation function would **warp the geometry** of the embedding space, distorting these relationships.
- As a result, vector arithmetic would no longer correspond cleanly to semantic relationships, and the dot product used by Softmax to measure similarity between center and context embeddings would become less meaningful.

### Complete Flow of Skip-gram (Full Softmax)

1. The input is the **one-hot vector** of the center word — only the center word's position is `1`, all other entries are `0`.

2. This one-hot vector is multiplied by the randomly-initialized **input embedding matrix `W`**, which simply selects the corresponding row of `W` (at the exact position where the `1` is in the one-hot vector).

3. This selected row becomes the **hidden layer output** (the current embedding of the center word) — no activation function is applied.

4. This embedding vector is then multiplied by the **output embedding matrix `W′`**, producing a score vector with one score for every vocabulary word.

5. The entire score vector is passed through **Softmax**, converting it into a probability distribution over the vocabulary.

6. The word with the highest probability is the predicted context word.

7. During training, the prediction is compared with the true context word using a loss function, and **backpropagation** updates both `W` and `W′` so future predictions become more accurate.

### Reference Links for Skip-gram (Full Softmax)

- [Complete Skip-gram + Full Softmax explanation](https://chatgpt.com/s/t_6a58f568cc3c8191a7d66a3759227fe3) ⭐ recommended
- [If you only read one link — read this one](https://chatgpt.com/s/t_6a58f791c4548191ac9c48bdae8d5771) ⭐ most recommended
- [Skip-gram's embedding matrix dimensions](https://chatgpt.com/s/t_6a58fb6e1e2c81918c6096ae4fe48c33)
- [Mathematical formulation](https://chatgpt.com/s/t_6a5886b08ad08191a16557337e789b3d)
- [The best full-chat version to address confusion](https://chatgpt.com/share/6a59802d-9ee0-83e8-bda7-322eb0b8b7c2)
- [Computational complexity](https://chatgpt.com/s/t_6a5886e15b0c819182dacd9402850ee6)
- [Entire Skip-gram algorithm — fully worked example](https://chatgpt.com/s/t_6a5900320bf48191940fc02426a93eb5)

> 💡 **Important caveat**: Full Softmax Skip-gram is **never used in practice** for real training — it's always replaced by Negative Sampling or Hierarchical Softmax (see below), because computing Softmax over the entire vocabulary is far too expensive at scale.
>
> "**Full**" in Full Softmax means the model computes probabilities over the **entire vocabulary** — a `|V|`-way classification problem. Negative Sampling replaces this expensive Softmax with multiple Sigmoid-based binary classification tasks, making training much faster.

---

## CBOW (Full Softmax Version)

In CBOW, the input is **not** a single center word — instead, the input consists of **all the surrounding context words** within the chosen window.

### The Flow

1. Each context word is first represented as a **one-hot vector**.

2. All of these one-hot vectors use the **same** input embedding matrix `W` (just one shared embedding matrix, exactly like in Skip-gram).

3. Multiplying each one-hot vector by `W` simply selects the corresponding row of `W`, producing one `d`-dimensional embedding vector **per context word**.

4. The embedding vectors of all context words are combined into a single hidden vector by **averaging** (or summing) them. This hidden vector is still `d`-dimensional and represents the combined meaning of the surrounding context.

5. **No activation function** is applied at the hidden layer, so the embedding space stays linear.

6. The hidden vector is multiplied by the output embedding matrix `W′` to produce a score vector — one score per vocabulary word.

7. These scores pass through **Full Softmax**, converting them into a probability distribution over the entire vocabulary.

8. The word with the highest probability is predicted as the **center (target) word**.

9. During training, predicted probabilities are compared with the true center word, loss is computed, and backpropagation updates both `W` and `W′`.

### Clearing Up the "Confusion of Neurons"

*Reference: [ChatGPT conversation clearing up neuron confusion](https://chatgpt.com/s/t_6a59845a2e488191b4d2db60fbe238f3)*

> 💡 **Worked example**: Suppose there are 3 context words, so we have 3 one-hot vectors. Each is independently multiplied by the same embedding matrix `W` (size `V × d`, where `V` is the vocabulary size and `d` is the embedding dimension).
>
> During each multiplication `xᵀW`, every hidden neuron (i.e., every column of `W`) computes one dot product with the one-hot vector. Since the input is one-hot, each dot product simply **picks the weight** at the position of the `1`. So the outputs of all hidden neurons together form the embedding vector for that word — exactly the corresponding row of `W`.
>
> This repeats for all 3 one-hot vectors, producing 3 embedding vectors. Finally, CBOW **averages** these 3 embedding vectors to obtain a single hidden vector `h`, which is passed to the output layer.

### Reference Links for CBOW

- [Mathematical formulation](https://chatgpt.com/s/t_6a59115f68908191bb9346517cc6fe2e)
- [Difference between CBOW and Skip-gram — math forms](https://chatgpt.com/s/t_6a59126136a4819186c2dad68c9a6419)
- [Entire CBOW algorithm — fully worked example](https://chatgpt.com/s/t_6a5914c326688191b1b82856937cf5c1)

---

## Negative Sampling (SGNS — Skip-Gram with Negative Sampling)

In real-world Word2Vec applications, **Negative Sampling** is predominantly used over full Softmax.

### Reframing the Prediction Problem

| Approach | The question it asks | Problem type |
|---|---|---|
| **Full Softmax** | *"Out of all `|V|` words in the vocabulary, which one is the context word?"* | A `|V|`-way classification problem |
| **Negative Sampling** | *"Is this particular (center, context) pair a REAL pair from the corpus, or a FAKE pair I generated randomly?"* | A binary (2-class) classification problem |

This transforms an expensive `|V|`-class problem into a cheap 2-class problem — the core reason negative sampling scales so much better.

### How a Training Example Is Prepared

Before each SGNS training iteration, the **training algorithm** (not the neural network itself) first prepares the training example:

1. It scans the corpus to choose a center word and one of its **true** context words, forming the **positive pair** `(center, context)` with label `1`.

2. It then randomly samples `k` **negative** words from the vocabulary using a noise distribution, creating `k` **fake pairs** `(center, negative)`, each with label `0`.

3. Once these positive and negative pairs are prepared, the center word alone is converted into a one-hot vector and fed into the network to retrieve its embedding from the input matrix `W`.

4. The already-selected positive and negative context words are later used at the output layer to retrieve their corresponding output embeddings from `W′`.

### Reference Links for Negative Sampling

- [Complete forward and backward flow of one SGNS training iteration](https://chatgpt.com/s/t_6a599ec3e4dc819188235351036e20b3)
- [Worked example: Skip-gram with Negative Sampling](https://chatgpt.com/s/t_6a592bab394c8191b2f02c643561c168)
- [Mathematical formulation](https://chatgpt.com/s/t_6a592cb2105081918ad8fd661c24bad6)

---

## GloVe (Global Vectors for Word Representation)

*Reference: [Core concepts of GloVe](https://chatgpt.com/s/t_6a59a546177881919f7ed810f25887e4)*

GloVe creates word embeddings by first **counting** how often every pair of words appears together across the entire corpus. It stores these counts in a large **co-occurrence matrix**, then compresses that matrix into small, dense word vectors (embeddings) using **matrix factorization**.

> 💡 **Key contrast with Word2Vec**: GloVe is a **count-based** algorithm that learns from **global** corpus statistics, whereas Word2Vec is a **prediction-based** algorithm that learns from individual local training pairs generated by a neural network.

### Complete Training Flow

1. Before training begins, the corpus is scanned **once** to construct a large word-word co-occurrence matrix, where each entry `X_ij` records how many times word `j` appears in the context window of word `i`.

2. Each **non-zero** entry of this matrix becomes one training example.

3. During one training iteration, the model takes a pair `(w_i, w_j)` together with its co-occurrence count `X_ij`, retrieves the word embedding of `w_i` from the word embedding matrix `W`, and the context embedding of `w_j` from the context embedding matrix `W̃`, then computes their **dot product**.

   > 💡 Unlike Word2Vec, there is **no** input one-hot vector, no hidden layer performing an embedding lookup, no output neurons, no softmax, and no sigmoid — GloVe's mechanics are fundamentally different from the neural prediction approach.

4. The dot product, together with learned bias terms, is trained to approximate the **logarithm of the co-occurrence count**, `log(X_ij)`.

5. The error between the predicted value and the target value is minimized using **gradient descent**, updating only the corresponding word and context embeddings.

6. After training on all co-occurring word pairs, words with similar global co-occurrence patterns end up with similar embedding vectors — capturing both semantic similarity and meaningful analogical relationships.

### Additional References

- [Complete working example](https://chatgpt.com/s/t_6a59a87e2b748191b2570a14fc0e3ba1)
- [Mathematics of GloVe](https://chatgpt.com/s/t_6a59bcb70d148191bf0d6a5f4b619556)
- [Detailed example: building X from scratch](https://chatgpt.com/s/t_6a59c048580c8191a872821988c1048d)

### GloVe Summary

GloVe is a **count-based** word embedding algorithm that learns dense vector representations by analyzing global word co-occurrence statistics across the entire corpus. It first constructs a word-word co-occurrence matrix, then learns embeddings so that the dot product of two word vectors approximates the logarithm of their co-occurrence count. The model is trained by minimizing a **weighted least-squares loss**, where a weighting function reduces the influence of both very rare and extremely frequent word pairs.

### Strengths of GloVe

- Uses **global corpus statistics** — captures information from the entire dataset, not just local context windows.

- Produces high-quality semantic embeddings, where similar words are located close together in embedding space.

- Captures many word analogies (e.g., `King − Man + Woman ≈ Queen`).

- More mathematically **interpretable**, since it directly models co-occurrence statistics.

- Efficient training once the co-occurrence matrix is built, since it optimizes only over observed (non-zero) word pairs.

- Distance-weighted co-occurrence counting gives greater importance to nearby context words.

### Weaknesses of GloVe

- Requires building and storing a large co-occurrence matrix — can consume substantial memory for very large vocabularies.

- Preprocessing is expensive: the entire corpus must be scanned before training to construct the co-occurrence matrix.

- Cannot easily incorporate new words — adding new data usually requires rebuilding the co-occurrence matrix and retraining.

- Learns **one fixed embedding per word**, so it cannot distinguish different meanings of polysemous words (e.g., "bank" as a financial institution vs. a river bank).

- Less suitable for streaming or continuously growing datasets, since it relies on complete global statistics.

### One-Line Comparison with Word2Vec

| Model | Approach |
|---|---|
| **Word2Vec (SGNS)** | Learns from local context windows using a neural network and negative sampling |
| **GloVe** | Learns from global co-occurrence statistics using matrix-based optimization |

*Reference: [Comparison of all static embeddings](https://chatgpt.com/s/t_6a59c80f34f48191a6fbb09a29bb4a09)*

---

> This wraps up everything on **static embeddings** (Word2Vec and GloVe) — vector representations where each word gets exactly one fixed embedding, regardless of context.
