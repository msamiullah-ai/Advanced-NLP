# Transformer Architecture: Complete Flow (End-to-End Summary)

*A condensed, start-to-finish walkthrough tying together the detailed Transformer notes into one continuous narrative.*

*Reference: [Gemini version of this walkthrough](https://chatgpt.com/s/t_6a5faa07ff80819185f7fbaff6d4d01a)*

---

## The Big Picture

The Transformer is an **Encoder–Decoder** architecture designed to transform an input sequence (e.g., an English sentence) into an output sequence (e.g., its Urdu translation).

- Unlike RNNs, it does **not** process words one by one.
- The **encoder** processes the entire input sentence **in parallel**.
- The **decoder** generates the output sentence **one word at a time**.

---

## Stage 1: Inside the Encoder — From Words to Position-Aware Vectors

The process begins in the encoder:

1. Each input word is first represented as a **one-hot vector**.

2. The **Embedding Lookup** layer converts these one-hot vectors into dense word embeddings, capturing the semantic meaning of the words.

3. Since the Transformer has no inherent notion of sequence order, **Positional Encodings** are added to the embeddings — so each word's representation contains both its **meaning** and its **position** in the sentence.

---

## Stage 2: Encoder Self-Attention

These position-aware vectors are then passed into the encoder's **Self-Attention** mechanism.

1. Each vector is multiplied by three learned weight matrices, producing **Query (Q)**, **Key (K)**, and **Value (V)** vectors.

2. For every input word, its **Query** vector is compared against the **Key** vectors of all input words, computing similarity scores.

3. After applying **Softmax**, these scores become attention weights, which are multiplied with the corresponding **Value** vectors.

4. The weighted Value vectors are **summed**, producing a context-aware representation of each word — letting every word gather information from the entire sentence.

5. A **Residual Connection** adds the original input representation back to the attention output, and **Layer Normalization** stabilizes the values.

**Result**: the final encoder outputs contain **word meaning**, **positional information**, and **contextual relationships** among all words — all fused into one representation per word.

---

## Stage 3: Inside the Decoder — Masked Self-Attention

These final encoder outputs are passed to the decoder, whose task is to generate the target sentence **one word at a time** (auto-regressively).

1. The decoder begins with the `<START>` token (or previously generated target words, during training).

2. Like the encoder, these words are converted into embeddings, combined with positional encodings, and passed through **Masked Self-Attention**.

3. The attention computation is **identical to the encoder**, except that **future target words are masked** — so each word can attend only to itself and previously generated words.

> 💡 **Why this matters**: this masking is precisely what ensures the decoder cannot "look ahead" while generating the sentence — without it, the model could cheat by peeking at the answer it's supposed to be predicting.

---

## Stage 4: Encoder–Decoder Attention (Cross-Attention)

The output of masked self-attention is then passed to **Encoder–Decoder Attention (Cross-Attention)**.

- Here, the **Query** comes from the **decoder**.
- The **Keys and Values** come from the **encoder outputs**.

This lets the decoder determine which parts of the input sentence are most relevant for predicting the current output word. As before, the attention output is followed by a **Residual Connection** and **Layer Normalization**.

---

## Stage 5: Final Prediction

1. The decoder output is passed through a **Linear layer**, converting the decoder representation into scores for every word in the target vocabulary.

2. A **Softmax** layer transforms these scores into probabilities, and the word with the **highest probability** is selected as the next output word.

3. This predicted word is fed back into the decoder as input for the **next** time step — and the process repeats until the model generates the `<EOS>` (End-of-Sequence) token.

---

## Putting It All Together

The **encoder** reads and understands the entire input sentence in parallel, producing rich contextual representations of every input word.

The **decoder** uses these representations — together with the words it has already generated — to predict the next target word, one step at a time.

Through **embeddings**, **positional encodings**, **self-attention**, **masked self-attention**, **encoder–decoder attention**, **residual connections**, and **layer normalization**, the Transformer can efficiently capture long-range dependencies and generate accurate output sequences — all without relying on recurrent networks.
