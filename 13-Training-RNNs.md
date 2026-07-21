# Training RNNs: BPTT, Teacher Forcing & Scheduled Sampling

---

## Reference Materials

*Slides (complete through before "Sequence Models Training Dynamics"): [Slide deck](https://pern-my.sharepoint.com/:b:/g/personal/namoos_qasmi_lums_edu_pk/IQAaOjeQvHGBQbVQALu3dt6JAeyCLt4aNFOUPWJ2sUyoYZU?e=Yt2pdo)*

*Explanation: [Gemini conversation](https://share.gemini.google/kmVq21pXT4sx)*

---

## Backpropagation Through Time (BPTT)

The **total error** at any hidden state is a combination of the mistake it made at its own time step, **combined with all the mistakes it caused in future steps** — because that hidden state's output fed forward and influenced everything downstream.

> 💡 **What we're actually computing**: at every stage of this process, we're interested in the derivative of the **loss's partial derivative with respect to the contribution we want to inspect** — i.e., how much did this specific weight or hidden state actually contribute to the final error, so we know how to adjust it.

### Full Derivations

- [Derivation: BPTT for `W_hh`](https://chatgpt.com/s/t_6a5c465ee7a481918696373ab8cca731) — the hidden-to-hidden weight matrix.
- [Derivation: BPTT for `W_xh`](https://chatgpt.com/s/t_6a5c481d904c8191b6deea5c972fe56c) — the input-to-hidden weight matrix.

---

## Truncated BPTT — The Practical, Real-World Fix

Truncated BPTT is the practical modification used to fix the massive computational problems of standard (full) BPTT.

### The Problem With Full BPTT

In full BPTT, the error signal has to travel all the way back to the very first step (`t = 1`). For a long sequence (an entire book, a long audio clip with thousands of steps), this causes three serious problems:

1. **Memory explodes** — the computer has to store every single hidden state vector in RAM for thousands of steps, just to do one backward pass.

2. **Speed slows down** — calculating those massive chains of derivatives takes forever.

3. **Vanishing/exploding gradients** — the longer the chain, the more likely the gradient shrinks toward zero or blows up.

### The Solution: Truncated BPTT

Instead of backpropagating across the entire history, Truncated BPTT breaks the sequence into **manageable chunks of length K** (typically 20 to 35 steps).

**The step-by-step training loop for a single chunk:**

1. **Chunk Forward Pass** — the network runs a forward pass for just `K` steps (e.g., steps 1 to 35), calculating and saving those 35 hidden states.

2. **Calculate Chunk Loss** — computes the total loss strictly over those `K` steps.

3. **Truncated Backward Pass** — the error signal flows backward through time, but stops completely after `K` steps. It does **not** go all the way back to the beginning of the actual document (`t = 1`).

4. **Update Shared Weights** — the accumulated gradients from this chunk are used to update the shared weight matrices (`W_xh`, `W_hh`, `W_hy`).

5. **Pass the Memory Forward** — the very last hidden state vector (`h_K`) from this chunk is saved and used as the **starting hidden state** for the next chunk. The network then moves to the next 35 steps and repeats.

> 💡 **The key mechanic to remember**: the *weights* are updated per-chunk (using only that chunk's local gradients), but the *hidden state memory itself* carries forward continuously across chunk boundaries. So the model doesn't fully "forget" everything between chunks — it just stops computing gradients that reach back that far.

### The Crucial Trade-off

**The Good:**
- Training becomes extremely fast.
- Memory usage stays low and constant.
- Gradients don't easily vanish (since the chain is short).

**The Bad:**
- Because the error signal is cut off after `K` steps, the network **cannot learn dependencies longer than `K` steps**.
- **Example**: if `K = 35`, and a word at step 100 relies on context provided at step 20, the network will completely miss that connection — the error signal simply cannot travel back that far.

---

## Teacher Forcing

**Teacher forcing** is a training technique used in sequence-to-sequence models where, after predicting a word, the decoder is **not** fed its own prediction as the next input. Instead, it's given the **correct previous word (ground truth)** from the training data.

- This prevents early prediction errors from propagating through the rest of the sequence, letting the model learn each prediction under the correct context.
- **During inference**, however, the ground-truth sequence is unavailable — so the decoder must feed its own predicted word back as the next input, making generation **autoregressive**.

> ⚠️ This mismatch between training (always sees ground truth) and inference (only ever sees its own predictions) is known as **exposure bias**.

### The Cost of Skipping Teacher Forcing

- It is possible to train a good model **without** teacher forcing, and it may lead to interesting results, because the model learns to handle its own mistakes directly.
- However, this comes with real costs: **convergence at the beginning of training will be slow**, leading to a longer overall training time — the model has to first learn to produce *anything* reasonable before it can learn from its own imperfect outputs.

---

## Scheduled Sampling

Teacher Forcing always feeds the correct previous word to the decoder during training — stable, but it creates **exposure bias**, since the model never practices handling its own mistakes.

**Scheduled Sampling** fixes this by **gradually replacing** the correct previous word with the model's own prediction, according to a probability `ε` (epsilon).

*Reference: [ChatGPT conversation on Scheduled Sampling](https://chatgpt.com/s/t_6a5c544d7be88191bf102496b77d7821)*

**How the schedule works:**

- At the **beginning** of training, `ε` is **high**, so the model mostly receives ground-truth tokens (behaving like standard teacher forcing).
- As training progresses, `ε` **decreases** according to a schedule (linear, exponential, or inverse sigmoid), forcing the decoder to increasingly rely on its own predictions.
- By the **end** of training, the model behaves much like it will during inference — making it better at recovering from its own mistakes, and reducing error accumulation.

> 💡 **In simple words**: Scheduled Sampling is a gradual "training wheels come off" strategy — start by holding the model's hand with correct answers, then slowly force it to walk on its own, so the training experience ends up matching real-world inference conditions.
