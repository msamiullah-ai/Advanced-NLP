# LoRA: Low-Rank Adaptation

---

## LoRA vs. Previous PEFT Methods

**All previous PEFT methods** (Prompt Tuning, Prefix Tuning, Adapter Tuning) add **new trainable components around** the transformer.

**LoRA** takes a different approach: it keeps the architecture **almost unchanged**, and instead learns efficient updates to the transformer's **existing** linear transformations.

---

## The Motivation

*Reference: [ChatGPT conversation on LoRA's motivation](https://chatgpt.com/s/t_6a6a48f7ffc08191b191f41ef2d160fa)*

LoRA is based on the empirical observation that **the changes learned during full fine-tuning lie in a low-dimensional subspace** — so the large weight update matrix `ΔW` can be accurately approximated by a **low-rank matrix**, instead of learning every weight independently.

### The Intuition: Matrices Can Contain Redundant Information

- Matrices that contain redundant information can be represented as a **low-rank product** of other matrices.
- Matrices are made of numbers derived from words, and words have deep relationships with other words — so, by extension, matrices have connections with other matrices (i.e., their rows/columns aren't all independent).

**Key ideas:**
- Matrices can contain some level of "duplicate information," in the form of **linear dependence**.
- We can exploit this via **factorization**, to represent a large matrix in terms of two smaller matrices.
- Similarly to how a large number can be represented as the multiplication of two smaller numbers, a matrix can be thought of as the multiplication of two smaller matrices.

*Reference: [High-level overview of LoRA](https://chatgpt.com/s/t_6a6aac6d17748191ac1be97153ed024b)*

---

## The Core Math

- `W` comes from the **pre-trained model**.
- `A` and `B` are **newly introduced trainable parameters**, created by LoRA.
- During training, **only `A` and `B` learn**; together, they produce the update `BA`, which is added to the frozen `W`.

**Starting point:**
```
W                          ← original pre-trained weight matrix
```

**During full fine-tuning, it would become:**
```
W′                         ← fully updated weights
```

**LoRA does not update `W`.** Instead, it learns two small matrices `A` and `B`, whose product approximates the weight update:
```
ΔW = BA
```

**Therefore, the effective weight matrix used during the forward pass is:**
```
W′ = W + BA
```
where:
- `W` = original pre-trained weights (**frozen**)
- `BA` = learned low-rank update (**trainable**)
- `W′` = effective weight used to compute the output

---

## Matrix Factorization, In Depth

LoRA is based on the idea that a large matrix doesn't always need to be stored directly. If the matrix contains a lot of redundant (duplicate) information, it can often be represented as the product of two much smaller matrices — this process is called **matrix factorization**.

**The rank of a matrix** is the number of linearly independent rows (or, equivalently, columns) — it represents the **true dimensionality** of the transformation.

### Shapes and Parameter Count

- Matrix `B` has shape `(d × r)`.
- Matrix `A` has shape `(r × d)`.
- The rank `r` is a **hyperparameter** you choose.

**Number of trainable parameters for LoRA:**
```
Storage (B + A) = (d × r) + (r × d) = 2 · d · r
```

> ⚠️ **LoRA only saves parameter space when `r < d/2`** — below this threshold, the combined size of `A` and `B` is smaller than the original `d × d` matrix; above it, you'd actually be using *more* storage than the original weight matrix.

> 💡 **The surprising empirical result**: even though you're fine-tuning less than 1% of the original matrix's parameter capacity, real-world experiments show the model performs **just as well** on downstream tasks as if you had updated 100% of the full matrix.

---

## Worked Example: The Base Transformer (Vaswani et al., 2017)

Using the base Transformer model:
- Transformer weight dimensions: `d × k = 512 × 64`
- So: `512 × 64 = 32,768` trainable parameters (in a full fine-tuning scenario for this one matrix).

**With LoRA at rank `r = 8`:**
- `A` has dimensions `r × k = 8 × 64 = 512` parameters.
- `B` has dimensions `d × r = 512 × 8 = 4,096` parameters.

**Total LoRA parameters**: `512 + 4,096 = 4,608`.

### Key Result

**Comparing 4,608 to the original 32,768 yields nearly an 86% parameter savings:**
```
1 − (4,608 / 32,768) ≈ 85.94%
```

> **86% reduction in parameters to train**, for this single weight matrix — and this saving compounds across every matrix LoRA is applied to throughout the model.

---

## We Only Care About `ΔW`, Not `W` Itself

> **We do not care whether the original weight matrix `W` is low-rank or full-rank. We only care whether the *change* `ΔW` is low-rank.**

The task-specific update needed to adapt a pre-trained model lies in a low-dimensional subspace, so `ΔW` can be represented as a low-rank matrix:
```
ΔW = BA
ΔW = W_finetuned − W_pretrained
```

> 💡 **Pre-trained matrices are high-rank, but the *changes* to them are low-rank.** This is the crucial distinction — `W` itself encodes vast, complex pretrained knowledge (genuinely high-dimensional), but the specific *adjustment* needed to specialize it for a new task turns out to be a much simpler, lower-dimensional operation.

### How This Was Discovered (Research Process)

**Before LoRA (the research that motivated it):**
```
Full Fine-tuning
        │
        ▼
ΔW = W_finetuned − W_pretrained
        │
        ▼
Run SVD (Singular Value Decomposition)
        │
        ▼
Observe ΔW is low-rank
```

**During LoRA training (applying the insight):**
```
Choose rank r (e.g., 8)
        │
        ▼
Create A and B
        │
        ▼
Train A and B only
        │
        ▼
ΔW = BA
```

### The SVD Connection

*Reference: [SVD connection to LoRA](https://chatgpt.com/s/t_6a6acbc921e0819185bb7bafca21a657)*

> 💡 **The singular values are the empirical proof that fine-tuning adaptations are low-dimensional** — when researchers ran SVD on `ΔW` matrices from full fine-tuning, they found that only a small number of singular values were significant, with the rest being negligibly small. This is the direct mathematical evidence that justified approximating `ΔW` with a low-rank product in the first place.

---

## Why One of the LoRA Matrices Must Be Initialized to Zero

*Reference: [ChatGPT conversation on zero initialization](https://chatgpt.com/s/t_6a6ad195032881919f23630fd911b514)*

Initializing **one** matrix (either `A` or `B`) to zero ensures that at the start of training, the product `A×B` is zero.

- This means the model **initially relies entirely on the frozen pre-trained weights** (`W`) — preventing the model's output from being corrupted by random initializations.
- If both matrices were initialized randomly, the model would start with a significant deviation from its pre-trained state. It would then waste time "recovering" or "training back" to the original performance level, before it could start effectively learning the downstream task.

### Why Not Initialize *Both* to Zero?

During backpropagation, the derivative with respect to `B` includes `A`, and the derivative with respect to `A` includes `B`.

- Therefore, if **both** are initialized to zero, there's **no gradient flow through either matrix** — because the partial derivative for each depends on the *other* matrix being non-zero.
- If **only one** is set to zero, the other remains non-zero, allowing gradients to flow, and training to proceed normally.

> 💡 **In simple words**: zero-initializing one matrix gives you a safe, no-op starting point (the model behaves exactly like the original pretrained model on day one), while zero-initializing both would freeze the whole update mechanism before it even starts — a subtle but critical distinction between "start from a safe baseline" and "start from a dead end."

**The underlying principle**: `ΔW` is approximately low-rank, so LoRA keeps only the important patterns by learning a low-rank update `BA`, instead of a full, unconstrained matrix.

---

## Computational Efficiency in the Forward Pass

*Reference: [ChatGPT conversation on computational efficiency](https://chatgpt.com/s/t_6a6ad454ee9481919c948970ee41b8d5)*

---

## Which Matrices Get LoRA?

LoRA can be applied to **any** trainable linear layer in a transformer, but the most common choices are the attention projection matrices — `W_Q` (Query), `W_K` (Key), `W_V` (Value), `W_O` (Output) — and the Feed-Forward Network (FFN) layers.

### The Original Paper's Finding

In the original LoRA paper, researchers experimented with adding LoRA to all four attention matrices, and found that applying it **only to `W_Q` and `W_V`** achieved **most** of the performance improvement.

> 💡 **The intuition**: `W_Q` determines *what information each token should attend to*, while `W_V` determines *what information is actually passed through* the attention mechanism. Since these two matrices have the greatest influence on the model's behavior, adapting them is often sufficient — while `W_K` and `W_O` play more of a supporting role in the mechanics of comparison and recombination.

**However**, later research showed that applying LoRA to **all** attention matrices (`W_Q, W_K, W_V, W_O`) — and even the FFN layers — can further improve performance, especially for tasks that require **larger behavioral changes** from the pre-trained model.

---

## The Road to Quantization

Although LoRA drastically reduces the number of **trainable** parameters, it does **not** reduce the memory required to **store** the original pre-trained weight matrices — which can still occupy hundreds of gigabytes for very large LLMs.

**Quantization** addresses this problem by reducing the **precision** used to represent these frozen weights.

- **Key observation**: transformer weights are typically normalized, and most values lie within a relatively small range (roughly -1 to 1) — so storing every weight as a 32-bit or 16-bit floating-point number is **unnecessarily precise**.
- Instead, quantization maps these high-precision continuous values into a much smaller set of discrete values (**quantization levels/buckets**), letting each weight be stored using fewer bits (e.g., 8-bit or 4-bit integers).
- This mapping can be **uniform** (fixed intervals) or **adaptive** (intervals chosen based on the actual weight distribution), to better preserve accuracy.

### LoRA + Quantization = QLoRA

By storing the **frozen model** in a low-precision format, while keeping only a small number of **trainable parameters** (the LoRA matrices) in higher precision, memory usage is reduced dramatically, with only a small loss in model performance.

- This makes it possible to load and run much larger language models on hardware with limited memory — consumer GPUs, and for sufficiently quantized models, even modern CPUs.

> This combination of **LoRA + Quantization (QLoRA)** is one of the most widely used techniques today for efficiently fine-tuning and deploying large language models.
