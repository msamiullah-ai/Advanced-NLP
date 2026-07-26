# T5, GPT, and the Economics of Scale

---

## The Research Question

> **Can we make pretraining cheaper and more accessible, while maintaining the benefits of these LLMs (or fine-tuning)?**

This question motivates much of what follows — from T5's unifying text-to-text framing, to GPT's simplified decoder-only design, to the more recent efficiency innovations like Mixture-of-Experts.

---

## T5 (Text-to-Text Transfer Transformer)

T5 kept the **full encoder-decoder** Transformer, but treated **every** NLP task as a text generation problem.

- Instead of separate architectures for translation, summarization, question answering, or classification, T5 converted **every** task into: `input text → output text`.

### How T5 Was Trained: "Span Corruption"

T5 combined the best of two popular training styles:

- **BERT style** (Masked Language Modeling): hiding words in a sentence, asking the model to fill in the blanks.
- **GPT style** (Generative Language Modeling): generating text token-by-token.

| | Borrowed from |
|---|---|
| The **encoder** sees both left and right context (bidirectional understanding) | Like BERT |
| The **decoder** autoregressively generates text one token at a time | Like GPT |

So T5 combines **BERT's strong understanding** with **GPT's text generation ability**.

### Final Takeaway on T5

T5 proved that **one** encoder-decoder model could solve every NLP task, by converting everything into text-to-text generation. It trained using span corruption — the encoder reads corrupted text, and the decoder reconstructs the missing spans — combining BERT's bidirectional understanding with GPT's generative decoding.

> ⚠️ However, **decoder-only** models eventually became the standard, because they are simpler, easier to scale, and support powerful in-context learning.

---

## Span Corruption (Denoising) — Step by Step

*Reference: [Full process walkthrough](https://chatgpt.com/s/t_6a642ad48f888191a57bfa6e44e69a59)*

### The Goal

Before T5 can perform specific tasks (translation, summarization), it needs **general language understanding**. Span corruption is how T5 learns the structure, grammar, and context of human language, by filling in missing chunks of text.

### Worked Example

Take a normal sentence: *"Thank you for inviting me to your party last week."*

**Step 1: Masking Contiguous Chunks (Spans)**

Instead of replacing individual random words (like BERT does), T5 identifies **contiguous sequences of words (spans)** — typically around 3 words on average — and removes them. Each missing span is replaced with a unique **sentinel token** (like `<extra_id_0>`, `<extra_id_1>`, etc.).

**Corrupted input to the model:**
```
"Thank you " + <extra_id_0> + " me to your " + <extra_id_1> + " week."
```

**Step 2: Generating Only the Missing Chunks**

Unlike standard autoencoders that try to reconstruct the **entire** original sentence, T5 skips repeating the known words. Instead, its target output is a list of **only** the missing words, tagged with their corresponding sentinel tokens:

**Target output from the model:**
```
<extra_id_0> + " for inviting " + <extra_id_1> + " party last " + <extra_id_2>
```

> 💡 **Why it's called "denoising"**: the model takes a noisy, incomplete sentence and learns to "clean" or recover the missing pieces — the corrupted input is the "noise," and reconstruction is the "denoising."
>
> Notice the efficiency gain here compared to a naive autoencoder: since T5 only has to *generate* the missing spans (not the whole sentence), the target sequence is much shorter than the input — making training computationally cheaper per example, while still forcing the model to learn deep contextual understanding to fill the gaps correctly.

---

## Decoder-Only Transformers: GPT

Unlike the original Transformer (which used both an Encoder and Decoder) or models like T5, **GPT models throw away the Encoder entirely** and use only the Decoder stack.

- **No Encoder**: the model processes text sequentially, rather than processing the entire input bidirectionally all at once.

- **No Cross-Attention**: since there's no Encoder to send representations over, the cross-attention mechanism is completely removed. Every layer consists purely of **self-attention** and **feed-forward networks**.

- To prevent the model from "cheating" during training, it uses a **causal mask** in its self-attention layers — exactly as covered in the Decoder section of the earlier Transformer notes, just without the cross-attention sub-layer this time.

### Training Objective

The model is trained using a simple **Causal Language Modeling (CLM)** objective: predict the next token, given all preceding tokens.

- The underlying loss is the **Negative Log-Likelihood (NLL) Loss** — the same core idea covered earlier for cross-entropy loss in the original Transformer training notes.

---

## Why Do These Models "Learn"? A Scale Perspective

Models don't learn because they're vastly smarter than humans at processing each individual word — they achieve high performance through **unfathomable scale**, reading centuries' worth of information in just a few weeks of computer processing.

### Tokens vs. Parameters: Data Quality Over Pure Scale

Models like **Qwen 2.5 (72B)** show that scaling **data** (18T tokens) on a medium-sized model can produce results competitive with massive models (like 405B) — making deployment drastically cheaper and faster.

> 💡 This echoes the earlier Chinchilla scaling-law insight: more tokens per parameter, rather than simply more parameters, can be the more efficient lever to pull.

### The Mixture-of-Experts (MoE) Efficiency Shift

**DeepSeek-V3** trained on roughly the same token count as LLaMA 3.1 (14.8T vs. 15T). However, by using a **Mixture-of-Experts** architecture (671B total weights, but only **37B active per token**), it achieved state-of-the-art performance using a fraction of the compute and GPU hours required for standard dense models.

### Diminishing Returns on Pure Text

Simply adding more raw web text reaches a point of **diminishing returns**. Modern performance leaps instead come from:
- Mixing in high-quality **synthetic data**.
- **Specialized code repositories**.
- **Step-by-step reasoning tokens**.
- **Post-training instruction tuning** (RLHF/GRPO).

---

## Dense vs. Mixture-of-Experts (MoE) Architectures

### Dense Architecture

A **dense** model activates **every** parameter in every layer, for every input token.

- Each token passes through the **same** feed-forward networks and attention layers — all parts of the model participate in every prediction.
- Dense models are **simple to design and train**, making them stable and effective — but they become computationally expensive as they grow, since increasing parameter count increases computation required **for every single token**.
- **Examples**: BERT, T5, GPT-2, GPT-3, GPT-4, Llama — the entire model is used regardless of input.

### Mixture of Experts (MoE)

An MoE model contains many specialized feed-forward networks called **experts**, but only a **small subset** is activated for each input token.

- A lightweight **router (gating network)** examines each token and decides which experts are most suitable — different tokens can use different experts.
- This makes MoE models much more **compute-efficient**, since only a fraction of total parameters are used per token, while still benefiting from a very large overall model capacity.
- As a result, MoE models can scale to **hundreds of billions or even trillions** of parameters, without requiring dense-model levels of computation for every prediction.

> 💡 **The core trade-off, in one line**: Dense = simple, stable, but computationally expensive at scale (every parameter works on every token). MoE = more complex to design and route correctly, but lets total model *capacity* grow far larger than the compute cost per token would otherwise allow.

---

## Why Decoder-Only Models Really Won

Decoder-only models didn't win because they were better at specific narrow benchmarks out of the box.

> They won because they turned **task execution into a simple text-completion problem**, allowing scale, hardware parallelization, and zero-shot prompting to replace task-specific training.
