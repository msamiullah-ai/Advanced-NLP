# Training the Transformer


## The Task

The original Transformer was trained on **machine translation** — specifically **English-to-German** and **English-to-French**.

- The training data consists of millions of sentence pairs, like: `("The cat sat on the mat", "Die Katze saß auf der Matte")`.
- **The goal**: given an English sentence, produce the German sentence.

---

## Step 1: Data Preparation

- **Corpus**: WMT 2014 — about **4.5 million** sentence pairs for English-German, and **36 million** for English-French.

- **Tokenization**: both source and target sentences are tokenized using **Byte-Pair Encoding (BPE)**.
  - For English-German, a **shared vocabulary** of about **37,000 tokens** across both languages.
  - Sharing means tokens like numbers and proper nouns get reused — and importantly, it allows **weight tying** between encoder and decoder embeddings.

- **Batching**: sentence pairs are grouped into batches by **approximate sequence length**.
  - This prevents wasting compute padding short sentences to match a very long one in the same batch.
  - Each batch contained roughly **25,000 source tokens** and **25,000 target tokens**.

### Notes: Padding

- GPUs process batches as **fixed-size tensors** — every sequence in a batch must have the same length.
- Since sentences vary in length, shorter sentences are extended with a special `<PAD>` token to match the longest sentence in the batch.
- Without padding, we'd need to process one sentence at a time, losing all parallelism.

**The padding mask** ensures these filler positions are invisible to the model:
- Padded positions are excluded from attention (their logits are set to `-∞`) and from the loss computation, so `<PAD>` tokens have **zero effect** on what the model learns or attends to.

> 💡 **Padding is a systems-level constraint, not an architectural one.** Pad tokens exist purely to satisfy the hardware's requirement for rectangular tensors — nothing in the Transformer architecture actually requires it.
> - Self-attention computes pairwise interactions between however many tokens you give it — 5 or 500, the mechanism is the same.
> - The feed-forward layers operate on each position independently. Positional encodings are added per-position. None of these components assume or require a fixed sequence length.
> - GPUs operate on rectangular tensors, so if you want to batch 32 sentences together in one matrix multiply, they must share a common length — that's a **hardware/implementation** concern.
> - In fact, if you were willing to process one sentence at a time (batch size 1), you'd never need padding at all — you'd just pay a steep price in GPU utilization.

---

## Step 2: Preparing a Single Training Example

Take one sentence pair:
- **Source**: "The cat sat on the mat" → tokenized to IDs: `[The, cat, sat, on, the, mat]` (6 tokens)
- **Target**: "Die Katze saß auf der Matte" → tokenized to IDs: `[Die, Katze, saß, auf, der, Matte]`

**The target is split into two versions:**

- **Decoder input** (what the decoder sees): prepend a start-of-sequence token → `[<SOS>, Die, Katze, saß, auf, der, Matte]`
- **Decoder label** (what the decoder should predict): append an end-of-sequence token → `[Die, Katze, saß, auf, der, Matte, <EOS>]`

So, at each position, the decoder is trained to predict the **next** token:
- When it sees `<SOS>`, it should output **Die**.
- When it sees `<SOS>, Die`, it should output **Katze**.
- ...and so on, until it produces `<EOS>` to signal completion.

> 💡 **This offset-by-one pairing is the entire trick behind teacher forcing** — the decoder input and decoder label are the *same* sequence, just shifted by one position, letting the model learn "predict the next token" at every position simultaneously.

---

## Step 3: Forward Pass in the Encoder

The source tokens are fed through the encoder.

### Embedding

1. Each source token ID is looked up in the embedding matrix (`|V| × 512`).
2. Then **scaled by `√512`**.
3. Then **added** to sinusoidal positional encodings.
4. The result is a matrix `X` of shape `(6 × 512)` — 6 tokens (`[The, cat, sat, on, the, mat]`), each represented as a 512-dimensional vector.

> 💡 **Why scale by `√512`?** This wasn't covered in the earlier intuition notes: the original paper multiplies embeddings by `√d_model` before adding positional encodings, so that the embedding values are on a comparable scale to the positional encoding values (which are bounded between -1 and 1) — without this scaling, the positional signal could be disproportionately large or small relative to the word-meaning signal.

### Encoder Blocks 1 Through 6

`X` passes sequentially through all 6 encoder blocks. Each block performs:

1. **Multi-head self-attention** over `X` (8 heads, each 64-dim).
2. **Residual + LayerNorm**.
3. **Feed-forward network** (512 → 2048 → 512, with ReLU).
4. **Residual + LayerNorm**.

- Every source token can attend to **every other source token** (no masking).
- After all 6 blocks, the output `Enc` is a `(6 × 512)` matrix, where each position now contains a **rich contextual representation** of that source token, informed by all the others.

---

## Step 4: Forward Pass in the Decoder

The decoder input `[<SOS>, Die, Katze, saß, auf, der, Matte]` is processed similarly, but with crucial differences.

### Embedding

Same process: lookup, scale by `√512`, add positional encoding. Result `Y` has shape `(7 × 512)`.

### Decoder Blocks 1 Through 6

`Y` passes through 6 decoder blocks. Each block performs three sub-layers:

#### Sub-layer 1: Masked Self-Attention

The decoder tokens attend to each other, **but with a causal mask**. This is the critical part. The mask is an upper-triangular matrix of `-∞` values, applied before softmax:

|  | `<SOS>` | Die | Katze | saß | auf | der | Matte |
|---|---|---|---|---|---|---|---|
| **`<SOS>`** | 0 | -∞ | -∞ | -∞ | -∞ | -∞ | -∞ |
| **Die** | 0 | 0 | -∞ | -∞ | -∞ | -∞ | -∞ |
| **Katze** | 0 | 0 | 0 | -∞ | -∞ | -∞ | -∞ |
| **saß** | 0 | 0 | 0 | 0 | -∞ | -∞ | -∞ |
| **auf** | 0 | 0 | 0 | 0 | 0 | -∞ | -∞ |
| **der** | 0 | 0 | 0 | 0 | 0 | 0 | -∞ |
| **Matte** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**When `-∞` goes through softmax, it becomes 0** — so those positions get zero attention weight.

- This ensures that when computing the representation for "Katze," the model can only look at `<SOS>`, "Die," and "Katze" itself — but **never** at future tokens like "saß" or "auf."

**Why is this necessary?**
- During inference, the decoder generates one token at a time, and genuinely doesn't have access to future tokens.
- If we let it see the future during training, it would learn to **cheat** (just copy the next token from the input) and fail completely at inference time.
- The mask **simulates the autoregressive constraint during training**, while still allowing the entire sequence to be processed in parallel for efficiency.

#### Sub-layer 2: Cross-Attention

- **Queries** come from the decoder's output of sub-layer 1.
- **Keys and Values** come from the encoder output `Enc`.
- This is how each decoder position "reads" the source sentence to decide what to translate.
- **No masking here**: every decoder position can attend to every encoder position.

#### Sub-layer 3: Feed-Forward Network

Same structure as the encoder (512 → 2048 → 512, with ReLU).

- Each sub-layer has its own **residual connection** and **LayerNorm**.

---

## Step 5: Output Projection

After all 6 decoder blocks, the output has shape `(7 × 512)`. This is projected to vocabulary size:

```
logits = Y_final · W_vocab      shape: (7 × vocab_size)
```

- `W_vocab` is **tied** with the embedding matrix (transposed).
- Each row of `logits` is a score for every token in the vocabulary, at that position.

**Apply softmax to each row:**
```
P = softmax(logits)             shape: (7 × vocab_size)
```

- `P[0]` is a probability distribution over the vocabulary for what should come after `<SOS>`.
- `P[1]` is what should come after "Die."
- ...and so on.

---

## Step 6: Loss Computation

The loss function is **cross-entropy**, computed between the predicted probability distributions and the actual target labels:

```
Loss = −Σᵢ log P[i][correct_token_i]
```

i.e., look up the probability of the correct token at position `i`, in the `[#output_tokens × |V|]` output matrix.

**For our example:**
- Position 0: correct token is "Die" → loss contribution: `-log P[0][Die]`
- Position 1: correct token is "Katze" → loss contribution: `-log P[1][Katze]`
- Position 2: correct token is "saß" → loss contribution: `-log P[2][saß]`
- ...
- Position 6: correct token is `<EOS>` → loss contribution: `-log P[6][<EOS>]`

**Explanation:**
- If the model is confident **and** correct (assigns high probability to the right token), `-log` of a number close to 1 is close to 0 — i.e., **low loss**.
- If the model is wrong or uncertain, `-log` of a small number is large — i.e., **high loss**.
- The total loss is **averaged** across all positions and all examples in the batch.

### Label Smoothing

Instead of the target being a hard `1.0` for the correct token and `0.0` for everything else, they used **`0.9`** for the correct token, and distributed the remaining **`0.1`** uniformly across all other tokens.

- This prevents the model from becoming **overconfident**, and improves generalization.
- It slightly hurts **perplexity** (the model can never be "perfectly sure"), but improved **BLEU scores**.

### Formalizing the Loss

At each decoder position `t`, the model produces a distribution `p̂(x)` over the vocabulary, and the true next token is `u_t`. Compute the cross-entropy loss:

```
L_t = −log p̂(u_t)
```

This is the **negative log probability** assigned to the correct token.
- If the model confidently predicts the right token, `p̂(u_t)` is close to 1, and the loss is close to 0.
- If it assigns low probability to the correct token, the loss is large.

**Average over all non-padded positions across the batch:**
```
L = (1/T) · Σ_{t=1}^{T} L_t
```
where `T` is the total number of non-padded target positions in the batch.

---

## Step 7: Backpropagation

Gradients of the loss with respect to **every parameter** in the model are computed via the chain rule. This flows backward through:

1. Output projection weights.
2. All 6 decoder blocks (FFN weights, cross-attention Q/K/V/O weights, self-attention Q/K/V/O weights).
3. All 6 encoder blocks (FFN weights, self-attention Q/K/V/O weights).
4. Both embedding matrices.
5. *(Positional encodings are fixed, so no gradients for them.)*

- The total parameter count for the **base model** is about **65 million**. Every single one receives a gradient update.

> 💡 **A key thing to understand**: even though the forward pass goes encoder → decoder **sequentially**, backpropagation computes gradients for **everything in one pass**.
>
> The encoder learns because its output affects the decoder's cross-attention, which affects the decoder's predictions, which affects the loss. So the encoder gets trained to produce representations that are **useful for the decoder**, even though the encoder itself never directly "sees" the loss — the gradient signal reaches it entirely through the cross-attention connection.

---

## Step 8: Optimizer Step — Adam

The optimizer is **Adam**, with specific hyperparameters: `β₁ = 0.9`, `β₂ = 0.98`, `ε = 10⁻⁹`.

```
θ_{t+1} = θ_t − lr · (m̂_t / (√v̂_t + ε))
```

The optimizer updates each parameter:
- It maintains a **running mean** (first moment) and **running variance** (second moment) of the gradients for each parameter.
- This gives each parameter an **adaptive learning rate**: parameters with consistently large gradients get smaller updates, and vice versa.

### Quick Adam Revision

Adam combines **two ideas**:

**Idea 1: Momentum.** Instead of using the raw gradient `g_t`, maintain a running average of past gradients (first moment `m_t`):
```
m_t = β1·m_{t-1} + (1 − β1)·g_t
```
This is an exponential moving average that smooths out noise and keeps the optimizer moving in a consistent direction.

> 💡 **Intuition**: pick up speed in a consistent direction, slow down along dimensions where direction switches often.

**Idea 2: Per-parameter adaptive learning rate.** Also maintain a running average of past **squared** gradients (second moment `v_t`):
```
v_t = β2·v_{t-1} + (1 − β2)·g_t²
```
Parameters with historically large gradients get a **smaller** effective learning rate (divide by `√v_t`), and parameters with small gradients get a larger one. This is the **RMSProp** idea.

> 💡 **Intuition**: apply the brakes if heading toward the minimum at a very high speed.

### Bias Correction

Both `m_t` and `v_t` are initialized to zero. In early steps, they haven't accumulated enough history, so they're **biased toward zero**. Adam corrects this:

```
m̂_t = m_t / (1 − β1^t)
v̂_t = v_t / (1 − β2^t)
```

- At step 1 with `β1 = 0.9`, the correction divides by `0.1` — a **10x boost**.
- By step 50, `1 − 0.9^50 ≈ 0.995`, and the correction is negligible.
- This is a one-line fix, but it matters: without it, early updates are systematically too small.

**The full update rule:**
```
θ_{t+1} = θ_t − lr · (m̂_t / (√v̂_t + ε))
```
where `ε = 10⁻⁹` just prevents division by zero when `v_t` is tiny, and `m̂_t`, `v̂_t` are the bias-corrected moments.

### Warmup — The Learning Rate Schedule

In the Transformer, `lr` is **not** a fixed constant — it's the value computed by a schedule at each step:

```
lr_t = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

- With `d_model = 512` and `warmup = 4000`, the learning rate at step 1 is tiny, climbs **linearly** to a peak around step 4000 (roughly `1.6 × 10⁻³`), and then **decays**.
- The Adam update uses whatever `lr_t` is at that step — there is **no separate base learning rate hyperparameter**.
- For the first 4,000 steps, the learning rate increases linearly (**warmup**). After that, it decays proportionally to the **inverse square root** of the step number.

> 💡 **Why warmup matters**: Adam's running averages of gradients (the momentum terms) need a few thousand steps to become reliable estimates. If you start with a high learning rate before these estimates stabilize, training can diverge.

### If Adam Already Corrects for Bias, Why Do We Also Need Warmup?

- **Bias correction** fixes the *statistics* of `m_t` and `v_t` — but the estimates themselves are still **noisy** when computed from only a handful of samples.
- Correcting a noisy estimate doesn't make it *accurate* — it just makes it *unbiased*.
- With a high learning rate, unbiased-but-noisy estimates can still cause destructively large steps.
- The **warmup** keeps the learning rate small while the running averages accumulate enough history to become reliable, and then the `1/√step` decay gradually reduces the learning rate as the model approaches convergence and needs finer adjustments.

> 💡 **The distinction in one line**: bias correction fixes *systematic* bias (a mathematical guarantee); warmup compensates for *statistical noise* (an empirical safeguard) — two different problems that happen to occur in the same early training window, requiring two different fixes.

---

## Regularization

Three mechanisms prevent overfitting:

### 1. Dropout

Apply dropout (rate **0.1**) after every sub-layer, after the attention weights, and after the embedding + positional encoding addition.

- During training, 10% of activations are randomly zeroed out, forcing the network to distribute information across multiple pathways.

### 2. Label Smoothing

(Recap from Step 6.) Instead of training against a hard target of `1` for the correct token and `0` for all others, soften the target:

- Assign `0.9` to the correct token, and distribute the remaining `0.1` uniformly across all **37,000** vocabulary entries.
- This penalizes overconfidence — the model learns not to put 99.99% probability on a single token, even when it's very sure.
- Slightly hurts perplexity, but improves BLEU score and generalization.

### 3. Residual Pathway

Not a regularizer *per se*, but the residual connections in every layer ensure gradients flow cleanly through 6 encoder and 6 decoder layers, without vanishing. This is a **prerequisite** for stable training at this depth.

---

## Step 9: Repeat (Training Loop)

Iterate over the entire corpus. One full pass through all sentence pairs is one **epoch**.

- The original Transformer was trained for about **100,000 steps** for the base model, and **300,000 steps** for the large model.
- With batch sizes of roughly 25,000 tokens, that's **billions of tokens** seen during training.

**Hardware and time:**
- The **base model** was trained on **8 NVIDIA P100 GPUs** for about **12 hours**.
- The **large model** took **3.5 days** on the same hardware.

- Each GPU processes a portion of the batch, gradients are accumulated across GPUs, and parameters are updated synchronously.
- The model sees each sentence pair multiple times across epochs, and shuffled batching ensures it sees different combinations of sentences together.
- Training continues until the loss on a **held-out validation set** stops improving.

---

## Step 10: Inference (After Training)

At inference time, the process is fundamentally different from training, because there's **no target sentence** to feed the decoder. Generation is **autoregressive**:

1. Encode the source sentence through all 6 encoder blocks → `Enc`.
2. Start the decoder with just `[<SOS>]`.
3. Run the decoder → get a probability distribution → sample or pick the highest-probability token → say it's "Die."
4. Now decoder input is `[<SOS>, Die]` → run again → predict "Katze."
5. Continue until `<EOS>` is generated, or max length is reached.

> 💡 **Efficiency note — KV caching**: each step technically requires a full decoder forward pass, but in practice, **KV caching** is used — the key and value matrices from previous positions are stored and reused, so each new token only needs to compute attention for **one new position**, rather than reprocessing the entire sequence from scratch.

**Beam search at inference**: the original paper used beam search with **beam width 4** for translation — instead of greedily picking the single best token at each step, the model maintains the top 4 candidate sequences and explores them in parallel, choosing the one with the highest overall probability at the end.

- A **length penalty** of `α = 0.6` was applied to avoid the model preferring shorter translations (since shorter sequences naturally accumulate less negative log-probability).

---

## The Key Training Insight

The most elegant aspect of the Transformer's training is **teacher forcing with masking**.

- During training, the decoder receives the **entire correct target sequence at once** and processes all positions in parallel.
- The **causal mask** prevents cheating.
- This means training is **massively parallelizable**, unlike RNNs, which had to process tokens one at a time, even during training.

> This parallelism is what made Transformers practical to train on large datasets, and is arguably the **single biggest reason** they replaced RNNs.

---

## What Comes Next? The Course Roadmap

*The instructor's framing for how the rest of the course builds on this foundation — presented as a chain of motivating questions, each topic answering the question raised by the one before it.*

### The Modern Transformer Block
*"What is the architecture, and how does each component work?"*

- **Architecture Split (BERT / GPT / T5)** — sets the stage; establishes decoder-only as the dominant paradigm. Everything that follows is about making this architecture better.
- **Positional Encoding (Sinusoidal → Learned → ALiBi → RoPE)** — *"We've chosen decoder-only. But attention is permutation-equivariant — how does the model know word order?"* Ends with RoPE as the modern standard.
- **Normalization (Post-LN → Pre-LN → RMSNorm)** — *"We have attention and position. But stacking 96 layers causes training to diverge — how do we stabilize it?"*
- **Activation Functions (ReLU → GELU → SwiGLU)** — *"Training is stable. But the FFN's ReLU is wasting capacity with dead neurons — what's better?"* Completes the picture of a single modern Transformer block.
- **Scaling Laws (Kaplan → Chinchilla)** — *"We have the architecture. Now the critical question: how big should the model be, and how much data does it need?"* Provides the principled framework before encountering the obstacles of scaling.

### Efficient Attention
*"Scaling reveals the quadratic bottleneck — how do we fix it?"*

- **Quadratic Attention + Flash Attention** — *"We want to scale to longer sequences, but attention is O(n²) in memory and compute."* Flash Attention solves the memory problem without changing the math.
- **Sliding Window Attention** — *"Flash Attention helps, but do we really need every token to attend to every other token?"* Introduces the local-attention alternative used by Mistral.
- **KV Cache + MQA + GQA** — *"At inference time, even with Flash Attention, the KV cache grows linearly and eats all the memory."* Transitions from training efficiency to inference efficiency.

### Scaling Further
*"We've made attention efficient. What else limits scaling?"*

- **Context Length Extension (Position Interpolation, YaRN, NTK, Ring Attention)** — *"Flash Attention + GQA solved memory. But RoPE breaks beyond the trained length — how do we extend to 128K+ tokens?"* Natural follow-up to both RoPE and efficient attention.
- **Mixture of Experts** — *"Can we scale parameters without scaling per-token compute?"* The final architectural innovation — after this, the model architecture is complete.

### From Logits to Text
*"The model produces probabilities. Now what?"*

- **Decoding Strategies (Greedy → Beam → Temperature → Top-k → Top-p → Min-p)** — *"The model outputs a distribution over 30,000 tokens at each step. How do you pick which one?"* This must come before alignment, because alignment changes *what* the model generates, and you need to understand *how* it generates first. *(This is exactly the content covered in the two "Decoding Strategies" note files.)*

### Alignment and Adaptation
*"The model generates text, but it's not helpful or safe — how do we fix that?"*

- **Alignment: SFT → RLHF → DPO** — *"A raw pre-trained model will happily generate harmful content. How do we align it with human values?"* The core alignment pipeline.
- **Constitutional AI / RLAIF** — *"RLHF requires expensive human feedback. Can AI provide the feedback instead?"* Direct extension of alignment, scaling it with AI.
- **LoRA / QLoRA** — *"Full fine-tuning (for SFT, RLHF, or any task) costs hundreds of GPUs. Can we adapt the model cheaply?"* Directly motivated by the cost of the fine-tuning described above.
- **Model Merging (Task Arithmetic → TIES → DARE)** — *"We've cheaply fine-tuned several LoRA variants for different tasks. Can we combine them into one model without retraining?"* Natural follow-up to LoRA.

### Deployment at Scale
*"The model is trained and aligned. How do we serve it affordably?"*

- **Quantization (PTQ, GPTQ, AWQ, QAT)** — *"A 70B model needs 140 GB in fp16. How do we fit it on one GPU?"* The first deployment bottleneck.
- **Speculative Decoding** — *"The model fits in memory but generates tokens slowly — each one requires a full forward pass. Can we generate multiple tokens at once?"* Addresses inference latency.
- **PagedAttention / vLLM** — *"We're serving thousands of concurrent users. How do we manage KV cache memory across all of them without 85% waste?"* The systems-level capstone of deployment.

### Advanced Reasoning
*"The model is deployed. Can it think harder on difficult problems?"*

- **Chain-of-Thought + Test-Time Compute (o1 paradigm)** — *"The model uses the same compute per token regardless of problem difficulty. Can it 'think longer' on hard problems?"* Introduces the reasoning frontier.
- **Process Reward Models + Tree-of-Thoughts** — *"CoT generates a single linear reasoning path. Can we evaluate each step and explore multiple paths systematically?"* Deepens the reasoning toolkit.

### Extending Capabilities Beyond the Weights
*"The model reasons well but has limited knowledge and can't take actions."*

- **Tool Use + RAG** — *"The model's knowledge is frozen at training time, it can't do reliable arithmetic, and it can't access private data. How do we extend it?"* Bridges internal capability with external resources.
- **LLM Agents (ReAct, Function Calling, Frameworks)** — *"Tool use handles single calls. But complex tasks require multi-step planning, observation, and adaptation."* The culmination of everything — agents combine generation, reasoning, and tools into autonomous systems.

### Measurement and Responsibility
*"We've built all this. How do we know if it works, and what can go wrong?"*

- **Evaluation & Benchmarks (Perplexity, BLEU, ROUGE, MMLU, LLM-as-Judge)** — *"We have models that can reason, use tools, and act as agents. How do we measure all of this?"* Comes near the end because students now understand all the capabilities being measured.
- **Safety & Red Teaming (Jailbreaks, Prompt Injection, Guardrails, Watermarking)** — *"These powerful models can be manipulated. What are the risks and defenses?"* The final topic — after understanding everything the system can do, understand how it can fail and be misused. Ends the course on a note of responsibility.
