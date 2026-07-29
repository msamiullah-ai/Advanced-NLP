# Parameter-Efficient Fine-Tuning (PEFT): Prompt, Prefix & Adapter Tuning

*Reference: [Complete Adam Optimizer — recap](https://chatgpt.com/s/t_6a6a145515508191841b795f713ac686)*

---

## What Is PEFT?

**PEFT (Parameter-Efficient Fine-Tuning)** is a set of techniques that adapt a pre-trained model by training only a **small subset** of parameters, while keeping the original model weights **frozen** — significantly reducing computational and memory costs.

> 💡 This is the direct, practical answer to the Aghajanyan finding covered in the previous notes: if only a small set of parameters is actually needed to adapt a model, PEFT methods are the concrete techniques that exploit that fact.

### Common PEFT Methods

- **LoRA (Low-Rank Adaptation)** — the most widely used.
- **QLoRA** — combines quantization with LoRA.
- **Adapters** — insert small trainable layers into the network.
- **Prefix Tuning**
- **Prompt Tuning**
- **P-Tuning v2**

---

## The Concept Behind PEFT

*Reference: [ChatGPT conversation on the concept behind PEFT](https://chatgpt.com/s/t_6a694ee99ee881918771606d4c3e868a)*

> 💡 **The "layer of paint" analogy**: PEFT recognizes that the structural foundation (the pre-trained LLM) is already solid — so you only need to apply a thin "layer of paint" (`ΔW`) in a targeted area to adapt it to a new task, rather than rebuilding the whole structure.

- PEFT methods freeze the **vast majority** of the pre-trained model's parameters, and update only a small, carefully selected subset.

**The key hypothesis**: the adaptation required for downstream tasks lies in a **low-dimensional subspace** of the full parameter space. The model doesn't need to relearn everything — it only needs **targeted adjustments**.

---

## 1. Prompt Tuning

> 💡 **The core intuition, informally**: *"I don't know the best prompt — tell me the best one, by finding the best prompt automatically."* Prompt Tuning is a PEFT technique where the computer finds the optimal prompt for you, rather than a human hand-writing it.

Instead of writing words, you **prepend a small set of trainable continuous vectors** (virtual "tokens") directly to the input embeddings.

### How It Works

Prompt Tuning is a PEFT technique in which the pre-trained model's weights are **frozen**, and only a small set of trainable continuous prompt embeddings (called **soft prompts**) is learned.

1. Given an input text prompt, the text is first tokenized and converted into its token embeddings, using the model's embedding layer.
2. The learned soft prompt embeddings are then **prepended** (placed before) these original token embeddings — creating an extended input embedding sequence, fed into the transformer.
3. During training, the model performs a forward pass, computes the task-specific loss, and backpropagates gradients **only** to the soft prompt embeddings — all original model parameters remain unchanged.
4. Through this process, the soft prompts learn to encode task-specific information that guides the frozen model toward the desired behavior.
5. **During inference**, the same learned soft prompt embeddings are prepended to the input's token embeddings, letting the model perform the task without modifying its original weights.

### Key Properties

- **Not human-readable**: these soft prompt vectors don't map to actual English words (like "summarize" or "classify") — they're continuous mathematical values.
- **Trained via gradient descent**: during training, the original LLM's weights stay frozen, but these **20–100** soft prompt vectors are updated until the model produces the best possible outputs for that task.

---

## 2. Prefix Tuning

Prefix Tuning **freezes the entire Transformer**, and learns small task-specific prefix vectors `P_K` and `P_V` for **every** Transformer layer.

- These vectors are prepended to the **Key** and **Value** matrices (**not** the Query matrix), allowing every attention layer to attend to learned task-specific information, without modifying the model's original weights.

### How It Works

Prefix Tuning prepends these learnable prefix vectors to the Key and Value matrices of the transformer's self-attention (and cross-attention, in encoder-decoder models), **at every transformer layer.**

- It learns a small set of trainable prefix vectors, which are transformed into additional Key and Value vectors, and prepended to the Key/Value matrices of every layer.
- During backpropagation, **only** these prefix parameters are updated — all original model parameters remain frozen.

> 💡 **Why Key/Value and not Query?** Since attention works by comparing a Query against Keys to weight Values (see the earlier Self-Attention notes), prepending trainable "fake" entries onto the Key and Value side effectively gives every real token extra, learnable context to attend to — without needing to alter what each token's own Query is asking for. This is a more structurally invasive technique than Prompt Tuning, since it touches **every layer**, not just the input embeddings.

### Further Reading

- [Details](https://chatgpt.com/s/t_6a69565db0e08191bc70903b34ee393c)
- [Mathematics](https://chatgpt.com/s/t_6a695760661c8191b36935ed51e643db)

---

## 3. Adapter Tuning

Adapter Tuning is a PEFT technique in which the pre-trained model's original weights are **frozen**, and small trainable neural network modules called **adapters** are inserted into each transformer layer (typically after the attention and/or feed-forward sub-layers).

### How It Works

1. During the forward pass, input activations pass through **both** the original transformer computations **and** these adapter modules.
2. The task-specific loss is computed, and during backpropagation, **only** the adapter parameters are updated — all original model parameters remain unchanged.
3. Over time, the adapters learn task-specific transformations of the hidden representations, letting the frozen model adapt to new tasks.
4. **During inference**, the learned adapters are loaded alongside the frozen base model — allowing efficient task switching by replacing only the adapter weights, rather than the entire model.

*Reference: [How Adapter Tuning works, in detail](https://chatgpt.com/s/t_6a6a36b5a99c8191b80ca9ce2c24dcc7)*

> 💡 **Structural contrast with Prefix Tuning**: while Prefix Tuning inserts learnable vectors *into the attention mechanism itself* (the K/V matrices), Adapter Tuning inserts entirely separate small **neural network modules** in between the transformer's existing sub-layers — a different point of intervention, even though both share the same "freeze everything except a small new piece" philosophy.

---

## The Common Pattern Across All PEFT Methods

Regardless of the specific PEFT method used, the standard approach is: **one LoRA, adapter, prefix, or soft prompt per task/domain**, while keeping the **same frozen base model**, and swapping in the task-specific module during inference.

> 💡 **Why this matters practically**: this means a single frozen base model (which might be many gigabytes) can serve dozens or hundreds of different tasks, by storing only the small task-specific modules (often just a few megabytes each) and swapping them in and out — a massive storage and deployment efficiency win compared to keeping a fully fine-tuned copy of the entire model for every task.
