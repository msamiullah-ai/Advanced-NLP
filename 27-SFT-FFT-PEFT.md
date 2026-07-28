# Supervised Fine-Tuning, Full Fine-Tuning & the Path to PEFT

---

## From Pretraining to Instruction-Following

When a base language model is initially trained (pre-trained), its **only** objective is statistical pattern matching: predicting the next word (or token), given a sequence of text.

- By fine-tuning the model across a **wide range of different instruction types**, it learns the general ability to **follow directions**.
- This lets the model **zero-shot or few-shot generalize** to entirely new instructions it hasn't explicitly seen during training.

> 💡 **A major shift in AI belief**: for a long time, the dominant assumption was that simply building bigger models (more parameters, more memory) was the best way to make them smarter. This lesson showed that teaching a model **how to follow diverse tasks** provides a far greater jump in capability than just throwing more computing power at a raw model.

---

## The FLAN Breakthrough

Instead of fine-tuning a model for just **one** specific job (like translation or sentiment analysis), **FLAN** demonstrated that if you train a model on a **wide mix of diverse tasks**, phrased as instructions, the model learns the **meta-skill of general instruction-following.**

- When tested on completely new, unseen tasks, FLAN performed surprisingly well — because it understood how to process an **instruction template**, not just memorize specific task patterns.

**It proved**: post-pretraining alignment and **instruction dataset diversity** are critical ingredients for building practical, usable AI systems.

> ⚠️ **In fine-tuning, there's no tolerance for noise** — since fine-tuning datasets are much smaller than pretraining corpora, most flaws in the data will be directly reflected in the final model's outcomes. This is very different from pretraining, where a bit of noisy text gets washed out by sheer volume.

---

## Supervised Fine-Tuning (SFT)

**SFT** is a machine learning process where a pre-trained AI model is taught to perform a specific task using a structured, **human-labeled dataset**.

- It bridges the gap between a general, all-purpose model and a highly specialized tool tailored to specific needs.
- **Teaches the model** what a good, helpful assistant writes — polite, direct, accurate, and safe responses.

### Fine-Tuning Data as a Competitive Moat

For general-purpose models (ChatGPT, Gemini, Claude), fine-tuning datasets are extremely diverse — and are **highly guarded, proprietary intellectual property** of those organizations, serving as a core **"moat"** (competitive advantage) distinguishing their models from others.

> 💡 By fine-tuning foundational models (like GPT or Gemma) into a specific product, the resulting behavior, knowledge, and functionality become **unique to that application**. This effectively acts as a "moat" — a defensible answer to why an investor should fund a project, rather than assuming it's simply a thin wrapper around an off-the-shelf API.

---

## The SFT Loss Function

*Reference: [ChatGPT conversation on the SFT loss function](https://chatgpt.com/s/t_6a66ce653a348191b5a4936ad897e078)*

The loss function measures how "surprised" or wrong the model is when trying to predict the correct next token.

- **During standard pre-training**: loss is calculated on **every single word** in a document.
- **During SFT**: we only care about teaching the model how to write the **response** — not the user's prompt.

> During SFT, the prompt/instruction tokens are **masked out** (ignored) when calculating loss gradients. The model is only penalized and updated based on its performance generating the **response** tokens.

### The Core Idea: Instruction Masking / Completion-Only Training

Each training example contains both the **instruction** (the user's prompt) and the **response** (the ideal assistant answer). However, loss (the penalty for wrong predictions) is calculated **only** on the response portion.

- **Instruction tokens = context**: the model reads the instruction tokens during the forward pass, to understand what's being asked — but it is **not** penalized or tested on predicting those instruction words.

> 💡 **Why this makes sense**: the primary goal of an assistant model is to generate **good answers** given a user prompt — not to learn *how users write prompts*. Training loss on the instruction tokens would waste capacity teaching the model to predict user phrasing, which is not the actual task it needs to perform.

### A Note on Training Duration

> ⚠️ **We don't fine-tune for too many epochs**, because otherwise the model will start **forgetting its core knowledge** — this is the exact catastrophic forgetting phenomenon covered in the earlier alignment/fine-tuning notes, and it's precisely why epoch count is a carefully tuned hyperparameter during SFT rather than something to maximize.

---

## Key Limitations of Supervised Fine-Tuning (SFT)

### 1. Quality Ceiling

- **Human dependency**: the performance of the fine-tuned model depends entirely on the quality of the human-written examples (demonstrations) provided to it.
- **Mediocre in = mediocre out**: if annotators write average, generic, or flawed responses, the model caps out at that level, and won't exceed the quality of its training data.

### 2. Hard to Demonstrate Some Behaviors

- **Subjectivity**: for complex or nuanced topics (like ethical dilemmas), there's rarely one single "perfect" response.
- **Annotator disagreement**: different human writers structure and phrase answers differently, based on personal perspective.
- **One-example constraint**: SFT forces you to pick a single definitive demonstration per prompt — restricting the model's ability to capture the nuances or variations of acceptable answers.

### 3. Imitation, Not Understanding (The Most Fundamental Limitation)

- **Surface-level pattern matching**: SFT trains the model to copy the style, formatting, and structural patterns of human examples (e.g., memorizing phrasing like "I'm not able to..." to reject a prompt), rather than understanding **why** a behavior or refusal is appropriate.
- **Lack of preference knowledge**: it teaches the model to imitate *what* was written, rather than grasp *why* one answer is preferred over another.

### Key Takeaway & the Transition Beyond SFT

> **Judging vs. demonstrating**: it's often much easier for humans to evaluate two different model responses and choose which is better, than it is to sit down and write the "perfect" response from scratch.

**This asymmetry is the core reason behind moving beyond standard SFT** — toward methods like **RLHF** (Reinforcement Learning from Human Feedback) and **DPO** (Direct Preference Optimization), which rely on human **preferences** rather than exact demonstrations.

---

## Full Fine-Tuning (FFT) — Mathematical Details

*Reference: [ChatGPT conversation on FFT mathematical details](https://chatgpt.com/s/t_6a66d9c961c48191a19349bddc75202c)*

Full Fine-Tuning starts with a pre-trained model, so **all** model parameters `θ` — which include every weight matrix `W` and bias vector `b` — are initialized with the pre-trained values `θ_PT`.

**The training loop:**

1. For each training example (or mini-batch), the model performs a **forward pass** to produce a prediction `f_θ(x)`, which is compared with the correct target `y` using a task-specific loss function `L_task`.

2. **Backpropagation** computes the gradient `∇L` with respect to `θ` — measuring how sensitive the loss is to each parameter, and indicating how each one should change to reduce error.

3. The gradient is scaled by the learning rate `η`, to control the update size, and parameters are updated via gradient descent:
   ```
   Δ = -η · ∇L
   θ_new = θ_old + Δ

   (equivalently: θ_new = θ_old - η · ∇L)
   ```

This process repeats over many mini-batches and epochs. Because **every** weight matrix and bias vector in the model is updated, the entire model gradually adapts to the downstream task — which is exactly why it's called **Full** Fine-Tuning.

> 💡 **Why FFT is expensive**: since every single parameter receives a gradient update, FFT requires storing gradients and optimizer states (e.g., Adam's `m` and `v` moments — see the Adam notes from the Transformer training lecture) for the **entire** model, not just a small piece of it. For a model with billions of parameters, this multiplies memory requirements several times over — motivating the more parameter-efficient approaches below.

---

## The Bridge to Parameter-Efficient Fine-Tuning (PEFT)

**Aghajanyan et al.** proved that only a **small set of parameters** from the pretrained model would be effective — or even needed — for fine-tuning.

> 💡 **Why this finding matters so much**: it directly challenges the assumption behind Full Fine-Tuning that *every* parameter needs to move for the model to adapt well to a new task. If a task's "true" adaptation can be captured in a much lower-dimensional space than the full parameter count, then updating the entire model is doing a lot of unnecessary (and expensive) work — you're paying the full memory and compute cost of FFT to achieve something that a much smaller set of updates could accomplish.
>
> This is the exact motivating insight behind **Parameter-Efficient Fine-Tuning (PEFT)** methods like **LoRA** and **QLoRA** (introduced in the earlier alignment notes): instead of updating the full weight matrices, these methods introduce small, low-rank "adapter" matrices that capture the task-specific adjustment, while the vast majority of the original pretrained weights stay **completely frozen**. This directly avoids both the memory cost of Full Fine-Tuning *and* the catastrophic forgetting risk discussed earlier — since the original pretrained knowledge, sitting in the frozen weights, is never directly overwritten at all.
