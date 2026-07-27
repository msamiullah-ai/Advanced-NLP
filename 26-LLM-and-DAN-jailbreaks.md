# LLM Jailbreaks, Alignment & Fine-Tuning Risks

> **A framing thought to start with**: an LLM is, in a sense, **the compression of the internet** — everything covered below (guardrails, alignment, hallucination) is fundamentally about managing what happens when you try to make that compressed knowledge behave safely and predictably.

---

## Guardrails

**Guardrails** are rules, constraints, and safety mechanisms placed around an AI system to ensure it behaves only within predefined boundaries.

- They control **what** the AI can say, do, access, and decide — making AI systems safer, more reliable, and aligned with user or business requirements.

> 💡 **In short**: guardrails are the safety barriers on a road — they don't change where you're capable of driving, but they keep you from veering off into dangerous territory.

---

## Jailbreaks

A **jailbreak** is an attempt to make an LLM ignore its intended rules or safety constraints, using specially crafted prompts.

- **DAN ("Do Anything Now")** was a well-known early jailbreak that relied on role-playing an "unrestricted AI" persona — but modern LLMs are designed to recognize and resist these kinds of prompt-based attacks.

### Normal LLM Behavior

```
User
   │
   ▼
System Instructions
   │
   ▼
Safety Policies
   │
   ▼
LLM
   │
   ▼
Safe Response
```

### A Jailbreak Attempt

```
User
   │
   ▼
Malicious Prompt
("Ignore all previous instructions...")
   │
   ▼
LLM
```

The goal is to get the model to **ignore or override** its intended constraints. Modern LLMs are specifically trained and engineered to resist these attempts.

---

## A Practical Warning About Using LLMs Dishonestly

> *"When something says don't use an LLM, don't use an LLM — despite the fact that I've just said [detection] can't be caught. Here's the reason: yes, we cannot catch the plagiarism happening with the use of LLMs at this moment. It's a cat-and-mouse game. But suppose you submitted an application today, and I'm running a plagiarism check on it a month later, two months later, or a year later. That plagiarism checker knows all the methods that were current, state-of-the-art a year ago. That plagiarism detection system will have very high precision and recall for a plagiarism which happened a year ago."*

> *"So if you're a university... suppose it says [not to use LLMs]... I'm glad that many universities and institutions and paper reviews are now becoming like this class, where usage is permitted. As long as knowledge is ensured, go ahead and use it. But if it's prohibited, and you still used it, and you didn't get caught, you got admitted — that university is also obviously not stupid. They've kept an audit process every six months of all applicants who applied six months ago, a year ago: 'let's run an audit whether any of these broke our rule.' And if they start applying this retrospectively... that percentage who broke the rule, they're going to make an example, terminate all of them. It's going to stop them. So don't do it. It's a realistic scenario. Don't do it."*

> 💡 **The core lesson**: detection technology only gets *better* over time, never worse — so a violation that's undetectable today can become trivially detectable in a retrospective audit a year from now. "Not caught yet" and "safe" are not the same thing.

---

## Alignment

> **Research context**: attaining LLM alignment across various dimensions — specifically making models **helpful, honest, and harmless** — is currently a very active area of research.

**Alignment** means whether the model behaves according to human intentions and values.

**An aligned model:**
- Answers helpfully.
- Admits when it doesn't know.
- Refuses dangerous requests.
- Avoids misinformation.
- Respects safety policies.

### Why Pretraining Alone Doesn't Give You Alignment

A pretrained LLM learns only **one** objective: predict the most likely next token given the context.

- This makes it highly capable — it acquires vast knowledge and fluent language generation — but it does **not** inherently understand concepts like helpfulness, honesty, or safety.
- As a result, **capability and alignment are separate properties**: pretraining provides the model's knowledge and skills, while **post-training** methods (supervised fine-tuning, human preference learning, guardrails) align its behavior with human intentions.

> 💡 **Why next-token prediction isn't enough on its own**: next-token prediction can *implicitly* teach behaviors like helpfulness, honesty, and refusals, because those patterns exist somewhere in the training data — but it does not *explicitly* optimize for them. Consequently, there's no guarantee a pretrained model will consistently behave helpfully, honestly, or safely across all situations. Additional alignment training exists specifically to make those behaviors more reliable and consistent.

---

## LLM Hallucinations

**Hallucination** in LLMs refers to outputs that appear fluent and coherent, but are factually incorrect, logically inconsistent, or entirely fabricated.

**Two primary types:**

1. **Factuality hallucination**: output contradicts real-world facts.
2. **Faithfulness hallucination**: output is unfaithful to the user's instructions or provided context (even if it happens to be factually true elsewhere).

> ⚠️ **Intent capturing is a very difficult task** — much of the alignment challenge comes down to the fact that correctly inferring what a user actually *wants* (as opposed to what they literally typed) is genuinely hard, and hallucination is one visible symptom of that deeper difficulty.

---

## The Alignment Pipeline

Alignment addresses all of the issues above — making the model **helpful, honest, harmless, and instruction-following.**

**The standard pipeline involves three stages, each building on the last:**

1. **Fine-Tuning (FT)** — also called Supervised Fine-Tuning (SFT).
2. **Reward Modeling**.
3. **Policy Optimization** (via RLHF or DPO).

> 💡 *(This pipeline is covered in much greater depth in the dedicated Alignment lecture notes — this section just situates it in context alongside jailbreaks and hallucination.)*

---

## Efficient Fine-Tuning: QLoRA

**QLoRA (Quantized Low-Rank Adaptation)** — introduced in the paper *"QLoRA: Efficient Finetuning of Quantized LLMs"* (2023) — introduces a method for **backpropagating gradients through a frozen, 4-bit quantized pretrained language model into Low-Rank Adapters.**

> 💡 This lets you fine-tune enormous models on much more modest hardware, since the bulk of the frozen model stays in a highly compressed (4-bit) form, and only small adapter matrices actually receive gradient updates.

---

## Prompt Engineering vs. Fine-Tuning

> **In prompt engineering, model weights are not altered.** It's specifically designed in a way that the model considers the prompt as part of its ordinary next-token prediction task — no parameters change; only the *input* changes.

This is the sharpest possible contrast with fine-tuning (including QLoRA above): prompt engineering shapes behavior entirely through **input framing**, while fine-tuning actually **modifies the model's parameters**.

---

## The Echo Chamber Effect

Just as social media algorithms create echo chambers by recommending content that aligns with a user's pre-existing worldviews and preferences, **LLMs can amplify this same effect.**

- Because models are trained to be helpful, and often attempt to "please" the user based on their specific context and history, they may end up reflecting a **skewed view of reality that confirms what the user already believes** — rather than providing an objective or balanced perspective.

> 💡 This is a subtle risk distinct from hallucination: the model can be factually accurate in each individual claim, while still systematically steering the *overall framing* toward whatever the user seems to want to hear.

---

## In-Context Learning: Fine-Tuning Without Changing Weights

> **Some of the best fine-tuning techniques don't change the weights at all.**

This is exactly what **in-context learning** (via prompting, few-shot examples, etc.) accomplishes — adapting the model's behavior for a task purely through the prompt, without ever touching the underlying parameters. It sits alongside prompt engineering as a weight-free alternative to true fine-tuning.

---

## Catastrophic Forgetting

**Catastrophic forgetting** is the phenomenon where a pretrained model **loses previously learned knowledge or abilities** when it's fine-tuned on a new task or dataset.

- The model doesn't simply *add* new knowledge — it updates the **same parameters (weights)** that stored old knowledge. Those updates can overwrite useful information learned during pretraining.

### Why Does This Happen, Mathematically?

Training always updates parameters via:
```
W_new = W_old − η·∇L(W)
```
- `W` = model weights
- `η` = learning rate
- `L` = loss

- **During pretraining**: `Loss = General Internet Data`.
- **During fine-tuning**: `Loss = Only [narrow new dataset]` (e.g., a legal dataset).

The gradients now optimize **only** for the new, narrow dataset — which can move parameters away from values that supported older, broader capabilities.

### The Two Major Risks of Fine-Tuning a Huge LLM on a Small Dataset

1. **Overfitting**: the model memorizes the small dataset and generalizes poorly to new examples of the same task.

2. **Catastrophic forgetting**: the optimizer updates the same parameters that encode the model's broad pretrained knowledge — potentially degrading unrelated abilities like coding, multilingual performance, or reasoning.

### Mitigations

Modern techniques widely used to reduce these effects:
- **LoRA** (Low-Rank Adaptation)
- **Parameter-Efficient Fine-Tuning (PEFT)** more broadly
- **Diverse fine-tuning data** (rather than a very narrow dataset)
- **Careful training settings** (e.g., lower learning rates, fewer epochs)

---

## Further Reading

*Reference: [Fine-tuned Language Net (FLAN)](https://chatgpt.com/s/t_6a657b3cf5988191abeea9422cddbc01)*
