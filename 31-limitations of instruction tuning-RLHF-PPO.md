# Alignment: RLHF and DPO

*Slides: [Full lecture slide deck](https://pern-my.sharepoint.com/:b:/g/personal/namoos_qasmi_lums_edu_pk/IQDi4yntA4eVRLXqY_sVM8i-AeXoZNV9u1bOEUOjWSmxIgs?e=bTxsfc)*

---

## Why SFT Isn't Enough

Standard Supervised Fine-Tuning (SFT) works well when there's a **single correct answer** (like answering a factual math problem, or translating a sentence). However, SFT fails to capture human **preference** for complex or creative prompts.

**SFT's three limitations:**
1. It falls short on **open-ended requests** that lack a single gold-standard answer.
2. It uses **word-level loss metrics** that cannot evaluate overall quality — like safety or coherence — as a holistic property.
3. It restricts the model to merely **mimicking** human-written examples, rather than directly optimizing for user satisfaction.

> **The likelihood objective has no slot for "this is better than that."** SFT's loss function can only ask "did you predict this exact token?" — it has no mechanism to express relative quality judgments at all.

To overcome these boundaries, model training must transition to **alignment techniques** (RLHF, DPO, KTO) that directly optimize models based on **human preferences**, rather than exact text matching.

*Reference: [Mathematical Objectives of Supervised Fine-Tuning](https://chatgpt.com/s/t_6a6c02c6b0c08191b33b6884bf65ff30)*

*Reference: [Alignment — Goals of Post-Training](https://chatgpt.com/s/t_6a6c02217d84819192d6e66e59bfa924)*

---

## Why RLHF Is Needed

RLHF is needed because SFT can only teach a model to **imitate** good responses, whereas RLHF teaches the model to **prefer** better responses over worse ones.

| | Learns from |
|---|---|
| **SFT** | Demonstrations (`prompt → ideal response`) |
| **RLHF** | Comparisons (`response A vs. response B`) |

This lets RLHF capture human preferences — helpfulness, honesty, harmlessness, appropriate style — that pure imitation can't.

> 💡 **The key insight**: recognizing which response is better is much **easier, faster, cheaper, and more consistent** for humans than writing the perfect response from scratch. This makes preference comparisons a richer and more scalable training signal than demonstrations alone — enabling behaviors that SFT simply cannot learn effectively.

---

## The Bradley-Terry Model

*Reference: [The Bradley-Terry Assumption](https://chatgpt.com/s/t_6a6c2f1b1518819189a82d6468451c28)*

Unlike SFT (which learns from demonstrations), RLHF and DPO learn directly from **human preferences**.

**Training data consists of triplets** `(x, y_w, y_l)`:
- `x` = the prompt
- `y_w` = the response humans preferred (**winner**)
- `y_l` = the less preferred response (**loser**)

The **Bradley-Terry model** assumes every response has an unknown (**latent**) reward `r*(x,y)`, and the probability that one response is preferred over another depends **only** on the *difference* between their rewards, not their absolute values:

```
P(y_w > y_l) = σ( r*(x, y_w) − r*(x, y_l) )
```

where `σ` is the sigmoid function, converting the reward difference into a probability between 0 and 1.

- A **larger** reward difference means the winner is much more likely to be preferred.
- **Equal** rewards imply a 50% preference for either response.

> 💡 **The crucial implication**: RLHF/DPO don't need to know the exact quality (absolute reward) of a response — they only need to learn **which response is better relative to another**. This relative framing is what makes the whole approach tractable, and it's the mathematical foundation both RLHF *and* DPO are built on (as you'll see below, DPO is really just a different way of optimizing this exact same equation).

---

## PPO (Proximal Policy Optimization)

Classical RLHF first trains a **reward model** `r_φ` that learns to assign higher reward to responses humans prefer, based on comparison data. Then, using **PPO**, the language model (**policy**) is updated to generate responses that maximize this learned reward.

> PPO balances **maximizing human preference** with **staying close to the reliable behavior learned during SFT.**

---

## RLHF: Step-by-Step

### Step 1: Train the Reward Model

- **Input**: a dataset of 100K–1M human preference comparisons — each example contains a prompt, a preferred (winning) response, and a less preferred (losing) response.
- The **Reward Model** is a separate neural network (usually initialized from the SFT model), whose job is to predict a **single scalar reward** (quality score) for a given `(prompt, response)` pair.
- During training, both the winning and losing responses are passed through the reward model to produce scores.
- **Goal**: train the Reward Model to assign a **higher** reward to responses humans prefer, and a **lower** reward to less preferred responses.
- **Output**: a trained Reward Model that can automatically score the quality of any generated response.

*Reference: [Training the Reward Model (RM)](https://chatgpt.com/s/t_6a6c3017a4c88191bca23a9f0798a7a6)*

### Step 2: Optimize the Policy Model (RL with PPO)

- **Input**: a set of unlabeled prompts (10K–100K) and the SFT model.
- **Process**: the SFT model generates a response for each prompt. The Reward Model evaluates each response and assigns it a reward score. PPO then updates the SFT model's parameters to generate responses that receive higher reward scores — while a **KL-divergence penalty** keeps the model close to the original SFT model, so it doesn't drift toward unnatural or low-quality responses (**reward hacking**).
- **Output**: a preference-aligned model that not only knows how to perform tasks (from SFT), but also generates responses that better match human preferences.

*Reference: [Fine-tune the policy with PPO + KL](https://chatgpt.com/s/t_6a6c30a356b881918d6d5d10cea3da79)*

---

## Complete RLHF Pipeline (as Used in InstructGPT)

### Step 1: Supervised Fine-Tuning (SFT)

A prompt is sampled from the dataset, and a human writes the ideal response (**demonstration**). These prompt-response pairs fine-tune the pretrained GPT using supervised learning (negative log-likelihood). The result is the **SFT model** — which learns to follow instructions and generate responses in the desired format. This model becomes the **initial policy** that RLHF will later improve.

### Step 2: Reward Model (RM) Training

For each prompt, the SFT model generates **multiple** candidate responses, instead of humans writing one from scratch. Human annotators simply **rank** these responses from best to worst. Using these preference rankings, a separate Reward Model is trained to assign higher reward scores to preferred responses and lower scores to less preferred ones — effectively learning to mimic human preferences.

### Step 3: Reinforcement Learning (PPO)

A new prompt is given to the SFT policy, which generates a response. The Reward Model evaluates that response and assigns it a reward score. Using PPO, the policy's weights are updated to produce responses that receive higher rewards, while a **KL penalty** keeps the policy close to the original SFT model, to prevent unnatural or unstable behavior. This process repeats iteratively until the policy becomes better aligned with human preferences.

*Reference: [RLHF (Classical PPO-Based) — Complete Pipeline](https://chatgpt.com/s/t_6a6c31bad1ac81919fc86973e2980ad2)*

---

## Key Ingredients of RLHF

To align a model using Reinforcement Learning, you need **five** core components:

1. The model being trained, `π_θ` (the **policy**).
2. A **frozen reference copy**, `π_ref` — to prevent drifting.
3. **Human preference rankings**, `(y_w, y_l)`.
4. An **automated judge model**, `r_φ` (the Reward Model).
5. A **safety constraint** (KL divergence), to keep outputs coherent.

**The flow:**
```
Preference Data → Reward Model:      Train the Reward Model to predict human preferences.
SFT Model → Copy → Policy Model:     Create the initial policy by copying the SFT model.
Policy Model:                        Given a new prompt, generate a response.
Reward Model:                        Evaluate that response and assign it a reward score.
PPO:                                 Use the reward score (+ KL constraint) to update the Policy Model.

Repeat steps 3-5 thousands of times, until the Policy Model consistently produces
responses that receive higher human-preference rewards.
```

> 💡 **In plain terms**: the instruct-tuned (SFT) model is copied to become the Policy Model, which generates answers; the Reward Model scores those answers using preference data; and PPO updates the Policy Model while a KL constraint prevents it from straying too far from its original base.

---

## Why the KL Term Is Essential

The reward model is a **proxy** for human preference — it is **not** human preference itself.

Without the KL constraint, the policy will discover **adversarial inputs** to the reward model — phrasings, formats, or tokens that game the score without actually improving quality:

- Ending every response with "Sure!" or other reward-correlated tokens.
- Inserting flattery ("Great question!").
- Producing repetitive scaffolding ("Let me think step by step…").
- Generating well-formed but factually empty text that happens to trigger high reward.

> **Goodhart's Law**: when a measure becomes a target, it ceases to be a good measure.

---

## Challenges of RLHF

RLHF is powerful, but **expensive and difficult to train**:

- Requires **multiple models in memory simultaneously**: policy, reference, reward model, and value model.
- Because PPO needs **fresh responses after every update**, training is computationally inefficient.
- Prone to **training instability**: reward hacking, mode collapse, excessive drift from the reference model.
- Collecting human preference data is **costly**, and human annotators often disagree (around **65–75% agreement**) — introducing noise into the training data.
- The Reward Model is only a **proxy** for true human preference — optimizing it too aggressively can increase reward scores while actually *reducing* the quality of the model's responses (**Goodhart's Law**, again).

---

## DPO (Direct Preference Optimization)

*This entire section was missing from the source notes — filled in below to complete the picture, since DPO is the natural and important counterpart to everything above.*

### The Motivation: Can We Skip the Reward Model and RL Entirely?

Looking back at RLHF's challenges above, the core pain points all stem from the same place: **RLHF needs a separate reward model, and a full reinforcement learning loop (PPO) to optimize against it.** This means:
- Four models in memory at once (policy, reference, reward, value).
- An unstable, iterative RL training process, requiring careful hyperparameter tuning.
- The reward model can be "gamed" (Goodhart's Law), requiring the KL constraint as a safety net.

**DPO's central question**: what if we could get the *exact same alignment result* as RLHF, using **only supervised learning** — no separate reward model, no RL loop at all?

### The Key Mathematical Insight

Recall the RLHF objective being optimized during the PPO phase: maximize expected reward, while staying close to the reference model (via KL penalty):

```
max_π  E[ r(x,y) ]  −  β · KL( π(·|x) || π_ref(·|x) )
```

where `β` controls how strongly the policy is penalized for drifting from the reference model.

**A known result from earlier RL/control theory work** (used as the starting point for DPO) is that this exact objective has a **closed-form optimal solution**:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )
```

where `Z(x)` is a normalization constant (a "partition function") that doesn't depend on `y`.

**The key trick — rearranging to solve for the reward, in terms of the policy:**

```
r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

> 💡 **What this equation is really saying**: the *optimal* policy that would result from RLHF training is mathematically determined by the reward function — but this relationship can be **inverted**. Given *any* policy `π`, you can back out an **implicit reward function** that `π` would be optimal for. In other words: **the language model itself secretly encodes a reward model, just by virtue of its own probabilities.** This is the core insight of the DPO paper (its title is literally *"Your Language Model is Secretly a Reward Model"*).

### Substituting Into the Bradley-Terry Model

Recall the Bradley-Terry preference probability from earlier:
```
P(y_w > y_l) = σ( r*(x, y_w) − r*(x, y_l) )
```

Now substitute the reparameterized reward `r(x,y) = β·log(π(y|x)/π_ref(y|x)) + β·log Z(x)` from above into this formula.

- Since `β·log Z(x)` depends only on the prompt `x` — **not** on which response (`y_w` or `y_l`) — it **cancels out exactly** when you take the difference `r(x,y_w) − r(x,y_l)`.

**This leaves a clean, reward-model-free expression:**

```
P(y_w > y_l) = σ( β·log(π(y_w|x)/π_ref(y_w|x)) − β·log(π(y_l|x)/π_ref(y_l|x)) )
```

> 💡 **Why this cancellation is the whole point**: it means we never actually need to know `Z(x)` (which would be extremely expensive to compute — it requires summing over every possible response in the vocabulary). The unknown, intractable normalization term simply disappears from the math, leaving something we *can* directly optimize.

### The DPO Loss Function

This expression is now just a **binary classification problem** — "is `y_w` preferred over `y_l`?" — that can be trained directly with a standard cross-entropy-style loss, over the policy `π` itself:

```
L_DPO(π; π_ref) = −E_{(x,y_w,y_l)} [ log σ( β·log(π(y_w|x)/π_ref(y_w|x)) − β·log(π(y_l|x)/π_ref(y_l|x)) ) ]
```

**What you need to compute this loss:**
- The policy model `π` (being trained).
- A **frozen** reference model `π_ref` (typically just the SFT model — exactly the same reference model role as in RLHF).
- A dataset of preference triplets `(x, y_w, y_l)` — the **same kind of data** RLHF's reward model would have been trained on.

**What you do NOT need:**
- ❌ A separate reward model.
- ❌ A PPO/reinforcement learning loop.
- ❌ Online sampling of new responses during training (DPO trains on a fixed, static dataset — like SFT does).

### Why This Is a Big Deal

| | RLHF (PPO-based) | DPO |
|---|---|---|
| **Models needed in memory** | 4 (policy, reference, reward, value) | 2 (policy, reference) |
| **Reward model** | Explicitly trained, separate network | **Implicit** — the policy itself defines it |
| **Training method** | Reinforcement learning (PPO) | Standard supervised-style gradient descent |
| **Training stability** | Prone to instability, reward hacking, mode collapse | Much more stable — it's just a classification loss |
| **Data usage** | Needs online sampling from the current policy during RL | Trains directly on a fixed offline preference dataset |
| **Compute cost** | High (generation + reward scoring + RL updates, repeated) | Much lower (single forward/backward pass per batch, like SFT) |

> 💡 **The intuition behind what the DPO loss is actually doing during training**: it increases the model's log-probability of the *winning* response `y_w` (relative to the reference model), and decreases its log-probability of the *losing* response `y_l` — directly, via gradient descent. There's no intermediate reward-scoring step; the adjustment happens straight on the policy's own output probabilities.

### The Gradient Intuition

The gradient of the DPO loss has a particularly elegant interpretation: it pushes up the likelihood of `y_w` and pushes down the likelihood of `y_l`, but the **strength** of this push is automatically scaled by how "wrong" the model's current implicit reward ranking is.

- If the model *already* strongly prefers `y_w` over `y_l` (i.e., the implicit reward gap is already large and correct), the gradient is small — little needs to change.
- If the model currently has it backwards (implicitly prefers the *loser*), the gradient is large — a strong correction is applied.

> 💡 This built-in, self-adjusting weighting is conceptually similar to how the reward model's score would have driven the *magnitude* of PPO's policy updates in classical RLHF — except here it emerges naturally from the math itself, with no separate reward network required to compute it.

### Limitations of DPO

DPO isn't a strict upgrade with no trade-offs:

- **Off-policy training**: DPO trains on a **fixed, pre-collected** dataset of preference pairs, rather than generating fresh responses from the current policy and getting them re-ranked (as RLHF's iterative loop does). This means DPO doesn't get to "explore" and correct based on its own current behavior — it only ever learns from whatever responses were in the original dataset.
- **Distribution shift risk**: if the policy drifts significantly from the distribution of responses in the original preference dataset during training, the implicit reward signal can become less reliable, since it was calibrated on a different distribution of outputs.
- **Still requires high-quality preference data**: DPO doesn't remove the need for good `(x, y_w, y_l)` comparison data — it only removes the need to train an *explicit* reward model from that data.
- **The `β` hyperparameter still matters**: exactly like the KL penalty coefficient in RLHF, `β` controls the same fundamental trade-off — how much the policy is allowed to move away from the reference model. Poorly tuned `β` can cause similar (though generally milder) issues to a poorly-tuned KL constraint in RLHF.

### The Big Picture: RLHF and DPO Are Mathematically Connected

DPO isn't a completely different idea from RLHF — it's a **reparameterization** of the exact same underlying optimization problem (the KL-constrained reward maximization objective), solved in a way that avoids ever needing to explicitly represent the reward model or run reinforcement learning. This is why DPO can achieve alignment results comparable to RLHF, despite being dramatically simpler to implement and train — it's optimizing for the same theoretical target, just via a much more direct mathematical path.
