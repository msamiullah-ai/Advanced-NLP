# The Architectural Split & BERT

---

## The Three Architectural Paradigms

Every Transformer-based model falls into one of three architectural categories, each suited to a different kind of task.

### 1. Encoder-only

```
Input → Encoder layers → Contextual representations
```

- Every token can attend to **all** other tokens (bidirectional attention).
- Produces rich representations, rather than generating text.

**Best for:**
- Text classification
- Sentiment analysis
- Named Entity Recognition (NER)
- Semantic search
- Question answering (extractive)

> 💡 In a Transformer, the encoder provides **bidirectional context** by processing the entire input sequence simultaneously, letting each token attend to all other tokens — both past and future — without any causal masking. This gives the model a holistic, "both left and right" understanding of the text.

### 2. Decoder-only

```
Input → Decoder layers → Predict next token
```

- Uses **causal (masked) self-attention**, so each token only sees previous tokens.
- Generates text **autoregressively** (one token at a time).

**Best for:**
- Chatbots
- Story writing
- Code generation
- Translation by prompting
- Modern LLMs generally

### 3. Encoder–Decoder

- **Encoder**: reads and understands the entire input.
- **Decoder**: uses the encoder's output, plus previously generated tokens, to produce the output sequence.

**Best for:**
- Machine translation
- Text summarization
- Paraphrasing
- Sequence-to-sequence tasks generally

---

## Why Did Decoder-Only Models Win?

The rise of decoder-only models happened because predicting the next token on massive amounts of internet text turned out to produce a model that learns **language, knowledge, reasoning, coding, and many other abilities — all with a single, scalable objective.**

- That's why nearly all modern foundation LLMs use the decoder-only architecture.

### But Can Encoder-Decoder Do Everything Decoder-Only Does?

Almost all tasks a decoder-only Transformer (GPT, Llama) performs can also be handled by an encoder-decoder architecture (BART, T5). Since encoder-decoder models both process inputs *and* generate outputs, they're highly capable of generating text too.

| | How it handles a task |
|---|---|
| **Decoder-only** | Treats everything as a continuation of the input sequence, predicting the next word one by one |
| **Encoder-decoder** | Uses its encoder to *fully understand* the prompt first, then uses its decoder to generate the requested output |

> 💡 **A useful analogy**: if GPT (decoder-only) is an expert writer who generates text word-by-word, BERT (encoder-only) is an expert reader who thoroughly understands every detail of a given text.

---

## BERT (Bidirectional Encoder Representations from Transformers)

The most famous and influential Encoder-only language model, created by Google in 2018.

### What Makes BERT Special?

Before BERT, models read text sequentially — either left-to-right or right-to-left.

- BERT introduced **true bi-directionality**: it reads the entire sequence of words at once, in both directions simultaneously.
- This lets BERT deeply understand a word's context based on **all** the surrounding words (both before and after it).

> 💡 **Worked example**: consider the word "bank" in *"He sat on the river bank"* vs. *"He deposited money at the bank."* A traditional left-to-right model seeing "He sat on the river..." only knows "bank" comes next — it has no idea which sense is intended yet. BERT looks at the whole sentence at once, and instantly understands which meaning of "bank" is intended, based on surrounding words like "river" or "money."

### What BERT Is Best Used For

Because BERT excels at *understanding* text rather than generating new text, it's widely used for:

- **Search Engine Ranking**: Google incorporated BERT directly into Google Search, to better understand complex, conversational search queries.
- **Text Classification**: spam detection, topic categorization, sentiment analysis.
- **Named Entity Recognition (NER)**: extracting names, dates, or locations from documents.

---

## Pre-training vs. Fine-tuning

- **Pre-training** is the process of training an LLM on a massive dataset, to learn linguistic patterns and knowledge — which can then be fine-tuned later for a specific application.
- **Fine-tuning** further trains the pretrained model on a smaller, task-specific dataset, to adapt it to a particular application.

---

## Pretraining Objectives by Architecture

Each Transformer architecture is trained with a pretraining objective that matches how it will actually be used later:

| Architecture | Objective | What it does |
|---|---|---|
| **Encoder-only (BERT)** | Masked Language Modeling (MLM) | Some words are masked/corrupted, and the model predicts the original words using **both** left and right context |
| **Encoder-only (BERT)** | Next Sentence Prediction (NSP) | Given two sentences, predict whether the second actually follows the first — teaches sentence-level relationships |
| **Decoder-only (GPT)** | Causal Language Modeling (CLM) | The model sees only previous tokens, and learns to predict the next one — enables left-to-right generation |
| **Encoder-decoder (T5)** | Denoising Seq2Seq (Span Corruption) | Entire **spans** of consecutive words are masked; encoder reads the corrupted input, decoder reconstructs the missing spans |

> 💡 **MLM vs. Span Corruption**: MLM masks individual tokens/words; Span Corruption masks entire **continuous spans** of tokens (e.g., hiding a chunk of 10 words at once). In Span Corruption, the model isn't even told *how many* words were removed — it might just see a single sentinel token representing the hidden span. So the model must predict both the **content** and the **length** of the missing segment, making it a significantly harder "exam" than simple fill-in-the-blank.

---

## Masked Language Modeling (MLM), In Depth

### 1. Core Idea: Fill in the Blanks

MLM works like a fill-in-the-blanks exercise:
- A sentence is taken, and a small percentage of its tokens (typically **15%**) are randomly replaced with a special `[MASK]` token.
- The model sees the entire sentence except for the masked token(s), and must predict the original hidden token.

### 2. Objective: Learning Bidirectional Context

Unlike traditional language models that read text only left-to-right, MLM lets the model use information from **both** directions:
- The words *before* the masked token.
- The words *after* the masked token.

By combining context from both sides, the model learns richer representations of grammar, meaning, and word relationships.

### 3. Subword Tokenization: WordPiece

BERT uses **WordPiece** tokenization, splitting words into smaller subword units instead of treating every word as a single token.

- Example: `"cats" → "cat" + "##s"`
- The `##` prefix indicates the token is a **continuation** of the previous subword.

Since MLM operates at the token level, it may mask only a root token (e.g., "mat") while leaving other subword tokens (e.g., "##s") visible — letting those remaining pieces provide additional context to help predict the masked token.

### Token-Level Masking — The Drawback

Standard MLM is performed at the **token** level: when a word is split into subwords, token-level masking treats each token independently.

> 💡 **Worked example**: `"mats" → "mat" + "##s"`. If only "mat" is masked: `"The cat ##s sat on the [MASK] ##s."`
>
> **The problem**: because the plural suffix "##s" is left *unmasked*, the model gets a huge hint! It becomes much easier to guess "mat" simply by looking at the remaining suffix "##s," rather than truly understanding the whole sentence's context.

> This exact leak is what **Whole Word Masking** (below) was designed to fix.

### The 80/10/10 Rule

During BERT training, 15% of all words are first selected as prediction targets. Regardless of what happens next, the model's job is **always** to predict the original selected word using the surrounding context.

However, BERT does **not** replace all selected words with `[MASK]`, because real-world text never contains `[MASK]` tokens — the model would become overly dependent on them during training. Instead:

- **80%** of selected words are replaced with `[MASK]` — forcing the model to infer the missing word from surrounding context.
- **10%** are replaced with a **random** word — exposing the model to incorrect/noisy input, so it learns not to blindly trust the current word and instead rely on context.
- **10%** are left **unchanged** — but the model is *still* expected to predict that same word, and isn't told it was intentionally kept.

> 💡 **Why the unchanged 10% matters**: this prevents BERT from assuming that *only* `[MASK]` positions require prediction, and encourages it to build strong contextual representations for **every** token — not just the visibly masked ones. As a result, BERT becomes robust to missing words, incorrect words, and normal text, while reducing the mismatch between its training setup and real-world usage.

> 💡 **Reframing "masked" as "test" tokens**: BERT randomly selects 15% of tokens in each input sequence to mask. This means BERT only compares these 15% tokens against the ground truth to evaluate loss. Instead of thinking of them as "masked tokens," it's more useful to think of them as **"test tokens."**

*Reference: [BERT MLM — worked example](https://chatgpt.com/s/t_6a62d4cbd8f4819186195d6b90c485d1)*

### Weaknesses of Masked Language Modeling (MLM)

1. **Training–inference mismatch**: during training, the model sees `[MASK]` tokens, but real-world text (during fine-tuning or inference) never contains `[MASK]`. The 80/10/10 strategy *reduces*, but doesn't *eliminate*, this mismatch.

2. **Cannot generate text naturally**: MLM predicts only the masked words, not the next word one by one — so BERT isn't suitable for chatbots or story generation.

3. **Only a fraction of tokens provide a prediction loss**: since only 15% of tokens are selected, the model directly learns from predicting only those tokens per training example, while the rest mainly serve as context.

4. **Requires specially prepared masked inputs**: training data must be modified by selecting and masking tokens before being fed to the model — more complex than standard next-token prediction.

5. **Less suitable for open-ended generation**: since MLM learns to fill in blanks using both left and right context, it's optimized for language *understanding*, not autoregressive generation.

---

## Whole Word Masking (WWM)

Instead of masking individual subword tokens, WWM masks **all** subword tokens belonging to the same word.

- In standard BERT training, masking happens at the subword token level. If a word like "discovers" is split into `dis + ##covers`, the model might mask `dis` while leaving `##covers` visible — a leak.
- **WWM fixes this**: if *any* subword token of a word is selected, the **entire word** is masked together.

> 💡 **Worked example**:
> - Original: *"I am playing football."*
> - WordPiece tokens: `I | am | play | ##ing | football`
> - After Whole Word Masking: `I | am | [MASK] | [MASK] | football`
> - Now the model must predict **both** "play" and "##ing," using only the surrounding context — no partial hints available.

**Objective**: force the model to learn the meaning of the entire word from context, rather than guessing it from visible subword pieces.

**Advantages:**
- Harder, more realistic prediction task.
- Prevents the model from exploiting visible subword fragments.
- Produces better contextual word representations.
- Often improves performance on downstream NLP tasks.

**Used by**: BERT (Whole Word Masking variant), RoBERTa (can be trained with WWM), and other BERT-based models using WordPiece tokenization.

---

## Causal Language Modeling (CLM)

In CLM, a language model is trained to predict the **very next** word/token in a sequence, based only on the words that came before it.

- **"Causal" (unidirectional)**: the model reads left-to-right. When predicting word `N`, it's strictly forbidden from looking ahead at word `N+1` or any future words — enforced via a **causal mask**.
- **Contrast with BERT**: unlike BERT (which looks both directions to fill in `[MASK]` tokens), CLM models only look **backward** at past tokens.

### Why Does CLM Achieve State-of-the-Art Performance?

CLM achieves SOTA performance because its **training objective matches its real-world usage exactly**: trained to predict the next token given all previous tokens, and at inference it generates text in exactly the same way — no train/inference mismatch.

- Unlike MLM (where only ~15% of tokens are prediction targets), CLM learns from **every** token in the training data — more efficient learning.
- This objective naturally teaches the model to generate coherent text, letting it perform a wide variety of tasks — conversation, coding, translation, summarization, reasoning — simply through prompting.
- CLM scales exceptionally well with larger models, more data, and more compute — which is why modern LLMs (GPT, Llama, Qwen) are built on this approach.

---

## Next Sentence Prediction (NSP)

BERT receives two sentences separated by a `[SEP]` token, and must predict whether sentence B actually follows sentence A in the original text, or is a randomly sampled sentence.

**Examples:**
- **Positive pair (IsNext)**: `[CLS] The cat sat on the mat [SEP] It purred softly [SEP]` → IsNext
- **Negative pair (NotNext)**: `[CLS] The cat sat on the mat [SEP] Stock prices rose sharply [SEP]` → NotNext

- Training data is constructed **50/50**: half the time B is the real next sentence, half the time it's random.
- The `[CLS]` token's output representation is fed into a binary classifier.

**The idea**: teach BERT inter-sentence coherence, for downstream tasks like question answering and natural language inference.

### NSP Turned Out to Be a Weak Objective

- **RoBERTa** showed that removing NSP entirely, and just using MLM with longer sequences, actually **improved** performance.
- **Likely reason**: the "NotNext" negatives are too easy — a random sentence from a different document is trivially distinguishable, so the model doesn't learn much useful signal from the task.

*Reference: [NSP Loss — details](https://chatgpt.com/s/t_6a62ded23c848191871eae4770f14140)*

---

## Encoder-Only vs. Decoder-Only vs. Encoder-Decoder — The Task-Type Mapping

| Architecture | Task type | Purpose |
|---|---|---|
| **Encoder-only** | NLU (Natural Language **Understanding**) | Developing a rich understanding of language |
| **Decoder-only** | NLG (Natural Language **Generation**) | Generating text |
| **Encoder-decoder** | Translation (and other seq2seq tasks) | Remains the primary choice, though more computationally expensive |

---

## BERT's Bidirectional Conditioning, Formally

- **Standard language models (like GPT)** calculate probability looking only to the **left**:
  ```
  −Σ log P(x_i | x_1 ... x_{i-1})
  ```
- **BERT** calculates probability looking at **both sides simultaneously**, conditioning on the full sequence `x = x_1, x_2, ..., x_n` (minus the masked positions).

> 💡 **Why it matters**: seeing context from both sides lets BERT capture richer dependencies and nuances — making it far superior for understanding full sentences, context, and word meanings (though, as covered above, unsuitable for autoregressive generation).

> 💡 **On input representation**: in the case of input IDs, it helps to think of them as a **one-hot vector**, where the specified position is `1` and everywhere else is `0` — the same representation convention used throughout tokenization and embeddings elsewhere in this course.

---

## The Pre-Training / Fine-Tuning Paradigm

BERT introduced a **two-phase workflow** that defined NLP for several years:

### Phase 1: Pre-training (expensive, done once)

- Train on massive unlabeled text, for hundreds of thousands of steps, on several GPUs in parallel.
- This produces general-purpose language representations.
- **Cost**: tens of thousands of dollars.

### Phase 2: Fine-tuning (cheap, done per task)

- Add a simple classification head (usually a single linear layer) on top of the `[CLS]` token's representation.
- Train on task-specific labeled data (e.g., 10K labeled sentiment examples) for a few epochs, on a single GPU.
- **Cost**: minutes to hours.

> 💡 **Reframing prompting through this lens**: prompting can be viewed as a way to frame a specific task such that the model — even if not explicitly trained for that exact objective — interprets the input through the lens of a task it's *already* capable of performing. This is conceptually a lightweight, training-free alternative to the fine-tuning phase above.

### Further Reading

- [BERT Architecture — Summarized](https://chatgpt.com/s/t_6a62e2a969948191a993eeb9329b6a77)
- [How BERT Transforms Plagiarism Detection](https://chatgpt.com/s/t_6a62e2a969948191a993eeb9329b6a77) *(applied use case)*

**Complete BERT Pretraining Pipeline** (7-part walkthrough):
1. [Part 1](https://chatgpt.com/s/t_6a62e4f42a34819182b3c40f6222cb49)
2. [Part 2](https://chatgpt.com/s/t_6a62e5594c788191bec87bdfb11d260b)
3. [Part 3](https://chatgpt.com/s/t_6a62e65762548191a70bb91badf53f67)
4. [Part 4](https://chatgpt.com/s/t_6a62e67a5edc8191984670ead5886b2d)
5. [Part 5](https://chatgpt.com/s/t_6a62e6e6ee00819196fc64a173dcb9cb)
6. [Part 6](https://chatgpt.com/s/t_6a62e80dcda8819189f42dfdbcb9c7bc)
7. [Part 7](https://chatgpt.com/s/t_6a62e8575878819188ee72d9883836e6)
- [Summary of the full pipeline](https://chatgpt.com/s/t_6a62e89dedfc8191896f0ea39bcbddbe)

**Fine-tuning BERT**: [ChatGPT conversation](https://chatgpt.com/s/t_6a62ea81f290819184e159800bffca3e)

---

## Summary: Why All These Objectives Exist

All of these techniques exist because the model has **no labels** telling it what language means — it only has raw text. The pretraining objective creates a **self-supervised learning task**, where the input text itself provides the supervision. By solving this artificial task millions or billions of times, the model learns grammar, vocabulary, semantics, facts, and relationships between words, **without human annotations**.

| Objective | What it teaches |
|---|---|
| **Masked Language Modeling (MLM)** | Understand language by predicting missing words using both left and right context — strong contextual representations for language understanding |
| **Whole Word Masking (WWM)** | Prevents cheating via visible subword pieces — forces inferring the meaning of the entire word from context |
| **Next Sentence Prediction (NSP)** | Relationships between sentences, via predicting whether two sentences naturally follow each other *(later found to be not very beneficial — RoBERTa removed it)* |
| **Causal Language Modeling (CLM)** | Predict the next token — exactly matching how modern LLMs generate text; ideal for chatbots, coding, reasoning |
| **Span Corruption (T5)** | Reconstruct entire missing spans instead of single words — understanding larger chunks of text, naturally fitting encoder-decoder architectures for translation/summarization |

**The overall purpose**: all these objectives are different ways of creating a learning signal from unlabeled text. They force the model to infer missing information from context — which is how it learns the statistical patterns, syntax, semantics, and world knowledge encoded in language.

**The choice of objective depends on the architecture and the model's intended use**:
- **MLM** → for understanding (encoder-only)
- **CLM** → for generation (decoder-only)
- **Span Corruption** → for sequence-to-sequence transformation (encoder-decoder)
