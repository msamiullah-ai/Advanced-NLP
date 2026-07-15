# Tokenization II: WordPiece, Unigram, Viterbi & Perplexity

> Related tokenization algorithms: **WordPiece, BPE, Unigram, and SentencePiece**

## WordPiece Tokenization

WordPiece is a **greedy, statistical algorithm** that turns text into a hybrid of characters and frequent subwords. It prioritizes tokens that form "tight" statistical bonds while keeping individual characters as a safety net to ensure **100% coverage** of all possible words.

### The Algorithm

**1) Initial Representation**

Given a training corpus, each word is initially represented as individual characters. The **first character is stored normally**, while every subsequent character is prefixed with **`##`** to indicate it is a continuation of the same word.

> 💡 **What `##` means**: Any character (or subword) that is **not** at the start of a word gets a `##` prefix. This tells the model: "Attach this directly to the preceding token without a space."

Example:
```
"playing" → p ##l ##a ##y ##i ##n ##g
```

**2) The Scoring Formula**

WordPiece doesn't just look for what is most common (unlike BPE) — it looks for pairs that **co-occur more than expected by chance**:

```
SCORE = P(pair) / (P(first) × P(second))
```

> 💡 **In simple words**: This is essentially asking "do these two pieces show up together *more often than random chance would predict*?" rather than just "how often do they show up together?" — this is the key mathematical difference from BPE, which only counts raw frequency.

**3) Score Every Pair** — calculate the score for every adjacent pair in the current corpus, alongside the individual score/count of each occurrence.

**4) Select** — pick the pair with the highest score.

**5) Merge** — combine that pair into a single new token (e.g., `##in + ##g → ##ing`).

**6) Update** — append this new token to your vocabulary.

**7) Re-scan** — replace all instances of the old pair in your training corpus with the new, single token.

> 💡 **Clarification**: This replacement happens in the *corpus* (e.g., wherever a word was broken into individual chars, `##i, ##g` becomes `##ig`) — but the *old individual tokens stay in the vocabulary itself*, they just aren't re-counted as separate entities for future merging.

### The "Safety Net" — Why Old Tokens Remain

Even after creating `##ing`, you do **not** delete the individual `##i`, `##n`, or `##g` tokens from your vocabulary.

**Why**: The vocabulary must remain capable of forming *any* word. If you encounter a rare word like "ignite," you can't use the `##ing` token because the letters are in a different sequence. You must be able to fall back on your "atomic" building blocks (`##i, ##g, ##n, ##t, ##e`) to represent the word accurately.

**8) Final Output**

The process concludes when the vocabulary reaches the desired size. You're left with:
- **A Base Vocabulary**: single characters plus common subword chunks (like `##ing`, `##ed`, `##ment`).
- **Tokenization Logic**: a system that always prefers the **longest possible matches** from your vocabulary — common patterns get single, efficient tokens, while rare words fall back to their smallest, safest components.

As the algorithm progresses, it "absorbs" smaller tokens into larger ones (like `##ig`), yet retains the original smaller building blocks (`##i`, `##g`) in its vocabulary list as a critical safety net. This keeps the model flexible — able to "spell out" any word, even ones that don't fit common patterns — balancing word-level understanding with character-level coverage.

### The Result: A "Hybrid" Vocabulary

Example outcome after 12 merges: a compact, intelligent vocabulary of 26 tokens, no longer just individual letters, but a blend of:
- **Original Characters**: used for the absolute basics (`p, h, r, ##a, ##d...`).
- **Meaningful Subwords**: high-frequency chunks representing common linguistic patterns (`##in, ##ing, ##ay, ##ping`).

### How It's Used — The Tokenization Phase

This vocabulary translates any new, unseen text into a format the AI can process, following a **"Longest-Match-First"** strategy.

**For a newer corpus**: WordPiece processes the word **left to right**. At each step, it considers the remaining unprocessed part of the word and searches the vocabulary for the longest token that matches the current position.
- If a match is found → output that token, move the pointer to its end.
- If no valid token matches → the entire word is replaced with `[UNK]`.

## Why Is BPE More Commonly Used Than WordPiece?

*Reference: [ChatGPT conversation link](https://chatgpt.com/s/t_6a54fdfb42ec81919f2d0a27d3b846cd)*

**The real reason**: Modern LLMs use **Byte-Level BPE**, which guarantees **100% coverage** of any input text (no `[UNK]` tokens), making it much more robust for real-world data.

| | WordPiece | Byte-Level BPE |
|---|---|---|
| **Biggest limitation** | If a word cannot be decomposed into known subwords → `[UNK]` | Never outputs `[UNK]` |
| **Starting point** | Characters | Bytes (0–255) |
| **Coverage** | Not guaranteed 100% | Guaranteed 100% — every possible text can always be represented |

## Unigram Tokenization

*Reference: [Gemini conversation link](https://share.gemini.google/KZx3bxskvz19)*

Unlike BPE (which **merges** characters upward) or WordPiece (which **grows** a vocabulary), **Unigram starts with everything and prunes away the useless stuff** — a top-down rather than bottom-up approach.

> 💡 **Core rule**: The best vocabulary is the one that **minimizes the "cost"** (Negative Log-Likelihood) of representing your data.

### The Step-by-Step Lifecycle

The algorithm runs in a repeating cycle of four main stages:

#### Stage A: Initial Inventory (The Candidate Pool)

The algorithm scans your corpus and collects **every unique character and substring** as a "candidate" token. It doesn't care if they're real words — it just records their frequency to establish a baseline.

#### Stage B: The E-Step (Expectation / Segmentation)

This is where the model "prices out" the best way to spell every word.

- **The Problem**: There are many ways to slice a word (e.g., `abc` could be `a+b+c`, `ab+c`, `a+bc`, or `abc`).
- **The Solution**: The model calculates the **Negative Log-Likelihood (NLL)** for every possible path.
- **The Choice**: It chooses the path with the **lowest NLL** (the "cheapest" path) — this is the path of highest probability.

> 💡 **"E-Step" name origin**: This mirrors the Expectation step in the classic Expectation-Maximization (EM) algorithm from statistics — you're figuring out the best "expected" segmentation given current token probabilities.

#### Stage C: The M-Step (Maximization / Re-estimation)

Once the model has "voted" on the best way to segment every word, it updates its tally:
- It counts how many times each token was actually **used** in those best segmentations.
- It recalculates the probability `p(x)` for each token based on those new counts.
- **Result**: Tokens frequently used in the best segmentations become more "valuable" (higher probability); others fade away.

#### Stage D: Pruning (The Efficiency Filter)

The "cleanup" stage. The model asks: *"If I delete this token, how much does my total cost (NLL) increase?"*
- It removes tokens that have the **lowest impact** on total cost.
- This forces the model to represent text using a smaller, more efficient set of tokens.

### The Repeating Loop

After pruning, the algorithm goes back to **Stage B (E-Step)**. Because the dictionary has changed (some tokens are gone), the model must re-evaluate the best way to segment words using the tools it has left.

This cycle continues until you hit your target vocabulary size.

> 💡 **Big picture summary**: Unigram is like starting with a giant, wasteful toolbox of every possible piece, then repeatedly asking "which tools do I actually use well?" and throwing out the ones that don't earn their place — until only the most efficient toolkit remains.

## Perplexity

**Perplexity** is a key metric for evaluating how well a language model predicts text.

### Relationship to Loss

Perplexity (`P`) is calculated by exponentiating the average negative log-likelihood (`H`):

```
P = e^H
```

> 💡 **In simple words**: If your model has a "loss" (uncertainty score), you convert it to perplexity to get a more intuitive number to reason about.

### What Perplexity Actually Means

It answers the question: *"On average, how many different choices does the model think are likely at each step?"*

- **High Perplexity**: The model is confused — it sees many equally likely (and potentially wrong) options for the next word.
- **Low Perplexity**: The model is confident and accurate — it has narrowed down the next word to a small number of likely candidates.

## The Viterbi Algorithm

*Reference: [ChatGPT conversation link](https://chatgpt.com/s/t_6a55274745b08191aa19d772067e9a99)*

Given the current vocabulary and token probabilities, **Viterbi finds the tokenization with the highest probability** — which is equivalently the tokenization with the **lowest Negative Log-Likelihood (NLL)**.

> 💡 **Where this fits in**: Viterbi is the algorithm that actually *performs* the segmentation search described in Unigram's E-Step (Stage B above) — it efficiently finds the best path without having to brute-force every possible way to slice a word.

### Core Assumption: Independence

The model treats every token as **independent**. If you break a word into a sequence of tokens `(x1, x2, ..., xn)`, the probability of that whole sequence is simply the product of the individual token probabilities:

```
P(x) = P(x1) × P(x2) × ... × P(xn)
```

**Key relationship**: LOWER NLL = BETTER SEGMENTATION = HIGHER PROBABILITY for a token sequence.

The algorithm is constantly trying to "find the path of least resistance" (lowest NLL) to represent any given word.

### The NLL Formula (Log Form)

```
NLL = −log P = −log P(x1) − log P(x2) − … − log P(xn)
```

> 💡 **Why use logs instead of multiplying probabilities directly?** To handle **numerical instability**. Multiplying many small probabilities together (each less than 1) quickly produces an extremely tiny number that computers struggle to represent precisely. Taking the negative log turns multiplication into addition and keeps the numbers in a stable, manageable range.
