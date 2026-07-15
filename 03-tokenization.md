# Tokenization

## What Is Tokenization?

Tokenization is the process of segmenting raw text into discrete units called **tokens**.
- These tokens can be **words**, **subwords**, **characters**, or **bytes**.

> 💡 **In simple words**: A tokenizer is like a pair of scissors that cuts a sentence into pieces a model can actually work with — the pieces just aren't always "whole words."

By breaking a sentence down into these hierarchical building blocks, a model moves from seeing "just text" to understanding the underlying rules of language. Once the model understands these building blocks, it can represent the sentence in a way that captures relationships and context, allowing it to perform tasks like translation or summarizing without getting confused by the complex, layered nature of human speech.

### Why Is Tokenization Done?
1. **Translating Text to Numbers** — models only understand numbers, not raw text.
2. **Creating a Vocabulary** — defining the finite set of units the model will ever know.
3. **Managing Input** — controlling sequence length and computational cost.

### The Four Main Types
1. Word-Level Tokenization
2. Character-Level Tokenization
3. Subword Tokenization
4. Byte-Level Tokenization

*(You have to decide on the tokenization strategy at the very beginning of building a model — it shapes everything downstream.)*

## The Linguistic Hierarchy

NLP understands language through a hierarchy of **six levels**, where each level builds on the previous one:

1. **Phonetics/Phonology** — physical speech signals converted into abstract **phonemes** (basic speech sounds).
2. **Morphology** — phonemes combine to form **morphemes**, the smallest units that carry meaning, which together create words.
3. **Syntax** — defines how words are arranged into grammatically correct sentences.
4. **Semantics** — determines the sentence's literal meaning and the conditions under which it is true.
5. **Pragmatics** — interprets the speaker's *intended* meaning using context, rather than relying only on literal words.
6. **Discourse** — connects multiple sentences together, enabling coherent communication by maintaining context, resolving references, and understanding the overall flow of a conversation.

> 💡 **In simple words**: Humans don't always mean exactly what they literally say — the same sentence can be interpreted differently depending on context. That's why pragmatics and discourse exist as separate levels above plain semantics.

## How Text Flows Through a Neural Network

```
Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Embedding Layer (lookup)
   ↓
Embedding Vectors
   ↓
Transformer / Neural Network
   ↓
Logits
   ↓
Softmax
   ↓
Predicted Token ID
   ↓
Token (Word) → Output Text
```

**Step by step:**
1. Input text is **tokenized** into smaller units (tokens).
2. Each token is assigned a unique **token ID** based on the model's vocabulary.
3. Token IDs pass through an **embedding layer**, which maps each ID to a fixed-length embedding vector (e.g., 768 or 1024 dimensions). These values are *learned* during training and capture semantic/syntactic relationships.
4. Embedding vectors become the actual numerical input to the neural network (e.g., a Transformer).
5. The network processes these vectors through multiple layers and produces a **logit** (score) for every token in the vocabulary — representing how likely each token is to be next.
6. **Softmax** converts logits into probabilities.
7. The token with the highest probability (or one sampled from the distribution) is selected as the prediction.
8. The predicted token ID is mapped back to its corresponding token (word/subword) to generate output text.

## Types vs. Tokens

- **Token** = each individual split entry from a sentence, no matter if it repeats or not.
- **Type** = a *unique* token (i.e., counted only once, regardless of how many times it appears).

> 💡 **In simple words**: "the cat sat on the mat" has 6 tokens but only 5 types (because "the" repeats).

### Type-Token Ratio (TTR)

TTR is how computational linguists and AI researchers mathematically measure **vocabulary variety/diversity**.

**Formula:**
```
TTR = (Number of unique words / Total number of words) = Types / Tokens
```
- Keeping the total number of tokens constant will mathematically increase the TTR when unique words increase.

**Interpretation:**
- **High TTR** (close to 1) → many unique words → rich vocabulary → you're encountering new, unique words frequently rather than repeating the same ones.
- **Low TTR** (close to 0) → many repeated words → less diverse vocabulary.

**Why does TTR decrease for longer texts?**
As a document gets longer:
- Common words ("the," "is," "and," "of," etc.) appear repeatedly.
- Tokens increase faster than Types.
- Therefore, the TTR value decreases over the length of a longer text.

## Vocabulary

**Formal definition**: The collection of all unique tokens used by the tokenizer, where each token has a unique ID.

- **Types**: unique tokens in a *specific* text or corpus.
- **Vocabulary**: the *complete set* of unique tokens defined by the tokenizer/model.

> 💡 **In simple words**: Think of vocabulary as "the list of words we decided to keep from the training corpus" — it's a fixed, pre-selected list, not just whatever appears in any given piece of text.

**Core vocabulary pipeline:**
1. Convert tokens to IDs.
2. Convert IDs back to tokens.
3. Create the embedding matrix.
4. Predict the next token.

**Important nuance**: In the context of Zipf's Law and vocabulary coverage, "vocabulary" means the set of unique words selected from the *training* corpus (usually the most frequent words) — **not** words from an unseen inference dataset.
- Before training, we choose which words to include in the vocabulary based on their frequencies.
- **Coverage** measures what percentage of all token occurrences in the training corpus belong to these selected vocabulary words.
- After the vocabulary is fixed, the neural network learns token IDs and embeddings **only** for these words.

## Word-Level Tokenization

### Algorithm
```
Input: Text string S
Output: List of word tokens T

1. Normalize the text
   a. Remove extra whitespace.
   b. (Optional) Convert text to lowercase.

2. Split the text into words using whitespace.

3. For each word:
   a. Separate punctuation from word boundaries.
   b. Handle special cases (e.g., contractions such as "don't" and hyphenated words).

4. Return the list of word tokens T.
```

### The Out-of-Vocabulary (OOV) Problem

> 💡 **Key clarification**: OOV occurs during **inference**, not during training.

- During training, a word-level tokenizer builds a fixed vocabulary from all unique words in the training corpus. The network learns embeddings only for words in this vocabulary.
- During **inference**, if the model receives a new input containing a word that was **not** present in the training vocabulary, the tokenizer cannot assign it a valid token ID.
- Instead, it replaces the unseen word with the special **`<UNK>`** (Unknown) token.
- As a result, the model loses the specific meaning of that word — this is the **Out-of-Vocabulary (OOV) problem**.

This is one of the main reasons modern LLMs use **subword tokenization** instead of pure word-level tokenization.

## Zipf's Law and Vocabulary Distribution

**Zipf's Law**: In natural language, the frequency of a word is inversely proportional to its rank in the frequency table.

```
f(r) = C / r
```
- `f(r)` = frequency of the word with rank `r`
- `r` = rank of the word (1 = most frequent, 2 = second most frequent, etc.)
- `C` = constant (approximately equal to the frequency of the most common word)

*Reference: [ChatGPT conversation link on Zipf's Law](https://chatgpt.com/s/t_6a53b8f5121c81919a0785f8aadf0fb2)*

- **Frequency** = the number of times a token (or word) appears in the corpus/dataset.
- **Rank** is *not* assigned based on the frequency value itself — rank is simply the position **after sorting** words by frequency.

### Long-Tail Distribution

Language follows a **long-tail distribution**:
- **Head**: A few very frequent words.
- **Tail**: Millions of rare words with very low frequencies.

> 💡 **In simple words (80/20 rule)**: A tiny percentage of words make up the vast majority of everything we ever say or write. A very small number of tokens appear most of the time, while a huge number of unique, rare words appear only once or twice. This is why engineers don't need to include every obscure word in a model's vocabulary — they focus on the most frequent ones, because that covers the most ground.

**A few words are extremely common. Most words are extremely rare.**

### Coverage — High Coverage at Low Cost

*Reference: [ChatGPT conversation link on Coverage](https://chatgpt.com/s/t_6a53e1781be48191b5d2d3921eea6957)*

**Coverage** = the percentage of token occurrences in a corpus (typically the training corpus) that are represented by the selected vocabulary — **not** a ratio between training tokens and inference tokens.

> 💡 **Worked example**: Suppose the training corpus contains 1,000,000 total tokens, and the word "the" appears 100,000 times. If our vocabulary currently contains only the word "the," it covers 100,000 out of 1,000,000 token occurrences — i.e., **10% coverage**.

- Vocabulary = the set of unique words/tokens selected from the training corpus to be learned by the model (not words from an unseen real-world corpus).
- During training, the network learns embeddings only for words in this vocabulary.
- During inference, this fixed vocabulary is used to tokenize new input.
- If a new word wasn't part of the training vocabulary, a pure word-level tokenizer maps it to `<UNK>` → the OOV problem again.

### The Wall of Diminishing Returns

*Reference: [ChatGPT conversation link on Diminishing Returns](https://chatgpt.com/s/t_6a53e6dcb3788191974a8b78746b2659)*

The tokenizer-building algorithm selects the vocabulary **automatically** from the training corpus:

```
Training Corpus
        ↓
Tokenizer algorithm counts word/subword frequencies
        ↓
Vocabulary is selected (e.g., top 30k or 50k tokens)
        ↓
Assign token IDs
        ↓
Train the neural network
```

The developer usually specifies:
- **Vocabulary size** = e.g. 30,000
- **Tokenization method** = BPE, WordPiece, or SentencePiece

Then the tokenizer algorithm automatically analyzes the training corpus and selects the vocabulary.

> 💡 **Why "diminishing returns"?** Because of the long-tail distribution, the first few thousand vocabulary entries cover most of the text (high coverage gain per word added). But adding more and more rare words to the vocabulary gives smaller and smaller coverage improvements — hence a "wall."

## Heaps' Law

**Heaps' Law**: As the size of a corpus (number of words/tokens) increases, the vocabulary size (number of unique words/types) also increases, but at a **slower rate**.

- The more text you read, the more new words you discover.
- However, new words become less frequent over time because many words start repeating.

**Parameters:**
- `K` = Constant (depends on the corpus)
- `β` = Usually between 0.4 and 0.6

> 💡 **In simple words**: At the beginning of reading a corpus, almost every sentence teaches you new words. As the corpus gets larger, most new text repeats words you already know — so new vocabulary keeps growing, but more and more slowly.

## Character-Level Tokenization

**Characteristics:**
- ✅ No OOV error (can spell anything using its small character set)
- ❌ No inherent semantic meaning per token
- ❌ Long input sequences
- ❌ More computation, more memory, slower training
- Can tokenize every word, no matter how rare or unseen

> 💡 **In simple words**: Characters have little semantic meaning individually, so the model must learn to *compose* characters into meaningful words — this is a much harder learning job than working with whole words or subwords.

### The Quadratic Cost Problem

Character-level tokenization increases the number of tokens in a sequence. Since Transformer self-attention has **quadratic complexity O(n²)**, increasing the sequence length by a factor of `k` increases the computational cost by `k²`.

| Sequence length increase | Compute cost increase |
|---|---|
| 2× longer | 4× more computation |
| 3× longer | 9× more computation |
| 5× longer | 25× more computation |

This quadratic explosion is one of the main reasons modern LLMs prefer **subword tokenization** over pure character-level tokenization.

## Subword Tokenization

Subword tokenization breaks text into smaller, meaningful chunks — **smaller than a full word but larger than a single character**. It's designed to balance:
- The **efficiency** of word-level models (fast, but huge/inflexible vocabularies)
- The **flexibility** of character-level models (can spell anything, but loses semantic meaning)

**Advantages:**
- ✅ Handles unknown words
- ✅ Smaller vocabulary than word-level
- ✅ Shorter sequences than character-level
- ✅ Captures meaningful word parts (roots, prefixes, suffixes)
- ✅ Better computational efficiency

> 💡 **Important clarification**: The main purpose of a tokenizer is **not** primarily to create a diverse vocabulary. Its primary purpose is to convert raw text into tokens that a model can process — diversity/coverage is a *side effect* of doing this well.

### Key Ideas to Take Home
1. Anything that appears frequently is worth putting into the vocabulary.
2. Keep frequent character sequences together as one token, and split only when necessary.
3. Subword tokenization compresses the vocabulary by replacing rare whole words with combinations of frequently occurring subword units.
4. Common words are kept whole; rare words are split into chunks.

- We can "spell" a large number of rare words using a small, fixed set of common parts.
- We maintain the efficiency of the **Head** while handling the complexity of the **Tail**.

**Coverage** (in this context) = the tokenizer's ability to represent all words in a corpus without producing unknown (`<UNK>`) tokens. Subword tokenization provides very high coverage because unseen words can be decomposed into known subword units.

### Why & How

**Why it's done**: To eliminate "out-of-vocabulary" errors, keep vocabulary sizes manageable, and allow the model to process any input by decomposing unknown words into known sub-units.

**How it's done**: Subword tokenization starts by representing every word in the corpus as the smallest units (characters or bytes). It repeatedly finds the most frequently occurring adjacent units across the entire corpus and merges them into larger subword tokens. This continues until the desired vocabulary size is reached. Once learned, any new or rare word is tokenized by breaking it into the **longest matching subwords** from the learned vocabulary, rather than treating the entire word as a unique token.

## Byte-Level Tokenization

Byte-Level Tokenization represents text as **UTF-8 bytes** (values 0–255) and learns frequent byte-sequence merges (typically using BPE), allowing it to tokenize **any** text without ever producing an unknown token.

- Gives a fixed initial vocabulary of **256 byte tokens**.
- Since every character in every language can be encoded into bytes, it can tokenize any input without producing an `<UNK>` token.

> 💡 **In simple words**: Byte-level tokenization is the "ultimate fallback" — no matter what language, emoji, or symbol appears, it can always be represented as bytes, so there's truly never an unknown token.

## Byte-Pair Encoding (BPE)

### The `</w>` End-of-Word Marker

The `</w>` symbol is a special token used by the BPE algorithm to mark the **end of a word**. This is a crucial pre-processing step for training LLMs.

**Why it's used — Defining Boundaries**: Without this token, the model wouldn't know where one word ends and the next begins. By attaching `</w>` to the end of every word (e.g., "old" becomes `old</w>`), the tokenizer explicitly encodes that the word has concluded at that point.

**Example — "estimate" vs. "highest":**
- In *"highest,"* the word ends after the `est` sequence, so it's represented as `est</w>`.
- In *"estimate,"* the `est` is **not** at the end of the word, so it's just `est` (no marker).
- This allows the model to correctly distinguish words with the same letter sequence but different grammatical functions or meanings.

### The BPE Training Algorithm (Complete, Step-by-Step)

**Goal**: Transform a corpus of text into a set of efficient, high-frequency "subword" tokens. This allows the model to handle any word — even unseen ones — by breaking them into familiar pieces.

#### Phase 1: Initialization

1. **Tokenize into Characters** — take every word in your corpus and split it into its individual constituent characters.
2. **Add Boundary Markers** — append `</w>` to every word, so the model can differentiate a character sequence that's part of a word from one that ends a word.
3. **Build Initial Vocabulary** — your starting vocabulary consists of every unique character found in your text, plus the boundary marker.
   - Example: if your text is "play," your initial tokens are: `p, l, a, y, </w>`.

#### Phase 2: The Iterative Merging Loop

Repeat until the vocabulary reaches your predefined target size (`B`):

1. **Count Frequencies** — scan the entire corpus and identify all adjacent token pairs (e.g., in `p l a y </w>`, the pairs are `pl`, `la`, `ay`, `y</w>`). Count the frequency of every pair across the whole document.
2. **Identify the Winner** — find the pair that appears most frequently. (Tie → pick one arbitrarily.)
3. **Merge** — combine the winning pair into a single, new, consolidated token (e.g., `p + l → pl`).
4. **Update Vocabulary** — add this new token to your master vocabulary list.
5. **Update Corpus** — scan through all words and replace every instance of the old separate tokens with the new combined token. Subtract the counts of the individual parts that were just merged, since they're now part of the new, larger token.
6. **Record the Rule** — save this merge as an official rule in your **Ordered Merge Table**, e.g.: `Merge: p + l -> pl`. You'll need this exact ordered list of rules later to tokenize new, unseen text.

#### Phase 3: Stopping Criteria

Continue iterating until you reach a predefined goal — either a specific total vocabulary size or a target number of iterations. As a general rule, you stop merging when no further significant frequency gains can be found, or your desired vocabulary size is achieved.

#### Phase 4: Final Output

- **Final Vocabulary**: A collection of subwords ranging from single characters to common morphemes (like "ing" or "ed").
- **Ordered Merge Rules**: A specific list of instructions that must be applied in the exact same order to any future, unseen text.

### A Concrete Worked Example

Training corpus: `{"playing", "played", "play"}`

**1. Setup:**
```
p l a y i n g </w>
p l a y e d </w>
p l a y </w>
```

**2. Merging (simplified steps):**

- **Merge 1**: The pair `p, l` appears most frequently.
  - Action: Merge `p` and `l` → `pl`.
  - Corpus update: `pl a y i n g </w>`, `pl a y e d </w>`, `pl a y </w>`.

- **Merge 2**: The pair `pl, a` now appears most frequently.
  - Action: Merge `pl` and `a` → `pla`.
  - Corpus update: `pla y i n g </w>`, `pla y e d </w>`, `pla y </w>`.

- **Merge 3**: The pair `pla, y` appears most frequently.
  - Action: Merge `pla` and `y` → `play`.
  - Corpus update: `play i n g </w>`, `play e d </w>`, `play </w>`.

### Why This Matters for New/Unseen Text

When you give your trained model a new word, like **"replayer"**:

1. **Character Split**: It starts as `r e p l a y e r </w>`.
2. **Sequential Rule Application**: The tokenizer applies your Ordered Merge Rules one by one (Rule 1, then Rule 2, etc.). It will eventually group `p+l` into `pl`, then `pl+a` into `pla`, then `pla+y` into `play`.
3. **Result**: The final tokenization for "replayer" might be `["re", "play", "er", "</w>"]`.

Because the model was trained on these merge rules, it can break down **any** word into learned subword building blocks — preventing the Out-of-Vocabulary problem entirely.
