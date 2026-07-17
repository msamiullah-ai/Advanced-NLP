# Word Embeddings: From Counting Words to Learning Meaning

---

## Why TF-IDF Isn't Enough

TF-IDF cannot answer *"what does 'bank' mean?"* or *"how similar are 'happy' and 'joyful'?"* — it can only answer *"how important is 'bank' in this document?"*

That gap is exactly why **distributional semantics** and word embeddings were developed.

*Reference: [From TF-IDF to Word Embeddings: Why Distributional Semantics Is Needed](https://chatgpt.com/s/t_6a583ba812dc81919eabad0382070a84)*

---

## The Distributional Hypothesis

> "You shall know a word by the company it keeps." — J.R. Firth (1957)

If two words consistently appear in **similar contexts**, they are likely to have similar meanings or belong to the same semantic category.

> 💡 **What "similar contexts" means**: two words frequently appear surrounded by the same neighboring words, sentences, or grammatical structures.

This became the standard way to represent word meaning in NLP, built up by several linguists:

- **Joos (1950), Harris (1954), Firth (1957)**: Two words that occur in very similar distributions (whose neighboring words are similar) have similar meanings.

- **Zellig Harris (1954)**: If A and B have almost identical environments, we say that they are synonyms.

The natural next step: represent words as **distributions over contexts** — i.e., words can be vectors too. Two words are similar in meaning **if and only if their context vectors are similar**.

---

## Representing Words as Word Vectors: The Word-Word Matrix

A **word-word matrix** (also called a **term-context matrix**) represents words as vectors based on their surrounding contexts. By comparing these vectors, NLP models can determine that words with similar context vectors — such as "digital" and "information" — are likely to have similar meanings.

**Quick comparison:**

| Matrix type | Columns represent |
|---|---|
| **Term–Document Matrix** | Documents |
| **Word–Word Matrix** | Context words |

Instead of representing a *document* by the words it contains, we now represent each *word* by the words that commonly appear around it.

- Rows = target words; Columns = context words (both from the corpus vocabulary) → matrix has `|V| × |V|` dimensions, where `|V|` is the vocabulary size.

- Each cell = the **co-occurrence count**: how many times the target word (row) and the context word (column) appear together in the chosen context.

**Two common ways to define "context":**

1. **Document context** — two words co-occur if they appear anywhere in the same document.

2. **Window context** — two words co-occur only if the context word appears within a fixed window (e.g., ±4 words) around the target word.

Since words with similar meanings usually appear with similar neighbors, they end up with similar context vectors — the foundation of distributional semantics and word embeddings.

---

## Vector Arithmetic & Semantic Relationships

Each word is represented as a vector in a high-dimensional space. During training, a model learns to place words so that similar meanings are close together, and similar *relationships* are represented by similar vector **differences**.

> 💡 **Note**: This directional-arithmetic property is most reliably observed in *learned* embeddings (like Word2Vec/GloVe) rather than raw co-occurrence counts — it's the payoff of learning compact, dense representations rather than just tallying counts.

The core idea: word relationships can be represented as vectors, where the **direction** and **magnitude** of the space between two words encodes a specific type of relationship (like gender, tense, or geography).

**Worked example:**

```
woman − man   → represents the gender relationship
queen − king  → represents the same gender relationship
```

Since these relationships are similar, the vectors are approximately equal:

```
woman − man ≈ queen − king
```

This structural consistency lets AI models understand, predict, and manipulate concepts based on their semantic context — rather than just literal keyword matching.

---

## From Representing Documents to Representing Words

This marks a major shift in NLP. Earlier, documents were represented as vectors of words (TF or TF-IDF). Now, each **word** is represented as a vector based on where it appears.

In a **term-document matrix**: rows = words, columns = documents, and each cell = the TF or TF-IDF value of a word in a document. Each **row** then becomes the vector representation of a word.

> 💡 **Worked example**: if "doctor" has high TF/TF-IDF values in medical documents and low values in legal documents, its vector might look similar to the vector for "hospital," because both words frequently occur in the same *types* of documents. Since their vectors are similar, cosine similarity would indicate they're semantically related.

This was an important milestone: it introduced the idea that a word's meaning can be represented by a vector describing its distribution — the foundation later built on by Word2Vec, GloVe, FastText, BERT, and modern language models. The key difference in later methods is that they replace *documents* with *contexts*, producing much richer semantic representations.

### Weaknesses of the Term-Document Matrix (as a word representation)

- Uses the **entire document** as context — too broad, so two words may appear in the same document without being closely related in meaning or usage.

- Captures **topical similarity** (same subject) rather than **local linguistic relationships** (words that actually occur together in sentences).

- Produces very **high-dimensional and sparse** vectors — most entries are zero, making storage and similarity computation inefficient.

- **Ignores word order** (bag-of-words approach).

- Sensitive to **document boundaries** — simply splitting or merging documents can change the word vectors.

---

## Term–Term (Word–Word) Co-occurrence Matrix

Represents each word by the words that appear **near** it, rather than by the documents it appears in. Instead of treating an entire document as context, a small **context window** (e.g., ±5 words) is used.

- Rows and columns both represent words from the vocabulary.

- Each cell stores how many times the column word appears within the context window of the row (target) word.

- Each **row** becomes a context vector describing the typical linguistic neighborhood of a word.

Because words with similar meanings usually occur with similar neighboring words, their vectors become similar — capturing **local semantic relationships** much better than the term-document matrix.

> 💡 **Example**: "doctor" frequently co-occurs with "prescribed," "patient," and "medicine" — making it possible to learn richer semantic information than topic-level similarity alone.

### Strengths

- A local context window captures meaningful **linguistic** relationships, not just topic similarity.

- Words with similar meanings or related concepts tend to have similar context vectors, making semantic similarity easier to detect.

- Easy to interpret — every dimension corresponds to a known context word.

### Weaknesses

- Extremely **large and sparse** (`|V| × |V|`) — expensive to store and compute.

- Dominated by very **frequent words** (the, of, and), whose high co-occurrence counts can overshadow meaningful content words.

- Raw co-occurrence counts can be misleading — they don't distinguish between words that co-occur due to a genuine semantic relationship vs. words that appear together simply because both are individually very frequent.

### A Few Important Notes

- **TF-IDF can automatically filter out noise words** — the ones with high probability of appearing across most documents.

- **Polysemy and homonymy limitation**: all senses of a word end up crammed into the **same single vector** — a raw co-occurrence matrix (like one-hot or TF-IDF-style representations) has no way to give "bank" (river) and "bank" (financial) separate vectors.

- **If vectors are already normalized** (unit length), **cosine similarity and dot product are exactly the same** — you don't need to compute cosine separately in that case.

---

## PPMI (Positive Pointwise Mutual Information) — Fixing the Frequency Problem

*Reference: [ChatGPT conversation on PPMI](https://chatgpt.com/s/t_6a5859d74b6c81918e7e3e4b1dd203ec)*

PPMI improves raw co-occurrence counts by measuring whether two words appear together **more often than expected by chance**, making it a statistically meaningful association measure.

### Strengths

- Naturally reduces the influence of common function words like "the," because their high frequency makes their co-occurrence less informative.

- Captures strong semantic relationships and performs well on word similarity tasks.

### Weaknesses

- Still produces a large, sparse word-word matrix — most word pairs have no observed relationship.

- Mainly stores **observed** co-occurrences and can't learn hidden semantic patterns — it struggles with unseen relationships (e.g., if "cardiologist" and "nephrologist" never appear together, PPMI can't know they're related medical professions).

- PMI can give extremely high values to **rare word pairs** that co-occur purely by chance — biasing it toward low-frequency words.

- Overall: PPMI improves *counting*, but doesn't create a *compressed understanding* of language the way modern embedding methods do.

---

## The Turning Point: From Counting Words to Learning Meaning

> "Nets are for fish; once you get the fish, you can forget the net. Words are for meaning; once you get the meaning, you can forget the words." — 庄子 (Zhuangzi), Chapter 26

Words are only a tool for carrying meaning. Once a model has learned the underlying meaning, it doesn't need to rely on the exact words themselves. Words are the *training signal*, but the goal is not to memorize the words themselves — the goal is to learn the underlying patterns and relationships the words represent.

**The progression:**

1. **Documents represented by words** (Term-Document Matrix, TF-IDF) — asked *"how frequently does a word appear in a document?"* This let models compare documents, but treated words as independent symbols and ignored deeper relationships.

2. **Words represented by their contexts** (Term-Term / Word-Word matrices) — asked *"what words usually appear around this word?"* This introduced the Distributional Hypothesis: words that occur in similar contexts tend to have similar meanings.

3. **Co-occurrence matrices and PPMI** improved this by capturing meaningful word associations, but still produced huge sparse matrices and could only store relationships that were *explicitly observed*.

**The major turning point**: realizing we don't want to *store* words and their relationships directly — we want to **learn compact numerical representations** (word embeddings) where meaning and relationships are encoded in the *geometry* of vectors. Words become points in a semantic space where similar concepts are close together, and relationships can be represented as directions.

```
Documents → Words → Contexts → Meaningful Vector Representations
```

The goal shifted from **counting** words to **learning the hidden structure** of language — laying the foundation for Word2Vec, GloVe, BERT, and modern large language models.

---

## Cosine Similarity

*Reference: [ChatGPT conversation on cosine similarity](https://chatgpt.com/s/t_6a58615b86d88191bc0447e41354d6d8)*

Modern NLP extensively uses **cosine similarity** — the standard mathematical method to compare how semantically alike two pieces of text are, by measuring the **angle** between their vector representations.

- A word's meaning is represented by the **direction** of its vector in semantic space.

- Cosine similarity checks whether two words point in similar directions — meaning they've learned similar contextual behavior.

### Interpreting the Score

| Value | Meaning |
|---|---|
| **+1** | Same direction → maximum similarity |
| **0** | No directional relationship → low/no similarity |
| **-1** | Opposite directions → opposing vector patterns (rare in word embeddings) |

- **TF-IDF / Raw Counts / PPMI**: cosine similarity is usually **0 to 1**, since these vectors contain no negative values.

- **Word2Vec / GloVe**: cosine similarity can be **-1 to 1**, since these vectors contain both positive and negative values.

> 💡 **Important caveat**: Negative similarity means opposite *vector orientation*, not necessarily opposite *meaning*. Word embeddings capture contextual similarity, not human logical relationships like antonyms — so don't assume a negative cosine score means "antonym."

### What Vectors Are We Comparing?

This depends entirely on the representation method in use:

1. Term-Document Matrix

2. Term-Term / PPMI Matrix

3. Word Embeddings

### Why Cosine Instead of Euclidean Distance?

**Euclidean distance is sensitive to vector magnitude.** Two semantically similar words might have very different embedding magnitudes purely due to frequency effects during training — making Euclidean distance an unreliable similarity measure for embeddings.

### Advantages of Cosine Similarity

- **Magnitude Invariance**: ignores vector length, focusing purely on semantic direction.

- **Bounded Range**: always in `[-1, 1]`, making comparisons interpretable.

- **Works Well in High Dimensions**: more robust than Euclidean distance in 100–300 dimensional embedding spaces.

- **Handles Sparse Vectors**: efficient for TF-IDF and other sparse representations.

### Normalized Vectors — A Computational Shortcut

If all embeddings are pre-normalized to unit length (`||v|| = 1`), cosine similarity simplifies to just the **dot product** — computationally cheaper, and commonly used in practice.

**Standard workflow:**

1. Normalize all vectors once.

2. Store the unit vectors.

3. Use a simple dot product for similarity searches.

### Why Cosine Similarity Remains the Standard in Modern NLP

Cosine similarity became standard because embeddings encode meaning as vector **directions**, and cosine provides a fast, scale-friendly way to measure how close those directions are. It's simple enough for **billions** of comparisons while still being effective for modern semantic applications. Since meaning in embedding space is mainly represented by direction, cosine similarity captures the most important property of embeddings — and normalization makes it extremely fast.

---

## Prediction-Based Encodings: Tackling Sparse Vectors at Scale

### Why Do We Need Word2Vec?

Before Word2Vec, NLP mostly represented words using sparse, high-dimensional vectors — one-hot encoding, TF-IDF, co-occurrence matrices. The problem: these representations only store word **identity or frequency** information, and don't capture **semantic similarity**.

> 💡 **Example**: In one-hot encoding, every word is an independent dimension, so the vectors for "king," "queen," and "banana" are all equally distant from each other — even though humans immediately know "king" and "queen" are related, while "banana" is not.

**Word2Vec's innovation**: learn **dense, low-dimensional** word embeddings (typically 100–300 dimensions) where words appearing in similar contexts are placed close together in vector space — directly implementing the Distributional Hypothesis: *"You shall know a word by the company it keeps."*

**The key insight**: we can learn word meanings by **predicting words from their contexts** (or vice versa).

Instead of manually storing word relationships through counts, Word2Vec uses a neural network to learn word representations by predicting words from their contexts (or predicting contexts from words). Through this prediction task, the model automatically learns the hidden semantic structure of language — producing vectors where relationships such as similarity and analogies emerge naturally.

```
Sparse word counts → Dense semantic embeddings
```

Word2Vec's breakthrough was changing NLP from **counting** word occurrences to **learning meaningful representations** from language patterns.

> Word2Vec represents each word as a dense vector whose values are learned by a neural network from the word's context patterns in a large corpus. Each number is a learned parameter (a dimension) the network adjusts during training so that words with similar contexts get similar vectors. It's the **complete vector**, not individual values, that captures a word's meaning and relationships learned from the corpus.

---

## CBOW (Continuous Bag of Words)

CBOW works by doing the **opposite** of Skip-gram: it uses the surrounding **context words** to predict a single target **"Center Word."**

### How It Works

1. **Inputs**: instead of one center word, you feed in multiple one-hot vectors — each representing one of the surrounding context words (e.g., "THE," "QUICK," "FOX," "JUMPS").

2. **Averaging (the "Bag" concept)**: the network multiplies these input vectors by the weight matrix (`W`) to get their individual embedding vectors, then **averages** these embeddings together.

   > 💡 **Why "Bag of Words"**: the specific *order* of the input context words doesn't matter — only their combined presence does. That's the "bag" — order-agnostic.

3. **Prediction**: the averaged vector is passed to the output layer, which uses a **Softmax** function to predict the probability of the missing "Center Word."

**In short**: CBOW learns by "filling in the blank" in the middle of a sequence, based on the averaged semantic information from the surrounding words.

### The Embedding Lookup Trick

Instead of performing expensive, full-scale matrix multiplications when inputting a word, the system treats the input vector as an **index selector**. Because the input is a one-hot vector (a single `1`, everything else `0`), the "multiplication" simplifies to simply **extracting a specific column** from the embedding matrix `W`.

> 💡 **In practice**: modern ML frameworks implement this as an `nn.Embedding` layer — essentially a highly optimized **lookup table**, rather than a literal matrix multiplication. This distinction is crucial: it lets models handle massive vocabularies (10,000+ words) quickly and efficiently, since you're just indexing into a table rather than multiplying huge sparse matrices.

---

## The Essence of Word2Vec

1. Go through the entire corpus. For every word, treat it as the **center word** and identify its neighboring (**context**) words.

2. Create training pairs: `(center word, neighbor word)`, `(center word, neighbor word)`, and so on — this becomes your training dataset.

3. Using these word–neighbor pairs, train a neural network via **backpropagation** until it can accurately predict the neighboring word, given the center word as input.

4. Once fully trained, take the learned weights (embeddings) from the **hidden layer**, and **discard the neural network itself**.

5. These learned vectors become the final vector representations (embeddings) of the words.

> This is exactly how the **Skip-gram** model in Word2Vec works. Each embedding vector is learned through backpropagation so it becomes the best possible representation for predicting a word's surrounding context words. **That's the essence of Word2Vec.**
