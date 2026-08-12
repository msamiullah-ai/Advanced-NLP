# Word Similarity, Relatedness & Early Vector Representations

## Synonymy vs. Similarity vs. Relatedness

These three concepts are often confused, but they sit on a spectrum from identical meaning to "loosely connected in the same world."

### Synonymy

Two words have the **same or nearly the same meaning** and can often replace each other in many contexts.
- Example: big ↔ large.

### Word Similarity

Similar words don't mean the exact same thing, but they share a common "element of meaning" or belong within the same category.

- **Car, bicycle**: not synonyms — different types of vehicles — but they share the element of being "means of personal transportation."
- **Cow, horse**: different animals, but they share the semantic element of being "large farm animals."

> 💡 **Why this matters for AI**: Computers need to understand these relationships to perform smart searches. If you search for "transportation," an AI that understands word similarity knows results about "cars" and "bicycles" are relevant, even though they aren't literally synonyms of the word "transportation."

### Word Relatedness

A **much broader** concept than similarity. Two words can be related simply because they "live" in the same world or functional environment — even if they aren't synonyms or interchangeable at all.

- **Similar AND related**: "coffee" and "tea" — both exist in the same category (hot beverages) and can often be swapped in a sentence ("I'd like a cup of...").
- **Related but NOT similar**: "surgery" and "scalpel" — a scalpel isn't a type of surgery, and you can't swap the words, but they're highly relevant to each other.

> 💡 **The big AI takeaway**: If a search engine only understood *similarity*, it might miss relevant connections. By understanding *relatedness*, an AI learns that if you're researching "surgery," a "scalpel" is a highly relevant concept to surface — even though it isn't "similar" to the word "surgeon."

### Quick Reference

| Concept | Meaning |
|---|---|
| **Synonymous** | Same meaning |
| **Related** | A much bigger circle — includes things that are similar *and* things that merely co-occur in the same domain |

## Paradigmatic vs. Syntagmatic Relations

### Paradigmatic Relations

Hold between words that can **substitute for each other** in the same context. These words usually belong to the same semantic category and are often similar in meaning.

- Example: doctor ↔ physician; car ↔ truck.
- *"The cat sat on the mat"* → *"The dog sat on the mat."* — "cat" and "dog" are paradigmatic alternatives; they fill the same syntactic slot and share selectional properties.

> A **semantic category** is a classification of words or phrases based on shared meanings or conceptual similarities.

### Syntagmatic Relations

Hold between words that **naturally occur together** in the same context — but you would never substitute one for the other.

- Example: In *"The cat sat on the mat,"* "cat" and "mat" are syntagmatically related — they appear together, but swapping "cat" for "mat" wouldn't make sense.
- Another example: doctor ↔ patient (they co-occur constantly, but they're not interchangeable).

> 💡 **Simple distinction**: Paradigmatic = "could replace this word." Syntagmatic = "tends to appear alongside this word."

## Semantic Field

A **semantic field** is a group of words that belong to the same conceptual domain or topic and are related in meaning — describing different aspects of the same area of knowledge or experience.

- Example: doctor, nurse, hospital, patient, medicine → the semantic field of **healthcare**.

> 💡 The opposite structural concept to a semantic field is **antonymy** (words that oppose each other in meaning), rather than words that cluster around a shared theme.

## Connotation

**Connotation** is the emotional or affective meaning associated with a word, beyond its literal (dictionary) meaning (**denotation**).

- A word may describe the same thing as another word but evoke a different feeling. Their *denotation* (e.g., "a copy of something") can be similar, while their *connotation* differs sharply.
- This is crucial for **sentiment analysis**, where AI determines whether text expresses positive, negative, or neutral emotion.

| Positive connotation | Negative connotation |
|---|---|
| happy | sad |
| elegant | fake |
| replica | knockoff |
| copy | forgery |
| reproduction | — |

## Foundations Before Embeddings

### 1. Distributional Hypothesis

States that **words appearing in similar contexts tend to have similar meanings**. This is the core principle behind word embeddings like Word2Vec and GloVe.

- Example: "doctor" and "physician" often appear in similar contexts.

> 💡 To represent words as vectors, we effectively need to capture three things: **(1)** how words are formed (morphology), **(2)** what they mean (semantics), and **(3)** how they relate to other words (lexical relations). Word embeddings learn all of this from the contexts words occur in — following the core idea: *"Words are known by their contexts."*

### 2. Collocation

A **collocation** is a pair or group of words that frequently occur together, more often than would be expected by chance.

- Example: "make a decision" (rather than, say, "do a decision").

### 3. Hapax Legomenon

A word that appears **only once** in an entire corpus or dataset. Such rare words provide very little information for learning good embeddings, since the model barely sees them in context.

- Example: the word "defenestration" appearing only once in a corpus.

## Representing Words as One-Hot Vectors

*Reference: [ChatGPT conversation on representing words as one-hot vectors](https://chatgpt.com/s/t_6a55e35b3b7881918c3c5826c0e20fd8)*

A **one-hot vector** is a sparse vector representation of a word in which:
- The vector length equals the vocabulary size.
- The entry corresponding to the word's assigned index is `1`.
- All other entries are `0`.

It **uniquely identifies** a word, but encodes **no** semantic or contextual information.

> 💡 **Vocabulary vs. one-hot vectors vs. embeddings**: Vocabulary is the dictionary of unique words built from the corpus. One-hot vectors are simply created *using* the vocabulary's index positions. Embeddings, by contrast, are *learned* from the corpus itself — they aren't just an index lookup.

### Limitations of One-Hot Vectors

- **No similarity captured**: "cat" and "dog" are just two completely different vectors — the computer has no mathematical way of knowing they're both animals.
- **Sparse & inefficient**: creates massive, mostly-empty vectors (e.g., 9,999 zeros for every 10,000-word vocabulary) — very inefficient in memory and computation for large vocabularies.

This limitation is exactly what motivates the move to **embeddings** — denser vectors that actually capture meaning and relationships between words, instead of treating them as disconnected labels.

### The Goal of Embeddings

*Reference: [ChatGPT conversation on the goal of embeddings](https://chatgpt.com/s/t_6a55e6ae291c8191ade88e86bba8a350)*

Word embeddings (Word2Vec, GloVe, FastText) learn **dense vectors from data** — the whole point being to move past sparse, meaningless one-hot representations toward vectors that actually encode a word's meaning and its relationships to other words.

## Word2Vec's Efficiency Trick: Negative Sampling

In the original Skip-gram Word2Vec, the input is a one-hot vector representing the target word, and the network learns by predicting its surrounding (context) words.

- To do this, it computes a probability for **every** word in the vocabulary using a softmax layer — making the computational cost **O(V)**, where V is the vocabulary size.
- Since vocabularies can contain hundreds of thousands of words, this is very expensive.

**Negative sampling** solves this by replacing the large multi-class prediction with a few **binary classification** tasks: the model learns to distinguish the true context word from a small number (`k ≈ 5–20`) of randomly sampled *non-context* words.

- This reduces the training cost from **O(V)** to **O(k)**, while still learning high-quality word embeddings.

## Self-Supervised Learning

**Self-supervised learning** is the "learning strategy" used to train AI models without needing humans to manually label data.

- Instead of a human labeling data as "this is a cat" or "this is a dog," the model teaches itself using the structure of the language it reads.

**Why it works:**
- **The "label" is built in**: the model treats the sequential structure of natural text as its own teacher — the words themselves serve as the data points.
- **The prediction game**: the model plays "guess the missing word" or "what comes next," based on the context surrounding a word.
- **No human effort needed**: since the model can do this on any raw text (books, articles, websites) automatically, it can learn from massive amounts of data without human intervention.

## Representing Documents: Term Frequency (TF) / Bag-of-Words

Instead of representing individual words, **Term Frequency (TF)** represents an entire **document** as a vector.

**Process:**
1. Build a vocabulary containing all unique words in the corpus.
2. Assign each word an index.
3. For each document, count how many times each vocabulary word appears.
4. These counts form the document's **Bag-of-Words vector**, where each position corresponds to a vocabulary word, and the value is its frequency.
5. Often, counts are **normalized** — dividing each word's count by the total number of words in the document — so the representation is independent of document length.

> 💡 A Bag-of-Words vector represents **one entire document** (or sentence), not an individual word. If you have 3 documents, you get 3 Bag-of-Words vectors — one per document.

**Worked example**: Bag-of-Words vector for document D1: `[2, 1, 1, 0, 0, 0]`
- Position 0 → "doctor" (value 2)
- Position 1 → "patient"
- Position 2 → "treats"
- ... and so on, where the value is simply the count.

### Formula — Normalized Term Frequency

The proportion of a document occupied by a word — computed by dividing the number of times a word appears in a document by the total number of words in that document, making documents of different lengths comparable:

```
TF(t, d) = n(t,d) / Σ_{t′∈V} n(t′,d)
```
- `t` = the term (word) whose frequency you want
- `d` = the document
- `n(t,d)` = number of times word `t` appears in document `d`
- `Σ_{t′∈V} n(t′,d)` = total number of words (tokens) in document `d`

### Weaknesses of TF / Bag-of-Words

- Represents **documents**, not individual words — no word-level semantic representation.
- Treats every word as **equally important**, letting common function words (the, is, and) dominate despite carrying little meaning.
- **Ignores word order**, losing syntax and compositional meaning — "Dog bites man" and "Man bites dog" appear identical.
- Assumes every vocabulary word is an **independent dimension**, so it can't capture semantic similarity between related words (e.g., "happy" and "joyful" are treated as totally unrelated).
- Vector size equals vocabulary size → document vectors are **high-dimensional and sparse**, making storage, computation, and generalization difficult.

> 💡 **The core failure**: the inability to distinguish informative words from ubiquitous ones. "The" appears everywhere and tells us nothing; "mitochondria" appears rarely and tells us a great deal. TF has no built-in mechanism for capturing this asymmetry — which is exactly the gap TF-IDF is designed to close.

## Representing Documents as TF-IDF Vectors

**TF-IDF (Term Frequency–Inverse Document Frequency)** improves on simple TF by considering not just how often a word appears in a document, but also **how common that word is across the entire corpus**.

- A word gets a **high weight** if it appears frequently in a particular document but only in a few documents overall — because such words are more useful for distinguishing one document from another.
- Conversely, words appearing in almost every document (the, is, and) get a **low weight** — they carry little useful distinguishing information.

Each document is still represented as a numerical vector, but instead of raw counts, each element holds the word's **TF-IDF weight**.

### The Formula

```
TF-IDF(t, d) = TF(t, d) × IDF(t)

IDF(t) = log(N / df(t))
```
- `N` = total number of documents in the corpus
- `df(t)` (document frequency) = number of documents containing the term

**Key behavior:**
- A word appearing in **every** document has `IDF = log(1) = 0` → its TF-IDF score is zero, regardless of frequency.
- A word appearing in only **one** document receives a much larger IDF → indicating it's highly informative.

> 💡 **Information-theoretic view**: IDF is the negative logarithm of the probability that a randomly selected document contains the term. This means rarer words carry more information, and therefore receive higher weights — a mathematically principled way of capturing the "the vs. mitochondria" asymmetry that plain TF misses.

*Reference: [ChatGPT conversation on TF-IDF mathematics](https://chatgpt.com/s/t_6a571da9c7d08191a60d8f06eb0f9cbf)*

### Limitations of TF-IDF

1. Ignores word meaning (semantics).
2. Ignores word order.
3. Cannot capture context (the same word can have different meanings in different places).
4. Rare words are not *always* important — rarity alone doesn't guarantee relevance.

### Key Takeaway

TF-IDF is an excellent technique for representing documents — it highlights words that are important *within* a document while reducing the influence of common words. However, it only considers word **frequency**, not meaning, context, or word order. For this reason, modern NLP models — Word2Vec, GloVe, FastText, BERT, and other Transformer-based models — are generally preferred whenever understanding the semantics and context of language actually matters.
