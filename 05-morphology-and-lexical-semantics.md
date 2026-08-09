# Morphology, Word Formation, Lexical Semantics & Ambiguity

## Why This Matters for NLP

Tokenization is a way of representing words, subwords, characters, or any unit produced by a tokenizer so they can be fed into a neural network. But *how* we split words isn't arbitrary — it connects directly to how human language itself is built from smaller meaningful pieces. That's what morphology studies

## Morphological Units

### Morpheme — The Foundation

A **morpheme** is the smallest unit of language that carries meaning. If you break it down further, it either loses that meaning or the meaning changes entirely.

- Example: In **"unhappiness,"** `un-`, `happy`, and `-ness` are all individual morphemes. You can't break "happy" into smaller parts that still mean "happy."

**Bound vs. Free morphemes:**
- **Free morpheme** — can stand alone as a word (e.g., `happy`).
- **Bound morpheme** — cannot stand alone; must attach to something else (e.g., `un-`, `-ness`).

> 💡 **Worked example — "unhappiness"** consists of three morphemes:
> - `un-` → cannot stand alone → **bound morpheme**
> - `happy` → can stand alone → **free morpheme**
> - `-ness` → cannot stand alone → **bound morpheme**

A morpheme can function as either a **root** or an **affix**.

### Root — The Core

The **root** is the primary lexical core of a word — the core morpheme that carries the word's main lexical meaning. It's what remains after you strip away *all* prefixes and suffixes.

> 💡 **How to find the root**: Remove affixes one by one until you can't remove any more — whatever remains is the root. It's typically the original word from which the term's etymology derives. As a test: if you can still strip something off a unit and it still carries meaning, that unit is *not* the root yet.

- Example: In "unhappiness," `happy` is the root. In "rewriting," `write` is the root.

### Stem — The Base

The **stem** is the "base" form to which you add the next piece (affix). A stem can sometimes *be* the root, but it can also be a larger chunk that already has a suffix attached.

> 💡 **Worked example — "unkindness":**
> - `kind` is the root.
> - Add `-ness` to `kind` → `kind` acts as the **stem**.
> - Add `un-` to `kindness` → `kindness` becomes the **stem** for that final prefix.
>
> Similarly, for **"unforgiveness"**: the **stem** is `forgiveness`, while the **root** is `forgive`.

### Affix — The Add-on

An **affix** is a morpheme attached to a root or stem to form a new word or modify its meaning. An affix can *never* stand alone — it is always a **bound morpheme**.

- Examples: `un-` (a prefix that negates), `-ness` (a suffix that creates a noun/quality), `-ing`, `-ed`.

## Word Formation Processes

### 1. Inflection — The "Grammar" Adjustment

Inflection modifies a word to fit the grammatical requirements of a sentence **without** changing what the word actually is or means.

- **Core goal**: making the word agree with the sentence (tense, plurality, etc.)
- **Same lexeme**: "walked" is still just the word "walk."
- **Highly regular**: you can almost always add `-s`, `-ed`, or `-ing` to a verb without it sounding wrong.
- **No part-of-speech change**: a noun stays a noun; a verb stays a verb.

Example: "walk" → "walks" / "walking" — still fundamentally the concept of "to walk."

### 2. Derivation — The "New Word" Creation

Derivation uses an affix to create an entirely **new word** — a new "lexeme."

- **Core goal**: changing the meaning or creating a new category of word.
- **New lexeme**: you've created a brand-new concept.
- **Often changes part of speech**: a verb ("walk") becomes a noun ("walker").
- **Irregular**: doesn't apply consistently to every word (e.g., we say "writer," but not "cooker" for someone who cooks — we just say "cook").

Example: "happy" (adjective) + `un-` → "unhappy" (opposite meaning, still an adjective). "happy" + `-ness` → "happiness" (adjective → noun).

### 3. Compounding

Smashing two or more **independent roots** together to create a new concept.

- Examples: "white" + "board" → "whiteboard"; "pick" + "pocket" → "pickpocket."

### 4. Cliticization

Involves **clitics** — small, reduced word forms that can't stand alone and must "hitchhike" by attaching to a host word.

- Examples: `'s` in "she's," `'re` in "we're." In Urdu or Arabic, this can include possessive markers or articles like `al-` in "al-shams" (the sun).

### 5. Reduplication

Repeats the whole word or part of it to shift meaning.

- Examples: "bye-bye" in English; "rumah-rumah" (houses) in Malay, to show plurality.

### 6. Blending, Clipping & Back-formation — The "Shortcuts" of Language

- **Blending**: smashing parts of two words together — e.g., breakfast + lunch = **brunch**.
- **Clipping**: lopping off part of a word to make it shorter — e.g., advertisement → **ad**.
- **Back-formation**: creating a new word by "de-constructing" an existing one, usually by incorrectly assuming it has a removable affix — e.g., "editor" existed first, and people later created the verb "edit" from it.

## Why Subword Tokens Align With Morphology

We use subword tokens because most word formation processes — compounding, shortening words, adding affixes — are built **on top of existing words**.

By breaking words into subword units, models can better capture the morphological relationships and internal structure of complex words, rather than treating them as entirely new, random strings the model has never seen. This helps the AI **generalize** better, by leveraging the meaning already carried in these smaller, constituent parts.

## Morpheme Concatenation — A Language Typology

Languages vary widely in how they combine morphemes to build words. They exist on a spectrum:

| Type | Behavior | Example languages |
|---|---|---|
| **Isolating** | Words typically consist of a single morpheme, with very few affixes | Mandarin, Vietnamese |
| **Agglutinative** | Distinct morphemes are glued together with clear, separable boundaries | Turkish, Swahili |
| **Fusional** | A single affix may represent multiple grammatical features at once, making them hard to separate | Spanish, Arabic |
| **Polysynthetic** | A single word can function as an entire sentence | Inuktitut |

## From Form to Meaning

### Lexeme vs. Word Form

A **lexeme** is an abstract vocabulary item representing a single word and *all* of its grammatical forms — essentially the entry you'd look up in a dictionary (**lexicon**).

| Lexeme | Word Forms |
|---|---|
| PLAY | play, plays, played, playing |
| RUN | run, runs, ran, running |

- The **lexeme** is the underlying word in the mental lexicon.
- The **word forms** are the actual spellings that appear in text.

### Sense, Polysemy & Homonymy

A **sense** is one meaning of a word. If a word has multiple meanings, there are two possibilities:
- If those meanings are **related** → **polysemy**
- If those meanings are **unrelated** → **homonymy**

So: **sense** = one meaning; **polysemy** = multiple *related* meanings of the same lexeme; **homonymy** = multiple *unrelated* meanings that happen to share the same word form.

> 💡 **Polysemy example — "head":** body part → leader → top of a nail that gets hammered → head of lettuce. All of these are metaphorically related to the core idea of "head" → **Polysemy**.
>
> 💡 **Homonymy example**: "bank" (financial institution) vs. "bank" (riverbank) — completely different etymologies. Also: "bat" (the animal) vs. "bat" (sports equipment) → **Homonymy**.

## Lexical Semantic Relations

| Relation | Definition | Example | NLP Use |
|---|---|---|---|
| **Synonymy** | Two or more words with the same or very similar meaning | big ↔ large, begin ↔ start | Helps search engines/chatbots understand that different words express the same idea (searching "large house" can also return "big house") |
| **Antonymy** | Two words with opposite meanings | hot ↔ cold, good ↔ bad | Used in sentiment analysis to distinguish positive vs. negative opinions |
| **Hypernymy** | A general/broader category | *animal* is a hypernym of *dog*; *fruit* is a hypernym of *apple* | — |
| **Hyponymy** | A specific member of a broader category | *dog* is a hyponym of *animal*; *apple* is a hyponym of *fruit* | Helps build taxonomies and perform inference (if something is a dog, it is also an animal) |
| **Meronymy** | A part of something | wheel → car, keyboard → computer, finger → hand | — |
| **Holonymy** | The whole that contains the parts | car → wheel, computer → keyboard, hand → finger | Helps knowledge graphs and AI understand part-whole relationships |

> 💡 **Relationship reminders**:
> - Hypernym = General, Hyponym = Specific ("dog *is-a* animal")
> - Meronym = Part, Holonym = Whole ("wheel is part of a car")

### Two Organizing Rules

**Rule 1 — Some word forms have many senses** *(this is Polysemy and Homonymy)*

```
Word Form
    │
    ├── Sense 1
    ├── Sense 2
    └── Sense 3
```

Example — **TRUNK**:
- Tree trunk / human body trunk → related senses → **Polysemy**
- Elephant trunk / swimming trunks → unrelated senses → **Homonymy**

→ **One word form → many senses**

**Rule 2 — One sense can be expressed by many word forms** *(this is Synonymy)*

```
Sense (Meaning)
      │
 ┌────┼────┐
 │    │    │
big large huge
```

Another example — Sense: "Container" → box, carton, crate, package (different word forms, same/similar sense).

→ **One sense → many word forms**

## Ambiguity

### 1. Lexical Ambiguity (Word-level Ambiguity)

A sentence is lexically ambiguous when one or more **words** have multiple meanings — the ambiguity comes from the word itself.

- Example: *"I went to the bank."*
  - Possible meanings: financial institution, or river bank.
  - The word "bank" is ambiguous.

### 2. Syntactic Ambiguity (Structural Ambiguity)

A sentence is syntactically ambiguous when the sentence **structure/grammar** allows more than one interpretation — the words each have only one meaning, but their *arrangement* is ambiguous.

- Example: *"I saw the man with a telescope."*
  - Interpretation 1: I used a telescope to see the man.
  - Interpretation 2: The man had a telescope.

### Parse

A **parse** is one complete grammatical and semantic interpretation of a sentence. When a sentence is ambiguous, each distinct interpretation is a separate parse.

> 💡 **Worked example**: *"One morning I shot an elephant in my pajamas"* has multiple parses because of:
> - **Syntactic ambiguity** — where does "in my pajamas" attach? (to "I" or to "elephant"?)
> - **Lexical ambiguity** — does "shot" mean *fired at* or *photographed*?
>
> Combining these ambiguities yields **eight possible semantic interpretations** of the same sentence.
