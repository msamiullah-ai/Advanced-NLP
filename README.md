# Advanced-NLP

Covering the full arc of modern NLP — from tokenization and static word embeddings, through RNNs and LSTMs, to the Transformer architecture in full mathematical detail, and finally decoding strategies, model architectures, and post-training alignment.

Each file corresponds to one lecture and is written to be read standalone — self-contained, worked examples, and diagrams included, so no prior context from other files is required.

---

## About These Notes

These notes started as personal lecture notes taken during the course, and were then expanded, reorganized, and clarified for readability. Formulas, diagrams, and worked examples from the original lectures are preserved wherever possible; explanatory callouts have been added throughout to fill gaps and make dense concepts easier to follow on a second read.

> **Disclaimer**: These are personal study notes, not official course material. They may contain simplifications, personal interpretations, or the occasional error. If you spot one, feel free to open an issue or a PR.

---

## Table of Contents

### 1 · Foundations
| File | Topic |
|---|---|
| [01 – Overview of NLP Evolution](<01-overview-of-nlp-evolution.md>) | From symbolic/statistical NLP to the foundation-model era; emergent abilities, scaling laws, paradigm shifts |
| [02 – Scaling & Computational Realities](<02-scaling-and-computational-realities.md>) | Test-time compute, reasoning models, Mixture-of-Experts, the curse of multilinguality |

### 2 · Tokenization
| File | Topic |
|---|---|
| [03 – Tokenization](<03-tokenization.md>) | Linguistic hierarchy, TTR, Zipf's & Heaps' Laws, word/character/subword/byte-level tokenization, full BPE algorithm |
| [04 – WordPiece, Unigram & Terminologies](<04-wordpiece-unigram-and-terminologies.md>) | WordPiece scoring, Unigram's E-M-prune cycle, Perplexity, the Viterbi algorithm |

### 3 · Morphology & Lexical Semantics
| File | Topic |
|---|---|
| [05 – Morphology & Lexical Semantics](<05-morphology-and-lexical-semantics.md>) | Morphemes, roots, stems, affixes; inflection vs. derivation; word formation processes; ambiguity |
| [06 – Lexical Semantics & Text Representation](<06-lexical-semantics-and-text-representation.md>) | Synonymy, similarity, relatedness, semantic fields, one-hot vectors, TF & TF-IDF |

### 4 · Word Embeddings
| File | Topic |
|---|---|
| [07 – Word Embeddings: Data Prep & Prediction-Based Encodings](<07-word-embeddings-data-preparation-prediction-encodings.md>) | Distributional semantics, co-occurrence matrices, PPMI, cosine similarity, why Word2Vec |
| [08 – Skip-gram, CBOW & GloVe](<08-skipgram-cbow-and-glove.md>) | Full/negative-sampling Skip-gram, CBOW, subsampling, GloVe's count-based training |
| [09 – Semantic Properties of Embeddings](<09-semantic-properties-of-embeddings.md>) | Vector arithmetic & analogies, embedding space geometry, isotropy, societal bias, WEAT |
| [10 – Mitigating Bias & Implications](<10-methods-of-mitigating-bias-and-implications.md>) | Post-hoc debiasing, CDA, "Lipstick on a Pig," and the FFNN language model bridge |

### 5 · Sequence Models (RNNs & LSTMs)
| File | Topic |
|---|---|
| [11 – FFNN for Next-Word Prediction](<11-FFNN-for-next-word-prediction.md>) | Bengio-style FFNN language model, frozen vs. fine-tuned embeddings, why FFNNs hit a wall |
| [12 – RNNs, Sequence Modeling & Encoder-Decoder](<12-rnn-seq-modeling-encoder-decoder.md>) | RNN memory & parameter sharing, information dilution, Bidirectional RNNs, autoregressive generation |
| [13 – Training RNNs](<13-Training-RNNs.md>) | Backpropagation Through Time, Truncated BPTT, Teacher Forcing, Scheduled Sampling |
| [14 – Problems with RNNs](<14-problems-with-RNNs.md>) | Professor Forcing, vanishing/exploding gradients, the Fundamental Dilemma, residual connections |
| [15 – LSTMs](<15-LSTMs.md>) | Cell state vs. hidden state, the three gates, how LSTM addresses vanishing gradients, variants |
| [16 – BiRNNs & Stacked Deep RNNs](<16-BiRNNs-and-stacked-deep-RNNs.md>) | Deep RNN stacking, per-layer recurrent states, a fully worked multi-layer architecture |

### 6 · Attention & Transformers
| File | Topic |
|---|---|
| [17 – Cross-Attention in Encoder-Decoder Models](<17-cross-attention-in-encoder-decoder-sequence-models.md>) | The information bottleneck, Bahdanau/Luong attention, alignment, score functions |
| [18 – Transformers: Intuition](<18-Transformers-intuition.md>) | Self-attention, Q/K/V, multi-head attention, positional encoding, encoder/decoder walkthroughs |
| [19 – Complete Flow of the Transformer Architecture](<19-complete-flow-of-transformers-architrcture.md>) | End-to-end summary tying encoder, decoder, and generation into one narrative |
| [20 – Transformers in Equations](<20-transformers-in-equations.md>) | The full architecture in exact formulas — embeddings, attention, FFN, LayerNorm, masking math |

### 7 · Decoding Strategies
| File | Topic |
|---|---|
| [21 – Decoding Strategies, Part 1](<21-decoding-strategies-part01.md>) | Greedy decoding, Beam Search & its variants, Pure Random Sampling, Temperature, Top-k/Top-p |
| [22 – Decoding Strategies, Part 2](<22-decoding-strategies-part02.md>) | Repetition/presence/frequency penalty, Min-p, Contrastive & Speculative Decoding |

### 8 · Training & Model Architectures
| File | Topic |
|---|---|
| [23 – Training the Transformer](<23-training-the-transformers.md>) | The full WMT training pipeline — data prep, masking, loss, Adam + warmup, inference |
| [24 – Architectural Split](<24-architectural-split (encoder only, decoder only and encoder-decoder).md>) | Encoder-only vs. decoder-only vs. encoder-decoder; BERT deep dive; MLM, WWM, NSP, CLM |
| [25 – GPT Pretraining & Model Comparisons](<25-GPT-pretraining-comparison-of-models.md>) | T5's span corruption, GPT's decoder-only design, Dense vs. MoE, why decoder-only won |

### 9 · Safety, Fine-Tuning & Alignment
| File | Topic |
|---|---|
| [26 – LLM & DAN Jailbreaks](<26-LLM-and-DAN-jailbreaks.md>) | Guardrails, jailbreak mechanics, alignment, hallucinations, catastrophic forgetting |
| [27 – SFT, Full Fine-Tuning & PEFT](<27-SFT-FFT-PEFT.md>) | FLAN, instruction masking, SFT's limitations, Full Fine-Tuning math, the bridge to PEFT |
| [28 – Prompt, Prefix & Adapter Tuning](<28-prompt-prefix-adapter-tunings.md>) | Soft prompts, prefix vectors on K/V, adapter modules — the core PEFT family |
| [29 – LoRA](<29-LoRA.md>) | Low-rank adaptation math, matrix factorization, zero-init, which matrices to target |
| [30 – Quantization](<30-quantization.md>) | Absmax quantization, NF4, block-wise quantization, Double Quantization, Paged Optimizers (QLoRA) |
| [31 – Instruction Tuning Limits, RLHF & PPO](<31-limitations of instruction tuning-RLHF-PPO.md>) | Why SFT isn't enough, Bradley-Terry, the full RLHF/PPO pipeline, and DPO |

---

## Repository Structure

```
Advanced-NLP/
├── 01-overview-of-nlp-evolution.md
├── 02-scaling-and-computational-realities.md
├── ...
├── 31-limitations of instruction tuning-RLHF-PPO.md
└── images/
    └── (diagrams referenced inline across the notes above)
```

Some notes embed diagrams pulled directly from lecture slides or hand-drawn during lecture; these live in an `images/` folder and are referenced with relative paths, so they render inline on GitHub automatically.

---

## Suggested Reading Order

The files are numbered in the order the course covered them, and each module builds on the one before it — reading front-to-back is the intended path. That said, each file is self-contained enough to be read on its own as a reference for a specific topic (e.g., jump straight to **29 – LoRA** if that's what you need today).

---

## License

These notes are shared for educational purposes. Feel free to fork, reference, or build on them — attribution appreciated.
