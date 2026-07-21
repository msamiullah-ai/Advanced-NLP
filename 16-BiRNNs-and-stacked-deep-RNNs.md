# Bidirectional RNNs & Stacked (Deep) RNNs

---

## Bidirectional RNNs — Why We Need Them

*Reference: [Why there is a need for Bidirectional RNN](https://chatgpt.com/s/t_6a5e207722b08191bf67cc3454ecf2cf)*

*Reference: [Equations for a BiRNN](https://chatgpt.com/s/t_6a5e214164ac8191a05bd5f75f108643)*

> 💡 **Quick recap** (see the earlier RNN notes for the full "Sarah" example): a standard left-to-right RNN can only use *past* context. A Bidirectional RNN runs two RNNs simultaneously — one forward, one backward — and combines their hidden states so each position has access to both past *and* future context.

---

## Deep (Stacked) RNNs

*Reference: [Deep Stacked RNNs](https://chatgpt.com/s/t_6a5e220e86348191b2d51c58b8dddf3d)*

Just as feedforward networks benefit from depth, RNNs can be **stacked** to create deeper architectures.

- In a deep RNN, the output of one recurrent layer becomes the **input** to the next.

<img width="1347" height="768" alt="image" src="https://github.com/user-attachments/assets/e750fd98-c0f9-4102-9428-12749f381542" />


A **stacked RNN** has multiple recurrent hidden layers, and **each hidden layer maintains its own hidden state across time**.

- At every time step, each layer receives **two** inputs:
  1. The output of the layer *below* it, from the *current* time step.
  2. Its *own* hidden state from the *previous* time step.

> 💡 **In simple words**: information flows in two directions at once through a stacked RNN — **upward** through the layers (like a normal deep network) and **forward** through time (like a normal RNN) — with each layer keeping its own independent memory thread. This lets each layer learn representations at different levels of abstraction, while preserving its own temporal memory independently.

### The Mathematical Definition

<img width="1375" height="780" alt="image" src="https://github.com/user-attachments/assets/5a9cb099-d668-4519-9206-c699000db97f" />


**For the first layer (`l = 1`):**
```
h_t^1 = f(W_hh^1 · h_{t-1}^1 + W_xh^1 · x_t + b_h^1)
```

**For every subsequent layer (`l = 2, ..., L`):**
```
h_t^l = f(W_hh^l · h_{t-1}^l + W_xh^l · h_t^{l-1} + b_h^l)
```

**Final output:**
```
y_t = g(W_hy · h_t^L + b_y)
```

> 💡 **Reading the notation**: the superscript `l` denotes which *layer* (depth), and the subscript `t` denotes which *time step*. Notice that layer `l`'s input isn't the raw `x_t` anymore (except for layer 1) — it's the *previous layer's* hidden state at the *same* time step, `h_t^{l-1}`. Meanwhile, each layer still recurrently feeds its *own* previous hidden state `h_{t-1}^l` forward through time. This is exactly the "two inputs per layer" rule described above, just written out formally.

*Source diagram reference: [cs231n.github.io/rnn/](https://cs231n.github.io/rnn/)*

---

## Full Worked Example: RNN Language Model With Pre-trained Embeddings — 3 Hidden Layers (Single Time Step)

*Reference: [Detailed breakdown of this figure](https://chatgpt.com/s/t_6a5e2580c07881919783920993f722c0)*

**Setup**: `|V| = 10,000`, embedding dim `d = 300`, hidden dim `h = 256`. Colors: Red = `W_hh1`, Blue = `W_hh2`, Green = `W_hh3`.

<img width="1205" height="779" alt="image" src="https://github.com/user-attachments/assets/79d55a86-6a33-49b2-8330-151d315d5957" />


### Layer-by-Layer Walkthrough

1. **Input Layer** — One-Hot, `|V| × 1`.

2. **Embedding Layer** — `d × |V|` (`d = 300`). **No activation** — pure row lookup via `W_emb`.

3. **Hidden Layer 1 (Recurrent)** — `h × d`, `h = 256`, **tanh** activation. Receives input from the embedding layer (via `W_eh`) *and* its own previous time step (via the red recurrent weights `W_hh1`).

4. **Hidden Layer 2 (Recurrent)** — `h × h`, `h = 256`, **tanh** activation. Receives input from Hidden Layer 1 (via `W_{h1→h2}`) *and* its own previous time step (via the blue recurrent weights `W_hh2`).

5. **Hidden Layer 3 (Recurrent)** — `h × h`, `h = 256`, **tanh** activation. Receives input from Hidden Layer 2 (via `W_{h2→h3}`) *and* its own previous time step (via the green recurrent weights `W_hh3`).

6. **Output Layer** — Softmax, `|V| × h`, producing the probability distribution over the vocabulary.

### The Weight Matrices, In Detail

| Matrix | Role | Dimensions | Formula |
|---|---|---|---|
| `W_emb` | Input → Embedding | `d × |V|` | `e^{[t]} = W_emb[w^{[t]}] → d × 1` (row lookup) |
| `W_eh` | Embedding → Hidden 1 | `d × h` | `W_eh · e^{[t]} = (h×d)(d×1) = h × 1` |
| `W_hh1` | Recurrent in Hidden 1 | `h × h` | `W_hh1 · h^{[1]}_{(t-1)} = (h×h)(h×1) = h × 1` |
| `W_hh2` | Recurrent in Hidden 2 | `h × h` | `W_hh2 · h^{[2]}_{(t-1)} = (h×h)(h×1) = h × 1` |
| `W_hh3` | Recurrent in Hidden 3 | `h × h` | `W_hh3 · h^{[3]}_{(t-1)} = (h×h)(h×1) = h × 1` |
| `W_{1→2}`, `W_{2→3}` | Between hidden layers | `h × h` each | `(h×h)(h×1) = h × 1` |
| `W_hy` | Hidden 3 → Output | `|V| × h` | `ŷ^{[t]} = softmax(W_hy · h^{[t]}_{(3)}) = (|V|×h)(h×1) = |V| × 1` |

**The recurrence equations for all three layers:**

```
Hidden Layer 1:  h^{[1]}_{(t)} = tanh( W_eh · e^{(t)} + W_hh1 · h^{[1]}_{(t-1)} + b_1 )   → h × 1 (256-dim)

Hidden Layer 2:  h^{[2]}_{(t)} = tanh( W_{1→2} · h^{[1]}_{(t)} + W_hh2 · h^{[2]}_{(t-1)} + b_2 )   → h × 1 (256-dim)

Hidden Layer 3:  h^{[3]}_{(t)} = tanh( W_{2→3} · h^{[2]}_{(t)} + W_hh3 · h^{[3]}_{(t-1)} + b_3 )   → h × 1 (256-dim)
```

### Key Structural Notes

- **Total weight matrices: 8** — `W_emb (d×|V|)`, `W_eh (h×d)`, `W_hh1 (h×h)`, `W_{1→2} (h×h)`, `W_hh2 (h×h)`, `W_{2→3} (h×h)`, `W_hh3 (h×h)`, `W_hy (|V|×h)`.

- **Recurrent connections exist only on the hidden layers.** The embedding layer is a simple lookup, with no recurrence or activation.

- **Each hidden layer maintains its own independent recurrent state**: `h^{[1]}_{(t-1)}`, `h^{[2]}_{(t-1)}`, `h^{[3]}_{(t-1)}` — three separate `h × 1` vectors, each carrying its own thread of memory through time, completely independent of the other layers' memories.
