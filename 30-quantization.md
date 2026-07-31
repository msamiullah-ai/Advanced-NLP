# Quantization & QLoRA (Quantized LoRA)

---

## The Memory Problem LoRA Doesn't Solve

Although LoRA drastically reduces the number of **trainable** parameters, it does **not** reduce the memory required to store the original pre-trained (base) model weights.

- During every forward pass, the model still needs these frozen weights to compute its outputs, so they must remain **loaded in GPU memory**.
- **Example**: a 70-billion parameter model stored in 16-bit floating-point (FP16) precision requires approximately **140 GB** of GPU memory, just to hold the frozen weights.

Consequently, even though LoRA trains only a small set of adapter matrices, the enormous memory required for the base model means fine-tuning still typically requires **multiple high-memory GPUs.**

---

## QLoRA's Solution: Quantizing the Frozen Weights

**QLoRA (Quantized LoRA)** solves this problem with a key observation: the base model weights are **frozen** and are **never updated** during backpropagation.

- High precision is primarily needed for parameters whose values are **continuously adjusted** using gradients during training.
- Since the frozen weights are used **only** during the forward pass (to compute activations), they can be stored with **much lower precision**, while maintaining almost the same behavior.

**QLoRA therefore quantizes the frozen base weights from 16-bit to 4-bit precision**, reducing their memory footprint by about **4×**.

- For a 70B model, this reduces memory usage from roughly **140 GB to 35 GB** — making it possible to fine-tune very large language models on a **single** high-memory GPU (an NVIDIA RTX 3090, RTX 4090, or a single A100), while preserving nearly the same fine-tuning performance as standard LoRA.

---

## Quantization: Core Concepts

*Reference: [Quantization — core concepts](https://chatgpt.com/s/t_6a6af53286f08191b06a3a382271139a) ⭐⭐*

> ⚠️ **A fundamental limitation to keep in mind throughout**: you can **never perfectly recover** the original, high-precision data once it has been quantized. Quantization is inherently a **lossy** process — the goal is always to minimize that loss, never to eliminate it entirely.

---

## Linear (Absmax) Scaling Quantization

*Reference: [Linear scaling quantization](https://chatgpt.com/s/t_6a6af9cc9c4481919249ab4155b6086b)*

Instead of scaling using the largest value that an FP32 number *could theoretically* represent, **Absmax quantization** scales using the **largest value the current tensor actually contains** — allowing the available INT8 range to be used much more effectively, and minimizing precision loss.

> 💡 **Why this matters**: if you scaled based on the theoretical maximum of the data type, and your actual weights only span a small fraction of that range, you'd be wasting most of your available integer levels on values that never occur — this is exactly the same "don't waste your quantization budget on empty regions" principle that shows up again in NF4 below.

### Dequantization — And Why It's Lossy

To reverse the quantization process, you perform the inverse mathematical operations:

1. Multiply the quantized value by the `abs_max` (the scale factor identified during quantization).
2. Divide by the target range maximum (e.g., **127** for INT8).

> ⚠️ **But this is a lossy process.** Because the original continuous values were rounded down into a finite set of discrete integer buckets during quantization, the exact original value is gone — dequantization can only recover an *approximation* of it, not the original bit-for-bit value.

---

## Lookup Tables: A Smarter Alternative to Linear Quantization

A **lookup table (LUT)** is a **non-linear** quantization method that replaces the fixed linear scaling formula with a **pre-defined mapping** from normalized values (`[-1, 1]`) to quantized values.

- Instead of spacing quantization buckets **uniformly**, the lookup table allows granular control — assigning specific ranges of values (e.g., `-0.9` to `-0.8`) to specific quantized levels.
- This means quantization levels can be placed **more densely** where higher precision is needed, and **more sparsely** elsewhere — reducing precision loss compared to uniform quantization.

> 💡 **QLoRA's NF4 quantization** (detailed below) uses exactly this lookup-table approach, where the mapping is designed to better match the actual distribution of neural network weights — resulting in higher accuracy at very low precision.

### The Fundamental Principle of Efficient Quantization

> You should allocate **more precision** to the regions where the model's weights are actually **concentrated** (the values that occur most frequently), and **lower precision** to the zones where weights are less likely to occur.

This approach avoids wasting memory on empty or sparse areas of the value distribution — effectively maximizing the utility of the limited bits available for representation.

---

## QLoRA's Three Key Innovations

The QLoRA paper introduces **three** distinct techniques that work together:

1. **4-bit NormalFloat (NF4)** — a new data type that is information-theoretically optimal for normally distributed weights.
2. **Double Quantization** — reduces the average memory footprint further, by quantizing the quantization constants themselves.
3. **Paged Optimizers** — manage memory spikes during training.

---

## Innovation 1: 4-bit NormalFloat (NF4)

### Why Normal Distribution Matters

Neural network weights, after training, are typically **approximately normally distributed** (a bell curve centered around zero, following a Gaussian distribution `N(0, σ)`) — most weight values cluster near zero, and taper off toward the extremes.

- Standard **uniform/linear** quantization is wasteful here: it allocates equal-width buckets across the entire value range.
- But under a Gaussian distribution, the region near zero contains far more actual weight values than the tails — so giving every region **equal-width** buckets means the sparse tail regions get the *same* resolution as the densely-populated center, wasting precision where it's least needed and under-serving where it's most needed.

### Quantile Quantization: The Core Idea

**NF4 solves this using quantile quantization**: instead of equal-**width** bins, use equal-**probability** bins.

- Each of the `2^4 = 16` discrete levels in NF4 is chosen so it represents an **equal proportion of the probability mass** under a standard normal distribution — not an equal slice of the numeric range.
- This is what makes NF4 **"information-theoretically optimal"** for normally distributed data: each of the 16 levels carries the same amount of "information" (in the sense of equally likely usage), maximizing the information conveyed per bit spent.

### How Is Quantile Quantization Achieved?

1. Take a theoretical standard normal distribution, `N(0, 1)`.
2. Using the **quantile function** (the inverse of the cumulative distribution function, CDF), find the boundary points that divide the distribution into **16 regions of equal probability mass**.
3. Because probability mass is denser near zero for a normal distribution, this naturally places quantization levels **closer together near zero**, and **more spread out** toward the tails — exactly matching where the actual weight values are concentrated.
4. To ensure that **exact zero** is representable (important for numerical stability), the QLoRA authors estimate the quantiles for the positive and negative halves of the distribution **separately**, then merge them into a final asymmetric set of 16 values that includes an exact zero point.

> 💡 **Why this two-sided estimation matters**: naively splitting 16 levels symmetrically around zero doesn't cleanly produce an exact-zero level (since 16 is even, a perfectly symmetric split leaves zero "between" two levels rather than exactly on one). Handling each half separately lets the final table include a precise zero — useful since many weight values are very close to (or exactly) zero, and losing that precision would be costly.

### The NF4 Lookup Table

Because these quantile boundaries are **precomputed once** (based on the fixed statistics of `N(0,1)`), NF4 quantization becomes, in practice, a simple **lookup table** operation:

- **Quantization**: take a normalized weight value, and find the nearest of the 16 fixed quantile levels → store just its 4-bit index (0–15).
- **Dequantization**: look up the stored 4-bit index in the table, to retrieve the corresponding float value → multiply by the block's scaling constant (see Block-wise Quantization below) to restore it to the correct magnitude.

> 💡 **Key distinction from Absmax quantization above**: Absmax quantization computes a *linear* scale and divides the range into *equal-width* integer buckets. NF4 instead uses a *fixed, precomputed, non-uniform* table of 16 values, chosen specifically to match a normal distribution's shape — a fundamentally different (and, for this specific use case, more accurate) strategy.

---

## Block-wise Quantization

### The Outlier Problem

If you quantized an **entire** weight tensor using a single scaling factor (one global `abs_max`), a single unusually large outlier weight would force the scale to stretch to accommodate it — wasting precision for every other, more typical value in that tensor.

### The Solution

Weights are divided into small, **contiguous blocks** (QLoRA uses a block size of **64** values per block).

- Each block gets its **own** scaling constant (its own local `abs_max`).
- This **localizes** the effect of outliers to just their own block, rather than distorting the scale for the entire tensor — giving much more accurate quantization for every region.

### Block-wise Quantization Pipeline

1. Take the full weight tensor, and split it into contiguous blocks of 64 values.
2. For each block, compute its local `abs_max` (the largest absolute value within that block).
3. Normalize the block's values by dividing by that local `abs_max`, mapping them into the `[-1, 1]` range.
4. Apply NF4 quantile quantization to the normalized values, mapping each to its nearest of the 16 fixed levels → store as a 4-bit index.
5. Store both: the 4-bit indices for every value in the block, **and** the block's own scaling constant (`abs_max`).
6. **At dequantization time**: look up each 4-bit index in the NF4 table to get its float value, then multiply by that block's stored scaling constant to recover the (approximate) original weight.

---

## Innovation 2: Double Quantization

### The Overhead Problem

Even after NF4 quantization, you still need to **store the per-block scaling constants** — one scaling factor for every block of 64 weights.

- These scaling constants are typically stored as 32-bit floats.
- For a block size of 64: that's `32 bits / 64 values` = **0.5 bits of overhead per parameter**, just to store the scaling constants — on top of the 4 bits already spent on the quantized weight itself.

### How Double Quantization Works

**Double Quantization** further quantizes these quantization constants **themselves.**

- The set of block-wise `abs_max` values (across the whole tensor) is itself a collection of numbers with its own distribution.
- These constants are quantized using an additional **8-bit** quantization step, using a **second, larger block size** (e.g., 256 constants per block) for this second layer of quantization.

**The net effect**: this reduces the average overhead from storing scaling constants from about **0.5 bits per parameter** down to roughly **0.127 bits per parameter** — a savings of about **0.37 bits per parameter.**

- **Combined total**: instead of needing roughly **4.5 bits per parameter** (4 bits for NF4 + 0.5 bits for unquantized scaling constants), QLoRA achieves an effective **~4.127 bits per parameter** on average for the frozen base weights.

> 💡 **Why "double" quantization**: it's quantization applied *twice*, in a nested fashion — first quantizing the weights themselves (into NF4), then quantizing the very numbers that describe how that first quantization was scaled. It's a small additional saving per parameter, but it adds up significantly across billions of parameters.

---

## Innovation 3: Paged Optimizers

### The Memory Spike Problem

During training, GPU memory usage can spike **unpredictably** — for example, when processing an unusually long sequence in a mini-batch, or during certain gradient checkpointing operations.

- These spikes can cause **out-of-memory (OOM) crashes**, even if the *average* memory usage across training is well within the GPU's capacity — the problem is the rare peak, not the typical case.

### How Paged Optimizers Work

Paged Optimizers use **NVIDIA's unified memory** feature to automatically transfer optimizer states (like Adam's momentum `m` and variance `v` — see the earlier Adam optimizer notes) between **GPU and CPU memory**, whenever GPU memory is about to overflow — and page them back to the GPU when needed.

> 💡 **The operating-system analogy**: this works very similarly to how an operating system pages memory between RAM and disk when RAM runs low. Instead of crashing when memory briefly spikes, the system temporarily "offloads" less urgently needed data to a slower but larger storage tier (CPU RAM, in this case), then brings it back when there's room again.

- This prevents OOM crashes during memory spikes, **without a significant performance penalty** — since paging only happens during those rare spike events, not throughout normal training.

---

## Putting It All Together

QLoRA combines all three innovations to make fine-tuning enormous models practical on a single consumer or prosumer GPU:

| Innovation | What it saves |
|---|---|
| **NF4** | Reduces the base weights themselves from 16-bit to ~4-bit, matched optimally to their normal distribution |
| **Double Quantization** | Shrinks the overhead of storing per-block scaling constants, saving an additional ~0.37 bits/parameter |
| **Paged Optimizers** | Prevents OOM crashes from unpredictable memory spikes during training, without slowing down normal training |
| **Block-wise Quantization** | The underlying mechanism (block size 64) that makes NF4 robust to outliers, applied throughout |

Together, these let a 70B-parameter model — which would need ~140 GB in FP16 — be fine-tuned with LoRA adapters on hardware with as little as **~35–48 GB** of GPU memory, while achieving performance close to full-precision, full fine-tuning.
