# Dynamic_Kronecker — Fourier-Position Kronecker (Problem #3)

## Glossary — what the terms mean

Read this once; the rest of the report uses these words constantly.

| Term | Meaning |
|---|---|
| **`pos_dim`** | Paper Kronecker’s **byte-position window** (number of columns in the 256×pos grid). Default **32**. Only the first `pos_dim` UTF-8 bytes of a token are kept. |
| **`crop@32` / `crop@N`** | **Fraction of corpus token instances** whose UTF-8 surface is **longer than N bytes** (N=32 = paper default). Those tokens are truncated by Classic. Shown as a rate or %. *Example from our run: English crop@32 = **0%**; Tamil = **0.469%**.* |
| **Hard crop** | The truncate itself: Classic **never sees** bytes past `pos_dim`. Different words that agree on the first 32 bytes get the **same κ forever**. |
| **Silent collision** | Two different strings → identical Classic codes because they only differed *after* the window. No error; the model cannot separate them. |
| **Fertility** | **Tokenizer tokens per whitespace word** = `(# subword tokens) / (# words)`. Higher ⇒ more pieces per word ⇒ more attention cost. *Our run: kn≈2.91 vs en≈1.59.* |
| **Bytes/tok** | Mean **UTF-8 byte length of a token’s surface** (what Kronecker encodes). Indic scripts often use 3 bytes/char, so this is much higher than English. |
| **`D` (code dim)** | Width of κ before the Linear. Classic: `D = 256 × pos_dim`. Fourier: `D = 256 × r`. *pos_dim=32 or r=32 ⇒ D=8192.* |
| **`r` / `pos_features`** | Fourier position-feature width (size of φ(p)). Analogous to `pos_dim` for budget matching, but **does not cap** token length. |
| **Matched D** | Fair compare: Classic and Fourier use the **same D** (same `Linear(D→d_model)` size). Only the codec differs. *e.g. `classic_pos32` vs `fourier_r32_abs`.* |
| **Emb-path params** | Trainable params in the **input path only** = `D × d_model` (the projection). Not the transformer body or LM head. |
| **Val loss / ppl** | Held-out cross-entropy (lower better); perplexity `exp(val loss)` (lower better). Absolute values are high on this tiny proxy — compare **deltas between arms**. |
| **Δ (C − F)** | `classic_val_loss − fourier_val_loss`. **Positive ⇒ Fourier wins.** *D=8192: Δ=+0.122.* |
| **Absolute vs relative** | Fourier position indexing: absolute uses byte index `0,1,2,…`; relative uses `p/(L−1)∈[0,1]`. Absolute won our ablation for long suffixes. |
| **κ / codec** | Deterministic byte→vector map (no learned weights). Then `emb = Linear(κ)`. |

---

## Technical report of the solution

### 1. Problem

Paper / Session 7 Kronecker embeddings build each token from its UTF-8 bytes on a fixed **256 × `pos_dim`** grid (byte value × byte position), then apply one shared

```text
Linear(D, d_model)   with   D = 256 · pos_dim
```

The shipped default is **`pos_dim = 32`**. That choice creates two structural failures:

1. **Hard crop** — bytes with position ≥ 32 never mark a cell. Distinct long Indic pieces that share a UTF-8 prefix become **identical codes forever** (silent collision). The projection cannot separate them.
2. **Wasted columns** — short tokens (`"a"`, `"the"`) leave most one-hot position columns as exact zeros, but the model still pays for full `D` in the projection.

Raising `pos_dim` to 64 or 128 only **moves the wall** and multiplies embedding-path parameters. It is not length-dynamic.

### 2. Solution

**Fourier-Position Kronecker** keeps the same compositional event — *byte × position* — but replaces the finite one-hot position basis \(e_p \in \mathbb{R}^{\texttt{pos}}\) with a continuous Fourier feature \(\phi(p) \in \mathbb{R}^{r}\) defined for every byte index:

```text
paper:   κ = (1/√L) · vec Σ_{p < pos}   c_b ⊗ e_p      # crop if L > pos
ours:    κ = (1/√L) · vec Σ_{p = 0..L-1} c_b ⊗ φ(p)     # no hard crop
         φ_{2i}=sin(p/θ^{…}),  φ_{2i+1}=cos(…)
         D = 256 · r
```

- Code width stays **fixed** (`D = 256·r`). Matched fair compares use the same `D` as Classic.
- Token length `L` is unrestricted (soft `max_bytes` guardrail only).
- The only trainable input object remains **`Linear(D, d_model)`** — same architectural contract as the paper module.

Optional ablation: `position_mode = relative` maps `p → p/(L−1) ∈ [0,1]` instead of absolute byte index.

### 3. How we prove it

| Stage | What |
|---|---|
| **Diagnose** | Fertility + Classic crop@32 on Wikipedia **en / hi / ta / te / kn** |
| **Train** | Same tiny LLM body; swap only the codec (Classic vs Fourier) |
| **Match budgets** | `D ∈ {4096, 8192, 16384, 32768}` i.e. width ∈ {16, 32, 64, 128} |
| **Ablate** | Absolute vs relative Fourier at `r=32` |
| **Readout** | Val loss / ppl deltas; param efficiency; expected vs got |

**Protocol (this run):** `xlm-roberta-base` · `wikimedia/wikipedia` 120k chars/lang · equal mix → **178,695** tokens · tiny causal LM `d_model=256`, `n_layer=4`, `n_head=4` · AdamW `3e-4` · **3 epochs** · batch 32 · CUDA.

**Nine curated runs** (not a full grid): Classic `pos_dim∈{16,32,64,128}` · Fourier `r∈{16,32,64,128}` absolute · + `r=32` relative.

### 4. Results (headline)

| Comparison | Classic | Fourier | Δ (C−F) | Winner |
|---|---:|---:|---:|---|
| **Matched D=8192** (paper default) | 7.7931 | **7.6712** | **+0.1219** | Fourier |
| Matched D=4096 | 7.8531 | 7.7553 | +0.0978 | Fourier |
| Matched D=16384 | 7.7307 | 7.6428 | +0.0879 | Fourier |
| Matched D=32768 | 7.6782 | **7.5949** | +0.0833 | Fourier |

**Efficiency punchline:** `fourier_r32_abs` (D=8192, **2.10M** emb-path params) **beats** `classic_pos128` (D=32768, **8.39M** params) — better val loss at **¼** the projection size.

**Absolute > relative** at r=32 (7.6712 vs 7.6777).

### 5. Fertility / crop (why Indic matters)

| Lang | Fertility (tok/word) | Bytes/tok | Classic crop@32 |
|---|---:|---:|---:|
| kn | **2.915** | 7.73 | 0.084% |
| te | 2.714 | 7.44 | 0.185% |
| ta | 2.661 | **8.61** | **0.469%** |
| en | 1.585 | 3.32 | **0.000%** |
| hi | 1.527 | 8.22 | 0.093% |

Indic UTF-8 is ~2.2–2.6× denser per token than English; Classic crop@32 is an Indic issue, not an English one.

### 6. Expected vs got

| Expectation | Got |
|---|---|
| Indic ≫ EN fertility & bytes; crop@32 ≈ 0 on EN | Confirmed |
| At D=8192 Fourier beats Classic | Confirmed (Δ=+0.122) |
| Fourier@small D competes with Classic@large D | Confirmed (8k beats 32k) |
| Absolute ≥ relative for suffixes | Confirmed |

### 7. Limits

Proxy scale (~179k tokens, 3 epochs). Absolute perplexities are high (large V vs data) — **trust deltas between arms**. Crop rates on this sample are modest (≤0.47%); Fourier still wins at matched D, so benefits are not only hard-collision removal. This does **not** claim a full V5 pretrain win.

---

## Interactive report

Open **[`index.html`](index.html)** in a browser (JSON already embedded):

- Technical report hero at the top  
- **Playground:** D duel (chips + slider), live Δ, winner highlight  
- Curve filters (All / Paper / Fourier), replay animation, D=8192 highlight  
- Fertility metric toggle + clickable languages  
- Click scoreboard / efficiency points / bars to spotlight runs  

Refresh after a new Colab run:

```bash
python embed_report.py results/report.json
```

---

## Files

```
Dynamic_Kronecker/
  README.md           ← this technical report + guide
  index.html          ← interactive E2E report
  kronecker.py        ← Classic + Fourier codecs (lightly rephrased docstrings)
  experiment.py       ← Colab runner → results/report.json
  embed_report.py     ← inject JSON into index.html
  requirements.txt
  results/report.json
  train.log
```

---

## Reproduce (Colab)

```python
!pip install -r requirements.txt
!python experiment.py colab --device cuda --epochs 3
# download results/report.json → embed locally
```

```bash
python embed_report.py results/report.json
```

### Shape cheat-sheet

```text
text → κ [D]
codec_table [V, D]
input_ids [B, T] → gather [B, T, D] → Linear → [B, T, d_model]
TinyLLM → logits [B, T, V] → CE on [B*T, V]
```
