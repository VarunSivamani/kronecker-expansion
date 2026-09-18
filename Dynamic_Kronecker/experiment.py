#!/usr/bin/env python3
"""
Problem #3 — Colab experiment runner
====================================

Produces ONE file: results/report.json  (schema fixed — see README)

  python experiment.py colab --device cuda --epochs 3

Curated valuable runs (not a full grid)
--------------------------------------
  Paper Kronecker : pos_dim ∈ {16, 32, 64, 128}
  Ours Fourier    : r ∈ {16, 32, 64, 128}  (absolute)
                  + one noteworthy ablation: r=32 relative

Also measures fertility on en/hi/ta/te/kn Wikipedia before training.

After Colab finishes, download report.json and run locally:
  python embed_report.py results/report.json
→ rewrites index.html with the JSON embedded.

Tensor shape pipeline (training)
--------------------------------
  raw wiki text (str)
    → tokenizer.encode → ids: List[int] length N_tokens
    → pack_blocks → x, y: [N_blocks, seq_len]
    → DataLoader batch → xb, yb: [B, T]   (T = seq_len)
    → TinyLLM(xb) → logits: [B, T, V]
    → cross_entropy(logits.view(B*T, V), yb.view(B*T)) → scalar loss
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kronecker import (
    ClassicKronecker,
    FourierKronecker,
    KroneckerEmbedding,
    build_codec_table,
    utf8_bytes,
)

LANGS = ("en", "hi", "ta", "te", "kn")  # India-first + English mix
WIKI_CONFIG = {
    "en": "20231101.en",
    "hi": "20231101.hi",
    "ta": "20231101.ta",
    "te": "20231101.te",
    "kn": "20231101.kn",
}
_WORD_RE = re.compile(r"\S+", re.UNICODE)  # fertility: count whitespace-separated tokens as "words"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_tokenizer(name: str):
    """Load HF tokenizer; ensure pad_token exists for completeness."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok


def vocab_surfaces(tokenizer) -> List[str]:
    """
    Surface string for every vocab id — what Kronecker encodes.

    out: list length V (= tokenizer.vocab_size), each entry a str
    """
    n = int(getattr(tokenizer, "vocab_size", len(tokenizer)))  # V
    out = []
    for tid in range(n):
        try:
            # decode([tid]) → actual text bytes this piece emits (not ▁ markers alone)
            out.append(tokenizer.decode([tid], clean_up_tokenization_spaces=False))
        except Exception:
            out.append("")
    return out  # len = V


def load_wiki_text(lang: str, max_chars: int, seed: int = 0) -> str:
    """
    Stream HF Wikipedia for one language until we have ~max_chars of text.

    out: one big str (length ≤ max_chars)
    """
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", WIKI_CONFIG[lang], split="train", streaming=True
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    chunks, total = [], 0
    for row in ds:
        t = (row.get("text") or "").strip()
        if not t:
            continue
        chunks.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]  # str


def build_multilingual_ids(
    tokenizer, chars_per_lang: int, mix: Dict[str, float], seed: int
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Load en/hi/ta/te/kn, tokenize each, interleave by mix weights.

    out:
      ids   : List[int] length N_tokens  (flat token stream for LM)
      meta  : per-lang char/token counts + total_tokens
    """
    rng = random.Random(seed)
    per_lang: Dict[str, List[int]] = {}  # lang → token id list
    meta: Dict[str, Any] = {}
    for lang, w in mix.items():
        if w <= 0:
            continue
        text = load_wiki_text(lang, chars_per_lang, seed=seed)          # str
        ids = tokenizer.encode(text, add_special_tokens=False)          # List[int]
        per_lang[lang] = ids
        meta[lang] = {"chars": len(text), "tokens": len(ids), "weight": w}
        print(f"  data[{lang}] chars={len(text):,} tokens={len(ids):,} w={w}")

    # Round-robin-ish sampling in chunks (keeps some language locality).
    cursors = {l: 0 for l in per_lang}
    out: List[int] = []
    chunk = 64  # tokens pulled per draw
    target = sum(len(v) for v in per_lang.values())  # aim to use ~all tokens once
    while len(out) < target:
        lang = rng.choices(list(per_lang), weights=[mix[l] for l in per_lang])[0]
        ids = per_lang[lang]                       # List[int] for this language
        c = cursors[lang]
        if c >= len(ids):
            cursors[lang] = 0
            c = 0
        take = ids[c : c + chunk]                  # up to 64 token ids
        cursors[lang] = c + len(take)
        out.extend(take)
        if all(cursors[l] >= len(per_lang[l]) for l in per_lang):
            break
    meta["total_tokens"] = len(out)                # N_tokens
    return out, meta                               # List[int], dict


# ---------------------------------------------------------------------------
# Fertility
# ---------------------------------------------------------------------------

def measure_fertility(
    tokenizer, chars_per_lang: int, pos_dim_ref: int, seed: int
) -> Dict[str, Any]:
    """
    Per language:
      fertility = (# tokenizer tokens) / (# whitespace words)
      classic_crop_rate = fraction of token *instances* whose UTF-8 surface
                          is longer than pos_dim_ref (paper window).
    """
    languages = {}
    for lang in LANGS:
        print(f"[fertility] {lang} …")
        text = load_wiki_text(lang, chars_per_lang, seed=seed)          # str
        words = len(_WORD_RE.findall(text))                             # #words
        ids = tokenizer.encode(text, add_special_tokens=False)          # List[int], len=N
        cache: Dict[int, str] = {}  # tid → surface str (decode once per unique id)
        cropped = 0
        byte_sum = 0
        for tid in ids:
            if tid not in cache:
                cache[tid] = tokenizer.decode(
                    [tid], clean_up_tokenization_spaces=False
                )
            blen = len(utf8_bytes(cache[tid]))                          # UTF-8 byte length of surface
            byte_sum += blen
            if blen > pos_dim_ref:
                cropped += 1                                            # would be truncated by Classic
        n_tok = len(ids)
        languages[lang] = {
            "chars": len(text),
            "words": words,
            "tokens": n_tok,
            "fertility_tokens_per_word": n_tok / max(words, 1),
            "chars_per_token": len(text) / max(n_tok, 1),
            "bytes_per_token": byte_sum / max(n_tok, 1),
            "classic_crop_rate": cropped / max(n_tok, 1),
            "classic_cropped_tokens": cropped,
            "unique_token_ids_seen": len(cache),
        }
        r = languages[lang]
        print(
            f"  fert={r['fertility_tokens_per_word']:.3f}  "
            f"bytes/tok={r['bytes_per_token']:.2f}  "
            f"crop@{pos_dim_ref}={100*r['classic_crop_rate']:.3f}%"
        )

    ranking = sorted(
        languages.items(),
        key=lambda kv: kv[1]["fertility_tokens_per_word"],
        reverse=True,
    )
    return {
        "classic_pos_dim_reference": pos_dim_ref,
        "chars_per_lang": chars_per_lang,
        "languages": languages,
        "fertility_ranking_high_to_low": [lang for lang, _ in ranking],
    }


# ---------------------------------------------------------------------------
# Tiny LLM
# ---------------------------------------------------------------------------

class TinyLLM(nn.Module):
    """
    Small causal LM for the proof.

    Shapes
    ------
    embedding : module mapping [B, T] → [B, T, d_model]  (KroneckerEmbedding here)
    pos       : nn.Embedding(seq_len, d_model)           absolute position table
    blocks    : TransformerEncoder, batch_first=True
    head      : Linear(d_model, vocab_size)

    forward(idx):
      idx     [B, T]
      → x     [B, T, d_model]
      → logits[B, T, V]
    """

    def __init__(self, vocab_size, d_model, n_layer, n_head, seq_len, embedding):
        super().__init__()
        self.embedding = embedding                                     # [B,T] → [B,T,d]
        self.pos = nn.Embedding(seq_len, d_model)                      # weight [seq_len, d]
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,                                          # expects [B, T, d]
            activation="gelu",
            norm_first=True,
            dropout=0.1,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)         # weight [V, d]
        # causal mask: True = blocked (PyTorch TransformerEncoder convention)
        # shape [seq_len, seq_len]; upper triangle is True
        self.register_buffer(
            "causal",
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(),  # [Tmax, Tmax]
            persistent=False,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        in:  idx    [B, T]           token ids
        out: logits [B, T, V]        next-token scores
        """
        B, T = idx.shape                                               # batch, time
        # token emb [B,T,d] + position emb [T,d] broadcast → [B,T,d]
        x = self.embedding(idx) + self.pos(torch.arange(T, device=idx.device))  # [B, T, d]
        x = self.blocks(x, mask=self.causal[:T, :T])                   # [B, T, d], mask [T, T]
        return self.head(self.ln(x))                                   # [B, T, V]


def pack_blocks(ids: List[int], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a flat token stream into fixed-length LM blocks (next-token pairs).

    in:  ids      List[int] length N
    out: x, y     each [N_blocks, seq_len]
         x[i] = ids[i*T : i*T + T]
         y[i] = ids[i*T + 1 : i*T + T + 1]   (shifted by 1 for next-token pred)
    """
    n = (len(ids) - 1) // seq_len                                      # N_blocks
    usable = n * seq_len
    x = torch.tensor(ids[:usable], dtype=torch.long).view(n, seq_len)            # [N, T]
    y = torch.tensor(ids[1 : usable + 1], dtype=torch.long).view(n, seq_len)     # [N, T]
    return x, y


def train_run(
    *,
    run_id: str,
    family: str,
    label: str,
    codec,
    surfaces: List[str],
    x_train,
    y_train,
    x_val,
    y_val,
    d_model: int,
    n_layer: int,
    n_head: int,
    seq_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Train ONE tiny LLM for one codec config.

    surfaces : len V
    x_train  : [N_train, T]
    y_train  : [N_train, T]
    x_val    : [N_val, T]
    y_val    : [N_val, T]
    """
    torch.manual_seed(seed)
    V = len(surfaces)                                                  # vocab size
    # Build frozen κ table once: [V, D]
    table = build_codec_table(surfaces, codec, show=True)              # [V, D]
    emb = KroneckerEmbedding(table, d_model=d_model)                   # gather+Linear
    model = TinyLLM(V, d_model, n_layer, n_head, seq_len, emb).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_params = emb.proj.weight.numel()                               # D * d_model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []

    def epoch_loss(xt, yt, train: bool) -> float:
        """
        One pass over blocks.

        xt, yt : [N_blocks, T]
        micro-batch xb, yb : [B, T]
        logits : [B, T, V] → reshape [B*T, V] for CE against yb [B*T]
        """
        model.train(train)
        total = count = 0
        order = torch.randperm(len(xt)) if train else torch.arange(len(xt))
        for i in range(0, len(xt), batch_size):
            sel = order[i : i + batch_size]
            xb, yb = xt[sel].to(device), yt[sel].to(device)              # [B, T], [B, T]
            if train:
                opt.zero_grad(set_to_none=True)
            logits = model(xb)                                         # [B, T, V]
            # flatten time into batch for token-level CE
            loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))  # scalar
            # logits.reshape(-1, V) → [B*T, V]
            # yb.reshape(-1)       → [B*T]
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.item()) * yb.numel()                   # sum CE * #tokens
            count += yb.numel()                                        # B*T
        return total / max(count, 1)                                   # mean token CE

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = epoch_loss(x_train, y_train, True)                        # scalar
        with torch.no_grad():
            va = epoch_loss(x_val, y_val, False)                       # scalar
        history.append(
            {
                "epoch": ep,
                "train_loss": tr,
                "val_loss": va,
                "val_ppl": math.exp(min(va, 20)),
            }
        )
        print(
            f"  [{run_id}] ep {ep}/{epochs}  "
            f"train={tr:.4f} val={va:.4f} ppl={math.exp(min(va, 20)):.2f}"
        )

    hp: Dict[str, Any] = {"code_dim": codec.code_dim}                  # D
    if family == "classic":
        hp["pos_dim"] = codec.pos_dim
        hp["position_mode"] = None
        hp["theta"] = None
    else:
        hp["pos_dim"] = codec.pos_features  # r  (stored under pos_dim for table symmetry)
        hp["pos_features"] = codec.pos_features
        hp["position_mode"] = codec.position_mode
        hp["theta"] = codec.theta

    return {
        "id": run_id,
        "family": family,
        "label": label,
        "hyperparams": hp,
        "trainable_params": n_params,
        "embedding_path_params": emb_params,
        "seconds": round(time.time() - t0, 1),
        "final_val_loss": history[-1]["val_loss"],
        "final_val_ppl": history[-1]["val_ppl"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# Curated run list (valuable / noteworthy only)
# ---------------------------------------------------------------------------

def curated_configs() -> List[Dict[str, Any]]:
    """
    Paper Kronecker: pos_dim 16/32/64/128
    Fourier:         r       16/32/64/128 (absolute)
    + noteworthy:    Fourier r=32 relative  (short vs long geometry ablation)

    Note: D_classic = 256 * pos_dim, D_fourier = 256 * r
          so pos_dim=32 and r=32 are a matched-budget pair (D=8192).
    """
    cfgs = []
    for p in (16, 32, 64, 128):
        cfgs.append(
            {
                "id": f"classic_pos{p}",
                "family": "classic",
                "label": f"Paper Kronecker · pos_dim={p} · D={256*p}",
                "codec": ClassicKronecker(pos_dim=p),
            }
        )
    for r in (16, 32, 64, 128):
        cfgs.append(
            {
                "id": f"fourier_r{r}_abs",
                "family": "fourier",
                "label": f"Ours Fourier · r={r} absolute · D={256*r}",
                "codec": FourierKronecker(
                    pos_features=r, position_mode="absolute", theta=10000.0
                ),
            }
        )
    cfgs.append(
        {
            "id": "fourier_r32_rel",
            "family": "fourier",
            "label": "Ours Fourier · r=32 relative · D=8192 (ablation)",
            "codec": FourierKronecker(
                pos_features=32, position_mode="relative", theta=10000.0
            ),
        }
    )
    return cfgs


def auto_findings(fertility: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic summary bullets filled into report.json for the HTML."""
    classic = [r for r in runs if r["family"] == "classic"]
    fourier = [r for r in runs if r["family"] == "fourier"]
    best_c = min(classic, key=lambda r: r["final_val_loss"]) if classic else None
    best_f = min(fourier, key=lambda r: r["final_val_loss"]) if fourier else None

    crop = {
        lang: fertility["languages"][lang]["classic_crop_rate"]
        for lang in fertility["languages"]
    }
    fert = {
        lang: fertility["languages"][lang]["fertility_tokens_per_word"]
        for lang in fertility["languages"]
    }

    bullets = []
    bullets.append(
        "Fertility ranking (tokens/word, high→low): "
        + " > ".join(fertility["fertility_ranking_high_to_low"])
        + "."
    )
    worst_crop_lang = max(crop, key=crop.get)
    bullets.append(
        f"Classic crop@{fertility['classic_pos_dim_reference']} is highest on "
        f"{worst_crop_lang} ({100*crop[worst_crop_lang]:.3f}% of token instances)."
    )
    if best_c and best_f:
        delta = best_c["final_val_loss"] - best_f["final_val_loss"]
        winner = best_f if best_f["final_val_loss"] <= best_c["final_val_loss"] else best_c
        bullets.append(
            f"Best paper Kronecker: {best_c['id']} val_loss={best_c['final_val_loss']:.4f}."
        )
        bullets.append(
            f"Best Fourier: {best_f['id']} val_loss={best_f['final_val_loss']:.4f}."
        )
        bullets.append(
            f"Head-to-head delta (classic_best − fourier_best) = {delta:+.4f} "
            f"(positive ⇒ Fourier wins). Overall best run: {winner['id']}."
        )
        # Matched D=8192 compare if present
        c32 = next((r for r in classic if r["id"] == "classic_pos32"), None)
        f32 = next((r for r in fourier if r["id"] == "fourier_r32_abs"), None)
        if c32 and f32:
            bullets.append(
                f"Matched D=8192: classic_pos32 val_loss={c32['final_val_loss']:.4f} vs "
                f"fourier_r32_abs val_loss={f32['final_val_loss']:.4f} "
                f"(Δ={c32['final_val_loss']-f32['final_val_loss']:+.4f})."
            )

    return {
        "bullets": bullets,
        "best_classic_id": best_c["id"] if best_c else None,
        "best_fourier_id": best_f["id"] if best_f else None,
        "fertility_by_lang": fert,
        "classic_crop_by_lang": crop,
    }


# ---------------------------------------------------------------------------
# Main Colab entry
# ---------------------------------------------------------------------------

def run_colab(
    tokenizer_name: str = "xlm-roberta-base",
    chars_per_lang: int = 120_000,
    epochs: int = 3,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    seq_len: int = 128,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 0,
    out: str = "results/report.json",
    skip_fertility: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end Colab job:
      fertility → multilingual ids → pack blocks → train all curated codecs
      → write results/report.json
    """
    assert 1 <= epochs <= 5, "epochs must be in 1..5 for this assignment run"

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = "cpu"

    mix = {lang: 1.0 for lang in LANGS}                                # equal language weights
    print("=" * 64)
    print("Problem #3 · Colab run · Paper Kronecker vs Fourier")
    print("=" * 64)

    tok = load_tokenizer(tokenizer_name)

    if skip_fertility:
        fertility = {
            "classic_pos_dim_reference": 32,
            "chars_per_lang": chars_per_lang,
            "languages": {},
            "fertility_ranking_high_to_low": [],
        }
    else:
        fertility = measure_fertility(tok, chars_per_lang, pos_dim_ref=32, seed=seed)

    print("\nBuilding multilingual train stream …")
    ids, mix_meta = build_multilingual_ids(tok, chars_per_lang, mix, seed)  # List[int], dict
    x, y = pack_blocks(ids, seq_len)                                   # [N, T], [N, T]
    n_val = max(1, len(x) // 10)                                       # ~10% val blocks
    x_train, y_train = x[:-n_val], y[:-n_val]                          # [N_train, T]
    x_val, y_val = x[-n_val:], y[-n_val:]                              # [N_val, T]
    print(f"Blocks: train={len(x_train)} val={len(x_val)}")            # N_train, N_val

    print("Decoding vocab surfaces …")
    surfaces = vocab_surfaces(tok)                                     # len V

    runs = []
    for cfg in curated_configs():
        print(f"\n===== {cfg['id']} =====")
        result = train_run(
            run_id=cfg["id"],
            family=cfg["family"],
            label=cfg["label"],
            codec=cfg["codec"],
            surfaces=surfaces,
            x_train=x_train,                                           # [N_train, T]
            y_train=y_train,                                           # [N_train, T]
            x_val=x_val,                                               # [N_val, T]
            y_val=y_val,                                               # [N_val, T]
            d_model=d_model,
            n_layer=n_layer,
            n_head=n_head,
            seq_len=seq_len,
            batch_size=batch_size,                                     # B
            epochs=epochs,
            lr=lr,
            device=device,
            seed=seed,
        )
        runs.append(result)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    findings = auto_findings(fertility, runs)
    report = {
        "schema_version": 1,
        "meta": {
            "problem": 3,
            "title": "Dynamic Kronecker — Paper vs Fourier-Position",
            "tokenizer": tokenizer_name,
            "dataset": "wikimedia/wikipedia (en,hi,ta,te,kn)",
            "device": device,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "epochs": epochs,
            "d_model": d_model,
            "n_layer": n_layer,
            "n_head": n_head,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
            "chars_per_lang": chars_per_lang,
            "mix": mix,
            "seed": seed,
            "tokens_total": mix_meta.get("total_tokens"),
            "mix_meta": mix_meta,
        },
        "fertility": fertility,
        "runs": runs,
        "findings": findings,
    }

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Wrote {path.resolve()}")
    print("Download this file from Colab, then locally run:")
    print(f"  python embed_report.py {path}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("colab", help="Full curated run → results/report.json")
    c.add_argument("--tokenizer", default="xlm-roberta-base")
    c.add_argument("--chars-per-lang", type=int, default=120_000)
    c.add_argument("--epochs", type=int, default=3)
    c.add_argument("--d-model", type=int, default=256)
    c.add_argument("--n-layer", type=int, default=4)
    c.add_argument("--n-head", type=int, default=4)
    c.add_argument("--seq-len", type=int, default=128)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--lr", type=float, default=3e-4)
    c.add_argument("--device", default="cuda")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="results/report.json")
    c.add_argument("--skip-fertility", action="store_true")

    f = sub.add_parser("fertility", help="Fertility only")
    f.add_argument("--tokenizer", default="xlm-roberta-base")
    f.add_argument("--chars-per-lang", type=int, default=200_000)
    f.add_argument("--out", default="results/fertility_only.json")
    f.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)

    if args.cmd == "colab":
        run_colab(
            tokenizer_name=args.tokenizer,
            chars_per_lang=args.chars_per_lang,
            epochs=args.epochs,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
            out=args.out,
            skip_fertility=args.skip_fertility,
        )
    elif args.cmd == "fertility":
        tok = load_tokenizer(args.tokenizer)
        fert = measure_fertility(tok, args.chars_per_lang, 32, args.seed)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fert, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
