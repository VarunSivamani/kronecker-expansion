"""Runs the full Kronecker-vs-Fourier comparison and writes results to
results.json / results.txt. No numbers in report.md should be hand-typed --
they are all read back from this run's output.
"""

from __future__ import annotations

import json

import torch
from torch import nn

from embedding_codecs import FourierByteCodec, KroneckerCodec
from data import ByteClassificationDataset
from tiny_transformer import TinyClassifier, count_trainable_params

D_MODEL = 64
EPOCHS = 2000
LR = 3e-3
SEED = 0


def count_collisions(vocab: list[bytes], codec) -> tuple[int, int]:
    """Returns (num_colliding_pairs, num_tokens_checked) using the codec's
    own collision_key (what it actually observes of each token)."""
    keys = [codec.collision_key(b) for b in vocab]
    seen: dict[bytes, int] = {}
    collisions = 0
    for k in keys:
        if k in seen:
            collisions += 1
        seen[k] = seen.get(k, 0) + 1
    return collisions, len(vocab)


def exact_vector_collisions(codec) -> int:
    """Ground truth: how many pairs of distinct tokens produce identical
    (or near-identical) pre-projection codes, measured directly on tensors."""
    codes = codec.codes  # [V, code_dim]
    n = codes.shape[0]
    collisions = 0
    # O(n^2) pairwise check -- fine at this vocab size
    for i in range(n):
        for j in range(i + 1, n):
            if torch.allclose(codes[i], codes[j], atol=1e-5):
                collisions += 1
    return collisions


def train_and_eval(codec_name: str, codec, dataset: ByteClassificationDataset):
    torch.manual_seed(SEED)
    model = TinyClassifier(codec, D_MODEL, dataset.num_classes)
    ids, labels = dataset.tensors()
    train_idx, val_idx = dataset.split(seed=SEED)
    train_idx_t = torch.tensor(train_idx)
    val_idx_t = torch.tensor(val_idx)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(ids[train_idx_t])
        loss = loss_fn(logits, labels[train_idx_t])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step()

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(-1) == labels[train_idx_t]).float().mean().item()
                val_logits = model(ids[val_idx_t])
                val_loss = loss_fn(val_logits, labels[val_idx_t]).item()
                val_acc = (val_logits.argmax(-1) == labels[val_idx_t]).float().mean().item()
            history.append(
                dict(epoch=epoch, train_loss=loss.item(), train_acc=train_acc,
                     val_loss=val_loss, val_acc=val_acc)
            )

    # Targeted eval: accuracy restricted to the "collide" group only
    model.eval()
    with torch.no_grad():
        all_logits = model(ids)
        preds = all_logits.argmax(-1)
        collide_mask = torch.tensor(
            [1 if i >= dataset.collide_start else 0 for i in range(len(dataset.vocab))],
            dtype=torch.bool,
        )
        collide_acc = (preds[collide_mask] == labels[collide_mask]).float().mean().item()
        short_acc = (preds[~collide_mask] == labels[~collide_mask]).float().mean().item()

    return dict(
        codec=codec_name,
        history=history,
        final_train_acc=history[-1]["train_acc"],
        final_val_acc=history[-1]["val_acc"],
        collide_group_acc=collide_acc,
        short_group_acc=short_acc,
        trainable_params=count_trainable_params(model),
        codec_projection_params=sum(p.numel() for p in codec.projection.parameters()),
        codec_code_dim=codec.code_dim,
    )


def multi_seed_summary(seeds: list[int]) -> dict:
    """Repeats the collide-group / short-group measurement across several
    independently-sampled datasets, for robustness beyond a single seed."""
    from embedding_codecs import FourierByteCodec, KroneckerCodec

    summary = {"kronecker": [], "fourier": []}
    for seed in seeds:
        ds = ByteClassificationDataset(seed=seed, n_short=64, n_pairs=32)
        kron = KroneckerCodec(ds.vocab, D_MODEL, pos_dim=32)
        four = FourierByteCodec(ds.vocab, D_MODEL, num_freqs=128)
        for name, codec in [("kronecker", kron), ("fourier", four)]:
            run = train_and_eval(name, codec, ds)
            summary[name].append(
                dict(seed=seed, collide_group_acc=run["collide_group_acc"],
                     short_group_acc=run["short_group_acc"])
            )
    return summary


def main():
    dataset = ByteClassificationDataset(seed=SEED, n_short=64, n_pairs=32)
    vocab = dataset.vocab

    kron = KroneckerCodec(vocab, D_MODEL, pos_dim=32)
    fourier = FourierByteCodec(vocab, D_MODEL, num_freqs=128)

    # Collision measurements (ground truth, from actual tensors)
    kron_key_collisions, n_tok = count_collisions(vocab, kron)
    fourier_key_collisions, _ = count_collisions(vocab, fourier)
    kron_vector_collisions = exact_vector_collisions(kron)
    fourier_vector_collisions = exact_vector_collisions(fourier)

    results = dict(
        vocab_size=n_tok,
        num_colliding_pairs_engineered=len(dataset.pairs),
        kronecker=dict(
            key_collisions=kron_key_collisions,
            vector_collisions=kron_vector_collisions,
        ),
        fourier=dict(
            key_collisions=fourier_key_collisions,
            vector_collisions=fourier_vector_collisions,
        ),
        runs={},
    )

    for name, codec in [("kronecker", kron), ("fourier", fourier)]:
        print(f"\n=== Training with {name} codec ===")
        run = train_and_eval(name, codec, dataset)
        results["runs"][name] = run
        print(f"{name}: final_val_acc={run['final_val_acc']:.3f} "
              f"collide_group_acc={run['collide_group_acc']:.3f} "
              f"short_group_acc={run['short_group_acc']:.3f} "
              f"projection_params={run['codec_projection_params']} "
              f"code_dim={run['codec_code_dim']}")

    print("\n=== Multi-seed robustness check ===")
    results["multi_seed"] = multi_seed_summary(seeds=[0, 1, 2, 3, 4])
    for name in ("kronecker", "fourier"):
        accs = [r["collide_group_acc"] for r in results["multi_seed"][name]]
        print(f"{name}: collide_group_acc per seed = {accs} "
              f"(mean={sum(accs)/len(accs):.3f})")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nWrote results.json")


if __name__ == "__main__":
    main()
