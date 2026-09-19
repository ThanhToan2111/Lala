"""Bird-MML complementarity audit on frozen embeddings (spec §25/§107).

Answers the gate question: does this benchmark have genuine multimodal
headroom (conditional complementarity), and which modality dominates?

Protocol:
    - stratified 70/15/15 train/val/test split per species (seed-fixed,
      saved to data/bird_mml/split.npz for every future experiment);
    - sklearn multinomial logistic probes on standardized frozen features;
    - all unimodal / pair / full subsets;
    - conditional gains (pair over best single, full over best pair);
    - nonlinear (parameter-matched MLP) probes for the key comparisons.

Output: results/birds/AUDIT.json + printed table.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def make_split(labels: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        train.append(idx[: int(0.7 * n)])
        val.append(idx[int(0.7 * n): int(0.85 * n)])
        test.append(idx[int(0.85 * n):])
    return {
        "train": np.sort(np.concatenate(train)),
        "val": np.sort(np.concatenate(val)),
        "test": np.sort(np.concatenate(test)),
    }


def linear_probe(x_tr, y_tr, x_te, y_te, max_iter=400):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, tol=1e-3))
    probe.fit(x_tr, y_tr)
    return probe.score(x_te, y_te)


def mlp_probe(x_tr, y_tr, x_te, y_te, hidden=256, epochs=30, seed=0, device="cuda"):
    import torch
    from torch import nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    mean = x_tr.mean(0, keepdims=True)
    std = x_tr.std(0, keepdims=True) + 1e-6
    x_tr_n = torch.tensor((x_tr - mean) / std, dtype=torch.float32)
    x_te_n = torch.tensor((x_te - mean) / std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)
    probe = nn.Sequential(nn.Linear(x_tr.shape[1], hidden), nn.GELU(), nn.Linear(hidden, y_tr.max() + 1)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    x_tr_g, y_tr_g = x_tr_n.to(device), y_tr_t.to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(y_tr), generator=gen).split(1024):
            loss = F.cross_entropy(probe(x_tr_g[idx]), y_tr_g[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    with torch.no_grad():
        return (probe(x_te_n.to(device)).argmax(1).cpu() == y_te_t).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/bird_mml/embeddings.npz")
    parser.add_argument("--out", default="results/birds")
    parser.add_argument("--n_probe", type=int, default=60000, help="cap train samples per probe")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.emb, allow_pickle=True)
    img, aud, txt, labels = data["img"], data["aud"], data["txt"], data["labels"]
    n_classes = len(np.unique(labels))
    print(f"embeddings: img{img.shape} aud{aud.shape} txt{txt.shape}; {n_classes} classes")

    split_path = os.path.join(os.path.dirname(args.emb), "split.npz")
    if os.path.exists(split_path):
        split = dict(np.load(split_path))
    else:
        split = make_split(labels)
        np.savez(split_path, **split)
    tr, te = split["train"], split["test"]
    rng = np.random.default_rng(0)
    tr_probe = np.sort(rng.choice(tr, size=min(args.n_probe, len(tr)), replace=False))

    sets = {
        "image": img, "audio": aud, "text": txt,
        "image+audio": np.concatenate([img, aud], 1),
        "image+text": np.concatenate([img, txt], 1),
        "audio+text": np.concatenate([aud, txt], 1),
        "all": np.concatenate([img, aud, txt], 1),
    }
    results = {"n_train_probe": len(tr_probe), "n_test": len(te), "n_classes": n_classes,
               "linear": {}, "nonlinear": {}}
    print("\n== linear probes (test acc) ==")
    for name, x in sets.items():
        acc = linear_probe(x[tr_probe], labels[tr_probe], x[te], labels[te])
        results["linear"][name] = acc
        print(f"  {name:<12} {acc*100:.2f}%", flush=True)

    print("\n== nonlinear probes (parameter-matched MLP, test acc) ==")
    for name in ["image", "audio", "text", "image+audio", "image+text", "audio+text", "all"]:
        x = sets[name]
        acc = mlp_probe(x[tr_probe], labels[tr_probe], x[te], labels[te])
        results["nonlinear"][name] = acc
        print(f"  {name:<12} {acc*100:.2f}%", flush=True)

    lin = results["linear"]
    results["conditional_gains"] = {
        "image+audio_over_best_single": lin["image+audio"] - max(lin["image"], lin["audio"]),
        "image+text_over_best_single": lin["image+text"] - max(lin["image"], lin["text"]),
        "audio+text_over_best_single": lin["audio+text"] - max(lin["audio"], lin["text"]),
        "all_over_best_pair": lin["all"] - max(lin["image+audio"], lin["image+text"], lin["audio+text"]),
    }
    print("\n== conditional gains (linear, pp) ==")
    for k, v in results["conditional_gains"].items():
        print(f"  {k:<30} {v*100:+.2f}")

    out_path = os.path.join(args.out, "AUDIT.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
