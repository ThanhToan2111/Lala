"""OGM-style modality-rebalancing baseline on frozen Bird-MML embeddings.

Same-topic control (Peng et al., CVPR 2022, "Balanced Multimodal Learning via
On-the-fly Gradient Modulation"): during joint training of an additive-logit
two-modality classifier, downscale the gradients of the dominant modality's
branch with a coefficient derived from per-batch per-modality softmax scores.

Our reproduction: frozen features, per-modality linear heads, logits summed
(image + text pair — the only pair with genuine headroom). Compare:
    plain      : standard training
    ogm        : k_i = (s_j / s_i)^alpha when modality i dominates, else 1
5 seeds, identical split; report test accuracy and per-modality probe.

This answers: does gradient rebalancing capture image-text complementarity
that plain joint training misses?
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def train_arm(ogm: bool, seed: int, args, x1, x2, y, x1t, x2t, yt, device) -> dict:
    torch.manual_seed(seed)
    d1, d2, C = x1.shape[1], x2.shape[1], int(y.max()) + 1
    norm1 = (x1.mean(0, keepdims=True), x1.std(0, keepdims=True) + 1e-6)
    norm2 = (x2.mean(0, keepdims=True), x2.std(0, keepdims=True) + 1e-6)
    X1 = torch.tensor((x1 - norm1[0]) / norm1[1], dtype=torch.float32, device=device)
    X2 = torch.tensor((x2 - norm2[0]) / norm2[1], dtype=torch.float32, device=device)
    X1t = torch.tensor((x1t - norm1[0]) / norm1[1], dtype=torch.float32, device=device)
    X2t = torch.tensor((x2t - norm2[0]) / norm2[1], dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.long, device=device)
    Yt = torch.tensor(yt, dtype=torch.long, device=device)

    h1 = nn.Linear(d1, 256).to(device); h2 = nn.Linear(d2, 256).to(device)
    c1 = nn.Linear(256, C).to(device); c2 = nn.Linear(256, C).to(device)
    params = list(h1.parameters()) + list(h2.parameters()) + list(c1.parameters()) + list(c2.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(args.epochs):
        for idx in torch.randperm(len(Y), generator=gen).split(1024):
            f1, f2 = h1(X1[idx]), h2(X2[idx])
            l1, l2 = c1(f1), c2(f2)
            logits = l1 + l2
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            if ogm:
                with torch.no_grad():
                    s1 = F.softmax(l1, -1).gather(1, Y[idx][:, None]).mean()
                    s2 = F.softmax(l2, -1).gather(1, Y[idx][:, None]).mean()
                    # OGM coefficient: downscale the dominant branch
                    k1 = 1.0 if s1 <= s2 else (s2 / s1).item() ** args.alpha
                    k2 = 1.0 if s2 <= s1 else (s1 / s2).item() ** args.alpha
                loss.backward()
                with torch.no_grad():
                    for p in list(h1.parameters()) + list(c1.parameters()):
                        if p.grad is not None:
                            p.grad.mul_(k1)
                    for p in list(h2.parameters()) + list(c2.parameters()):
                        if p.grad is not None:
                            p.grad.mul_(k2)
            else:
                loss.backward()
            opt.step()
    h1.eval(); h2.eval(); c1.eval(); c2.eval()
    with torch.no_grad():
        acc = ((c1(h1(X1t)) + c2(h2(X2t))).argmax(1) == Yt).float().mean().item()
        acc1 = (c1(h1(X1t)).argmax(1) == Yt).float().mean().item()
        acc2 = (c2(h2(X2t)).argmax(1) == Yt).float().mean().item()
    return {"ogm": ogm, "seed": seed, "acc_joint": acc, "acc_image": acc1, "acc_text": acc2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="data/bird_mml/embeddings_subset.npz")
    parser.add_argument("--out", default="results/birds")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.emb, allow_pickle=True)
    img, txt, labels = data["img"], data["txt"], data["labels"]
    split = dict(np.load("data/bird_mml/split.npz"))
    tr, te = split["train"], split["test"]

    runs = []
    for ogm in (False, True):
        for seed in range(1, args.seeds + 1):
            r = train_arm(ogm, seed, args, img[tr], txt[tr], labels[tr],
                          img[te], txt[te], labels[te], device)
            runs.append(r)
            print(f"[ogm={ogm} seed={seed}] joint={r['acc_joint']*100:.2f}% "
                  f"img={r['acc_image']*100:.2f}% txt={r['acc_text']*100:.2f}%", flush=True)
    for ogm in (False, True):
        sub = [r for r in runs if r["ogm"] == ogm]
        acc = [r["acc_joint"] for r in sub]
        print(f"ogm={ogm}: joint {statistics.mean(acc)*100:.2f} ± {statistics.stdev(acc)*100:.2f}")
    plain = [r["acc_joint"] for r in runs if not r["ogm"]]
    ogmd = [r["acc_joint"] for r in runs if r["ogm"]]
    diff = [o - p for o, p in zip(ogmd, plain)]
    md, sd = statistics.mean(diff), statistics.stdev(diff)
    t = md / (sd / len(diff) ** 0.5) if sd > 0 else float("inf")
    print(f"paired Δ (OGM - plain): {md*100:+.3f} ± {sd*100:.3f} (t={t:+.2f})")
    path = os.path.join(args.out, "OGM_FROZEN.json")
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": runs,
                   "paired_delta_pp": md * 100, "t": t}, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
