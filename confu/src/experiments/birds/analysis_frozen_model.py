"""Hard-sample + missing-modality analysis on a saved frozen-feature model.

#6: bin test samples by lower-order probe confidence; measure full-vs-lower
    accuracy gain per bin (hypothesis: interaction helps uncertain samples).
#7: drop each modality at probe time; compare ConFu++ vs ConFu degradation.

Uses the saved ConFu+U+S1+PairU seed-1 checkpoint and the ConFu baseline
(retrained identically for analysis with --save_model if missing).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.experiments.birds.confu_frozen import ConFuFrozen, MODALITY

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: str, embed_dim=256, hidden=512, n_classes=71):
    model = ConFuFrozen([2048, 768, 384], embed_dim, hidden, n_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def embed(model, feats, idx):
    tensors = [torch.tensor(feats[m][idx], dtype=torch.float32, device=device) for m in MODALITY]
    outs = []
    for s in range(0, len(idx), 4096):
        outs.append({k: v.cpu().numpy() for k, v in model([t[s:s+4096] for t in tensors]).items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


@torch.no_grad()
def head_logits(model, out):
    base = np.concatenate([out["z1"], out["z2"], out["z3"]], 1)
    full = np.concatenate([base, out["z12"], out["z13"], out["z23"]], 1)
    return model.head_base(torch.tensor(base, device=device)).cpu().numpy(), \
           model.head_full(torch.tensor(full, device=device)).cpu().numpy()


def softmax_conf(logits):
    from scipy.special import softmax
    p = softmax(logits, axis=1)
    return p.max(1)


def main():
    data = np.load("data/bird_mml/embeddings_subset.npz", allow_pickle=True)
    feats = {"image": data["img"], "audio": data["aud"], "text": data["txt"]}
    labels = data["labels"]
    split = dict(np.load("data/bird_mml/split.npz"))
    te = split["test"]

    model = load_model("results/birds/model_s1_0.5_lu_1.0_lp_1.0_seed_1.pt")
    out = embed(model, feats, te)
    lb, lf = head_logits(model, out)
    y = labels[te]
    pred_b, pred_f = lb.argmax(1), lf.argmax(1)
    conf = softmax_conf(lb)

    # --- hard-sample analysis ---
    bins = np.quantile(conf, [0, 1/3, 2/3, 1.0])
    print("== Hard-sample analysis (base-confidence bins) ==")
    result = {"bins": []}
    for i in range(3):
        mask = (conf >= bins[i]) & (conf <= bins[i+1] + 1e-9)
        acc_b = (pred_b[mask] == y[mask]).mean()
        acc_f = (pred_f[mask] == y[mask]).mean()
        n = mask.sum()
        result["bins"].append({"bin": i, "conf_range": [float(bins[i]), float(bins[i+1])],
                               "n": int(n), "acc_base": float(acc_b), "acc_full": float(acc_f),
                               "gain": float(acc_f - acc_b)})
        print(f"  bin{i} conf[{bins[i]:.2f},{bins[i+1]:.2f}] n={n}: base={acc_b*100:.1f}% full={acc_f*100:.1f}% gain={(acc_f-acc_b)*100:+.2f}pp")

    # --- missing-modality robustness (probe-level, model with task heads) ---
    print("\n== Missing-modality at test time (full head) ==")
    result["missing"] = {}
    acc_all = (pred_f == y).mean()
    print(f"  all modalities: {acc_all*100:.2f}%")
    result["missing"]["all"] = float(acc_all)
    for i, name in enumerate(MODALITY):
        out_m = {k: v.copy() for k, v in out.items()}
        out_m[f"z{i+1}"] = np.zeros_like(out_m[f"z{i+1}"])
        _, lf_m = head_logits(model, out_m)
        acc_m = (lf_m.argmax(1) == y).mean()
        result["missing"][name] = float(acc_m)
        print(f"  drop {name:<6}: {acc_m*100:.2f}%  (Δ {(acc_all-acc_m)*100:+.2f}pp)")

    os.makedirs("results/birds", exist_ok=True)
    with open("results/birds/HARD_MISSING_ANALYSIS.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nsaved results/birds/HARD_MISSING_ANALYSIS.json")


if __name__ == "__main__":
    main()
