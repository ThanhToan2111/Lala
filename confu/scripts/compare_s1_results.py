"""Aggregate S1 sweep results and compare with canonical AV-MNIST baselines."""

import glob
import json
import statistics
import sys

BASELINES = {
    "Additive (canonical)": {"accuracy": (70.448, 0.522)},
    "Concat+MLP (canonical)": {"accuracy": (70.518, 0.453)},
    "ConFu-style (canonical)": {"accuracy": (70.886, 0.578)},
    "Prev SynergyFormer": {"accuracy": (70.762, 0.338)},
    "P1 utility w0.2 (canonical)": {"accuracy": (70.296, 0.545), "gain12": (0.234, 0.263)},
}

KEYS = [
    "accuracy", "gain12", "zero_drop12", "shuffle_drop12",
    "shuffle_modality1_drop", "shuffle_modality2_drop",
    "net_correction", "linear_probe_gain", "nonlinear_probe_gain",
]


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def main(patterns):
    print("=== Baselines (canonical, 5 seeds) ===")
    for name, metrics in BASELINES.items():
        line = f"  {name:<28}"
        for k, (m, s) in metrics.items():
            line += f" {k}={m:.3f}±{s:.3f}"
        print(line)
    print()
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"no match: {pattern}")
            continue
        runs = []
        for path in files:
            try:
                runs.append(json.load(open(path)))
            except Exception as exc:
                print(f"  unreadable {path}: {exc}")
        if not runs:
            continue
        exp = runs[0]["experiment"]
        print(f"=== {exp} ({len(runs)} seeds) ===")
        for k in KEYS:
            stats = mean_std([r.get(k) for r in runs])
            if stats:
                print(f"  {k:<24} {stats[0]:+.4f} ± {stats[1]:.4f}")
        pred = mean_std([
            r["predictability"]["r1"]["r2"] for r in runs
            if r.get("predictability") and "r1" in r["predictability"]
        ])
        pred2 = mean_std([
            r["predictability"]["r2"]["r2"] for r in runs
            if r.get("predictability") and "r2" in r["predictability"]
        ])
        if pred:
            print(f"  {'R2(r1 -> r12)':<24} {pred[0]:.4f} ± {pred[1]:.4f}")
        if pred2:
            print(f"  {'R2(r2 -> r12)':<24} {pred2[0]:.4f} ± {pred2[1]:.4f}")
        acc = mean_std([r.get("accuracy") for r in runs])
        if acc:
            delta = acc[0] - BASELINES["P1 utility w0.2 (canonical)"]["accuracy"][0] + 70.296 - 70.296
            print(f"  vs ConFu-style: {acc[0]*100 - 70.886:+.3f} pp | vs prev SynergyFormer: {acc[0]*100 - 70.762:+.3f} pp")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
