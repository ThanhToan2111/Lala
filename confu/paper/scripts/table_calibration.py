"""Paper table: v6.0 and v6.1 calibration summaries."""

import json
from pathlib import Path


def main():
    root = Path("results")
    rows = []
    nulls = json.loads((root / "calibration/v60/null_distributions.json").read_text())["settings"]
    for name, value in nulls.items():
        summary = value["summary"]
        rows.append((name, summary["mean"], summary["q99"], summary["max"], "v6.0 B=10"))
    c1 = root / "calibration/v61/C1_MOSEI_NULL.json"
    if c1.exists():
        payload = json.loads(c1.read_text())
        summary = payload["null"]
        rows.append(("MOSEI_VA_to_T", summary["mean"], summary["q99"], summary["max"], "v6.1-C1 B=50"))
    print("| Setting | Null mean | q99 | Max | Audit |")
    print("|---|---:|---:|---:|---|")
    for name, mean, q99, maximum, audit in rows:
        print(f"| `{name}` | {mean:.5f} | {q99:.5f} | {maximum:.5f} | {audit} |")


if __name__ == "__main__":
    main()
