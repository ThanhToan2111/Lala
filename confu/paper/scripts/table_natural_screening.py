"""Emit a compact natural-screening table from frozen JSON reports."""
import json
from pathlib import Path


def main():
    root = Path("results")
    rows = []
    mosei = json.loads((root / "mosei/identifiability_exact/mosei_identifiability.json").read_text())
    rows.append(("MOSEI", "VA_to_T", sum(r["metrics"]["valid"]["joint_advantage"] for r in mosei["records"] if r["mapping"] == "VA_to_T") / 3))
    for dataset, rel in (("MUStARD", "mustard/v54/mustard_two_gate.json"), ("MELD", "meld/v55/meld_two_gate.json")):
        payload = json.loads((root / rel).read_text())
        for mapping, record in payload["g1"].items():
            rows.append((dataset, mapping, record["summary"]["valid"]["joint_advantage"]["mean"]))
    print("| Dataset | Mapping | J_cross |")
    print("|---|---|---:|")
    for dataset, mapping, value in rows:
        print(f"| {dataset} | `{mapping}` | {value:.5f} |")


if __name__ == "__main__":
    main()
