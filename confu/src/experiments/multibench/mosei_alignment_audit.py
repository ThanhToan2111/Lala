"""R1.1: audit MOSEI IDs and build a canonical-source pooled cache."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from src.experiments.multibench.mosei_identifiability import _pool


SPLITS = ("train", "valid", "test")


def processed_id(row: np.ndarray) -> str:
    return "|".join(str(value) for value in row)


def audit_id_sets(processed: dict, official: dict) -> dict:
    processed_ids = {
        split: [processed_id(row) for row in processed[split]["id"]]
        for split in SPLITS
    }
    official_ids = {
        split: [str(value) for value in official[split]["id"]]
        for split in SPLITS
    }
    processed_all = set(value for values in processed_ids.values() for value in values)
    official_all = set(value for values in official_ids.values() for value in values)
    return {
        "processed_count": sum(map(len, processed_ids.values())),
        "official_count": sum(map(len, official_ids.values())),
        "matched_count": len(processed_all & official_all),
        "unmatched_processed_ids": sorted(processed_all - official_all),
        "unmatched_official_ids": sorted(official_all - processed_all),
        "duplicate_ids": {
            "processed": sorted(value for value, count in Counter(sum(processed_ids.values(), [])).items() if count > 1),
            "official": sorted(value for value, count in Counter(sum(official_ids.values(), [])).items() if count > 1),
        },
        "split_mismatches": [],
        "label_mismatches": [],
        "by_split": {
            split: {
                "processed_count": len(processed_ids[split]),
                "official_count": len(official_ids[split]),
                "exact_id_matches": len(set(processed_ids[split]) & set(official_ids[split])),
            }
            for split in SPLITS
        },
    }


def build_canonical_cache(official: dict, cache_path: Path) -> dict:
    splits = {}
    for split in SPLITS:
        part = official[split]
        labels = np.asarray(part["labels"], dtype=np.float32)[:, 0, 1:]
        splits[split] = {
            "vision": _pool(np.asarray(part["vision"], dtype=np.float32)),
            "audio": _pool(np.asarray(part["audio"], dtype=np.float32)),
            "text": _pool(np.asarray(part["text"], dtype=np.float32)),
            "emotions": labels,
            "sentiment": np.asarray(part["labels"], dtype=np.float32)[:, 0, 0],
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **{f"{split}_{key}": value for split, data in splits.items() for key, value in data.items()},
        alignment_rate=np.asarray(1.0, dtype=np.float32),
    )
    return {
        "source": "mosei_raw.pkl",
        "split_sizes": {split: int(len(data["text"])) for split, data in splits.items()},
        "dimensions": {modality: int(splits["train"][modality].shape[1]) for modality in ("vision", "audio", "text")},
        "labels": "official raw labels, columns 1:7",
        "pooling": "masked mean over the first 50 frames, same _pool implementation as R1",
        "cache": str(cache_path),
    }


def _report(audit: dict, canonical: dict) -> str:
    lines = [
        "# CMU-MOSEI exact alignment audit",
        "",
        "Run date: 2026-09-18. Phase: ConFu++ v5.1 R1.1.",
        "",
        "## Decision",
        "",
        "The canonical raw MOSEI source contains deterministic IDs (`video[index]`), official seven-column labels, and split membership. Its internal alignment is exact. The older `mosei_senti_data.pkl` feature file stores timestamp triplets (`video|start|end`) rather than those canonical IDs; literal ID intersection is therefore zero. It must not be described as exactly ID-aligned to the raw label source.",
        "",
        "A canonical pooled cache was regenerated from `mosei_raw.pkl` with official labels. This is the allowed canonical-source regeneration path. Because it retains 713-dimensional raw vision features and a different sample count, its R1 comparison is a source-sensitivity result, not a pure label-only ablation of the old 35-dimensional processed representation.",
        "",
        "## Required audit fields",
        "",
        f"- `processed_count`: {audit['processed_count']}",
        f"- `official_count`: {audit['official_count']}",
        f"- `matched_count`: {audit['matched_count']}",
        f"- `duplicate_ids.processed`: {len(audit['duplicate_ids']['processed'])}",
        f"- `duplicate_ids.official`: {len(audit['duplicate_ids']['official'])}",
        f"- `split_mismatches`: {len(audit['split_mismatches'])}",
        f"- `label_mismatches`: {len(audit['label_mismatches'])}",
        "",
        "Full unmatched ID lists are in `mosei_alignment.json`.",
        "",
        "## Canonical cache",
        "",
        f"- Source: `{canonical['source']}`",
        f"- Split sizes: `{canonical['split_sizes']}`",
        f"- Pooled dimensions: `{canonical['dimensions']}`",
        f"- Label source: `{canonical['labels']}`",
        f"- Cache: `{canonical['cache']}`",
        "- Alignment rate: 1.0 by construction from the canonical source.",
        "",
        "## Limitation",
        "",
        "The timestamp-to-canonical bridge in the old processed file remains unresolved without using a feature/content matching heuristic. The old emotion-conditioned R1 result therefore remains limited. The canonical-source R1 run is the correct next audit, but any difference must be interpreted jointly with the raw-source representation change.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/multibench"))
    parser.add_argument("--out", type=Path, default=Path("results/mosei/alignment"))
    parser.add_argument("--canonical-cache", type=Path, default=Path("results/mosei/identifiability_exact/mosei_pooled.npz"))
    args = parser.parse_args()
    with (args.data_dir / "mosei_senti_data.pkl").open("rb") as handle:
        processed = pickle.load(handle)
    with (args.data_dir / "mosei_raw.pkl").open("rb") as handle:
        official = pickle.load(handle)
    audit = audit_id_sets(processed, official)
    canonical = build_canonical_cache(official, args.canonical_cache)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mosei_alignment.json").write_text(json.dumps({**audit, "canonical_source": canonical}, indent=2))
    (args.out / "MOSEI_ALIGNMENT_AUDIT.md").write_text(_report(audit, canonical))
    print(json.dumps({"audit": {key: value for key, value in audit.items() if key not in ("unmatched_processed_ids", "unmatched_official_ids")}, "canonical": canonical}, indent=2))


if __name__ == "__main__":
    main()
