#!/usr/bin/env python3
"""Compute SNQI seed variance from benchmark episodes JSONL.

This utility groups episodes by a dotted record path (default: ``scenario_params.algo``),
computes the mean SNQI per seed, and then reports variability across seed means
for each group. The output is a JSON mapping group → summary statistics.

Typical usage (via uv):

    uv run python scripts/seed_variance.py \
      --episodes results/episodes.jsonl \
      --out results/seed_variance.json \
      --group-by scenario_params.algo

Inputs
    - episodes: Path to an episodes JSONL produced by the benchmark runner.
    - group-by: Dotted path into the record used to group episodes.

Outputs
    - JSON with per-group fields: ``seeds``, ``snqi_mean``, ``snqi_std``, ``snqi_cv``.

Notes
    - Records without SNQI are skipped; see aggregate helpers for recomputation options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.aggregate import _get_nested, read_jsonl


def compute_seed_variance(
    records: list[dict[str, Any]],
    group_by: str,
) -> dict[str, dict[str, float | int]]:
    """Compute variability of SNQI across seeds for each group.

    For each group (derived from ``group_by`` dotted path), compute the mean SNQI
    per seed, then compute summary statistics across these seed means.

    Parameters
    ----------
    records
        Episode dictionaries as loaded from the episodes JSONL.
    group_by
        Dotted path used to group episodes (e.g., ``scenario_params.algo``).

    Returns
    -------
    dict
        Mapping group → {"seeds": int, "snqi_mean": float, "snqi_std": float, "snqi_cv": float}.
    """
    groups: dict[str, dict[int, list[float]]] = {}
    for rec in records:
        g = _get_nested(rec, group_by, default=str(rec.get("scenario_id", "unknown")))
        seed = int(rec.get("seed", -1))
        snqi = _get_nested(rec, "metrics.snqi")
        if snqi is None:
            continue
        groups.setdefault(str(g), {}).setdefault(seed, []).append(float(snqi))

    out: dict[str, dict[str, float | int]] = {}
    for g, by_seed in groups.items():
        # Aggregate per-seed first, then compute variance across seed means
        seed_means = [float(np.mean(vals)) for vals in by_seed.values() if len(vals) > 0]
        if len(seed_means) == 0:
            continue
        arr = np.asarray(seed_means, dtype=float)
        out[g] = {
            "seeds": len(seed_means),
            "snqi_mean": float(np.mean(arr)),
            "snqi_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "snqi_cv": float(np.std(arr, ddof=1) / (np.mean(arr) + 1e-9)) if len(arr) > 1 else 0.0,
        }
    return out


def main():
    """CLI entry point for seed-level SNQI variance computation."""
    ap = argparse.ArgumentParser(description="Compute seed-level variance for SNQI")
    ap.add_argument("--episodes", required=True, help="Input episodes JSONL path")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--group-by",
        default="scenario_params.algo",
        help="Grouping key (dotted path); defaults to algorithm",
    )
    args = ap.parse_args()

    records = read_jsonl(args.episodes)
    summary = compute_seed_variance(records, args.group_by)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "groups": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
