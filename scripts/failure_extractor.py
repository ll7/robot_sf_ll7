#!/usr/bin/env python3
"""Extract worst episodes by a chosen metric from episodes JSONL.

This script selects the top-k worst episodes according to a dotted metric path
(e.g., ``metrics.snqi`` or ``metrics.collisions``). Direction can be set to
``min`` (lowest values are worst) or ``max`` (highest values are worst).

Typical usage:

        # Lowest SNQI (worst), top-20
        uv run python scripts/failure_extractor.py \
            --episodes results/episodes.jsonl \
            --out results/worst_snqi.json \
            --metric metrics.snqi \
            --direction min \
            --top-k 20

Inputs
        - episodes: Episodes JSONL path.
        - metric: Dotted path to metric inside episode record.
        - direction: ``min`` or ``max``.
        - top-k: Number of episodes to include.

Outputs
        - JSON list with entries containing: ``score``, ``episode_id``, ``scenario_id``,
            ``seed``, ``algo``, and full ``metrics``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import _get_nested, read_jsonl


def pick_worst(
    records: list[dict[str, Any]],
    metric_path: str,
    top_k: int,
    reverse: bool,
) -> list[tuple[float, dict[str, Any]]]:
    """Score and select the top-k worst episodes by a metric.

    Parameters
    ----------
    records
        Episode dictionaries as loaded from the episodes JSONL.
    metric_path
        Dotted path to the metric (e.g., ``metrics.snqi``).
    top_k
        Number of episodes to select.
    reverse
        If True, higher metric values are treated as worse (descending sort).

    Returns
    -------
    list of (float, dict)
        A list of (score, episode_record) pairs, truncated to top_k.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        v = _get_nested(rec, metric_path)
        if v is None:
            continue
        try:
            score = float(v)
        except (TypeError, ValueError):
            continue
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=reverse)
    return scored[:top_k]


def main():
    """CLI entry point for extracting worst episodes."""
    ap = argparse.ArgumentParser(description="Extract top-k worst episodes by metric")
    ap.add_argument("--episodes", required=True, help="Input episodes JSONL")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--metric",
        default="metrics.snqi",
        help="Dotted path metric (default: metrics.snqi)",
    )
    ap.add_argument(
        "--direction",
        choices=["min", "max"],
        default="min",
        help="min selects lowest values as worst; max selects highest",
    )
    ap.add_argument("--top-k", type=int, default=20, help="Number of episodes to select")
    args = ap.parse_args()

    records = read_jsonl(args.episodes)

    reverse = args.direction == "max"
    worst = pick_worst(records, args.metric, args.top_k, reverse=reverse)

    # Project minimal info for convenience
    payload = []
    for score, rec in worst:
        payload.append(
            {
                "score": score,
                "episode_id": rec.get("episode_id"),
                "scenario_id": rec.get("scenario_id"),
                "seed": rec.get("seed"),
                "algo": _get_nested(rec, "scenario_params.algo"),
                "metrics": rec.get("metrics"),
            },
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path), "count": len(payload)}, indent=2))


if __name__ == "__main__":
    main()
