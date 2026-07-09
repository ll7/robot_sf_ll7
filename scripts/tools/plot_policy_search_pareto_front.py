#!/usr/bin/env python3
"""Plot a success versus collision Pareto view for policy-search runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument(
        "--promotion-gates",
        type=Path,
        default=Path("configs/policy_search/promotion_gates.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/policy_search/pareto/pareto.png")
    )
    return parser.parse_args()


def main() -> int:
    """Render a success-versus-collision Pareto plot."""
    args = parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, float, float]] = []
    for summary_path in args.summary_json:
        payload = _load_json(summary_path)
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        rows.append(
            (
                str(payload.get("candidate", summary_path.stem)),
                float(summary.get("collision_rate", 0.0)),
                float(summary.get("success_rate", 0.0)),
            )
        )

    gate_payload = yaml.safe_load(args.promotion_gates.read_text(encoding="utf-8")) or {}
    baselines = gate_payload.get("baselines", {}) if isinstance(gate_payload, dict) else {}
    for name, baseline in baselines.items():
        if not isinstance(baseline, dict):
            continue
        rows.append(
            (
                str(name),
                float(baseline.get("collision_rate", 0.0)),
                float(baseline.get("success_rate", 0.0)),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, collision, success in rows:
        ax.scatter(collision, success, s=60)
        ax.annotate(label, (collision, success), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Collision Rate")
    ax.set_ylabel("Success Rate")
    ax.set_title("Policy Search Pareto View")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(json.dumps({"plot": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
