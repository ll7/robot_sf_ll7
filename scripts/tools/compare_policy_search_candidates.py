#!/usr/bin/env python3
"""Compare multiple policy-search candidate summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

_ACTUATION_METRICS = (
    "command_clip_fraction_mean",
    "yaw_rate_saturation_fraction_mean",
    "signed_braking_peak_m_s2_mean",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", nargs="+", type=Path)
    parser.add_argument(
        "--promotion-gates",
        type=Path,
        default=Path("configs/policy_search/promotion_gates.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/policy_search/comparison"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Returns:
        dict[str, Any]: Parsed YAML mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _optional_float(raw: Any) -> float | None:
    """Return a numeric value when one is present, otherwise ``None``."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def main() -> int:
    """Compare candidate summaries against each other and baseline gates."""
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_payload = _load_yaml(args.promotion_gates)
    baselines_raw = gate_payload.get("baselines")
    baselines = baselines_raw if isinstance(baselines_raw, dict) else {}

    rows: list[dict[str, Any]] = []
    for summary_path in args.summary_json:
        payload = _load_json(summary_path)
        summary_raw = payload.get("summary")
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        family_raw = summary.get("scenario_family")
        family = family_raw if isinstance(family_raw, dict) else {}
        classic_raw = family.get("classic")
        classic = classic_raw if isinstance(classic_raw, dict) else {}
        francis_raw = family.get("francis2023")
        francis = francis_raw if isinstance(francis_raw, dict) else {}
        synthetic_actuation_raw = summary.get("synthetic_actuation")
        synthetic_actuation = (
            synthetic_actuation_raw if isinstance(synthetic_actuation_raw, dict) else {}
        )
        rows.append(
            {
                "candidate": str(payload.get("candidate", "unknown")),
                "stage": str(payload.get("stage", "unknown")),
                "success_rate": float(summary.get("success_rate", 0.0)),
                "collision_rate": float(summary.get("collision_rate", 0.0)),
                "near_miss_rate": float(summary.get("near_miss_rate", 0.0)),
                "classic_collision_rate": float(classic.get("collision_rate", 0.0)),
                "francis_collision_rate": float(francis.get("collision_rate", 0.0)),
                **{
                    metric_name: _optional_float(synthetic_actuation.get(metric_name))
                    for metric_name in _ACTUATION_METRICS
                },
                "summary_json": str(summary_path),
            }
        )

    for baseline_name, baseline_raw in baselines.items():
        baseline = baseline_raw if isinstance(baseline_raw, dict) else {}
        rows.append(
            {
                "candidate": str(baseline_name),
                "stage": "baseline_reference",
                "success_rate": float(baseline.get("success_rate", 0.0)),
                "collision_rate": float(baseline.get("collision_rate", 0.0)),
                "near_miss_rate": baseline.get("near_miss_rate"),
                "classic_collision_rate": baseline.get("classic_collision_rate"),
                "francis_collision_rate": baseline.get("francis_collision_rate"),
                **dict.fromkeys(_ACTUATION_METRICS),
                "summary_json": None,
            }
        )

    rows.sort(key=lambda row: (-row["success_rate"], row["collision_rate"], row["candidate"]))
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# Policy Search Comparison",
        "",
        "| Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. | Command Clip | Yaw Saturation | Signed Braking Peak |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['stage']} | {row['success_rate']:.4f} | {row['collision_rate']:.4f} | "
            f"{row['near_miss_rate'] if row['near_miss_rate'] is not None else 'n/a'} | "
            f"{row['classic_collision_rate'] if row['classic_collision_rate'] is not None else 'n/a'} | "
            f"{row['francis_collision_rate'] if row['francis_collision_rate'] is not None else 'n/a'} | "
            f"{row['command_clip_fraction_mean'] if row['command_clip_fraction_mean'] is not None else 'n/a'} | "
            f"{row['yaw_rate_saturation_fraction_mean'] if row['yaw_rate_saturation_fraction_mean'] is not None else 'n/a'} | "
            f"{row['signed_braking_peak_m_s2_mean'] if row['signed_braking_peak_m_s2_mean'] is not None else 'n/a'} |"
        )
    md_path = output_dir / "comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
