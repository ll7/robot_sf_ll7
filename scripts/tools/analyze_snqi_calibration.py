#!/usr/bin/env python3
"""Analyze SNQI v3 calibration robustness against weight and anchor variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.snqi.calibration import (
    analyze_snqi_calibration,
    derive_planner_rows_from_episodes,
    load_episode_jsonl,
    write_calibration_csv,
    write_calibration_markdown,
)
from robot_sf.benchmark.snqi.campaign_contract import (
    collect_episodes_from_campaign_runs,
    resolve_weight_mapping,
    sanitize_baseline_stats,
)
from robot_sf.common.artifact_paths import get_repository_root


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--campaign-root",
        type=Path,
        help="Completed camera-ready campaign root with reports/campaign_summary.json.",
    )
    input_group.add_argument(
        "--episodes",
        type=Path,
        help="Episode JSONL file with planner_key, kinematics, and metrics fields.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("configs/benchmarks/snqi_weights_camera_ready_v3.json"),
        help="SNQI v3 weight asset to use as the baseline comparison point.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("configs/benchmarks/snqi_baseline_camera_ready_v3.json"),
        help="SNQI v3 normalization baseline asset to use as the fixed anchor.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="Local weight perturbation fraction used around each v3 weight.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _load_campaign(campaign_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing campaign summary: {summary_path}")
    summary = _load_json(summary_path)
    runs = summary.get("runs") if isinstance(summary.get("runs"), list) else []
    planner_rows = (
        summary.get("planner_rows") if isinstance(summary.get("planner_rows"), list) else []
    )
    episodes = collect_episodes_from_campaign_runs(runs, repo_root=get_repository_root())
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), dict) else {}
    return episodes, planner_rows, str(campaign.get("campaign_id", campaign_root.name))


def main(argv: list[str] | None = None) -> int:
    """Run SNQI calibration analysis and write JSON, Markdown, and CSV artifacts."""
    args = _parse_args(argv)
    if args.epsilon <= 0.0 or args.epsilon >= 1.0:
        raise ValueError("--epsilon must be in the open interval (0, 1).")

    weights_path = args.weights.resolve()
    baseline_path = args.baseline.resolve()
    weights = resolve_weight_mapping(_load_json(weights_path))
    baseline, baseline_warnings = sanitize_baseline_stats(_load_json(baseline_path))

    if args.campaign_root is not None:
        input_path = args.campaign_root.resolve()
        episodes, planner_rows, input_id = _load_campaign(input_path)
        input_kind = "campaign_root"
    else:
        input_path = args.episodes.resolve()
        episodes = load_episode_jsonl(input_path)
        planner_rows = derive_planner_rows_from_episodes(episodes)
        input_id = input_path.stem
        input_kind = "episodes_jsonl"

    payload = analyze_snqi_calibration(
        episodes,
        weights=weights,
        baseline=baseline,
        planner_rows=planner_rows,
        epsilon=args.epsilon,
    )
    payload["input"] = {
        "kind": input_kind,
        "id": input_id,
        "path": str(input_path),
        "weights_path": str(weights_path),
        "weights_sha256": _sha256_file(weights_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha256_file(baseline_path),
        "baseline_adjustments": baseline_warnings,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    write_calibration_markdown(args.output_md, payload)
    write_calibration_csv(args.output_csv, payload)

    print(
        json.dumps(
            {
                "snqi_calibration_json": str(args.output_json),
                "snqi_calibration_md": str(args.output_md),
                "snqi_calibration_csv": str(args.output_csv),
                "recommendation": payload["recommendation"]["decision"],
                "episodes": payload["episodes"],
                "planners": payload["planners"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
