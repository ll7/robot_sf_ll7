#!/usr/bin/env python3
"""Measure paired legacy/footprint definitions on release scenarios and dev seeds.

Uses the canonical campaign path, unchanged release geometry/planner settings,
and opt-in diagnostic metadata. This is metric-definition evidence, not a release
run or calibrated ranking. No held-out seed can be selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.benchmark.footprint_metrics import FOOTPRINT_MARKER, FOOTPRINT_SCHEMA
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import (
    REPO_ROOT,
    SUITES,
    assert_dev_seeds,
    verify_head,
)

METRICS = (
    "wall_collisions",
    "agent_collisions",
    "time_to_collision_min",
    "space_compliance",
    "shortest_path_len",
    "path_efficiency",
    "time_to_goal_ideal_ratio",
    "comfort_exposure",
    "force_exceed_events",
    "force_exceed_near_robot",
)


def derive_inputs(out: Path, seeds: list[int], arms: list[str], workers: int) -> Path:
    """Copy source inputs, adding only development seeds and metric opt-in metadata.

    Returns:
        Path to the diagnostic campaign configuration.
    """
    source = REPO_ROOT / SUITES["main"]
    payload = yaml.safe_load(source.read_text())
    matrix = REPO_ROOT / payload["scenario_matrix"]
    scenarios = load_scenarios(matrix, base_dir=matrix.parent)
    for scenario in scenarios:
        scenario.setdefault("metadata", {})[FOOTPRINT_MARKER] = FOOTPRINT_SCHEMA
        scenario["seeds"] = seeds
        if scenario.get("map_file"):
            raw = Path(scenario["map_file"])
            candidates = (matrix.parent / raw, REPO_ROOT / raw)
            scenario["map_file"] = str(next(p.resolve() for p in candidates if p.is_file()))
    out.mkdir(parents=True, exist_ok=True)
    matrix_out = out / "scenarios_dev.yaml"
    matrix_out.write_text(yaml.safe_dump({"scenarios": scenarios}, sort_keys=False))
    payload["scenario_matrix"] = str(matrix_out.resolve())
    payload["seed_policy"] = {"mode": "fixed-list", "seeds": assert_dev_seeds(seeds)}
    payload["workers"] = workers
    payload["resume"] = False
    payload["stop_on_failure"] = False
    payload["record_simulation_step_trace"] = False
    # The additive block is not calibrated; score only unchanged canonical fields.
    payload["snqi_weights"] = payload["snqi_baseline"] = None
    payload.pop("snqi_v2", None)
    for slot in ("release_tag", "doi"):
        payload[slot] = "unpublished-footprint-diagnostic"
    if arms:
        known = {p["key"] for p in payload["planners"]}
        if set(arms) - known:
            raise ValueError(f"unknown planner arms: {set(arms) - known}")
        payload["planners"] = [p for p in payload["planners"] if p["key"] in arms]
    config = out / "campaign_dev.yaml"
    config.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config


def summarize(root: Path) -> dict[str, Any]:  # noqa: C901 - explicit paired availability accounting
    """Report paired finite means, availability changes and every execution failure.

    Returns:
        Compact diagnostic summary, with missing values kept distinct from zero.
    """
    rows = []
    for file in sorted(root.rglob("episodes.jsonl")):
        rows.extend(json.loads(line) for line in file.read_text().splitlines() if line.strip())
    paired: dict[str, list[list[float]]] = {key: [] for key in METRICS}
    available: dict[str, list[int]] = {key: [0, 0] for key in METRICS}
    failures = []
    unavailable_references = []
    for row in rows:
        metrics = row["metrics"]
        current = metrics.get("footprint_metrics")
        if not current or current.get("schema_version") != FOOTPRINT_SCHEMA:
            raise ValueError("missing opted-in footprint metric block")
        for key in METRICS:
            before = metrics.get(key)
            if key == "space_compliance":
                before = current.get("legacy_comparison", {}).get(key)
            if key == "shortest_path_len":
                before = current.get("legacy_comparison", {}).get(key)
            if key == "force_exceed_near_robot":
                before = current.get("legacy_comparison", {}).get(key)
            after = current.get(key)
            valid = [isinstance(x, int | float) and np.isfinite(x) for x in (before, after)]
            available[key][0] += int(valid[0])
            available[key][1] += int(valid[1])
            if all(valid):
                paired[key].append([before, after])
        identity = {key: row.get(key) for key in ("scenario_id", "seed", "algo")}
        if current["reference_status"] != "available":
            unavailable_references.append(identity)
        outcome = row.get("outcome", {})
        if not bool(metrics.get("success")):
            failures.append(
                {
                    **identity,
                    "classification": "collision"
                    if outcome.get("collision_event")
                    else "timeout_or_noncompletion",
                    "outcome": outcome,
                }
            )
    table = {}
    for key, pairs in paired.items():
        arr = np.asarray(pairs).reshape(-1, 2)
        table[key] = {
            "paired_n": len(pairs),
            "finite_before": available[key][0],
            "finite_after": available[key][1],
            "before": float(arr[:, 0].mean()) if len(pairs) else None,
            "after": float(arr[:, 1].mean()) if len(pairs) else None,
            "mean_change": float(np.diff(arr, axis=1).mean()) if len(pairs) else None,
        }
    return {
        "status": "diagnostic-only",
        "episodes": len(rows),
        "metrics": table,
        "episode_non_successes": failures,
        "unavailable_references": unavailable_references,
    }


def main() -> int:
    """Run or summarize a source-bound development campaign.

    Returns:
        Zero after measurement; runner failures remain explicit in campaign artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="*", default=["goal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1001, 1031)))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    head = verify_head(args.head_sha)
    seeds = assert_dev_seeds(args.seeds)
    out = args.output_dir.resolve()
    if not args.summarize_only:
        config = derive_inputs(out, seeds, args.arms, args.workers)
        run_campaign(
            load_campaign_config(config),
            out_dir=out / "campaign",
            label="footprint-dev",
            campaign_id="footprint-dev",
            skip_preflight=False,
        )
    summary = summarize(out / "campaign")
    summary.update(head_sha=head, seeds=seeds, arms=args.arms)
    summary["input_sha256"] = hashlib.sha256((out / "campaign_dev.yaml").read_bytes()).hexdigest()
    (out / "metric_changes.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"episodes": summary["episodes"], "metrics": summary["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
