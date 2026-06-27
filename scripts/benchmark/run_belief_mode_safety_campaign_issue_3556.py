#!/usr/bin/env python3
"""Run the ScenarioBelief drop-vs-retain safety contrast through the REAL benchmark runner (#3556).

Plain-language summary: #3471 (PR #3553) showed, in a controlled scripted scenario, that *dropping*
uncertain (out-of-field-of-view) agents from the ``stream_gap`` planner is less safe than *retaining*
them. PR #3561 wired the three belief modes into the real benchmark runner. This script closes #3556:
it runs the ``oracle`` / ``uncertain_retained`` / ``uncertain_dropped`` contrast through
``robot_sf.benchmark.map_runner.run_map_batch`` on a predeclared crossing scenario family + seed
matrix, computes episode-level safety metrics from the real benchmark records, and classifies the
result (``revise`` / ``retention_dominates`` / ``inconclusive`` / ``inconclusive_oracle_unsafe``).

The three modes differ only in the ``belief_mode`` algo-config knob; everything else is identical so
the contrast is attributable to dropping vs retaining out-of-FOV agents.

Evidence boundary: real-benchmark-runner evidence on a bounded crossing family. The FOV uncertainty
source is a configurable rule, not a calibrated perception model; no paper-grade claim until the
predeclared matrix is run at the seed-sufficiency budget and reviewed via a claim card.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.training.scenario_loader import load_scenarios

SCHEMA_VERSION = "belief-mode-safety-campaign.v1"
ISSUE = 3556
SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

#: Belief modes -> planner uncertainty gate (oracle/retained keep agents; dropped drops them).
MODES = ("oracle", "uncertain_retained", "uncertain_dropped")
DEFAULT_SCENARIO_SET = "configs/scenarios/sets/issue_3635_near_safe_occlusion_bearing_crossing.yaml"
DEFAULT_SEEDS = list(range(101, 113))
DEFAULT_FOV = 120.0
NEAR_MISS_NOTE = (
    "real benchmark near_misses + collisions + min_clearance (no scripted unsafe-commit)"
)
NEAR_SAFE_ORACLE_COLLISION_RATE = 0.25


def load_campaign_scenarios(set_path: Path, seeds: list[int]) -> list[dict[str, Any]]:
    """Load the scenario family with absolute map paths and the predeclared seed matrix applied."""
    scenarios = load_scenarios(set_path, base_dir=set_path)
    prepared: list[dict[str, Any]] = []
    for scenario in scenarios:
        entry = dict(scenario)
        entry["seeds"] = list(seeds)
        map_file = entry.get("map_file")
        if isinstance(map_file, str) and map_file.strip() and not Path(map_file).is_absolute():
            entry["map_file"] = str((set_path.parent / map_file).resolve())
        prepared.append(entry)
    return prepared


def write_algo_config(mode: str, out_dir: Path, *, fov_degrees: float) -> Path:
    """Write the stream_gap algo config for a belief mode; return its path."""
    config = {
        "algo": "stream_gap",
        "allow_testing_algorithms": True,
        "belief_mode": mode,
        "belief_fov_degrees": fov_degrees,
    }
    path = out_dir / f"algo_{mode}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run_mode(
    mode: str,
    scenarios: list[dict[str, Any]],
    out_dir: Path,
    *,
    fov_degrees: float,
    horizon: int,
    dt: float,
    workers: int,
) -> list[dict[str, Any]]:
    """Run all scenarios x seeds for one belief mode through the real runner; return episode records."""
    algo_path = write_algo_config(mode, out_dir, fov_degrees=fov_degrees)
    episodes_path = out_dir / f"episodes_{mode}.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    run_map_batch(
        scenarios,
        episodes_path,
        schema_path=Path(SCHEMA_PATH),
        algo="stream_gap",
        algo_config_path=str(algo_path),
        horizon=horizon,
        dt=dt,
        record_forces=False,
        workers=workers,
        resume=False,
        benchmark_profile="experimental",
    )
    return [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]


def _metric(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a metric from an episode record's metrics block."""
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode safety metrics for one belief mode."""
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    collisions = [_metric(r, "total_collision_count") for r in records]
    near_misses = [_metric(r, "near_misses") for r in records]
    min_clear = [_metric(r, "min_clearance", default=float("nan")) for r in records]
    success = [_metric(r, "success") for r in records]
    valid_clear = [c for c in min_clear if not np.isnan(c)]
    return {
        "episodes": n,
        "collision_rate": round(sum(1 for c in collisions if c > 0) / n, 4),
        "total_collisions": int(sum(collisions)),
        "total_near_misses": int(sum(near_misses)),
        "mean_min_clearance": round(float(np.mean(valid_clear)), 4) if valid_clear else None,
        "worst_min_clearance": round(float(np.min(valid_clear)), 4) if valid_clear else None,
        "success_rate": round(sum(success) / n, 4),
    }


def classify_decision(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify the dropped-vs-retained safety contrast for a maintainer decision."""
    oracle = by_mode.get("oracle", {})
    retained = by_mode.get("uncertain_retained", {})
    dropped = by_mode.get("uncertain_dropped", {})
    if not (oracle.get("episodes") and retained.get("episodes") and dropped.get("episodes")):
        return {"decision": "blocked", "reason": "one or more modes produced no episodes"}

    coll_delta = dropped["collision_rate"] - retained["collision_rate"]
    nm_delta = dropped["total_near_misses"] - retained["total_near_misses"]
    worse = coll_delta > 0 or nm_delta > 0
    # "Near-safe oracle" makes the effect attributable to dropping, not scenario difficulty.
    oracle_unsafe = oracle["collision_rate"] > 0.25

    if worse and not oracle_unsafe:
        decision, reason = (
            "revise",
            (
                f"dropping uncertain agents raised collisions ({retained['collision_rate']}->"
                f"{dropped['collision_rate']}) and/or near-misses (+{nm_delta}) vs retention, with a "
                f"near-safe oracle ({oracle['collision_rate']}); revise/block the dropping default"
            ),
        )
    elif worse and oracle_unsafe:
        decision, reason = (
            "inconclusive_oracle_unsafe",
            (
                f"dropping looks worse but the oracle baseline is itself unsafe "
                f"(collision_rate {oracle['collision_rate']}); effect not cleanly attributable"
            ),
        )
    elif coll_delta == 0 and nm_delta == 0:
        decision, reason = "inconclusive", "no measurable safety difference at this matrix"
    else:
        decision, reason = "retention_dominates", "dropping did not increase unsafe outcomes here"
    return {
        "decision": decision,
        "reason": reason,
        "collision_rate_delta_dropped_minus_retained": round(coll_delta, 4),
        "near_miss_delta_dropped_minus_retained": nm_delta,
    }


def classify_screened_decision(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify contrast while exposing #3556 scenario-screening gates.

    The campaign must not interpret a dropped-vs-retained contrast unless the
    oracle baseline is near-safe and the dropped mode differs from retention.
    """
    oracle = by_mode.get("oracle", {})
    retained = by_mode.get("uncertain_retained", {})
    dropped = by_mode.get("uncertain_dropped", {})
    if not (oracle.get("episodes") and retained.get("episodes") and dropped.get("episodes")):
        return {
            "decision": "blocked",
            "reason": "one or more modes produced no episodes",
            "screening_status": "missing_episodes",
            "oracle_near_safe": False,
            "mode_is_discriminating": False,
        }

    coll_delta = dropped["collision_rate"] - retained["collision_rate"]
    nm_delta = dropped["total_near_misses"] - retained["total_near_misses"]
    worse = coll_delta > 0 or nm_delta > 0
    oracle_collision_rate = float(oracle["collision_rate"])
    oracle_unsafe = oracle_collision_rate > NEAR_SAFE_ORACLE_COLLISION_RATE

    if oracle_unsafe:
        decision = "inconclusive_oracle_unsafe"
        reason = (
            "oracle baseline is itself unsafe "
            f"(collision_rate {oracle_collision_rate}); effect not cleanly attributable"
        )
        screening_status = "oracle_unsafe"
    elif worse:
        decision = "revise"
        reason = (
            f"dropping uncertain agents raised collisions ({retained['collision_rate']}->"
            f"{dropped['collision_rate']}) and/or near-misses (+{nm_delta}) vs retention, "
            f"with a near-safe oracle ({oracle_collision_rate}); revise/block dropping default"
        )
        screening_status = "near_safe_discriminating"
    elif coll_delta == 0 and nm_delta == 0:
        decision = "inconclusive"
        reason = "no measurable safety difference at this matrix"
        screening_status = "near_safe_nondiscriminating"
    else:
        decision = "retention_dominates"
        reason = "dropping did not increase unsafe outcomes here"
        screening_status = "near_safe_nonworsening"

    return {
        "decision": decision,
        "reason": reason,
        "screening_status": screening_status,
        "oracle_collision_rate": round(oracle_collision_rate, 4),
        "oracle_near_safe": not oracle_unsafe,
        "oracle_near_safe_threshold": NEAR_SAFE_ORACLE_COLLISION_RATE,
        "mode_is_discriminating": worse,
        "collision_rate_delta_dropped_minus_retained": round(coll_delta, 4),
        "near_miss_delta_dropped_minus_retained": nm_delta,
    }


def run_campaign(
    set_path: Path,
    seeds: list[int],
    out_dir: Path,
    *,
    fov_degrees: float,
    horizon: int,
    dt: float,
    workers: int,
) -> dict[str, Any]:
    """Run all three belief modes through the real runner and assemble the classified report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = load_campaign_scenarios(set_path, seeds)
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        records = run_mode(
            mode,
            scenarios,
            out_dir,
            fov_degrees=fov_degrees,
            horizon=horizon,
            dt=dt,
            workers=workers,
        )
        by_mode[mode] = aggregate(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "nominal_benchmark",
        "claim_boundary": (
            "Real benchmark-runner evidence on a bounded crossing family; FOV uncertainty source is "
            "not a calibrated perception model; not paper-grade until the predeclared matrix is run "
            "at seed-sufficiency budget with claim-card review."
        ),
        "runner": "robot_sf.benchmark.map_runner.run_map_batch",
        "scenario_set": str(set_path),
        "scenario_names": [s.get("name") for s in scenarios],
        "seeds": seeds,
        "fov_degrees": fov_degrees,
        "metric_note": NEAR_MISS_NOTE,
        "by_mode": by_mode,
        "decision": classify_screened_decision(by_mode),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-set", type=Path, default=Path(DEFAULT_SCENARIO_SET))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output/issue_3556_belief_mode_campaign")
    )
    parser.add_argument("--fov-degrees", type=float, default=DEFAULT_FOV)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the real-runner belief-mode safety campaign and emit the report."""
    args = parse_args(argv)
    report = run_campaign(
        args.scenario_set,
        args.seeds,
        args.out_dir,
        fov_degrees=args.fov_degrees,
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
