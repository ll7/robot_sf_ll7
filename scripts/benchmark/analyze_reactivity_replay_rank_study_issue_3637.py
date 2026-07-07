#!/usr/bin/env python3
"""Analyze post-run reactivity-vs-replay rank-study outputs for issue #3637.

This script is the post-run checker for the paper-grade candidate plan. It does
not run the benchmark. It consumes the predeclared launch packet plus campaign
JSONL outputs, validates the expected paired planner/arm/seed/scenario matrix,
and writes compact analysis artifacts plus the frozen seed-sufficiency gate
input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Keep the script runnable directly from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.reactivity_ablation import (  # noqa: E402
    REACTIVITY_ARMS,
    REPLAY_IS_TRAJECTORY_PLAYBACK,
    REPLAY_LIMITATION,
    ReactivityContrast,
    assess_reactivity_ablation,
)
from robot_sf.benchmark.reactivity_replay_preflight import (  # noqa: E402
    build_preflight_manifest,
    run_plan_from_packet,
)
from scripts.tools.seed_sufficiency_gate import SeedGateInput, decide_seed_gate  # noqa: E402

ISSUE = 3637
SOURCE_CAMPAIGN_ISSUE = 3573
SCHEMA_VERSION = "reactivity-replay-rank-study-analysis.v1"
DEFAULT_PACKET = Path(
    "configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml"
)
DEFAULT_CAMPAIGN_DIR = Path("output/issue_3637_reactivity_rank_study")
DEFAULT_CAMPAIGN_REPORT = DEFAULT_CAMPAIGN_DIR / "report.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CAMPAIGN_DIR / "analysis"
RANK_METRIC = "collision_rate"


class AnalysisInputError(ValueError):
    """Raised when campaign artifacts do not satisfy the frozen issue #3637 contract."""


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{path}: expected YAML mapping")
    return payload


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisInputError(f"missing campaign report: {path}") from exc
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{path}: expected JSON object")
    return payload


def _scenario_ids(packet: dict[str, Any], packet_path: Path) -> tuple[str, ...]:
    scenario_set = packet.get("scenario_set")
    if not isinstance(scenario_set, str) or not scenario_set:
        raise AnalysisInputError("packet scenario_set must be a non-empty string")
    scenario_path = (packet_path.parent / scenario_set).resolve()
    if not scenario_path.exists():
        scenario_path = (Path.cwd() / scenario_set).resolve()
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"{scenario_path}: expected scenario-set mapping")
    selected = payload.get("select_scenarios")
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(x, str) for x in selected)
    ):
        raise AnalysisInputError(
            f"{scenario_path}: select_scenarios must be a non-empty string list"
        )
    return tuple(selected)


def _metric(record: dict[str, Any], key: str) -> float:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict) or key not in metrics:
        raise AnalysisInputError(f"episode missing metrics.{key}")
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnalysisInputError(f"metrics.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisInputError(f"metrics.{key} must be finite")
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise AnalysisInputError(f"missing campaign episode file: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise AnalysisInputError(f"{path}:{line_no}: expected JSON object")
            records.append(payload)
    if not records:
        raise AnalysisInputError(f"{path}: no episode records")
    return records


def _episode_identity(
    record: dict[str, Any],
    *,
    planner: str,
    condition: str,
) -> tuple[str, str, str, int]:
    scenario = record.get("scenario_id") or record.get("scenario")
    seed = record.get("seed")
    if not isinstance(scenario, str) or not scenario:
        raise AnalysisInputError("episode missing non-empty scenario_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AnalysisInputError("episode missing integer seed")
    _metric(record, "total_collision_count")
    _metric(record, "near_misses")
    _metric(record, "min_clearance")
    return (planner, condition, scenario, seed)


def _validate_matrix(
    records: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    planners: tuple[str, ...],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
) -> list[dict[str, Any]]:
    expected = {
        (planner, condition, scenario, seed)
        for planner in planners
        for condition in conditions
        for scenario in scenarios
        for seed in seeds
    }
    observed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for (planner, condition), rows in records.items():
        for row in rows:
            key = _episode_identity(row, planner=planner, condition=condition)
            if key in observed:
                raise AnalysisInputError(
                    "duplicate episode row for "
                    f"planner={key[0]} condition={key[1]} scenario={key[2]} seed={key[3]}"
                )
            observed[key] = row
    observed_keys = set(observed)
    missing = sorted(expected - observed_keys)
    extra = sorted(observed_keys - expected)
    if missing:
        planner, condition, scenario, seed = missing[0]
        raise AnalysisInputError(
            "missing episode row for "
            f"planner={planner} condition={condition} scenario={scenario} seed={seed}"
        )
    if extra:
        planner, condition, scenario, seed = extra[0]
        raise AnalysisInputError(
            "unexpected episode row for "
            f"planner={planner} condition={condition} scenario={scenario} seed={seed}"
        )
    return [observed[key] for key in sorted(expected)]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise AnalysisInputError("cannot aggregate empty episode rows")
    count = len(rows)
    collision_rate = sum(1 for row in rows if _metric(row, "total_collision_count") > 0) / count
    near_miss_rate = sum(1 for row in rows if _metric(row, "near_misses") > 0) / count
    min_separation_m = sum(_metric(row, "min_clearance") for row in rows) / count
    return {
        "collision_rate": round(collision_rate, 6),
        "near_miss_rate": round(near_miss_rate, 6),
        "min_separation_m": round(min_separation_m, 6),
        "episodes": count,
    }


def _rank_by_collision(
    per_planner: dict[str, dict[str, dict[str, float | int]]], condition: str
) -> dict[str, int]:
    ordered = sorted(
        per_planner,
        key=lambda planner: (float(per_planner[planner][condition]["collision_rate"]), planner),
    )
    return {planner: index + 1 for index, planner in enumerate(ordered)}


def _per_seed_condition_rate(
    records: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    planner: str,
    condition: str,
    sampled_seeds: np.ndarray,
) -> float:
    rows_by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records[(planner, condition)]:
        seed = row["seed"]
        if not isinstance(seed, int):
            raise AnalysisInputError("episode seed must be integer")
        rows_by_seed[seed].append(row)
    sampled_rows = [row for seed in sampled_seeds for row in rows_by_seed[int(seed)]]
    return float(
        sum(1 for row in sampled_rows if _metric(row, "total_collision_count") > 0)
        / len(sampled_rows)
    )


def _bootstrap_rank_summary(
    records: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    planners: tuple[str, ...],
    seeds: tuple[int, ...],
    full_data_ranks: dict[str, dict[str, int]],
    resamples: int,
    stability_threshold: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(3637)
    seed_array = np.array(seeds)
    full_sensitive = full_data_ranks["reactive"] != full_data_ranks["replay"]
    sensitive_matches = 0
    max_abs_deltas: list[float] = []
    unstable_rank_samples = 0
    for _ in range(resamples):
        sample = rng.choice(seed_array, size=len(seed_array), replace=True)
        sample_rates: dict[str, dict[str, float]] = {condition: {} for condition in REACTIVITY_ARMS}
        for planner in planners:
            for condition in REACTIVITY_ARMS:
                sample_rates[condition][planner] = _per_seed_condition_rate(
                    records,
                    planner=planner,
                    condition=condition,
                    sampled_seeds=sample,
                )
        sample_ranks = {
            condition: {
                planner: index + 1
                for index, planner in enumerate(
                    sorted(planners, key=lambda p: (sample_rates[condition][p], p))
                )
            }
            for condition in REACTIVITY_ARMS
        }
        sample_sensitive = sample_ranks["reactive"] != sample_ranks["replay"]
        if sample_sensitive == full_sensitive:
            sensitive_matches += 1
        if sample_ranks != full_data_ranks:
            unstable_rank_samples += 1
        max_abs_deltas.append(
            max(
                abs(sample_rates["reactive"][planner] - sample_rates["replay"][planner])
                for planner in planners
            )
        )
    low, high = np.percentile(max_abs_deltas, [2.5, 97.5])
    stability_rate = sensitive_matches / resamples
    return {
        "resamples": resamples,
        "rank_effect_stability_rate": round(stability_rate, 6),
        "max_collision_delta_ci_half_width": round(float((high - low) / 2), 6),
        "rank_instability_observed": stability_rate < stability_threshold,
        "rank_flip_observed": stability_rate < stability_threshold,
        "unstable_full_rank_sample_rate": round(unstable_rank_samples / resamples, 6),
        "seed_resampling_unit": "paired seed across all planners, arms, and scenarios",
        "rank_direction": "lower collision_rate is better; ties break by planner name",
    }


def _read_campaign_records(
    campaign_dir: Path,
    *,
    planners: tuple[str, ...],
    conditions: tuple[str, ...],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for planner in planners:
        for condition in conditions:
            path = campaign_dir / f"episodes_{planner}_{condition}.jsonl"
            records[(planner, condition)] = _read_jsonl(path)
    return records


def _write_csv(path: Path, per_planner: dict[str, dict[str, dict[str, float | int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "planner",
                "condition",
                "episodes",
                "collision_rate",
                "near_miss_rate",
                "min_separation_m",
            ],
        )
        writer.writeheader()
        for planner in sorted(per_planner):
            for condition in REACTIVITY_ARMS:
                row = dict(per_planner[planner][condition])
                writer.writerow({"planner": planner, "condition": condition, **row})


def _write_readme(path: Path, analysis: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Issue #3637 Reactivity Replay Rank Study Analysis",
                "",
                "This is a post-run checker artifact, not a paper-facing claim by itself.",
                "",
                f"- Evidence status: `{analysis['claim_decision']}`",
                f"- Episode rows checked: `{analysis['episode_count']}`",
                f"- Replay limitation: {analysis['replay_limitation']['note']}",
                f"- Claim boundary: {analysis['claim_boundary']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def analyze(
    *,
    packet_path: Path,
    campaign_dir: Path,
    campaign_report: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate campaign outputs and emit issue #3637 post-run analysis artifacts."""
    packet = _load_mapping(packet_path)
    plan = run_plan_from_packet(packet)
    preflight = build_preflight_manifest(plan)
    if preflight["status"] != "ready":
        raise AnalysisInputError("launch packet preflight status must be ready before analysis")
    campaign = _load_json_mapping(campaign_report)
    if campaign.get("replay_limitation", {}).get(
        "is_trajectory_playback", REPLAY_IS_TRAJECTORY_PLAYBACK
    ):
        raise AnalysisInputError(
            "campaign report replay limitation must state is_trajectory_playback=false"
        )
    planners = tuple(plan.planners)
    seeds = tuple(next(iter(plan.arm_seeds.values())))
    scenarios = _scenario_ids(packet, packet_path)
    records = _read_campaign_records(campaign_dir, planners=planners, conditions=REACTIVITY_ARMS)
    all_rows = _validate_matrix(
        records,
        planners=planners,
        conditions=REACTIVITY_ARMS,
        seeds=seeds,
        scenarios=scenarios,
    )
    per_planner = {
        planner: {
            condition: _aggregate(records[(planner, condition)]) for condition in REACTIVITY_ARMS
        }
        for planner in planners
    }
    contrasts = [
        ReactivityContrast(
            planner=planner,
            reactive_collision_rate=float(per_planner[planner]["reactive"]["collision_rate"]),
            replay_collision_rate=float(per_planner[planner]["replay"]["collision_rate"]),
            reactive_near_miss_rate=float(per_planner[planner]["reactive"]["near_miss_rate"]),
            replay_near_miss_rate=float(per_planner[planner]["replay"]["near_miss_rate"]),
            reactive_min_separation_m=float(per_planner[planner]["reactive"]["min_separation_m"]),
            replay_min_separation_m=float(per_planner[planner]["replay"]["min_separation_m"]),
        )
        for planner in planners
    ]
    rank_effect = assess_reactivity_ablation(contrasts)
    full_data_ranks = {
        condition: _rank_by_collision(per_planner, condition) for condition in REACTIVITY_ARMS
    }
    rank_config = dict(plan.rank_stability_analysis)
    bootstrap = _bootstrap_rank_summary(
        records,
        planners=planners,
        seeds=seeds,
        full_data_ranks=full_data_ranks,
        resamples=int(rank_config["bootstrap_resamples"]),
        stability_threshold=float(rank_config["rank_effect_stability_threshold"]),
    )
    gate_input = SeedGateInput(
        schedule=str(rank_config["schedule"]),
        ci_half_width=float(bootstrap["max_collision_delta_ci_half_width"]),
        target_ci_half_width=float(rank_config["target_ci_half_width"]),
        rank_flip_observed=bool(bootstrap["rank_flip_observed"]),
        heldout_delta_abs=None,
        heldout_delta_threshold=None,
        invalid_row_count=0,
    )
    gate_decision = decide_seed_gate(gate_input)
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "source_campaign_issue": SOURCE_CAMPAIGN_ISSUE,
        "campaign_report": str(campaign_report),
        "planners": list(planners),
        "conditions": list(REACTIVITY_ARMS),
        "seeds": list(seeds),
        "scenario_set": plan.scenario_set,
        "scenarios": list(scenarios),
        "episode_count": len(all_rows),
        "expected_episode_count": len(planners)
        * len(REACTIVITY_ARMS)
        * len(seeds)
        * len(scenarios),
        "per_planner": per_planner,
        "full_data_ranks": full_data_ranks,
        "rank_effect": rank_effect,
        "paired_seed_bootstrap": bootstrap,
        "seed_sufficiency_gate_input": asdict(gate_input),
        "seed_sufficiency_gate_decision": asdict(gate_decision),
        "replay_limitation": {
            "is_trajectory_playback": False,
            "note": packet.get("replay", {}).get("limitation", REPLAY_LIMITATION),
        },
        "claim_boundary": rank_config["claim_boundary"],
        "claim_decision": "post_run_gate_input_ready",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "frozen_gate_input.json").write_text(
            json.dumps(asdict(gate_input), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "seed_gate_decision.json").write_text(
            json.dumps(asdict(gate_decision), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "rank_bootstrap_summary.json").write_text(
            json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(output_dir / "per_planner_condition_metrics.csv", per_planner)
        _write_readme(output_dir / "README.md", analysis)
    return analysis


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN_DIR)
    parser.add_argument("--campaign-report", type=Path, default=DEFAULT_CAMPAIGN_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the issue #3637 analyzer command-line interface."""
    args = _parse_args(argv)
    try:
        analysis = analyze(
            packet_path=args.packet,
            campaign_dir=args.campaign_dir,
            campaign_report=args.campaign_report,
            output_dir=args.output_dir,
        )
    except AnalysisInputError as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "ready",
                "analysis": str(args.output_dir / "analysis.json"),
                "episode_count": analysis["episode_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
