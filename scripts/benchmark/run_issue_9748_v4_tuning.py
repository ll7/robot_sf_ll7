#!/usr/bin/env python3
"""Run the predeclared hybrid v4 search on the #9748 development split only.

This produces diagnostic rows and a structured log for the #9908 validator.
The log must be committed after this source commit before validation. No
release-seed execution, automatic parameter freeze, or release claim occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import yaml

from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

pin_thread_env_for_determinism()

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_PATH = ROOT / "configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml"
SEARCH_PATH = ROOT / "configs/policy_search/issue_9748_v4_tuning_search_v1.yaml"
SCHEMA_PATH = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
# Camera-ready's map runner anchors repository-relative map_file paths here.
SCENARIO_ANCHOR = ROOT / "scoped_scenarios.json"
LOG_SCHEMA = "issue_9748.tuning_log.v1"
SEARCH_SCHEMA = "issue_9748.v4_search.v1"
TUNABLE = frozenset(
    {
        "v4_slow_clearance_human",
        "v4_moderate_clearance_human",
        "v4_braking_margin",
        "dynamic_clearance_weight",
        "goal_progress_weight",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_mapping(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_source_commit() -> str:
    changed = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if changed:
        raise ValueError("tuning requires a clean tracked source checkout and no untracked files")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_search() -> list[dict[str, Any]]:
    payload = yaml.safe_load(SEARCH_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SEARCH_SCHEMA:
        raise ValueError("invalid #9748 search schema")
    if payload.get("development_only") is not True:
        raise ValueError("#9748 search must be development-only")
    trials = payload.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("#9748 search must contain trials")
    if trials[0] != {"id": "baseline", "params": {}}:
        raise ValueError("#9748 search must start with the unmodified baseline")
    seen: set[str] = set()
    for trial in trials:
        trial_id = _validate_trial(trial)
        if trial_id in seen:
            raise ValueError(f"duplicate trial id: {trial_id}")
        seen.add(trial_id)
    return trials


def _validate_trial(trial: Any) -> str:
    if not isinstance(trial, dict) or set(trial) != {"id", "params"}:
        raise ValueError("each search trial needs exactly id and params")
    trial_id, params = trial["id"], trial["params"]
    if not isinstance(trial_id, str) or not trial_id.replace("_", "").isalnum():
        raise ValueError("trial id must be a simple nonempty identifier")
    if not isinstance(params, dict) or set(params) - TUNABLE:
        raise ValueError(f"trial {trial_id} contains an unapproved parameter")
    if trial_id != "baseline" and len(params) != 1:
        raise ValueError("search perturbations must change exactly one parameter")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
        for value in params.values()
    ):
        raise ValueError(f"trial {trial_id} has a nonfinite or nonnumeric parameter")
    return trial_id


def _choose(requested: list[Any], available: list[Any], *, label: str) -> list[Any]:
    if not requested:
        return available
    if len(requested) != len(set(requested)) or set(requested) - set(available):
        raise ValueError(f"{label} must be unique members of the frozen development set")
    return [item for item in available if item in requested]


def _candidate_manifest(path: Path, params: dict[str, float]) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("params"), dict):
        raise ValueError(f"invalid v4 candidate manifest: {path}")
    result = deepcopy(manifest)
    result["params"].update(params)
    return result


def _completion_time_s(record: dict[str, Any], *, dt: float) -> float:
    """Recover simulated goal time from the canonical success-only metric."""
    normalized = record.get("metrics", {}).get("time_to_goal_norm_success_only")
    horizon = record.get("horizon")
    if (
        not isinstance(normalized, (int, float))
        or not math.isfinite(normalized)
        or not 0.0 <= normalized < 1.0
        or not isinstance(horizon, int)
        or horizon <= 0
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("invalid success-only normalized goal time, horizon, or dt")
    return float(normalized) * horizon * dt


def _quiet_logs() -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")


def _effective_config_hash(
    *, manifest: dict[str, Any], candidate_path: Path, scenario: dict[str, Any]
) -> str:
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
        _resolve_policy_search_candidate_runtime,
    )

    algo, resolved = _resolve_policy_search_candidate_runtime(
        default_algo="hybrid_rule_local_planner",
        algo_config_path=str(candidate_path),
        scenario=scenario,
        algo_config=manifest,
    )
    if algo != "hybrid_rule_local_planner" or resolved.get("planner_variant") != (
        "hybrid_rule_v4_clearance_braking"
    ):
        raise ValueError("search trial did not resolve to the v4 hybrid planner")
    return _hash_mapping(resolved)


def _run_cell(
    job: tuple[dict[str, Any], int, dict[str, Any], str, str, int, float],
) -> dict[str, Any]:
    scenario, seed, manifest, candidate_path, scenario_anchor, horizon, dt = job
    started = time.perf_counter()
    try:
        _quiet_logs()
        from robot_sf.benchmark.map_runner.map_runner import _run_map_episode, build_map_policy

        # Use the same episode wrapper as the release batch. It captures the
        # parser-consumed map input and binds selected_map_identity in each row.
        record = _run_map_episode(
            scenario=scenario,
            seed=seed,
            horizon=horizon,
            dt=dt,
            record_forces=True,
            snqi_weights=None,
            snqi_baseline=None,
            algo="hybrid_rule_local_planner",
            scenario_path=Path(scenario_anchor),
            algo_config=manifest,
            algo_config_path=candidate_path,
            policy_builder=build_map_policy,
        )
    except (
        AssertionError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
        }
    return {"seed": seed, "record": record, "wall_seconds": time.perf_counter() - started}


def _write_group(
    *,
    output_root: Path,
    candidate: str,
    trial: dict[str, Any],
    scenario: dict[str, Any],
    seeds: list[int],
    jobs: list[tuple[dict[str, Any], int, dict[str, Any], str, str, int, float]],
    executor: ProcessPoolExecutor | None,
    schema: dict[str, Any],
) -> dict[str, Any]:
    from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
    from robot_sf.benchmark.map_runner.map_runner_jsonl import write_validated_to_handle

    scenario_id = scenario["name"]
    dt = jobs[0][-1]
    group_dir = output_root / candidate / trial["id"] / scenario_id
    group_dir.mkdir(parents=True)
    rows_path = group_dir / "episodes.jsonl"
    errors_path = group_dir / "errors.jsonl"
    counts = {"written": 0, "errors": 0, "degraded": 0, "collisions": 0, "completions": 0}
    completed_times: list[float] = []
    results = map(_run_cell, jobs) if executor is None else executor.map(_run_cell, jobs)
    started = time.perf_counter()
    with (
        rows_path.open("w", encoding="utf-8") as rows,
        errors_path.open("w", encoding="utf-8") as errors,
    ):
        for result in results:
            record = result.get("record")
            if record is None:
                counts["errors"] += 1
                errors.write(json.dumps(result, sort_keys=True) + "\n")
                continue
            try:
                write_validated_to_handle(rows, schema, record)
                counts["written"] += 1
                marker = runtime_fallback_or_degraded_marker(
                    record.get("algorithm_metadata"),
                    expected_algorithm="hybrid_rule_local_planner",
                    algorithm_metadata=record.get("algorithm_metadata"),
                )
                if marker is not None:
                    counts["degraded"] += 1
                outcome = record["outcome"]
                counts["collisions"] += int(outcome["collision_event"])
                if outcome["route_complete"]:
                    counts["completions"] += 1
                    try:
                        completed_times.append(_completion_time_s(record, dt=dt))
                    except ValueError as exc:
                        counts["errors"] += 1
                        errors.write(json.dumps({"seed": result["seed"], "error": str(exc)}) + "\n")
            except (KeyError, TypeError, ValueError) as exc:
                counts["errors"] += 1
                errors.write(
                    json.dumps({"seed": result["seed"], "error": f"invalid record: {exc}"}) + "\n"
                )
        rows.flush()
        errors.flush()
        os.fsync(rows.fileno())
        os.fsync(errors.fileno())
    return {
        "candidate": candidate,
        "trial_id": trial["id"],
        "scenario_id": scenario_id,
        "seeds": seeds,
        "counts": counts,
        "mean_time_to_goal_s": (
            sum(completed_times) / len(completed_times) if completed_times else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "episodes_path": rows_path.relative_to(output_root).as_posix(),
        "episodes_sha256": _sha256(rows_path),
        "errors_path": errors_path.relative_to(output_root).as_posix(),
        "errors_sha256": _sha256(errors_path),
        "completed_times": completed_times,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--trial", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tuning-log", type=Path, required=True)
    return parser


def _rank_trials(groups: list[dict[str, Any]], candidates: list[str]) -> dict[str, Any]:
    """Summarize the predeclared safety-first order without freezing a release arm."""
    rankings: dict[str, Any] = {}
    for candidate in candidates:
        trials = sorted({group["trial_id"] for group in groups if group["candidate"] == candidate})
        scored: list[dict[str, Any]] = []
        for trial_id in trials:
            selected = [
                group
                for group in groups
                if group["candidate"] == candidate and group["trial_id"] == trial_id
            ]
            counts = {
                key: sum(group["counts"][key] for group in selected)
                for key in ("written", "errors", "degraded", "collisions", "completions")
            }
            times = [value for group in selected for value in group["completed_times"]]
            planned = sum(len(group["seeds"]) for group in selected)
            eligible = (
                counts["written"] == planned and counts["errors"] == 0 and counts["degraded"] == 0
            )
            scored.append(
                {
                    "trial_id": trial_id,
                    "eligible": eligible,
                    "planned": planned,
                    **counts,
                    "mean_time_to_goal_s": sum(times) / len(times) if times else None,
                }
            )
        scored.sort(
            key=lambda item: (
                not item["eligible"],
                item["collisions"],
                -item["completions"],
                item["mean_time_to_goal_s"]
                if item["mean_time_to_goal_s"] is not None
                else math.inf,
                item["trial_id"],
            )
        )
        rankings[candidate] = scored
    return rankings


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915
    """Execute the selected dev cells and write raw rows plus a commit-ready log."""
    args = _parser().parse_args(argv)
    _quiet_logs()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output_root = args.output_root.resolve()
    log_path = args.tuning_log.resolve()
    if output_root == ROOT or ROOT in output_root.parents:
        raise ValueError("raw output must be outside the source checkout")
    if ROOT not in log_path.parents or log_path.exists():
        raise ValueError("the tuning log must be a new file inside the source checkout")
    if output_root.exists():
        raise ValueError("output root already exists; refuse to overwrite earlier trials")
    source_commit = _git_source_commit()

    from robot_sf.benchmark.camera_ready._config import (
        _load_campaign_scenarios,
        load_campaign_config,
    )
    from scripts.validation.check_issue_9748_dev_split import (
        EXPECTED_DEV_SEEDS,
        EXPECTED_PLANNER_CONFIGS,
        EXPECTED_SCENARIO_IDS,
        validate,
    )

    validate(CAMPAIGN_PATH)
    cfg = load_campaign_config(CAMPAIGN_PATH)
    if cfg.horizon != 600 or cfg.dt != 0.1:
        raise ValueError("#9748 runner requires the frozen H600 / dt=0.1 contract")
    scenarios = _load_campaign_scenarios(cfg)
    if {row["name"] for row in scenarios} != EXPECTED_SCENARIO_IDS or len(scenarios) != 4:
        raise ValueError("resolved development scenario identities changed")
    if any(tuple(row["seeds"]) != EXPECTED_DEV_SEEDS for row in scenarios):
        raise ValueError("resolved development seeds changed")
    planners = {planner.key: planner for planner in cfg.planners}
    if set(planners) != set(EXPECTED_PLANNER_CONFIGS):
        raise ValueError("development candidate roster changed")
    seeds = _choose(args.seed, list(EXPECTED_DEV_SEEDS), label="seed")
    candidates = _choose(args.candidate, list(planners), label="candidate")
    search = _load_search()
    trials = _choose(args.trial, [trial["id"] for trial in search], label="trial")
    selected_trials = [trial for trial in search if trial["id"] in trials]
    smoke_only = (
        len(seeds) != len(EXPECTED_DEV_SEEDS)
        or len(candidates) != len(planners)
        or len(trials) != len(search)
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    provenance = {
        "source_commit": source_commit,
        "campaign_config_sha256": _sha256(CAMPAIGN_PATH),
        "scenario_manifest_sha256": _sha256(cfg.scenario_matrix_path),
        "candidate_configs": {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}
            for key, path in EXPECTED_PLANNER_CONFIGS.items()
        },
        "search_spec_sha256": _sha256(SEARCH_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "python_version": sys.version.split()[0],
        "host": socket.gethostname(),
        "artifact_root": str(output_root),
    }
    output_root.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    started = time.perf_counter()
    executor = (
        ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
        if args.workers > 1
        else None
    )
    try:
        for candidate in candidates:
            candidate_path = EXPECTED_PLANNER_CONFIGS[candidate]
            for trial in selected_trials:
                manifest = _candidate_manifest(candidate_path, trial["params"])
                for scenario in scenarios:
                    effective_hash = _effective_config_hash(
                        manifest=manifest, candidate_path=candidate_path, scenario=scenario
                    )
                    jobs = [
                        (
                            scenario,
                            seed,
                            manifest,
                            str(candidate_path),
                            str(SCENARIO_ANCHOR),
                            cfg.horizon,
                            cfg.dt,
                        )
                        for seed in seeds
                    ]
                    group = _write_group(
                        output_root=output_root,
                        candidate=candidate,
                        trial=trial,
                        scenario=scenario,
                        seeds=seeds,
                        jobs=jobs,
                        executor=executor,
                        schema=schema,
                    )
                    groups.append(group)
                    entries.append(
                        {
                            "candidate": candidate,
                            "candidate_config_sha256": provenance["candidate_configs"][candidate][
                                "sha256"
                            ],
                            "scenario_id": group["scenario_id"],
                            "seeds": seeds,
                            "trial_id": trial["id"],
                            "parameter_overrides": trial["params"],
                            "effective_config_sha256": effective_hash,
                            "episodes_path": group["episodes_path"],
                            "episodes_sha256": group["episodes_sha256"],
                            "errors_path": group["errors_path"],
                            "errors_sha256": group["errors_sha256"],
                            "counts": group["counts"],
                            "mean_time_to_goal_s": group["mean_time_to_goal_s"],
                            "elapsed_seconds": group["elapsed_seconds"],
                        }
                    )
                    print(
                        f"{candidate}/{trial['id']}/{group['scenario_id']}: "
                        f"{group['counts']}, {group['elapsed_seconds']:.2f}s",
                        flush=True,
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    rankings = _rank_trials(groups, candidates)
    log = {
        "schema_version": LOG_SCHEMA,
        "provenance": provenance,
        "claim_boundary": (
            "diagnostic pipeline smoke only; no tuning selection or release claim"
            if smoke_only
            else "development tuning only; no held-out or release claim"
        ),
        "entries": entries,
        "rankings": rankings,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    overall = {
        "status": "diagnostic_pipeline_smoke" if smoke_only else "development_tuning",
        "source_commit": source_commit,
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "output_root": str(output_root),
        "selected_candidates": candidates,
        "selected_trials": trials,
        "selected_seeds": seeds,
        "planned_cells": len(candidates) * len(trials) * len(scenarios) * len(seeds),
        "completed_groups": len(groups),
        "error_cells": sum(group["counts"]["errors"] for group in groups),
        "degraded_cells": sum(group["counts"]["degraded"] for group in groups),
        "wall_seconds": time.perf_counter() - started,
        "rankings": rankings,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, indent=2, sort_keys=True), flush=True)
    return 2 if overall["error_cells"] or overall["degraded_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
