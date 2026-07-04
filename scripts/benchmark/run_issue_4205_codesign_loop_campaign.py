#!/usr/bin/env python3
"""Run the issue #4205 static-constriction co-design-loop campaign.

This runner consumes the public benchmark contract and a private hydration
manifest. It does not submit compute jobs; callers decide where to launch it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.runner import load_scenario_matrix
from scripts.validation.check_issue_4205_static_constriction_codesign_loop import (
    EXPECTED_ARM_KEYS,
    EXPECTED_SCENARIOS,
    EXPECTED_SEEDS,
    HYDRATION_MANIFEST_SCHEMA,
    REPO_ROOT,
    ContractError,
    _load_json,
    _load_yaml,
    _repo_relative,
    _sha256_path,
    _validate_checkpoint_hydration_manifest,
    validate_config,
)

DEFAULT_BENCHMARK_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_4205_static_constriction_codesign_loop_v1.yaml"
)
DEFAULT_RESEARCH_CONFIG = (
    REPO_ROOT / "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/benchmarks/issue_4205_codesign_loop_campaign"
DEFAULT_EPISODE_SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
CAMPAIGN_SCHEMA = "robot_sf.issue_4205.static_constriction_codesign_campaign.v1"
DEFAULT_BENCHMARK_PROFILE = "experimental"
PER_EPISODE_FIELDS = [
    "arm_key",
    "scenario_id",
    "seed",
    "episode_id",
    "row_status",
    "success",
    "collision",
    "near_miss",
    "snqi",
    "deadlock_count",
    "low_progress_window",
    "recenter_activation_count",
    "distance_to_goal_delta",
    "local_minimum_indicator",
    "wrapper_intervention_rate",
    "cbf_status_counts",
    "episode_jsonl_path",
]
PER_ARM_FIELDS = [
    "arm_key",
    "episode_count",
    "success_count",
    "success_rate",
    "collision_count",
    "collision_rate",
    "near_miss_count",
    "near_miss_rate",
    "mean_snqi",
    "deadlock_count",
    "static_deadlock_trace_rows",
    "row_status_counts",
]
FAILURE_MODE_FIELDS = ["arm_key", "failure_mode", "count"]


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text with stable final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    """Serialize rows as CSV with stable line endings."""
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return handle.getvalue()


def _sha256_entries(paths: list[Path]) -> list[str]:
    """Return SHA256SUMS entries using repository-relative paths when possible."""
    entries: list[str] = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest} {_repo_relative(path)}")
    return entries


def _validate_fail_closed_policy(benchmark_config: dict[str, Any]) -> None:
    """Validate benchmark rows cannot promote fallback/degraded evidence."""
    authorization = benchmark_config.get("campaign_authorization")
    if not isinstance(authorization, dict):
        raise ContractError("benchmark campaign_authorization must be a mapping")
    if authorization.get("compute_submit_authorized") is not False:
        raise ContractError("benchmark compute_submit_authorized must stay false")
    row_status_policy = benchmark_config.get("row_status_policy")
    if not isinstance(row_status_policy, dict):
        raise ContractError("benchmark row_status_policy must be a mapping")
    for key in ("fallback_rows_are_success_evidence", "degraded_rows_are_success_evidence"):
        if row_status_policy.get(key) is not False:
            raise ContractError(f"benchmark {key} must stay false")


def _validate_benchmark_contract(
    benchmark_config: dict[str, Any],
    benchmark_config_path: Path,
    research_report: dict[str, Any],
) -> dict[str, Any]:
    """Validate benchmark-side campaign identity against the pre-registration."""
    expected_scalars = {
        "schema_version": "robot_sf.issue_4205_static_constriction_benchmark.v1",
        "issue": 4205,
        "loop_id": research_report["loop_id"],
    }
    for key, expected in expected_scalars.items():
        if benchmark_config.get(key) != expected:
            raise ContractError(f"benchmark {key} drifted from pre-registration")
    expected_lists = {
        "scenario_ids": EXPECTED_SCENARIOS,
        "seeds": EXPECTED_SEEDS,
        "arms": EXPECTED_ARM_KEYS,
    }
    for key, expected in expected_lists.items():
        values = benchmark_config.get(key) or ()
        if key == "seeds":
            values = tuple(int(seed) for seed in values)
        if tuple(values) != expected:
            raise ContractError(f"benchmark {key} drifted from pre-registration")
    _validate_fail_closed_policy(benchmark_config)
    return {
        "path": _repo_relative(benchmark_config_path),
        "sha256": _sha256_path(benchmark_config_path),
        "scenario_ids": list(EXPECTED_SCENARIOS),
        "seeds": list(EXPECTED_SEEDS),
        "arms": list(EXPECTED_ARM_KEYS),
    }


def _arm_runtime_by_key(research_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-arm wrapper/CBF runtime metadata from the research contract."""
    arms = research_config.get("arms")
    if not isinstance(arms, list):
        raise ContractError("research arms must be a list")
    by_key: dict[str, dict[str, Any]] = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ContractError("research arm must be a mapping")
        key = str(arm.get("key"))
        by_key[key] = {
            "algo": str(arm["algo"]),
            "algo_config": str(arm["algo_config"]),
            "safety_wrapper": dict(arm.get("safety_wrapper") or {}),
            "cbf_safety_filter": dict(arm.get("cbf_safety_filter") or {}),
        }
    if tuple(by_key) != EXPECTED_ARM_KEYS:
        raise ContractError("research arms drifted from pre-registration")
    return by_key


def _scenario_key(scenario: dict[str, Any]) -> str:
    """Return the stable scenario identifier used by runner records."""
    return str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "")


def _campaign_scenarios(
    matrix_path: Path, scenario_ids: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    """Load and filter the pre-registered scenario matrix."""
    scenarios = [dict(scenario) for scenario in load_scenario_matrix(matrix_path)]
    selected = [scenario for scenario in scenarios if _scenario_key(scenario) in set(scenario_ids)]
    selected_ids = [_scenario_key(scenario) for scenario in selected]
    if selected_ids != scenario_ids:
        raise ContractError(f"scenario matrix selected {selected_ids}, expected {scenario_ids}")
    for scenario in selected:
        scenario["seeds"] = list(seeds)
    return selected


def _nested(record: dict[str, Any], *keys: str) -> Any:
    """Read a nested value from the first matching key path."""
    current: Any = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(record: dict[str, Any], *paths: tuple[str, ...] | str) -> Any:
    """Return the first present scalar from top-level or nested paths."""
    for path in paths:
        if isinstance(path, str):
            if path in record:
                return record[path]
            continue
        value = _nested(record, *path)
        if value is not None:
            return value
    return None


def _boolish(value: Any) -> bool:
    """Interpret common benchmark boolean/count values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "success", "collision", "near_miss"}
    return False


def _float_or_none(value: Any) -> float | None:
    """Return a finite float-like value when possible."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records written by the map runner."""
    if not path.exists():
        raise ContractError(f"runner did not write expected episode JSONL: {_repo_relative(path)}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ContractError(f"{path}:{line_number} JSONL row must be an object")
        rows.append(payload)
    return rows


def _episode_summary_rows(arm_key: str, episode_jsonl_path: Path) -> list[dict[str, Any]]:
    """Convert raw episode records to stable campaign per-episode rows."""
    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(episode_jsonl_path):
        scenario = record.get("scenario") if isinstance(record.get("scenario"), dict) else {}
        scenario_id = _first(
            record,
            "scenario_id",
            ("scenario", "name"),
            ("scenario", "scenario_id"),
            ("scenario", "id"),
        ) or _scenario_key(scenario)
        collision = _boolish(
            _first(record, "collision", ("outcome", "collision"), ("metrics", "collision"))
        )
        near_miss = _boolish(
            _first(
                record,
                "near_miss",
                "near_misses",
                ("metrics", "near_miss"),
                ("metrics", "near_misses"),
                ("metrics", "near_miss_count"),
            )
        )
        success = _boolish(
            _first(record, "success", ("outcome", "success"), ("metrics", "success"))
        )
        snqi = _float_or_none(_first(record, "snqi", ("metrics", "snqi")))
        deadlock_count = _first(record, "deadlock_count", ("metrics", "deadlock_count"))
        row_status = str(
            _first(
                record, "row_status", ("availability", "row_status"), ("provenance", "row_status")
            )
            or "completed"
        )
        rows.append(
            {
                "arm_key": arm_key,
                "scenario_id": scenario_id,
                "seed": _first(record, "seed", ("scenario", "seed")),
                "episode_id": _first(record, "episode_id", "id"),
                "row_status": row_status,
                "success": success,
                "collision": collision,
                "near_miss": near_miss,
                "snqi": "" if snqi is None else snqi,
                "deadlock_count": int(deadlock_count or 0),
                "low_progress_window": _first(
                    record, "low_progress_window", ("metrics", "low_progress_window")
                ),
                "recenter_activation_count": _first(
                    record,
                    "recenter_activation_count",
                    ("metrics", "recenter_activation_count"),
                ),
                "distance_to_goal_delta": _first(
                    record,
                    "distance_to_goal_delta",
                    ("metrics", "distance_to_goal_delta"),
                ),
                "local_minimum_indicator": _first(
                    record,
                    "local_minimum_indicator",
                    ("metrics", "local_minimum_indicator"),
                ),
                "wrapper_intervention_rate": _first(
                    record,
                    "wrapper_intervention_rate",
                    ("metrics", "wrapper_intervention_rate"),
                ),
                "cbf_status_counts": json.dumps(
                    _first(record, "cbf_status_counts", ("metrics", "cbf_status_counts")) or {},
                    sort_keys=True,
                ),
                "episode_jsonl_path": _repo_relative(episode_jsonl_path),
            }
        )
    return rows


def _rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate for compact CSV output."""
    return round(numerator / denominator, 6) if denominator else 0.0


def _aggregate_per_arm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-episode rows into the required per-arm metric table."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm_key"])].append(row)
    output: list[dict[str, Any]] = []
    for arm_key in EXPECTED_ARM_KEYS:
        arm_rows = grouped.get(arm_key, [])
        episode_count = len(arm_rows)
        success_count = sum(1 for row in arm_rows if _boolish(row["success"]))
        collision_count = sum(1 for row in arm_rows if _boolish(row["collision"]))
        near_miss_count = sum(1 for row in arm_rows if _boolish(row["near_miss"]))
        snqi_values = [_float_or_none(row["snqi"]) for row in arm_rows]
        snqi_values = [value for value in snqi_values if value is not None]
        deadlock_count = sum(int(row.get("deadlock_count") or 0) for row in arm_rows)
        trace_rows = sum(
            1
            for row in arm_rows
            if row.get("low_progress_window") not in {"", None}
            or row.get("local_minimum_indicator") not in {"", None}
        )
        row_status_counts = Counter(str(row.get("row_status") or "completed") for row in arm_rows)
        output.append(
            {
                "arm_key": arm_key,
                "episode_count": episode_count,
                "success_count": success_count,
                "success_rate": _rate(success_count, episode_count),
                "collision_count": collision_count,
                "collision_rate": _rate(collision_count, episode_count),
                "near_miss_count": near_miss_count,
                "near_miss_rate": _rate(near_miss_count, episode_count),
                "mean_snqi": (round(sum(snqi_values) / len(snqi_values), 6) if snqi_values else ""),
                "deadlock_count": deadlock_count,
                "static_deadlock_trace_rows": trace_rows,
                "row_status_counts": json.dumps(
                    dict(sorted(row_status_counts.items())), sort_keys=True
                ),
            }
        )
    return output


def _failure_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count compact failure modes per arm."""
    counters: dict[str, Counter[str]] = {arm_key: Counter() for arm_key in EXPECTED_ARM_KEYS}
    for row in rows:
        arm_key = str(row["arm_key"])
        modes: list[str] = []
        row_status = str(row.get("row_status") or "")
        if row_status not in {"", "completed", "success"}:
            modes.append(f"row_status:{row_status}")
        if _boolish(row.get("collision")):
            modes.append("collision")
        if _boolish(row.get("near_miss")):
            modes.append("near_miss")
        if int(row.get("deadlock_count") or 0) > 0:
            modes.append("deadlock")
        if _boolish(row.get("local_minimum_indicator")):
            modes.append("local_minimum")
        if not modes:
            modes.append("none")
        counters[arm_key].update(modes)
    output: list[dict[str, Any]] = []
    for arm_key in EXPECTED_ARM_KEYS:
        for failure_mode, count in sorted(counters[arm_key].items()):
            output.append({"arm_key": arm_key, "failure_mode": failure_mode, "count": count})
    return output


def _write_outputs(
    *,
    output_root: Path,
    metadata: dict[str, Any],
    per_episode_rows: list[dict[str, Any]],
    per_arm_rows: list[dict[str, Any]],
    failure_mode_rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Write campaign output tables and checksums."""
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "metadata": output_root / "run_metadata.json",
        "per_episode_rows": output_root / "per_episode_rows.csv",
        "per_arm_metric_table": output_root / "per_arm_metric_table.csv",
        "failure_mode_counts": output_root / "failure_mode_counts.csv",
    }
    _write_text(output_paths["metadata"], json.dumps(metadata, indent=2, sort_keys=True))
    _write_text(output_paths["per_episode_rows"], _csv_text(PER_EPISODE_FIELDS, per_episode_rows))
    _write_text(output_paths["per_arm_metric_table"], _csv_text(PER_ARM_FIELDS, per_arm_rows))
    _write_text(
        output_paths["failure_mode_counts"], _csv_text(FAILURE_MODE_FIELDS, failure_mode_rows)
    )
    sha_path = output_root / "SHA256SUMS"
    _write_text(sha_path, "\n".join(_sha256_entries(list(output_paths.values()))))
    output_paths["sha256sums"] = sha_path
    return {key: _repo_relative(path) for key, path in output_paths.items()}


def _run_arm(  # noqa: PLR0913
    *,
    arm_key: str,
    arm_runtime: dict[str, Any],
    scenarios: list[dict[str, Any]],
    output_root: Path,
    benchmark_profile: str,
    workers: int,
    horizon: int | None,
    dt: float | None,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one campaign arm and summarize its episode rows."""
    episode_jsonl_path = output_root / "episodes" / f"{arm_key}.jsonl"
    summary = run_map_batch(
        scenarios,
        episode_jsonl_path,
        DEFAULT_EPISODE_SCHEMA,
        horizon=horizon,
        dt=dt,
        record_forces=True,
        algo=arm_runtime["algo"],
        algo_config_path=arm_runtime["algo_config"],
        benchmark_profile=benchmark_profile,
        safety_wrapper=arm_runtime["safety_wrapper"],
        cbf_safety_filter=arm_runtime["cbf_safety_filter"],
        record_simulation_step_trace=True,
        workers=workers,
        resume=resume,
    )
    failures = summary.get("failures") if isinstance(summary, dict) else None
    if failures:
        raise ContractError(f"{arm_key}: map runner failed closed with failures: {failures}")
    return summary, _episode_summary_rows(arm_key, episode_jsonl_path)


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """Validate contracts, execute arm runs, and write campaign outputs."""
    benchmark_config_path = Path(args.config)
    benchmark_config = _load_yaml(benchmark_config_path)
    research_config_path = Path(benchmark_config["research_contract"])
    if not research_config_path.is_absolute():
        research_config_path = REPO_ROOT / research_config_path
    research_config = _load_yaml(research_config_path)
    research_report = validate_config(research_config)
    benchmark_identity = _validate_benchmark_contract(
        benchmark_config,
        benchmark_config_path,
        research_report,
    )
    manifest_path = Path(args.hydration_manifest)
    hydration = _validate_checkpoint_hydration_manifest(
        research_report,
        _load_json(manifest_path),
        manifest_path=manifest_path,
    )
    if hydration.get("schema_version") != HYDRATION_MANIFEST_SCHEMA:
        raise ContractError("hydration manifest schema drifted")
    scenario_ids = list(EXPECTED_SCENARIOS)
    seeds = list(EXPECTED_SEEDS)
    if args.smoke:
        scenario_ids = scenario_ids[:1]
        seeds = seeds[:1]
    matrix_path = Path(benchmark_config["scenario_matrix"])
    if not matrix_path.is_absolute():
        matrix_path = REPO_ROOT / matrix_path
    scenarios = _campaign_scenarios(matrix_path, scenario_ids, seeds)
    arm_runtime = _arm_runtime_by_key(research_config)
    output_root = Path(args.output_root)
    per_episode_rows: list[dict[str, Any]] = []
    arm_summaries: dict[str, Any] = {}
    for arm_key in EXPECTED_ARM_KEYS:
        summary, arm_rows = _run_arm(
            arm_key=arm_key,
            arm_runtime=arm_runtime[arm_key],
            scenarios=scenarios,
            output_root=output_root,
            benchmark_profile=args.benchmark_profile,
            workers=args.workers,
            horizon=args.horizon,
            dt=args.dt,
            resume=not args.no_resume,
        )
        arm_summaries[arm_key] = summary
        per_episode_rows.extend(arm_rows)
    expected_episode_count = len(EXPECTED_ARM_KEYS) * len(scenario_ids) * len(seeds)
    full_grid_completed = not args.smoke and len(per_episode_rows) == len(EXPECTED_ARM_KEYS) * len(
        EXPECTED_SCENARIOS
    ) * len(EXPECTED_SEEDS)
    if len(per_episode_rows) != expected_episode_count:
        raise ContractError(
            f"campaign wrote {len(per_episode_rows)} episode rows, expected {expected_episode_count}"
        )
    per_arm_rows = _aggregate_per_arm(per_episode_rows)
    failure_mode_rows = _failure_modes(per_episode_rows)
    metadata = {
        "schema_version": CAMPAIGN_SCHEMA,
        "issue": 4367,
        "source_issue": 4205,
        "loop_id": research_report["loop_id"],
        "mode": "smoke" if args.smoke else "full",
        "benchmark_profile": args.benchmark_profile,
        "benchmark_evidence": bool(full_grid_completed),
        "claim_boundary": (
            "Campaign execution table only; no Slurm/GPU submission, retraining, interpretation, "
            "or paper/dissertation claim edit is made by this runner."
        ),
        "benchmark_contract": benchmark_identity,
        "research_contract": {
            "path": _repo_relative(research_config_path),
            "sha256": _sha256_path(research_config_path),
        },
        "hydration_manifest": {
            "path": _repo_relative(manifest_path),
            "sha256": _sha256_path(manifest_path),
            "checkpoint_sha256": hydration["checkpoint_sha256"],
            "schema_version": hydration["schema_version"],
        },
        "scenario_ids": scenario_ids,
        "seeds": seeds,
        "arms": list(EXPECTED_ARM_KEYS),
        "expected_episode_count": expected_episode_count,
        "observed_episode_count": len(per_episode_rows),
        "arm_summaries": arm_summaries,
    }
    outputs = _write_outputs(
        output_root=output_root,
        metadata=metadata,
        per_episode_rows=per_episode_rows,
        per_arm_rows=per_arm_rows,
        failure_mode_rows=failure_mode_rows,
    )
    return {"metadata": metadata, "outputs": outputs}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_BENCHMARK_CONFIG), help="Benchmark contract."
    )
    parser.add_argument(
        "--hydration-manifest",
        required=True,
        help="Private frozen-PPO hydration manifest; required fail-closed input.",
    )
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Campaign output root."
    )
    parser.add_argument("--smoke", action="store_true", help="Run one seed x one scenario per arm.")
    parser.add_argument("--workers", type=int, default=1, help="Map-runner worker count.")
    parser.add_argument(
        "--benchmark-profile",
        default=DEFAULT_BENCHMARK_PROFILE,
        help=(
            "Benchmark readiness profile for the episode runner. Defaults to experimental "
            "because #4205 pre-registered PPO arms are exploratory diagnostic evidence."
        ),
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="Optional episode horizon override."
    )
    parser.add_argument("--dt", type=float, default=None, help="Optional episode dt override.")
    parser.add_argument("--no-resume", action="store_true", help="Disable map-runner resume.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        result = run_campaign(args)
    except ContractError as exc:
        print(f"issue #4367 campaign runner failed closed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        metadata = result["metadata"]
        print(
            "issue #4367 campaign runner complete: "
            f"mode={metadata['mode']} episodes={metadata['observed_episode_count']} "
            f"benchmark_evidence={metadata['benchmark_evidence']} "
            f"output_root={_repo_relative(Path(args.output_root))}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
