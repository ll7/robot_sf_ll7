"""Bounded optimization of a registered planner configuration.

The CLI deliberately reuses the policy-search candidate loader and episode
evaluator. It optimizes planner configuration values only; scenario parameters
remain fixed by separate train and held-out suites.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerConfig,
    build_hybrid_rule_local_planner_config,
)
from scripts.validation import run_policy_search_candidate as policy_search
from scripts.validation.policy_search_common import (
    normalize_scenario_exclusion,
    summarize_policy_search_records,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA = "planner_optimizer_config.v1"
MANIFEST_SCHEMA = "planner_optimizer_run.v1"
OBJECTIVE_NAMES = (
    "valid_episode_fraction",
    "collision_free_fraction",
    "near_miss_free_fraction",
    "success_fraction",
    "time_to_goal_score",
    "comfort_force_score",
)
_VALID_TERMINATIONS = {"success", "collision", "terminated", "truncated", "max_steps"}


@dataclass(frozen=True)
class ParameterSpec:
    """One bounded top-level planner parameter."""

    name: str
    kind: str
    low: float | int
    high: float | int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ParameterSpec:
        """Parse and validate a float or integer search bound."""
        name = str(raw.get("name", "")).strip()
        kind = str(raw.get("type", "")).strip().lower()
        if not name or "." in name:
            raise ValueError("parameter names must be non-empty top-level config keys")
        if kind not in {"float", "int"}:
            raise ValueError(f"unsupported parameter type for {name!r}: {kind!r}")
        low = raw.get("low")
        high = raw.get("high")
        if isinstance(low, bool) or isinstance(high, bool):
            raise TypeError(f"boolean bounds are not valid for numeric parameter {name!r}")
        if kind == "int":
            if not isinstance(low, int) or not isinstance(high, int):
                raise TypeError(f"integer parameter {name!r} requires integer bounds")
            lower: float | int = low
            upper: float | int = high
        else:
            if not isinstance(low, int | float) or not isinstance(high, int | float):
                raise TypeError(f"float parameter {name!r} requires numeric bounds")
            lower = float(low)
            upper = float(high)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"float parameter {name!r} requires finite bounds")
        if lower > upper:
            raise ValueError(f"parameter {name!r} has low > high")
        return cls(name=name, kind=kind, low=lower, high=upper)

    def suggest(self, trial: Any) -> float | int:
        """Ask an Optuna trial for a value inside this bound."""
        if self.kind == "int":
            return int(trial.suggest_int(self.name, int(self.low), int(self.high)))
        return float(trial.suggest_float(self.name, float(self.low), float(self.high)))


@dataclass(frozen=True)
class EvaluationSuite:
    """A fixed benchmark suite assembled by the canonical policy-search loader."""

    name: str
    matrix_path: Path
    scenario_filter: tuple[str, ...]
    seeds: tuple[int, ...]
    horizon: int
    dt: float
    workers: int
    benchmark_profile: str

    def load_scenarios(self) -> list[dict[str, Any]]:
        """Materialize the explicitly filtered scenario and seed list."""
        loaded = policy_search._load_stage_scenarios(
            self.matrix_path,
            seed_manifest=None,
            seed_list=list(self.seeds),
            scenario_filter=list(self.scenario_filter),
        )
        if not isinstance(loaded, list) or not loaded:
            raise ValueError(f"suite {self.name!r} did not materialize a scenario list")
        return loaded

    def identities(self, scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return stable scenario/seed identities used by this suite."""
        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_id = str(
                scenario.get("name")
                or scenario.get("scenario_id")
                or scenario.get("id")
                or "unknown"
            )
            scenario_seeds = scenario.get("seeds", self.seeds)
            if not isinstance(scenario_seeds, list | tuple) or not scenario_seeds:
                raise ValueError(f"suite scenario {scenario_id!r} has no explicit seeds")
            rows.extend({"scenario_id": scenario_id, "seed": int(seed)} for seed in scenario_seeds)
        return rows

    def as_dict(self, scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Serialize suite scope and materialized identities for provenance."""
        identities = self.identities(scenarios)
        return {
            "name": self.name,
            "scenario_matrix": _display_path(self.matrix_path),
            "scenario_matrix_sha256": _sha256_file(self.matrix_path),
            "scenario_filter": list(self.scenario_filter),
            "seeds": list(self.seeds),
            "horizon": self.horizon,
            "dt": self.dt,
            "workers": self.workers,
            "benchmark_profile": self.benchmark_profile,
            "episode_count": len(identities),
            "episode_identities": identities,
            "identity_sha256": _sha256_json(identities),
            "materialized_scenario_sha256": _sha256_json(list(scenarios)),
        }


@dataclass(frozen=True)
class OptimizerConfig:
    """Validated finite optimizer and benchmark contract."""

    run_id: str
    baseline_candidate: str
    candidate_registry: Path
    trials_per_method: int
    random_seed: int
    tpe_seed: int
    tpe_startup_trials: int
    parameters: tuple[ParameterSpec, ...]
    train: EvaluationSuite
    heldout: EvaluationSuite


@dataclass(frozen=True)
class OptimizationInputs:
    """Resolved candidate, suites, and identities for one optimization run."""

    baseline_entry: dict[str, Any]
    baseline_payload: dict[str, Any]
    baseline_config: dict[str, Any]
    baseline_config_path: Path
    base_config_path: Path | None
    algo: str
    train_scenarios: list[dict[str, Any]]
    heldout_scenarios: list[dict[str, Any]]
    train_identities: list[dict[str, Any]]
    heldout_identities: list[dict[str, Any]]
    train_suite: dict[str, Any]
    heldout_suite: dict[str, Any]


@dataclass(frozen=True)
class MethodRunContext:
    """Immutable inputs shared by Random and TPE candidate evaluation."""

    config: OptimizerConfig
    baseline_payload: Mapping[str, Any]
    baseline_config: Mapping[str, Any]
    train_scenarios: list[dict[str, Any]]
    train_identities: list[dict[str, Any]]
    expected_episodes: int
    output_dir: Path
    evaluator: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class Selection:
    """Training-only selection and canonical exported config paths."""

    random_best: dict[str, Any] | None
    tpe_best: dict[str, Any] | None
    winner: Mapping[str, Any] | None
    winner_method: str
    improves_over_baseline: bool
    payload: dict[str, Any]
    effective_config: dict[str, Any]
    candidate_name: str
    selected_train: dict[str, Any]
    candidate_path: Path
    registry_path: Path


def _repo_path(value: Any, *, anchor: Path) -> Path:
    """Resolve a configured repository path relative to config or repository root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    for candidate in ((anchor / path).resolve(), (REPO_ROOT / path).resolve()):
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject other top-level values."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected YAML mapping in {path}")
    return value


def _suite_from_mapping(
    name: str,
    raw: Mapping[str, Any],
    *,
    anchor: Path,
    evaluation: Mapping[str, Any],
) -> EvaluationSuite:
    scenario_filter_raw = raw.get("scenario_filter")
    seeds_raw = raw.get("seeds")
    if not isinstance(scenario_filter_raw, list) or not scenario_filter_raw:
        raise ValueError(f"suite {name!r} requires a non-empty scenario_filter")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"suite {name!r} requires a non-empty fixed seed list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds_raw):
        raise TypeError(f"suite {name!r} seeds must be integers")
    if len({int(seed) for seed in seeds_raw}) != len(seeds_raw):
        raise ValueError(f"suite {name!r} has duplicate seeds")
    scenario_ids = tuple(str(item).strip() for item in scenario_filter_raw)
    if any(not item for item in scenario_ids) or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError(f"suite {name!r} scenario_filter has empty or duplicate IDs")
    return EvaluationSuite(
        name=name,
        matrix_path=_repo_path(raw.get("scenario_matrix"), anchor=anchor),
        scenario_filter=scenario_ids,
        seeds=tuple(int(seed) for seed in seeds_raw),
        horizon=int(evaluation.get("horizon", 120)),
        dt=float(evaluation.get("dt", 0.1)),
        workers=int(evaluation.get("workers", 1)),
        benchmark_profile=str(evaluation.get("benchmark_profile", "experimental")),
    )


def _parse_search_configuration(
    raw: Mapping[str, Any],
) -> tuple[int, int, int, int, tuple[ParameterSpec, ...]]:
    trials = raw.get("trials_per_method")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("search.trials_per_method must be a positive integer")
    parameters_raw = raw.get("parameters")
    if not isinstance(parameters_raw, list) or not parameters_raw:
        raise ValueError("search.parameters must be a non-empty list")
    parameters = tuple(ParameterSpec.from_mapping(item) for item in parameters_raw)
    if len({parameter.name for parameter in parameters}) != len(parameters):
        raise ValueError("search parameter names must be unique")
    startup = int(raw.get("tpe_startup_trials", 1))
    if startup < 1:
        raise ValueError("search.tpe_startup_trials must be positive")
    return (
        trials,
        int(raw.get("random_seed", 1)),
        int(raw.get("tpe_seed", 2)),
        startup,
        parameters,
    )


def _parse_evaluation_configuration(raw: Mapping[str, Any]) -> dict[str, Any]:
    horizon = int(raw.get("horizon", 120))
    dt = float(raw.get("dt", 0.1))
    workers = int(raw.get("workers", 1))
    if horizon < 1 or not math.isfinite(dt) or dt <= 0 or workers != 1:
        raise ValueError("evaluation requires positive horizon/dt and exactly one worker")
    return {
        "horizon": horizon,
        "dt": dt,
        "workers": workers,
        "benchmark_profile": str(raw.get("benchmark_profile", "experimental")),
    }


def load_optimizer_config(path: str | Path) -> OptimizerConfig:
    """Load and validate the versioned optimizer configuration."""
    config_path = Path(path).resolve()
    raw = _load_yaml(config_path)
    if raw.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"config schema must equal {CONFIG_SCHEMA!r}")
    run_id = str(raw.get("run_id", "")).strip()
    if not run_id or any(char in run_id for char in "/\\"):
        raise ValueError("run_id must be a non-empty single path component")
    baseline = str(raw.get("baseline_candidate", "")).strip()
    if not baseline:
        raise ValueError("baseline_candidate is required")
    search = raw.get("search")
    evaluation = raw.get("evaluation")
    suites = raw.get("suites")
    if (
        not isinstance(search, dict)
        or not isinstance(evaluation, dict)
        or not isinstance(suites, dict)
    ):
        raise TypeError("search, evaluation, and suites must be YAML mappings")
    trial_count, random_seed, tpe_seed, tpe_startup, parameters = _parse_search_configuration(
        search
    )
    evaluation_config = _parse_evaluation_configuration(evaluation)
    train_raw = suites.get("train")
    heldout_raw = suites.get("heldout")
    if not isinstance(train_raw, dict) or not isinstance(heldout_raw, dict):
        raise TypeError("suites.train and suites.heldout must be mappings")
    return OptimizerConfig(
        run_id=run_id,
        baseline_candidate=baseline,
        candidate_registry=_repo_path(raw.get("candidate_registry"), anchor=config_path.parent),
        trials_per_method=trial_count,
        random_seed=random_seed,
        tpe_seed=tpe_seed,
        tpe_startup_trials=tpe_startup,
        parameters=parameters,
        train=_suite_from_mapping(
            "train", train_raw, anchor=config_path.parent, evaluation=evaluation_config
        ),
        heldout=_suite_from_mapping(
            "heldout", heldout_raw, anchor=config_path.parent, evaluation=evaluation_config
        ),
    )


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths in durable artifacts."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _portable_artifact_paths(value: Any, *, key: str = "") -> Any:
    """Replace local absolute artifact paths with repository-relative paths when possible."""
    if isinstance(value, Mapping):
        return {
            str(child_key): _portable_artifact_paths(child, key=str(child_key))
            for child_key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [_portable_artifact_paths(child, key=key) for child in value]
    if isinstance(value, Path):
        return _display_path(value) if key.endswith(("_path", "_dir")) else str(value)
    if isinstance(value, str) and key.endswith(("_path", "_dir")):
        path = Path(value)
        return _display_path(path) if path.is_absolute() else value
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(value: Any) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    )
    return _sha256_bytes(serialized.encode("utf-8"))


def _json_safe(value: Any) -> Any:
    """Normalize optional and non-finite metric values for strict JSON artifacts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except (TypeError, ValueError):
            return str(value)
    return value


def _git_revision() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status() -> list[str]:
    import subprocess

    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _validate_candidate_domain(
    config: OptimizerConfig,
    baseline_payload: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    *,
    algo: str,
) -> None:
    if algo != "hybrid_rule_local_planner":
        raise ValueError("planner optimizer v1 currently supports hybrid_rule_local_planner only")
    if any(
        isinstance(baseline_payload.get(key), dict) and baseline_payload.get(key)
        for key in ("scenario_overrides", "family_overrides", "scenario_algo_overrides")
    ):
        raise ValueError(
            "planner optimizer v1 requires one global config, without scenario overrides"
        )
    valid_parameter_names = {item.name for item in fields(HybridRuleLocalPlannerConfig)}
    for spec in config.parameters:
        if spec.name not in valid_parameter_names:
            raise ValueError(f"unknown hybrid-rule planner parameter: {spec.name!r}")
        value = baseline_config.get(spec.name)
        if value is None:
            raise ValueError(f"search parameter {spec.name!r} is absent from the baseline config")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"baseline parameter {spec.name!r} is not numeric")
        if spec.kind == "int" and not isinstance(value, int):
            raise TypeError(f"baseline parameter {spec.name!r} is not an integer")
        if spec.kind == "float" and not isinstance(value, int | float):
            raise TypeError(f"baseline parameter {spec.name!r} is not a float")
        if float(value) < float(spec.low) or float(value) > float(spec.high):
            raise ValueError(
                f"baseline value for {spec.name!r} is outside the configured search bounds"
            )
    build_hybrid_rule_local_planner_config(dict(baseline_config))


def _apply_params(
    baseline_payload: Mapping[str, Any], parameters: Mapping[str, float | int], *, name: str
) -> dict[str, Any]:
    payload = deepcopy(dict(baseline_payload))
    payload["name"] = name
    raw_params = payload.get("params")
    if not isinstance(raw_params, dict):
        raw_params = {}
    payload["params"] = {**raw_params, **dict(parameters)}
    return payload


def _invalid_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    scenario_id = row.get("scenario_id")
    seed = row.get("seed")
    if not isinstance(scenario_id, str) or not scenario_id.strip():
        reasons.append("episode_scenario_identity_missing")
    if isinstance(seed, bool) or not isinstance(seed, int):
        reasons.append("episode_seed_identity_missing")
    if not isinstance(row.get("metrics"), Mapping):
        reasons.append("episode_metrics_missing")
    termination = str(row.get("termination_reason", "")).strip().lower()
    if termination not in _VALID_TERMINATIONS:
        reasons.append(f"invalid_termination:{termination or 'missing'}")
    if policy_search._row_is_fallback_or_degraded(row):
        reasons.append("fallback_or_degraded_execution")
    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping) or any(
        not isinstance(outcome.get(field), bool)
        for field in ("route_complete", "collision_event", "timeout_event")
    ):
        reasons.append("episode_outcome_missing_or_malformed")
    exclusion = normalize_scenario_exclusion(row)
    if exclusion is not None:
        reasons.append(f"scenario_exclusion:{exclusion['status']}")
    integrity = row.get("integrity")
    contradictions = integrity.get("contradictions") if isinstance(integrity, Mapping) else None
    if not isinstance(contradictions, list):
        reasons.append("episode_integrity_missing_or_malformed")
    elif contradictions:
        reasons.append("episode_integrity_contradiction")
    return reasons


def _finite_metric(row: Mapping[str, Any], *names: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    for name in names:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return None


def _identity_audit(
    records: Sequence[Mapping[str, Any]],
    expected_episodes: int,
    expected_identities: Sequence[Mapping[str, Any]] | None,
) -> tuple[set[int], dict[str, int], int]:
    if expected_identities is None:
        return set(), {}, 0
    if len(expected_identities) != expected_episodes:
        raise ValueError("expected_identities length must equal expected_episodes")
    remaining = Counter((str(row["scenario_id"]), int(row["seed"])) for row in expected_identities)
    invalid_indices: set[int] = set()
    for index, row in enumerate(records):
        scenario_id = row.get("scenario_id")
        seed = row.get("seed")
        identity = (
            (str(scenario_id), int(seed))
            if isinstance(scenario_id, str) and isinstance(seed, int) and not isinstance(seed, bool)
            else None
        )
        if identity is None or remaining.get(identity, 0) <= 0:
            invalid_indices.add(index)
        else:
            remaining[identity] -= 1
    missing = {
        f"{scenario_id}::seed{seed}": count
        for (scenario_id, seed), count in sorted(remaining.items())
        if count > 0
    }
    return invalid_indices, missing, len(invalid_indices)


def score_records(
    records: Sequence[Mapping[str, Any]],
    expected_episodes: int,
    expected_identities: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute explicit lexicographic components from canonical episode records."""
    if expected_episodes < 1:
        raise ValueError("expected_episodes must be positive")
    identity_invalid_indices, missing_identities, unexpected_identity_count = _identity_audit(
        records, expected_episodes, expected_identities
    )
    valid_rows = [
        row
        for index, row in enumerate(records)
        if not _invalid_reasons(row) and index not in identity_invalid_indices
    ]
    invalid_counts: Counter[str] = Counter()
    for index, row in enumerate(records):
        invalid_counts.update(_invalid_reasons(row))
        if index in identity_invalid_indices:
            invalid_counts["unexpected_or_duplicate_episode_identity"] += 1
    for missing_count in missing_identities.values():
        invalid_counts["missing_episode_identity"] += missing_count
    if len(records) < expected_episodes:
        invalid_counts["missing_episode_record"] += expected_episodes - len(records)
    elif len(records) > expected_episodes:
        invalid_counts["unexpected_extra_episode_record"] += len(records) - expected_episodes
    # A missing or extra row invalidates completeness for this evaluation; keep all rows
    # in raw evidence but give no evaluation credit until the expected denominator is met.
    evaluation_complete = (
        len(records) == expected_episodes
        and not missing_identities
        and unexpected_identity_count == 0
    )
    valid_count = (
        len(valid_rows) if evaluation_complete else min(len(valid_rows), expected_episodes - 1)
    )
    denominator = max(valid_count, 1)
    colliding = 0
    successes = 0
    for row in valid_rows[:valid_count]:
        reason = str(row.get("termination_reason", "")).strip().lower()
        collisions = _finite_metric(row, "collisions", "total_collision_count") or 0.0
        outcome = row.get("outcome")
        collision_event = isinstance(outcome, Mapping) and outcome.get("collision_event") is True
        if reason == "collision" or collision_event or collisions > 0.0:
            colliding += 1
        if reason == "success" and collisions <= 0.0:
            successes += 1
    valid_fraction = valid_count / float(expected_episodes)
    collision_free_fraction = (valid_count - colliding) / float(denominator)
    valid_subset = valid_rows[:valid_count]
    valid_summary = summarize_policy_search_records([dict(row) for row in valid_subset])
    near_miss_metric_missing_count = sum(
        _finite_metric(row, "near_misses", "min_clearance") is None for row in valid_subset
    )
    near_miss_free_fraction = (
        1.0 - float(valid_summary.get("near_miss_rate", 0.0))
        if valid_count and near_miss_metric_missing_count == 0
        else None
    )
    successful_rows = [
        row
        for row in valid_subset
        if str(row.get("termination_reason", "")).strip().lower() == "success"
    ]
    success_fraction = successes / float(denominator)
    time_to_goal = [
        value
        for row in successful_rows
        for value in [
            _finite_metric(row, "time_to_goal_ideal_ratio", "time_to_goal_norm_success_only")
        ]
        if value is not None
    ]
    comfort_force = [
        value
        for row in valid_subset
        for value in [_finite_metric(row, "ped_force_q95", "force_q95")]
        if value is not None
    ]
    time_mean = (
        sum(time_to_goal) / len(time_to_goal)
        if successful_rows and len(time_to_goal) == len(successful_rows)
        else None
    )
    comfort_mean = (
        sum(comfort_force) / len(comfort_force)
        if valid_count and len(comfort_force) == valid_count
        else None
    )
    return {
        "expected_episodes": expected_episodes,
        "observed_episodes": len(records),
        "evaluation_complete": evaluation_complete,
        "expected_episode_identities": list(expected_identities)
        if expected_identities is not None
        else None,
        "observed_episode_identities": [
            {"scenario_id": str(row.get("scenario_id")), "seed": row.get("seed")} for row in records
        ],
        "missing_episode_identities": missing_identities,
        "unexpected_or_duplicate_episode_identity_count": unexpected_identity_count,
        "valid_episode_count": valid_count,
        "invalid_episode_count": expected_episodes - valid_count,
        "invalid_reason_counts": dict(sorted(invalid_counts.items())),
        "valid_episode_fraction": valid_fraction,
        "collision_free_fraction": collision_free_fraction,
        "colliding_valid_episode_count": colliding,
        "near_miss_free_fraction": near_miss_free_fraction,
        "near_miss_metric_missing_episode_count": near_miss_metric_missing_count,
        "success_fraction": success_fraction,
        "success_count": successes,
        "time_to_goal_ideal_ratio_mean_success_only": time_mean,
        "time_metric_missing_success_episode_count": len(successful_rows) - len(time_to_goal),
        "comfort_ped_force_q95_mean": comfort_mean,
        "comfort_metric_missing_valid_episode_count": valid_count - len(comfort_force),
        "comfort_metric_available": comfort_mean is not None,
        "selection_tuple": [
            valid_fraction,
            collision_free_fraction,
            near_miss_free_fraction,
            success_fraction,
            -time_mean if time_mean is not None else None,
            -comfort_mean if comfort_mean is not None else None,
        ],
        "sampler_objectives": [
            valid_fraction,
            collision_free_fraction,
            near_miss_free_fraction if near_miss_free_fraction is not None else -1.0e9,
            success_fraction,
            -time_mean if time_mean is not None else -1.0e9,
            -comfort_mean if comfort_mean is not None else -1.0e9,
        ],
        "objective_names": list(OBJECTIVE_NAMES),
        "metric_source": {
            "collision": "termination_reason and metrics.collisions/total_collision_count",
            "near_miss": "policy_search_common surface-clearance near-miss semantics",
            "completion": "termination_reason == success",
            "efficiency": "metrics.time_to_goal_ideal_ratio, success episodes only",
            "comfort": "metrics.ped_force_q95 (fallback: metrics.force_q95)",
        },
    }


def _selection_key(score: Mapping[str, Any]) -> tuple[float, ...]:
    raw = score.get("selection_tuple")
    if not isinstance(raw, list):
        return tuple(float("-inf") for _ in OBJECTIVE_NAMES)
    return tuple(float(value) if value is not None else float("-inf") for value in raw)


def select_best_trial(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select the best validly evaluated row by the explicit objective tuple."""
    candidates = [
        row
        for row in rows
        if row.get("status") == "complete" and isinstance(row.get("score"), Mapping)
    ]
    if not candidates:
        raise ValueError("no validly evaluated candidates are available for selection")
    return max(
        candidates,
        key=lambda row: (
            _selection_key(row["score"]),
            -int(row.get("trial_index", 0)),
        ),
    )


def _best_or_none(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    try:
        return select_best_trial(rows)
    except ValueError:
        return None


def _trial_candidate_config(
    baseline_payload: Mapping[str, Any],
    baseline_config: Mapping[str, Any],
    parameters: Mapping[str, float | int],
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_payload = _apply_params(baseline_payload, parameters, name=name)
    effective = deepcopy(dict(baseline_config))
    effective.update(parameters)
    build_hybrid_rule_local_planner_config(effective)
    return candidate_payload, effective


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")


def _candidate_evaluation(
    *,
    candidate_payload: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    scenarios: list[dict[str, Any]],
    expected_episodes: int,
    output_dir: Path,
    tag: str,
    suite: EvaluationSuite,
    evaluator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    candidate_path = output_dir / "candidate.yaml"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(
        yaml.safe_dump(dict(candidate_payload), sort_keys=False), encoding="utf-8"
    )
    started = time.perf_counter()
    try:
        result = evaluator(
            scenarios_or_path=scenarios,
            algo="hybrid_rule_local_planner",
            algo_cfg=dict(effective_config),
            out_dir=output_dir,
            tag=tag,
            horizon=suite.horizon,
            dt=suite.dt,
            workers=suite.workers,
            benchmark_profile=suite.benchmark_profile,
        )
        records = result.get("records", [])
        if not isinstance(records, list):
            records = []
        score = score_records(records, expected_episodes, suite.identities(scenarios))
        summary_path = output_dir / "stage_summary.json"
        _write_json(
            summary_path,
            _portable_artifact_paths(result.get("summary") or {}),
        )
        return {
            "status": "complete"
            if score["evaluation_complete"] and score["invalid_episode_count"] == 0
            else "invalid",
            "score": score,
            "runtime_sec": float(max(time.perf_counter() - started, 0.0)),
            "candidate_config_path": _display_path(candidate_path),
            "effective_config_sha256": _sha256_json(effective_config),
            "episode_jsonl_path": _display_path(Path(result["jsonl_path"]))
            if isinstance(result.get("jsonl_path"), str)
            else None,
            "stage_summary_path": _display_path(summary_path),
        }
    except Exception as exc:  # noqa: BLE001 - failed evaluations are preserved as invalid trials.
        return {
            "status": "invalid",
            "score": score_records([], expected_episodes, suite.identities(scenarios)),
            "runtime_sec": float(max(time.perf_counter() - started, 0.0)),
            "candidate_config_path": _display_path(candidate_path),
            "effective_config_sha256": _sha256_json(effective_config),
            "episode_jsonl_path": None,
            "stage_summary_path": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def _optuna_study(method: str, *, seed: int, startup_trials: int) -> Any:
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - project dependency is installed in normal use.
        raise RuntimeError("Install the project dependencies to use Optuna planner search") from exc
    if method == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif method == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials)
    else:
        raise ValueError(f"unknown optimizer method: {method}")
    return optuna.create_study(
        directions=["maximize"] * len(OBJECTIVE_NAMES),
        sampler=sampler,
        study_name=f"planner-{method}-{seed}",
    )


def _run_method(
    method: str,
    seed: int,
    context: MethodRunContext,
) -> list[dict[str, Any]]:
    import optuna

    config = context.config
    study = _optuna_study(method, seed=seed, startup_trials=config.tpe_startup_trials)
    rows: list[dict[str, Any]] = []
    trials_path = context.output_dir / "trials.jsonl"
    for trial_index in range(config.trials_per_method):
        trial_started = time.perf_counter()
        trial = study.ask()
        params = {spec.name: spec.suggest(trial) for spec in config.parameters}
        candidate_name = f"{config.run_id}_{method}_{trial_index:03d}"
        trial_dir = context.output_dir / "trials" / method / f"trial_{trial_index:03d}"
        candidate_payload = _apply_params(context.baseline_payload, params, name=candidate_name)
        try:
            candidate_payload, effective = _trial_candidate_config(
                context.baseline_payload,
                context.baseline_config,
                params,
                name=candidate_name,
            )
            result = _candidate_evaluation(
                candidate_payload=candidate_payload,
                effective_config=effective,
                scenarios=context.train_scenarios,
                expected_episodes=context.expected_episodes,
                output_dir=trial_dir,
                tag=f"{method}_trial_{trial_index:03d}",
                suite=config.train,
                evaluator=context.evaluator,
            )
        except Exception as exc:  # noqa: BLE001 - malformed trials remain visible in JSONL.
            trial_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = trial_dir / "candidate.yaml"
            candidate_path.write_text(
                yaml.safe_dump(candidate_payload, sort_keys=False), encoding="utf-8"
            )
            result = {
                "status": "invalid",
                "score": score_records([], context.expected_episodes, context.train_identities),
                "runtime_sec": float(max(time.perf_counter() - trial_started, 0.0)),
                "candidate_config_path": str(candidate_path),
                "effective_config_sha256": None,
                "episode_jsonl_path": None,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        result["evaluation_runtime_sec"] = result["runtime_sec"]
        result["runtime_sec"] = float(max(time.perf_counter() - trial_started, 0.0))
        row = {
            "method": method,
            "trial_index": trial_index,
            "sampler_seed": seed,
            "parameters": params,
            "candidate_name": candidate_name,
            **result,
        }
        rows.append(row)
        _append_jsonl(trials_path, row)
        if result["status"] == "complete":
            study.tell(trial, values=result["score"]["sampler_objectives"])
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
    return rows


def _selected_parameters(row: Mapping[str, Any]) -> dict[str, Any]:
    parameters = row.get("parameters")
    return dict(parameters) if isinstance(parameters, Mapping) else {}


def _score_observation(score: Mapping[str, Any]) -> str:
    expected = int(score.get("expected_episodes", 0))
    valid = int(score.get("valid_episode_count", 0))
    collisions = int(score.get("colliding_valid_episode_count", 0))
    successes = int(score.get("success_count", 0))
    near_miss_free = score.get("near_miss_free_fraction")
    near_miss_text = "unknown" if near_miss_free is None else f"{near_miss_free:.3f}"
    time_mean = score.get("time_to_goal_ideal_ratio_mean_success_only")
    time_text = "unknown" if time_mean is None else f"{time_mean:.3f}"
    comfort_mean = score.get("comfort_ped_force_q95_mean")
    comfort_text = "unknown" if comfort_mean is None else f"{comfort_mean:.3f}"
    return (
        f"valid {valid}/{expected}; collision episodes {collisions}/{valid}; "
        f"near-miss-free {near_miss_text}; successes {successes}/{valid}; "
        f"successful time ratio {time_text}; mean episode p95 force {comfort_text}"
    )


def _write_report(path: Path, manifest: Mapping[str, Any]) -> None:
    methods = manifest["method_results"]
    selected = manifest["selected"]
    heldout = manifest["heldout_comparison"]
    train = manifest["train_baseline"]
    lines = [
        f"# Planner optimizer bounded pilot: {manifest['run_id']}",
        "",
        f"- Source revision: `{manifest['source_revision']}`",
        "- Evidence class: diagnostic finite-budget simulator pilot; not nominal or release benchmark evidence",
        f"- Baseline candidate: `{manifest['baseline_candidate']}`",
        f"- Training episodes per optimizer trial: {manifest['budget']['episodes_per_trial']}",
        f"- Trial budget: {manifest['budget']['trials_per_method']} per method, equal across Random and TPE",
        f"- Total search evaluations: {manifest['budget']['search_evaluations']} episodes",
        f"- Held-out identities: `{manifest['suites']['heldout']['identity_sha256']}`",
        "",
        "## Objective order",
        "",
        "1. Valid benchmark execution/admissibility",
        "2. Collision-free fraction",
        "3. Near-miss-free fraction using the policy-search surface-clearance semantics",
        "4. Task completion fraction",
        "5. Lower successful time-to-goal ideal ratio",
        "6. Lower pedestrian 95th-percentile force (when available)",
        "",
        "The recorded selection tuple preserves this order. Optuna receives the same component vector; no weighted aggregate is used.",
        "Missing secondary metrics remain null in the evidence and receive a tied sentinel only in the sampler interface.",
        "",
        "## Training results",
        "",
        f"- Baseline: `{train['score']['selection_tuple']}` ({train['status']}); {_score_observation(train['score'])}.",
    ]
    for method in ("random", "tpe"):
        result = methods[method]
        best = result["best_trial"]
        if best is None:
            lines.append(
                f"- {method.upper()}: no valid trial; all {result['trial_count']} trials are preserved as invalid."
            )
        else:
            lines.append(
                f"- {method.upper()} best trial `{best['trial_index']}`: `{best['score']['selection_tuple']}` ({best['status']}); {_score_observation(best['score'])}; parameters `{json.dumps(best['parameters'], sort_keys=True)}`"
            )
        lines.append(
            f"  Invalid trials: {result['invalid_trial_count']} / {result['trial_count']}; runtime `{result['runtime_sec']:.3f}s`."
        )
    lines.extend(
        [
            "",
            "## Frozen held-out comparison",
            "",
            f"- Selected method/config: `{selected['method']}` / `{selected['candidate_name']}`.",
            f"- Baseline: `{heldout['baseline']['score']['selection_tuple']}` ({heldout['baseline']['status']}); {_score_observation(heldout['baseline']['score'])}.",
            f"- Selected: `{heldout['selected']['score']['selection_tuple']}` ({heldout['selected']['status']}); {_score_observation(heldout['selected']['score'])}.",
            f"- Training-set objective improvement over baseline: `{selected['improves_over_baseline']}`.",
            f"- Held-out objective improvement over baseline: `{heldout['selected_improves_over_baseline']}`.",
            "",
            "## Interpretation boundary",
            "",
            "The lexicographic improvement is limited to near-miss rate and force on this tiny pilot: task completion did not improve, and the training collision remained. Efficiency is unknown because no episode completed successfully. Three trials per method are too few to infer that one search method is better. This demonstrates only observed planner/config behavior on these exact rows; it does not establish global optimality, feasibility for other inputs, or real-world safety. No mathematical feasibility oracle is claimed.",
            "",
            f"Canonical config export: `{manifest['selected_candidate_config']}`.",
            f"Existing policy-search runner registry: `{manifest['candidate_registry_export']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_optimization_inputs(config: OptimizerConfig) -> OptimizationInputs:
    baseline_entry, baseline_payload, baseline_config, candidate_path = (
        policy_search.load_candidate_definition(
            config.candidate_registry, config.baseline_candidate
        )
    )
    algo = str(baseline_payload.get("algo", "")).strip().lower()
    _validate_candidate_domain(config, baseline_payload, baseline_config, algo=algo)
    train_scenarios = config.train.load_scenarios()
    heldout_scenarios = config.heldout.load_scenarios()
    train_identities = config.train.identities(train_scenarios)
    heldout_identities = config.heldout.identities(heldout_scenarios)
    train_ids = {(row["scenario_id"], row["seed"]) for row in train_identities}
    heldout_ids = {(row["scenario_id"], row["seed"]) for row in heldout_identities}
    if not train_ids or not heldout_ids:
        raise ValueError("training and held-out suites must both materialize episode identities")
    overlap = sorted(train_ids & heldout_ids)
    if overlap:
        raise ValueError(f"training and held-out scenario/seed identities overlap: {overlap}")
    return OptimizationInputs(
        baseline_entry=baseline_entry,
        baseline_payload=baseline_payload,
        baseline_config=baseline_config,
        baseline_config_path=candidate_path,
        base_config_path=policy_search._resolve_path(
            candidate_path.parent, baseline_payload.get("base_config_path")
        ),
        algo=algo,
        train_scenarios=train_scenarios,
        heldout_scenarios=heldout_scenarios,
        train_identities=train_identities,
        heldout_identities=heldout_identities,
        train_suite=config.train.as_dict(train_scenarios),
        heldout_suite=config.heldout.as_dict(heldout_scenarios),
    )


def _create_run_directory(output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    if list(target.iterdir()):
        raise FileExistsError(
            f"optimizer output directory is non-empty; inspect existing evidence instead of rerunning: {target}"
        )
    lock_path = target / ".run.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(f"pid={__import__('os').getpid()}\n")
    except FileExistsError as exc:
        raise FileExistsError(f"optimizer output is already locked: {target}") from exc
    return target, lock_path


def _initial_manifest(
    config_path: str | Path,
    config: OptimizerConfig,
    inputs: OptimizationInputs,
    *,
    source_revision: str,
    source_tracked_changes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    try:
        optuna_version: str | None = version("optuna")
    except PackageNotFoundError:
        optuna_version = None
    resolved_config_path = Path(config_path).resolve()
    train_count = len(inputs.train_identities)
    heldout_count = len(inputs.heldout_identities)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "running",
        "run_id": config.run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": source_revision,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "optuna_version": optuna_version,
            "pyproject_sha256": _sha256_file(REPO_ROOT / "pyproject.toml"),
            "uv_lock_sha256": _sha256_file(REPO_ROOT / "uv.lock")
            if (REPO_ROOT / "uv.lock").is_file()
            else None,
        },
        "source_tracked_changes_at_start": source_tracked_changes,
        "config_path": _display_path(resolved_config_path),
        "config_sha256": _sha256_file(resolved_config_path),
        "baseline_candidate": config.baseline_candidate,
        "baseline_registry_path": _display_path(config.candidate_registry),
        "baseline_registry_sha256": _sha256_file(config.candidate_registry),
        "baseline_candidate_config_path": _display_path(inputs.baseline_config_path),
        "baseline_candidate_config_sha256": _sha256_file(inputs.baseline_config_path),
        "baseline_base_config_path": (
            _display_path(inputs.base_config_path) if inputs.base_config_path else None
        ),
        "baseline_base_config_sha256": (
            _sha256_file(inputs.base_config_path) if inputs.base_config_path else None
        ),
        "baseline_algo": inputs.algo,
        "parameter_space": [
            {"name": item.name, "type": item.kind, "low": item.low, "high": item.high}
            for item in config.parameters
        ],
        "objective": {
            "kind": "lexicographic_components_v1",
            "order": list(OBJECTIVE_NAMES),
            "selection_rule": (
                "maximize validity, then collision-free, near-miss-free, completion, "
                "and minimize successful time-to-goal and comfort force in that order"
            ),
        },
        "budget": {
            "methods": ["random", "tpe"],
            "trials_per_method": config.trials_per_method,
            "episodes_per_trial": train_count,
            "search_evaluations": 2 * config.trials_per_method * train_count,
            "additional_train_baseline_evaluations": train_count,
            "heldout_baseline_and_selected_evaluations_max": 2 * heldout_count,
            "evaluation_budget_per_method": config.trials_per_method * train_count,
            "random_seed": config.random_seed,
            "tpe_seed": config.tpe_seed,
            "tpe_startup_trials": config.tpe_startup_trials,
        },
        "suites": {"train": inputs.train_suite, "heldout": inputs.heldout_suite},
        "suite_split": {
            "train_episode_count": train_count,
            "heldout_episode_count": heldout_count,
            "disjoint_scenario_seed_identity_count": train_count + heldout_count,
            "overlap": [],
        },
        "files": {"trials": "trials.jsonl", "episode_records": "trials/ and heldout/"},
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _evaluate_training(
    config: OptimizerConfig,
    inputs: OptimizationInputs,
    output_dir: Path,
    evaluator: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_baseline = _candidate_evaluation(
        candidate_payload=inputs.baseline_payload,
        effective_config=inputs.baseline_config,
        scenarios=inputs.train_scenarios,
        expected_episodes=len(inputs.train_identities),
        output_dir=output_dir / "train_baseline",
        tag="train_baseline",
        suite=config.train,
        evaluator=evaluator,
    )
    train_baseline_row = {
        "method": "baseline",
        "trial_index": -1,
        "sampler_seed": None,
        "parameters": {},
        "candidate_name": config.baseline_candidate,
        **train_baseline,
    }
    _write_json(output_dir / "train_baseline.json", train_baseline_row)
    context = MethodRunContext(
        config=config,
        baseline_payload=inputs.baseline_payload,
        baseline_config=inputs.baseline_config,
        train_scenarios=inputs.train_scenarios,
        train_identities=inputs.train_identities,
        expected_episodes=len(inputs.train_identities),
        output_dir=output_dir,
        evaluator=evaluator,
    )
    random_rows = _run_method("random", config.random_seed, context)
    tpe_rows = _run_method("tpe", config.tpe_seed, context)
    return train_baseline_row, random_rows, tpe_rows


def _export_candidate(
    config: OptimizerConfig,
    inputs: OptimizationInputs,
    output_dir: Path,
    train_baseline: Mapping[str, Any],
    random_rows: list[dict[str, Any]],
    tpe_rows: list[dict[str, Any]],
) -> Selection:
    random_raw = _best_or_none(random_rows)
    tpe_raw = _best_or_none(tpe_rows)
    random_best = dict(random_raw) if random_raw is not None else None
    tpe_best = dict(tpe_raw) if tpe_raw is not None else None
    baseline_key = _selection_key(train_baseline["score"])
    for row in (random_best, tpe_best):
        if row is not None:
            row["improves_over_baseline"] = _selection_key(row["score"]) > baseline_key
    available = [
        (method, row)
        for method, row in (("random", random_best), ("tpe", tpe_best))
        if row is not None
    ]
    if available:
        winner_method, winner = max(
            available,
            key=lambda item: (_selection_key(item[1]["score"]), item[0] == "random"),
        )
        improves = _selection_key(winner["score"]) > baseline_key
    else:
        winner_method, winner, improves = "baseline", None, False
    if improves and winner is not None:
        payload, effective = _trial_candidate_config(
            inputs.baseline_payload,
            inputs.baseline_config,
            _selected_parameters(winner),
            name=f"{config.run_id}_best_{winner_method}",
        )
        selected_name = str(payload["name"])
        selected_train = dict(winner)
        method = winner_method
    else:
        payload = deepcopy(inputs.baseline_payload)
        effective = deepcopy(inputs.baseline_config)
        selected_name = config.baseline_candidate
        selected_train = dict(train_baseline)
        method = "baseline"
    candidate_path = output_dir / "best_candidate.yaml"
    candidate_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    registry_payload = {
        "version": 1,
        "candidates": {
            selected_name: {
                "candidate_config_path": candidate_path.name,
                "status": "experimental_spike",
                "family": "hybrid_rule_based",
                "claim_scope": "diagnostic_only",
            }
        },
    }
    registry_path = output_dir / "candidate_registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry_payload, sort_keys=False), encoding="utf-8")
    return Selection(
        random_best=random_best,
        tpe_best=tpe_best,
        winner=winner,
        winner_method=method,
        improves_over_baseline=improves,
        payload=payload,
        effective_config=effective,
        candidate_name=selected_name,
        selected_train=selected_train,
        candidate_path=candidate_path,
        registry_path=registry_path,
    )


def _evaluate_heldout(
    config: OptimizerConfig,
    inputs: OptimizationInputs,
    selection: Selection,
    output_dir: Path,
    evaluator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    expected = len(inputs.heldout_identities)
    baseline = _candidate_evaluation(
        candidate_payload=inputs.baseline_payload,
        effective_config=inputs.baseline_config,
        scenarios=inputs.heldout_scenarios,
        expected_episodes=expected,
        output_dir=output_dir / "heldout" / "baseline",
        tag="heldout_baseline",
        suite=config.heldout,
        evaluator=evaluator,
    )
    if selection.effective_config == inputs.baseline_config:
        selected = {
            **baseline,
            "runtime_sec": 0.0,
            "candidate_config_path": _display_path(selection.candidate_path),
            "episode_jsonl_path": baseline.get("episode_jsonl_path"),
            "alias_of": "heldout/baseline",
        }
    else:
        selected = _candidate_evaluation(
            candidate_payload=selection.payload,
            effective_config=selection.effective_config,
            scenarios=inputs.heldout_scenarios,
            expected_episodes=expected,
            output_dir=output_dir / "heldout" / "selected",
            tag="heldout_selected",
            suite=config.heldout,
            evaluator=evaluator,
        )
    return {
        "baseline": baseline,
        "selected": selected,
        "selected_improves_over_baseline": _selection_key(selected["score"])
        > _selection_key(baseline["score"]),
        "same_configuration": selection.effective_config == inputs.baseline_config,
        "selection_was_frozen_before_heldout": True,
    }


def _compact_method_results(
    rows_by_method: Mapping[str, list[dict[str, Any]]], selection: Selection
) -> dict[str, Any]:
    best_rows = {"random": selection.random_best, "tpe": selection.tpe_best}
    results: dict[str, Any] = {}
    for method, rows in rows_by_method.items():
        results[method] = {
            "trial_count": len(rows),
            "invalid_trial_count": sum(row["status"] != "complete" for row in rows),
            "runtime_sec": sum(float(row["runtime_sec"]) for row in rows),
            "best_trial": best_rows[method],
            "trials": [
                {
                    "trial_index": row["trial_index"],
                    "status": row["status"],
                    "parameters": row["parameters"],
                    "score": row["score"],
                    "runtime_sec": row["runtime_sec"],
                    "episode_jsonl_path": row["episode_jsonl_path"],
                    "error": row.get("error"),
                }
                for row in rows
            ],
        }
    return results


def _finalize_run(
    output_dir: Path,
    manifest: dict[str, Any],
    inputs: OptimizationInputs,
    train_baseline: dict[str, Any],
    rows_by_method: Mapping[str, list[dict[str, Any]]],
    selection: Selection,
    heldout: dict[str, Any],
) -> dict[str, Any]:
    method_results = _compact_method_results(rows_by_method, selection)
    valid_methods = sum(best is not None for best in (selection.random_best, selection.tpe_best))
    validity = {
        "train_baseline_valid": train_baseline["status"] == "complete",
        "valid_optimizer_methods": valid_methods,
        "both_methods_have_valid_trials": valid_methods == 2,
        "heldout_baseline_valid": heldout["baseline"]["status"] == "complete",
        "heldout_selected_valid": heldout["selected"]["status"] == "complete",
    }
    run_valid = all(
        (
            validity["train_baseline_valid"],
            validity["both_methods_have_valid_trials"],
            validity["heldout_baseline_valid"],
            validity["heldout_selected_valid"],
        )
    )
    trials = [row for rows in rows_by_method.values() for row in rows]
    actual_rows = [train_baseline, *trials, heldout["baseline"]]
    if "alias_of" not in heldout["selected"]:
        actual_rows.append(heldout["selected"])
    actual_episodes = sum(
        int((row.get("score") or {}).get("observed_episodes", 0)) for row in actual_rows
    )
    manifest["budget"]["actual_search_evaluation_episodes"] = sum(
        int(row["score"].get("observed_episodes", 0)) for row in trials
    )
    manifest.update(
        {
            "status": "complete" if run_valid else "incomplete_evaluations",
            "completed_at": datetime.now(UTC).isoformat(),
            "runtime_sec": sum(float(row["runtime_sec"]) for row in actual_rows),
            "actual_evaluation_episode_count": actual_episodes,
            "train_baseline": train_baseline,
            "method_results": method_results,
            "evaluation_validity": validity,
            "selected": {
                "method": selection.winner_method,
                "candidate_name": selection.candidate_name,
                "parameters": _selected_parameters(selection.winner)
                if selection.improves_over_baseline and selection.winner
                else {},
                "score": selection.selected_train["score"],
                "improves_over_baseline": selection.improves_over_baseline,
                "candidate_config_path": "best_candidate.yaml",
            },
            "heldout_comparison": heldout,
            "selected_candidate_config": "best_candidate.yaml",
            "candidate_registry_export": "candidate_registry.yaml",
            "integration_contract": {
                "manifest_schema": MANIFEST_SCHEMA,
                "best_candidate_yaml": "best_candidate.yaml",
                "candidate_registry_yaml": "candidate_registry.yaml",
                "trial_rows_jsonl": "trials.jsonl",
                "heldout_selection_frozen": True,
            },
            "source_candidate_entry": dict(inputs.baseline_entry),
        }
    )
    _write_report(output_dir / "report.md", manifest)
    _write_json(output_dir / "heldout_comparison.json", heldout)
    manifest["files"].update(
        {
            "manifest": "run_manifest.json",
            "report": "report.md",
            "best_candidate": "best_candidate.yaml",
            "candidate_registry": "candidate_registry.yaml",
            "heldout_comparison": "heldout_comparison.json",
        }
    )
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def run_planner_optimization(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a finite Random-vs-TPE search and write replayable evidence.

    A non-empty output directory is rejected. An interrupted run keeps its lock
    and partial records so a retry cannot silently repeat its simulations.
    """
    config = load_optimizer_config(config_path)
    source_revision = _git_revision()
    source_changes = _git_status()
    inputs = _load_optimization_inputs(config)
    target, lock_path = _create_run_directory(output_dir)
    evaluator_fn = evaluator or policy_search._run_stage_eval
    manifest = _initial_manifest(
        config_path,
        config,
        inputs,
        source_revision=source_revision,
        source_tracked_changes=source_changes,
        output_dir=target,
    )
    baseline, random_rows, tpe_rows = _evaluate_training(config, inputs, target, evaluator_fn)
    rows_by_method = {"random": random_rows, "tpe": tpe_rows}
    selection = _export_candidate(config, inputs, target, baseline, random_rows, tpe_rows)
    heldout = _evaluate_heldout(config, inputs, selection, target, evaluator_fn)
    result = _finalize_run(
        target,
        manifest,
        inputs,
        baseline,
        rows_by_method,
        selection,
        heldout,
    )
    try:
        lock_path.unlink()
    except OSError:
        pass
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "OBJECTIVE_NAMES",
    "EvaluationSuite",
    "OptimizerConfig",
    "ParameterSpec",
    "load_optimizer_config",
    "run_planner_optimization",
    "score_records",
    "select_best_trial",
]
