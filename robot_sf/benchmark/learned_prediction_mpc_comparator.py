"""Preflight contract for issue #4013 learned-prediction MPC comparator configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "issue_4013_learned_prediction_mpc_comparator.v1"
MODEL_FREE_ROLE = "model_free_comparator"
MODEL_BASED_ROLE = "model_based_candidate"
DIAGNOSTIC_BOUNDARY = "diagnostic_smoke_only"
WORLD_MODEL_EXCLUSIONS = (
    "dreamerv3",
    "planet",
    "td_mpc2",
    "large_generative_world_model",
    "paper_grade_claim",
)


@dataclass(frozen=True)
class ComparatorPreflight:
    """Issue #4013 paired-smoke comparator preflight result."""

    schema_version: str
    status: str
    model_free_config: str
    model_based_config: str
    scenario_matrix: str
    seeds: list[int]
    model_free_planner: str
    model_based_planner: str
    predictor_config: str
    predictor_source: str
    fallback_status: str
    claim_boundary: str
    world_model_exclusions: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report payload."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "model_free_config": self.model_free_config,
            "model_based_config": self.model_based_config,
            "scenario_matrix": self.scenario_matrix,
            "seeds": self.seeds,
            "model_free_planner": self.model_free_planner,
            "model_based_planner": self.model_based_planner,
            "predictor_config": self.predictor_config,
            "predictor_source": self.predictor_source,
            "fallback_status": self.fallback_status,
            "claim_boundary": self.claim_boundary,
            "world_model_exclusions": list(self.world_model_exclusions),
            "blockers": list(self.blockers),
        }


def build_comparator_preflight(
    *,
    model_free_config: Path,
    model_based_config: Path,
    repo_root: Path = Path("."),
) -> ComparatorPreflight:
    """Validate paired issue #4013 smoke configs without running a benchmark campaign.

    Returns:
        ComparatorPreflight: Structured readiness or blocker report.
    """

    repo_root = repo_root.resolve()
    model_free_path = _resolve(repo_root, model_free_config)
    model_based_path = _resolve(repo_root, model_based_config)
    model_free = _load_yaml(model_free_path)
    model_based = _load_yaml(model_based_path)
    blockers: list[str] = []

    scenario_matrix = _shared_value(
        model_free,
        model_based,
        "scenario_matrix",
        blockers,
        "scenario_matrix_mismatch",
    )
    seeds = _seed_list(model_free, blockers, "model_free")
    model_based_seeds = _seed_list(model_based, blockers, "model_based")
    if seeds != model_based_seeds:
        blockers.append("seed_list_mismatch")

    model_free_planner = _single_planner(model_free, blockers, "model_free")
    model_based_planner = _single_planner(model_based, blockers, "model_based")
    _require_planner_role(model_free_planner, MODEL_FREE_ROLE, blockers, "model_free")
    _require_planner_role(model_based_planner, MODEL_BASED_ROLE, blockers, "model_based")

    if model_free_planner.get("algo") != "prediction_mpc":
        blockers.append("model_free_algo_not_prediction_mpc")
    if model_based_planner.get("algo") != "learned_prediction_mpc":
        blockers.append("model_based_algo_not_learned_prediction_mpc")

    predictor_config_rel = str(model_based_planner.get("algo_config", ""))
    predictor_config_path = _resolve(repo_root, Path(predictor_config_rel))
    predictor_config = _load_yaml(predictor_config_path) if predictor_config_rel else {}
    predictor_source, fallback_status = _predictor_status(predictor_config, blockers)

    claim_boundary = _claim_boundary(model_free_planner, model_based_planner, blockers)
    status = "ready_diagnostic_smoke" if not blockers else "blocked"
    return ComparatorPreflight(
        schema_version=SCHEMA_VERSION,
        status=status,
        model_free_config=_display_path(model_free_path, repo_root),
        model_based_config=_display_path(model_based_path, repo_root),
        scenario_matrix=str(scenario_matrix),
        seeds=seeds,
        model_free_planner=str(model_free_planner.get("key", "")),
        model_based_planner=str(model_based_planner.get("key", "")),
        predictor_config=predictor_config_rel,
        predictor_source=predictor_source,
        fallback_status=fallback_status,
        claim_boundary=claim_boundary,
        world_model_exclusions=WORLD_MODEL_EXCLUSIONS,
        blockers=tuple(blockers),
    )


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    if path.is_relative_to(repo_root):
        return str(path.relative_to(repo_root))
    return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _shared_value(
    left: dict[str, Any],
    right: dict[str, Any],
    key: str,
    blockers: list[str],
    blocker: str,
) -> Any:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value != right_value:
        blockers.append(blocker)
    return left_value


def _seed_list(config: dict[str, Any], blockers: list[str], label: str) -> list[int]:
    seed_policy = config.get("seed_policy")
    if not isinstance(seed_policy, dict):
        blockers.append(f"{label}_seed_policy_missing")
        return []
    if seed_policy.get("mode") != "fixed-list":
        blockers.append(f"{label}_seed_policy_not_fixed_list")
    seeds = seed_policy.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        blockers.append(f"{label}_seeds_missing")
        return []
    return [int(seed) for seed in seeds]


def _single_planner(config: dict[str, Any], blockers: list[str], label: str) -> dict[str, Any]:
    planners = config.get("planners")
    if not isinstance(planners, list) or len(planners) != 1 or not isinstance(planners[0], dict):
        blockers.append(f"{label}_must_define_one_planner")
        return {}
    return planners[0]


def _require_planner_role(
    planner: dict[str, Any],
    expected_role: str,
    blockers: list[str],
    label: str,
) -> None:
    if planner.get("issue_4013_role") != expected_role:
        blockers.append(f"{label}_issue_4013_role_missing")
    if planner.get("claim_boundary") != DIAGNOSTIC_BOUNDARY:
        blockers.append(f"{label}_claim_boundary_missing")


def _predictor_status(config: dict[str, Any], blockers: list[str]) -> tuple[str, str]:
    allow_untrained = bool(config.get("allow_untrained_smoke", False))
    fallback_to_cv = bool(config.get("fallback_to_constant_velocity", False))
    checkpoint_path = config.get("checkpoint_path")
    model_id = config.get("model_id")
    if fallback_to_cv:
        blockers.append("model_based_constant_velocity_fallback_enabled")
        return "constant_velocity_fallback", "fallback_enabled_not_benchmark_evidence"
    if checkpoint_path or model_id:
        return "configured_checkpoint", "native_learned_predictor_required"
    if allow_untrained:
        return "diagnostic_untrained_smoke", "diagnostic_only_not_benchmark_evidence"
    blockers.append("model_based_predictor_missing_checkpoint_or_diagnostic_mode")
    return "unavailable", "blocked"


def _claim_boundary(
    model_free_planner: dict[str, Any],
    model_based_planner: dict[str, Any],
    blockers: list[str],
) -> str:
    boundaries = {
        model_free_planner.get("claim_boundary"),
        model_based_planner.get("claim_boundary"),
    }
    if boundaries != {DIAGNOSTIC_BOUNDARY}:
        blockers.append("claim_boundary_not_diagnostic_only")
    return DIAGNOSTIC_BOUNDARY
