#!/usr/bin/env python3
"""Validate learned-local-policy eligibility metadata.

This helper implements the checklist in
``docs/context/policy_search/contracts/learned_local_policy_eligibility.md`` as
a lightweight completeness preflight. A passing result means the candidate
metadata records the required verdict inputs; it is not benchmark evidence,
adapter readiness, or a safety claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.models.registry import validate_registry_entry_benchmark_promotion

ALLOWED_VERDICTS = {
    "eligible_for_adapter",
    "eligible_for_research_only",
    "training_only_or_oracle",
    "monitor_only",
    "reject_for_benchmark",
}

ELIGIBLE_VERDICTS = {"eligible_for_adapter", "eligible_for_research_only"}

ACTION_FAMILY_REQUIRED_FIELDS = {
    "velocity_command": {
        "frame",
        "units",
        "bounds",
        "kinematics_compatibility",
        "projection_policy",
    },
    "bounded_residual_command": {
        "base_planner",
        "residual_bounds",
        "clamp_projection_order",
        "fallback_behavior",
    },
    "waypoint_or_subgoal": {
        "frame",
        "horizon",
        "goal_tolerance",
        "local_planner",
        "timeout_behavior",
    },
    "short_trajectory": {
        "horizon",
        "timestep",
        "frame",
        "feasibility_projection",
        "collision_handling",
        "first_action_extraction",
    },
    "motion_primitive_scores": {
        "primitive_library",
        "score_normalization",
        "tie_breaking",
        "selected_primitive_execution",
        "fallback",
    },
}

OBSERVATION_CLASSIFICATIONS = {
    "deployment_observable",
    "training_only",
    "forbidden_evaluation_time",
}

PROVENANCE_FIELDS = {
    "training_data_source",
    "validation_split",
    "test_split",
    "checkpoint_or_model_provenance",
    "privileged_training_inputs",
    "privileged_training_inputs_enter_evaluation",
    "normalization_statistics",
    "normalization_statistics_fit_on_training_only",
    "evidence_source",
}

LOGGING_FIELDS = {
    "raw_model_action",
    "adapted_action",
    "post_guard_action",
    "guard_applied",
    "guard_or_fallback_reason",
    "observation_level",
    "planner_observation_mode",
    "action_bounds",
    "action_projection_metadata",
}


@dataclass(frozen=True)
class EligibilityIssue:
    """One checklist validation issue."""

    path: str
    message: str


def _is_missing(value: Any) -> bool:
    """Return whether a YAML value is absent for checklist purposes."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a mapping, otherwise an empty mapping."""
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list, otherwise an empty list."""
    return list(value) if isinstance(value, list) else []


def _require_mapping(
    issues: list[EligibilityIssue],
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Require a top-level mapping and return it when present."""
    value = payload.get(key)
    if not isinstance(value, dict):
        issues.append(EligibilityIssue(key, "must be a non-empty mapping"))
        return {}
    if not value:
        issues.append(EligibilityIssue(key, "must be a non-empty mapping"))
    return dict(value)


def _require_fields(
    issues: list[EligibilityIssue],
    mapping: dict[str, Any],
    *,
    prefix: str,
    fields: set[str],
) -> None:
    """Require non-empty values for all fields in ``mapping``."""
    for field in sorted(fields):
        if _is_missing(mapping.get(field)):
            issues.append(EligibilityIssue(f"{prefix}.{field}", "is required"))


def _validate_verdict(payload: dict[str, Any], issues: list[EligibilityIssue]) -> str | None:
    """Validate and return the top-level verdict."""
    verdict = payload.get("verdict")
    if not isinstance(verdict, str) or verdict not in ALLOWED_VERDICTS:
        issues.append(
            EligibilityIssue(
                "verdict",
                f"must be one of {', '.join(sorted(ALLOWED_VERDICTS))}",
            )
        )
        return None
    return verdict


def _validate_observation_gate(
    payload: dict[str, Any],
    *,
    verdict: str | None,
    issues: list[EligibilityIssue],
) -> None:
    """Validate observation timing and input-field classifications."""
    if _is_missing(payload.get("observation_t")):
        issues.append(EligibilityIssue("observation_t", "is required"))

    observation_fields = _require_mapping(issues, payload, "observation_fields")
    for classification in sorted(OBSERVATION_CLASSIFICATIONS):
        if classification not in observation_fields:
            issues.append(
                EligibilityIssue(
                    f"observation_fields.{classification}",
                    "must be present, even when the list is empty",
                )
            )
        elif not isinstance(observation_fields[classification], list):
            issues.append(
                EligibilityIssue(
                    f"observation_fields.{classification}",
                    "must be a list",
                )
            )

    deployment_fields = _as_list(observation_fields.get("deployment_observable"))
    forbidden_fields = _as_list(observation_fields.get("forbidden_evaluation_time"))
    if not deployment_fields:
        issues.append(
            EligibilityIssue(
                "observation_fields.deployment_observable",
                "must list at least one deployment-time input field",
            )
        )
    if forbidden_fields and verdict in ELIGIBLE_VERDICTS:
        issues.append(
            EligibilityIssue(
                "observation_fields.forbidden_evaluation_time",
                "eligible verdicts cannot require forbidden evaluation-time fields",
            )
        )


def _validate_split_provenance(
    payload: dict[str, Any],
    *,
    verdict: str | None,
    issues: list[EligibilityIssue],
) -> None:
    """Validate split, provenance, and leakage metadata."""
    split_provenance = _require_mapping(issues, payload, "split_provenance")
    _require_fields(
        issues,
        split_provenance,
        prefix="split_provenance",
        fields=PROVENANCE_FIELDS,
    )
    if (
        split_provenance.get("privileged_training_inputs_enter_evaluation") is True
        and verdict in ELIGIBLE_VERDICTS
    ):
        issues.append(
            EligibilityIssue(
                "split_provenance.privileged_training_inputs_enter_evaluation",
                "eligible verdicts must keep privileged training inputs out of evaluation",
            )
        )
    if (
        split_provenance.get("normalization_statistics_fit_on_training_only") is False
        and verdict in ELIGIBLE_VERDICTS
    ):
        issues.append(
            EligibilityIssue(
                "split_provenance.normalization_statistics_fit_on_training_only",
                "eligible verdicts require normalization statistics fit on training data only",
            )
        )


def _validate_action_contract(payload: dict[str, Any], issues: list[EligibilityIssue]) -> None:
    """Validate the selected action-output family and adapter metadata."""
    action_contract = _require_mapping(issues, payload, "action_contract")
    output_family = action_contract.get("output_family")
    if output_family not in ACTION_FAMILY_REQUIRED_FIELDS:
        issues.append(
            EligibilityIssue(
                "action_contract.output_family",
                f"must be one of {', '.join(sorted(ACTION_FAMILY_REQUIRED_FIELDS))}",
            )
        )
    else:
        _require_fields(
            issues,
            action_contract,
            prefix="action_contract",
            fields=ACTION_FAMILY_REQUIRED_FIELDS[output_family],
        )
    _require_fields(
        issues,
        action_contract,
        prefix="action_contract",
        fields={"raw_to_robot_sf_action", "guard_or_projection_policy"},
    )


def _validate_logging(payload: dict[str, Any], issues: list[EligibilityIssue]) -> None:
    """Validate required per-step learned-policy diagnostics."""
    logging = _require_mapping(issues, payload, "per_step_logging")
    _require_fields(
        issues,
        logging,
        prefix="per_step_logging",
        fields=LOGGING_FIELDS,
    )


def _validate_registry_boundary(payload: dict[str, Any], issues: list[EligibilityIssue]) -> None:
    """Validate registry-readiness fields when a registry entry is planned."""
    registry = _as_mapping(payload.get("candidate_registry"))
    if registry.get("entry_planned") is True:
        _require_fields(
            issues,
            registry,
            prefix="candidate_registry",
            fields={
                "candidate_config_path",
                "adapter_path",
                "smoke_or_validation_command",
                "missing_checkpoint_policy",
                "unsupported_observation_policy",
                "guard_activation_policy",
            },
        )


def _validate_benchmark_promotion(payload: dict[str, Any], issues: list[EligibilityIssue]) -> None:
    """Validate optional benchmark-promotion observation-track metadata."""
    if "benchmark_promotion" not in payload:
        return
    registry_like_entry = {
        "model_id": payload.get("model_id", "candidate"),
        "tags": ["learned-policy", "promoted"],
        "benchmark_promotion": payload.get("benchmark_promotion"),
    }
    issues.extend(
        EligibilityIssue(issue.path, issue.message)
        for issue in validate_registry_entry_benchmark_promotion(registry_like_entry)
    )


def validate_learned_policy_eligibility(payload: dict[str, Any]) -> list[EligibilityIssue]:
    """Return checklist-completeness and consistency issues for one candidate spec."""
    issues: list[EligibilityIssue] = []
    verdict = _validate_verdict(payload, issues)
    _validate_observation_gate(payload, verdict=verdict, issues=issues)
    _validate_split_provenance(payload, verdict=verdict, issues=issues)
    _validate_action_contract(payload, issues)
    _validate_logging(payload, issues)
    _validate_registry_boundary(payload, issues)
    _validate_benchmark_promotion(payload, issues)
    return issues


def load_candidate_spec(path: Path) -> dict[str, Any]:
    """Load one YAML or JSON learned-policy eligibility spec."""
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Error loading {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Error parsing {path} at line {exc.lineno}: {exc.msg}") from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line_text = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"Error parsing {path}{line_text}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="+", type=Path, help="YAML/JSON eligibility spec files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text",
    )
    return parser.parse_args()


def main() -> int:
    """Run the learned-policy eligibility preflight."""
    args = parse_args()
    results: dict[str, list[dict[str, str]]] = {}

    for spec_path in args.spec:
        issues = validate_learned_policy_eligibility(load_candidate_spec(spec_path))
        results[str(spec_path)] = [
            {"path": issue.path, "message": issue.message} for issue in issues
        ]

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for spec_path, issues in results.items():
            if not issues:
                print(f"{spec_path}: PASS")
                continue
            print(f"{spec_path}: FAIL")
            for issue in issues:
                print(f"  - {issue['path']}: {issue['message']}")

    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
