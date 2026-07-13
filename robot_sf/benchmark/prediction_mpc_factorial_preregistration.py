"""Pre-registration harness for issue #5355 prediction-MPC 2x2 factorial.

The harness is deliberately CPU-only and no-submit: it validates a tracked
pre-registration config, checks the factorial arms through the config builder,
and emits deterministic planned rows. It does not execute benchmark episodes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from robot_sf.planner.prediction_mpc import (
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

SCHEMA_VERSION = "robot_sf.issue_5355_prediction_mpc_factorial_preregistration.v1"
EXPECTED_FACTORIAL_ARMS = ("A0_B0", "A0_B1", "A1_B0", "A1_B1")
PAIRING_KEY_FIELDS = ("scenario_id", "seed")

# Dependency-status tokens that count as resolved for the campaign-readiness gate.
# A declared blocking dependency must reach one of these before GPU submission is
# authorized; anything else (e.g. ``open``, ``in_progress``, blank) keeps the gate
# fail-closed. See ``docs/context/issue_5355_factorial_preregistration.md`` §6.
RESOLVED_DEPENDENCY_STATES = frozenset(
    {"resolved", "closed", "merged", "done", "landed", "complete", "satisfied"}
)

# The evidence registry that pins the exact preregistration config by sha256,
# relative to the repository root (prereg §6, "The campaign config lands with
# sha256 in the evidence registry").
DEFAULT_REGISTRY_RELATIVE_PATH = (
    "docs/context/evidence/issue_5355_prediction_mpc_factorial_preregistration"
    "/preregistration_config_registry.json"
)


def load_factorial_preregistration_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an issue #5355 factorial pre-registration config.

    Returns:
        Validated configuration payload.
    """

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"factorial pre-registration config must be mapping: {config_path}")
    return validate_factorial_preregistration_config(payload, config_path=config_path)


def validate_factorial_preregistration_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the no-submit prediction-MPC 2x2 factorial contract.

    Returns:
        Shallow-normalized configuration payload.
    """

    normalized = dict(config)
    if normalized.get("schema_version") != "prediction-mpc-factorial.v1":
        raise ValueError("schema_version must be 'prediction-mpc-factorial.v1'")
    if int(normalized.get("issue", 0)) != 5355:
        raise ValueError("issue must be 5355")
    _reject_transient_routing_state(normalized)
    _validate_factorial_arms(normalized.get("factorial_arms"), config_path=config_path)
    _validate_seed_policy(normalized.get("seed_policy"))
    _validate_seed_budget(normalized.get("seed_budget"))
    _validate_scenario_provenance(normalized, config_path=config_path)
    return normalized


def build_preregistration_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic planned rows for the 2x2 factorial.

    Returns:
        Planned-row packet with pair-check summary.
    """

    validated = validate_factorial_preregistration_config(config)
    arms = _normalized_arms(validated["factorial_arms"])
    seed_policy = validated["seed_policy"]
    seed_sets_path = seed_policy.get("seed_sets_path", "configs/benchmarks/seed_sets_v1.yaml")
    seed_set_name = seed_policy.get("seed_set", "eval")
    seeds = _load_seeds(seed_sets_path, seed_set_name)
    scenario_matrix = str(validated.get("scenario_matrix", ""))
    scenario_ids = _load_scenario_ids(scenario_matrix) if scenario_matrix else []

    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        for seed in seeds:
            for arm in arms:
                rows.append(
                    {
                        "study_id": str(validated["study_id"]),
                        "factorial_arm": str(arm["key"]),
                        "factor_a_prediction": bool(arm["factor_a_prediction"]),
                        "factor_b_constraints": bool(arm["factor_b_constraints"]),
                        "algo_config": str(arm["algo_config"]),
                        "scenario_matrix": scenario_matrix,
                        "scenario_id": str(scenario_id),
                        "seed": int(seed),
                    }
                )

    pair_check = check_planned_rows(rows)
    return {
        "schema_version": "robot_sf.issue_5355_prediction_mpc_factorial_plan.v1",
        "source_schema_version": SCHEMA_VERSION,
        "issue": 5355,
        "study_id": str(validated["study_id"]),
        "status": "planned_rows_cpu_smoke_only",
        "benchmark_evidence": False,
        "claim_boundary": str(validated.get("claim_boundary", "")),
        "pairing_key_fields": list(PAIRING_KEY_FIELDS),
        "row_count": len(rows),
        "pair_check": pair_check,
        "rows": rows,
    }


def check_planned_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check that each planned pair differs only by factorial arm.

    Returns:
        Pair-completeness report.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [field for field in (*PAIRING_KEY_FIELDS, "factorial_arm") if field not in row]
        if missing:
            invalid_rows.append({"row_index": index, "fields": missing})
            continue
        arm = str(row["factorial_arm"])
        key = tuple(row[field] for field in PAIRING_KEY_FIELDS)
        grouped.setdefault(key, {}).setdefault(arm, []).append(row)

    incomplete_pairs: list[dict[str, Any]] = []
    for key, by_arm in grouped.items():
        if set(by_arm) != set(EXPECTED_FACTORIAL_ARMS) or any(
            len(arm_rows) != 1 for arm_rows in by_arm.values()
        ):
            incomplete_pairs.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "factorial_arms": sorted(by_arm),
                }
            )

    complete = bool(rows) and not invalid_rows and not incomplete_pairs
    return {
        "complete": complete,
        "row_count": len(rows),
        "pair_count": len(grouped),
        "expected_factorial_arms": list(EXPECTED_FACTORIAL_ARMS),
        "invalid_rows": invalid_rows,
        "incomplete_pairs": incomplete_pairs,
    }


def validate_arm_configs(config_path: str | Path) -> dict[str, Any]:
    """Validate that all four factorial arm algo configs build valid adapters.

    Returns:
        Per-arm validation results.
    """

    config = load_factorial_preregistration_config(config_path)
    results: dict[str, Any] = {}
    for arm in config.get("factorial_arms", []):
        key = str(arm.get("key", ""))
        algo_config_path = Path(arm.get("algo_config", ""))
        if not algo_config_path.exists():
            results[key] = {"valid": False, "error": f"algo_config not found: {algo_config_path}"}
            continue
        try:
            algo_cfg = yaml.safe_load(algo_config_path.read_text(encoding="utf-8")) or {}
            pc_config = build_prediction_mpc_config(algo_cfg)
            expected = {
                "A0_B0": ("none", False, 4.5),
                "A0_B1": ("none", True, 0.0),
                "A1_B0": ("constant_velocity", False, 4.5),
                "A1_B1": ("constant_velocity", True, 0.0),
            }.get(key)
            actual = (
                pc_config.predictor_backend,
                pc_config.hard_pedestrian_constraints_enabled,
                pc_config.pedestrian_clearance_weight,
            )
            if expected is None or actual != expected:
                raise ValueError(
                    f"{key} must be {expected!r} for "
                    "(predictor_backend, hard_constraints, pedestrian_clearance_weight); "
                    f"got {actual!r}"
                )
            adapter = PredictionMPCPlannerAdapter(config=pc_config)
            diag = adapter.diagnostics()
            results[key] = {
                "valid": True,
                "predictor_backend": pc_config.predictor_backend,
                "hard_pedestrian_constraints_enabled": pc_config.hard_pedestrian_constraints_enabled,
                "pedestrian_clearance_weight": pc_config.pedestrian_clearance_weight,
                "factorial_toggles": diag.get("factorial_toggles", {}),
            }
        except (ValueError, TypeError, KeyError, OSError) as exc:
            results[key] = {"valid": False, "error": str(exc)}
    return results


def write_preregistration_plan(plan: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic planned-row JSON artifact.

    Returns:
        Path to the written JSON plan.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "issue_5355_prediction_mpc_factorial_preregistration_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def dependency_blockers(dependencies: Any) -> list[str]:
    """Return human-readable blockers for unresolved *blocking* dependencies.

    A dependency entry blocks the campaign when it declares a truthy ``blocking``
    reason and its ``status`` is not in :data:`RESOLVED_DEPENDENCY_STATES`. The
    ``dependencies`` block is optional; a missing or empty block yields no
    blockers (nothing is declared to wait on).

    Returns:
        Ordered blocker strings, one per unresolved blocking dependency.
    """

    if dependencies in (None, ""):
        return []
    if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
        raise ValueError("dependencies must be a list of mappings")

    blockers: list[str] = []
    for entry in dependencies:
        if not isinstance(entry, Mapping):
            raise ValueError("each dependencies entry must be a mapping")
        blocking_value = entry.get("blocking")
        blocking_reason = str(blocking_value).strip() if blocking_value is not None else ""
        if not blocking_reason:
            continue
        status_value = entry.get("status")
        status = str(status_value).strip().lower() if status_value is not None else ""
        if status in RESOLVED_DEPENDENCY_STATES:
            continue
        issue = _coerce_optional_issue_id(entry.get("issue"))
        issue_label = issue if issue is not None else "?"
        blockers.append(
            f"dependency #{issue_label} unresolved (status={status or 'unset'!s}): {blocking_reason}"
        )
    return blockers


# Canonical issue-state truth for the #5355 factorial dependencies. This is the
# reconciliation source of truth from #5483: #5353 (matched-capability fairness
# contract, delivered by #5370) is resolved. #5351 remains an open blocker until
# the required hierarchical analysis input exists. Any dependency entry declaring
# #5353 with a non-resolved status is drifted and must fail the gate.
RECONCILED_CLOSED_DEPENDENCY_ISSUES = frozenset({5353})


def _coerce_optional_issue_id(value: Any) -> int | None:
    """Normalize an optional dependency issue identifier or fail closed.

    Returns:
        Positive integer issue ID, or ``None`` when the optional value is absent.
    """

    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("dependency issue must be a positive integer")
    try:
        issue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("dependency issue must be a positive integer") from exc
    if issue < 1:
        raise ValueError("dependency issue must be a positive integer")
    return issue


def check_dependency_state_consistency(dependencies: Any) -> dict[str, Any]:
    """Detect dependency-status drift against the reconciled-closed truth.

    Returns:
        A report with ``consistent`` bool, the ``closed_issues`` reconciled set,
        and an ``inconsistent`` list naming any declared dependency whose issue
        is reconciled-closed but whose declared ``status`` is not a resolved
        token. Used by the #5483 regression guard so a future edit cannot silently
        reset a closed dependency back to ``open``.
    """

    report: dict[str, Any] = {
        "consistent": True,
        "closed_issues": sorted(RECONCILED_CLOSED_DEPENDENCY_ISSUES),
        "inconsistent": [],
    }
    if dependencies in (None, ""):
        return report
    if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
        raise ValueError("dependencies must be a list of mappings")
    for entry in dependencies:
        if not isinstance(entry, Mapping):
            raise ValueError("each dependencies entry must be a mapping")
        issue = _coerce_optional_issue_id(entry.get("issue"))
        if issue is None or issue not in RECONCILED_CLOSED_DEPENDENCY_ISSUES:
            continue
        status_value = entry.get("status")
        status = str(status_value).strip().lower() if status_value is not None else ""
        if status not in RESOLVED_DEPENDENCY_STATES:
            report["consistent"] = False
            report["inconsistent"].append(
                f"#{issue} reconciled-resolved but declared status={status or 'unset'!s}"
            )
    return report


def _readiness_dependency_blockers(dependencies: Any) -> list[str]:
    """Combine declared dependency blockers with reconciled-state drift checks.

    Returns:
        Ordered blocker messages, including any reconciled-state drift.
    """

    blockers = dependency_blockers(dependencies)
    consistency = check_dependency_state_consistency(dependencies)
    if not consistency["consistent"]:
        blockers.extend(f"dependency state drift: {item}" for item in consistency["inconsistent"])
    return blockers


def assess_campaign_readiness(
    config_path: str | Path,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed CPU readiness gate for the issue #5355 factorial campaign.

    Aggregates the CPU-checkable acceptance criteria the preregistration
    (``docs/context/issue_5355_factorial_preregistration.md`` §6) requires
    *before* any GPU submission into a single verdict. This function never runs
    benchmark episodes, touches GPU/Slurm, or promotes any benchmark/paper
    claim; it only inspects tracked config and evidence artifacts.

    Every criterion is treated as ``blocked`` unless it can be positively
    verified, so an unreadable config, a stale registry digest, an invalid arm
    config, or an unresolved declared dependency each keep ``ready`` False. This
    encodes prereg §6 in code: declared blocking dependencies are checked, and
    the reconciled #5353 state is also checked for metadata drift before a ready
    verdict can be returned.

    Args:
        config_path: Path to the tracked factorial preregistration config.
        registry_path: Optional override for the sha256 evidence registry. When
            omitted, the default evidence registry is resolved relative to the
            config's repository root.

    Returns:
        Readiness report: ``ready`` bool, per-criterion status mapping, the
        ordered ``blockers`` list, and a ``claim_boundary`` string.
    """

    config_path = Path(config_path)
    criteria: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []

    def _record(name: str, ok: bool, detail: str) -> None:
        criteria[name] = {"ready": bool(ok), "detail": detail}
        if not ok:
            blockers.append(f"{name}: {detail}")

    # Criterion 1: the preregistration config parses and satisfies the contract.
    config: dict[str, Any] | None = None
    try:
        config = load_factorial_preregistration_config(config_path)
        _record("preregistration_config_valid", True, f"validated {config_path}")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        _record("preregistration_config_valid", False, str(exc))

    # Criterion 2: all four arm configs build and realize the 2x2 truth table.
    if config is not None:
        try:
            arm_results = validate_arm_configs(config_path)
            invalid = {k: v.get("error") for k, v in arm_results.items() if not v.get("valid")}
            if len(arm_results) == len(EXPECTED_FACTORIAL_ARMS) and not invalid:
                _record("arm_configs_valid", True, "all four arm configs valid")
            else:
                _record("arm_configs_valid", False, f"invalid arms: {invalid or 'incomplete set'}")
        except (ValueError, OSError, yaml.YAMLError) as exc:
            _record("arm_configs_valid", False, str(exc))
    else:
        _record("arm_configs_valid", False, "skipped: preregistration config invalid")

    # Criterion 3: the exact config is pinned by sha256 in the evidence registry.
    _record("evidence_registry_pinned", *_check_registry_pinned(config_path, registry_path))

    # Criterion 4: every declared blocking dependency is resolved and the
    # reconciled #5353 state has not drifted.
    if config is not None:
        try:
            dep_blockers = _readiness_dependency_blockers(config.get("dependencies"))
            if dep_blockers:
                _record("dependencies_resolved", False, "; ".join(dep_blockers))
            else:
                _record("dependencies_resolved", True, "no unresolved blocking dependencies")
        except ValueError as exc:
            _record("dependencies_resolved", False, str(exc))
    else:
        _record("dependencies_resolved", False, "skipped: preregistration config invalid")

    return {
        "schema_version": "robot_sf.issue_5355_prediction_mpc_factorial_readiness.v1",
        "issue": 5355,
        "config_path": str(config_path),
        "ready": not blockers,
        "criteria": criteria,
        "blockers": blockers,
        "claim_boundary": (
            "CPU readiness gate only; no benchmark, paper, or release claim and no "
            "GPU/Slurm submission is authorized by a ready verdict."
        ),
    }


def _check_registry_pinned(config_path: Path, registry_path: str | Path | None) -> tuple[bool, str]:
    """Check that the sha256 evidence registry pins the exact config bytes.

    Returns:
        ``(ok, detail)`` where ``ok`` is True only when the registry exists,
        pins the given config path, and its digest matches the config bytes.
    """

    if registry_path is None:
        resolved = _resolve_existing_path(DEFAULT_REGISTRY_RELATIVE_PATH, config_path=config_path)
    else:
        resolved = Path(registry_path)
    if not resolved.is_file():
        return False, f"evidence registry not found: {resolved}"
    try:
        registry = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable evidence registry: {exc}"
    if not isinstance(registry, dict):
        return False, "evidence registry must be a JSON object/dictionary"
    pinned_config = _resolve_existing_path(
        str(registry.get("config_path", "")), config_path=resolved
    )
    if not pinned_config.is_file():
        return False, f"registry config_path missing: {registry.get('config_path')}"
    actual = hashlib.sha256(pinned_config.read_bytes()).hexdigest()
    if actual != str(registry.get("config_sha256", "")):
        return False, "registry config_sha256 does not match config bytes"
    if pinned_config.resolve() != config_path.resolve():
        return False, f"registry pins a different config: {pinned_config}"
    return True, f"config pinned by sha256 in {resolved}"


def _validate_factorial_arms(value: Any, *, config_path: str | Path | None) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("factorial_arms must be list")
    if not all(isinstance(arm, Mapping) for arm in value):
        raise ValueError("each factorial_arms entry must be mapping")
    by_key = {str(arm.get("key")): arm for arm in value}
    if tuple(by_key) != EXPECTED_FACTORIAL_ARMS:
        raise ValueError(f"factorial_arms must be ordered {list(EXPECTED_FACTORIAL_ARMS)!r}")
    baselines = [key for key, arm in by_key.items() if bool(arm.get("baseline", False))]
    if baselines != ["A0_B0"]:
        raise ValueError("A0_B0 must be the only baseline arm")
    for key, arm in by_key.items():
        if "algo_config" not in arm or not str(arm["algo_config"]).strip():
            raise ValueError(f"{key}.algo_config must be non-empty string")
        if config_path is not None:
            algo_path = _resolve_existing_path(
                str(arm["algo_config"]), config_path=Path(config_path)
            )
            if not algo_path.exists():
                raise ValueError(f"{key}.algo_config path does not exist: {algo_path}")


def _validate_seed_policy(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("seed_policy must be mapping")
    if value.get("mode") != "seed-set" or value.get("seed_set") != "paper_eval_s30":
        raise ValueError("seed_policy must select the preregistered paper_eval_s30 seed set")


def _validate_seed_budget(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("seed_budget must be mapping")
    if value.get("mode") != "paired" or value.get("seed_set") != "paper_eval_s30":
        raise ValueError("seed_budget must use paired paper_eval_s30")
    if int(value.get("seeds_per_arm", 0)) != 30:
        raise ValueError("seed_budget.seeds_per_arm must be 30")


def _validate_scenario_provenance(
    config: Mapping[str, Any], *, config_path: str | Path | None
) -> None:
    expected_digest = str(config.get("scenario_matrix_sha256", ""))
    if len(expected_digest) != 64 or any(
        char not in "0123456789abcdef" for char in expected_digest
    ):
        raise ValueError("scenario_matrix_sha256 must be a lowercase SHA-256 digest")
    matrix_path = str(config.get("scenario_matrix", ""))
    if not matrix_path:
        raise ValueError("scenario_matrix must be set")
    if config_path is None:
        return
    resolved = _resolve_existing_path(matrix_path, config_path=Path(config_path))
    if not resolved.is_file():
        raise ValueError(f"scenario_matrix path does not exist: {resolved}")
    actual_digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("scenario_matrix_sha256 does not match scenario_matrix")


def _normalized_arms(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for arm in arms:
        normalized.append(
            {
                "key": str(arm["key"]),
                "baseline": bool(arm.get("baseline", False)),
                "factor_a_prediction": bool(arm.get("factor_a_prediction", False)),
                "factor_b_constraints": bool(arm.get("factor_b_constraints", False)),
                "algo_config": str(arm["algo_config"]),
            }
        )
    return normalized


def _load_seeds(seed_sets_path: str, seed_set_name: str) -> list[int]:
    path = Path(seed_sets_path)
    if not path.exists():
        return [111, 112, 113]
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    seeds = payload.get(seed_set_name, [111, 112, 113])
    return [int(s) for s in seeds]


def _load_scenario_ids(scenario_matrix_path: str) -> list[str]:
    path = Path(scenario_matrix_path)
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = payload.get("scenarios", [])
    if isinstance(scenarios, list):
        return [str(s.get("id", s.get("name", f"scenario_{i}"))) for i, s in enumerate(scenarios)]
    return []


def _resolve_existing_path(path_text: str, *, config_path: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    for parent in config_path.resolve().parents:
        parent_candidate = parent / candidate
        if parent_candidate.exists():
            return parent_candidate
    return cwd_candidate


def _reject_transient_routing_state(config: Mapping[str, Any]) -> None:
    forbidden_keys = {"target_host", "packet_lineage", "queue_route", "submit_host"}
    found = sorted(forbidden_keys & set(config))
    if found:
        raise ValueError(f"transient queue-routing state is forbidden in tracked config: {found}")


__all__ = [
    "DEFAULT_REGISTRY_RELATIVE_PATH",
    "EXPECTED_FACTORIAL_ARMS",
    "PAIRING_KEY_FIELDS",
    "RESOLVED_DEPENDENCY_STATES",
    "SCHEMA_VERSION",
    "assess_campaign_readiness",
    "build_preregistration_plan",
    "check_dependency_state_consistency",
    "check_planned_rows",
    "dependency_blockers",
    "load_factorial_preregistration_config",
    "validate_arm_configs",
    "validate_factorial_preregistration_config",
    "write_preregistration_plan",
]
