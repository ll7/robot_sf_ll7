#!/usr/bin/env python3
"""Portable cross-host conformance capsule and comparator (issue #8914).

Runs one bounded public workflow (fixed scenario, planner, seed, short horizon)
on two hosts, emits sanitized receipts, and compares structure, identities,
statuses, and declared numeric tolerances.  Fields are classified as ``exact``,
``tolerance``, ``set`` (order-independent), ``unavailable``, or
``informational``.  The comparator returns exactly one of ``conformant``,
``structural_mismatch``, ``identity_mismatch``, ``numeric_out_of_tolerance``,
``environment_incompatible``, or ``unknown``; hidden fallback or degraded
execution always fails the comparison.

Composition: fixed-episode execution, thread pinning, and identity checks reuse
``run_per_context_determinism_smoke.run_single_episode_trace``; hidden
fallback/degraded detection reuses ``runtime_fallback_or_degraded_marker``;
host aliases and sanitization reuse ``compare_execution_environments``.

Claim boundary: conformance is an engineering smoke between receipts of the same
declared capsule.  It does not establish statistical repeatability, simulator
validity, planner quality, scientific reproducibility, or bitwise cross-platform
determinism.  The step-trace checksum is informational because bitwise
cross-platform equality is not claimed.

Usage: ``capsule.py run --config <spec.json> --host-alias host_a --output <out>``
and ``capsule.py compare <receipt_a.json> <receipt_b.json> --json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.validation.compare_execution_environments import HOST_LABEL_RE, sanitize_text
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from compare_execution_environments import HOST_LABEL_RE, sanitize_text

CAPSULE_SCHEMA_VERSION = "cross_host_conformance_capsule.v1"
RECEIPT_SCHEMA_VERSION = "cross_host_conformance_receipt.v1"
REPORT_SCHEMA_VERSION = "cross_host_conformance_report.v1"
FIELD_CLASSES = ("exact", "tolerance", "set", "unavailable", "informational")
STATUS_PRECEDENCE = (
    "environment_incompatible identity_mismatch structural_mismatch "
    "numeric_out_of_tolerance unknown conformant"
).split()
CLAIM_BOUNDARY = (
    "Conformance is an engineering smoke between receipts of the same declared capsule; it "
    "does not establish statistical repeatability, simulator validity, planner quality, "
    "scientific reproducibility, or bitwise cross-platform determinism."
)
REQUIRED_CONTROLS = (
    ("pin_thread_env", True),
    ("workers", 1),
    ("resume", False),
    ("record_simulation_step_trace", True),
)
SPEC_KEYS = (
    "schema_version capsule_id source_commit scenario planner seed horizon "
    "deterministic_controls numeric_policy expected_structure"
).split()
SCENARIO_KEYS = ("path", "id", "schema_path")
STRUCTURE_KEYS = (
    "min_step_count max_step_count required_pedestrian_id_prefix "
    "allowed_row_statuses allowed_termination_reasons"
).split()
ARCH_MAP = {"amd64": "x86_64", "arm64": "aarch64", "armv7l": "arm32", "i386": "x86", "i686": "x86"}
OS_MAP = {"linux": "linux", "darwin": "darwin", "mac": "darwin", "windows": "windows"}
FAILURE_EXCEPTIONS = (ArithmeticError, AssertionError, ImportError, IndexError, KeyError)
FAILURE_EXCEPTIONS += (OSError, RuntimeError, TypeError, ValueError)


class CapsuleContractError(ValueError):
    """Raised when a capsule spec or receipt cannot be trusted."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise CapsuleContractError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _step_list(trace: Any) -> list[Any]:
    steps = _as_dict(trace).get("steps")
    return steps if isinstance(steps, list) else []


def _decl(class_name: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {"class": class_name, "value": value, **extra}


def load_capsule_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate one versioned capsule spec JSON object."""
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapsuleContractError(f"cannot load capsule spec: {exc}") from exc
    _require(isinstance(spec, dict) and set(spec) == set(SPEC_KEYS), "capsule spec keys invalid")
    _require(spec["schema_version"] == CAPSULE_SCHEMA_VERSION, "capsule schema mismatch")
    _require(_non_empty_str(spec["capsule_id"]) and _non_empty_str(spec["planner"]), "id required")
    source = spec["source_commit"]
    _require(source is None or _non_empty_str(source), "source_commit must be null or a string")
    scenario = spec["scenario"]
    _require(isinstance(scenario, dict) and tuple(scenario) == SCENARIO_KEYS, "scenario invalid")
    _require(
        all(_non_empty_str(scenario[key]) for key in SCENARIO_KEYS), "scenario fields required"
    )
    _require(_int_value(spec["seed"]), "seed must be an integer")
    _require(
        _int_value(spec["horizon"]) and 1 <= spec["horizon"] <= 10_000,
        "horizon must be an integer between 1 and 10000",
    )
    controls = spec["deterministic_controls"]
    _require(
        isinstance(controls, dict) and all(controls.get(k) == v for k, v in REQUIRED_CONTROLS),
        "deterministic controls drifted",
    )
    _require(_valid_policy(spec["numeric_policy"]), "numeric_policy is invalid")
    _require(_valid_structure(spec["expected_structure"]), "expected_structure is invalid")
    return dict(spec)


def _valid_policy(raw: Any) -> bool:
    if not _as_dict(raw) or set(raw) != {"default_abs_tol", "default_rel_tol"}:
        return False
    return all((number := _finite_number(raw[key])) is not None and number >= 0.0 for key in raw)


def _valid_structure(raw: Any) -> bool:
    if not _as_dict(raw) or set(raw) != set(STRUCTURE_KEYS):
        return False
    low, high = raw["min_step_count"], raw["max_step_count"]
    prefix = raw["required_pedestrian_id_prefix"]
    return (
        _int_value(low)
        and _int_value(high)
        and 0 <= low <= high
        and all(isinstance(item, str) for item in raw["allowed_row_statuses"])
        and all(isinstance(item, str) for item in raw["allowed_termination_reasons"])
        and (prefix is None or _non_empty_str(prefix))
    )


def resolve_capsule(spec: dict[str, Any], source_commit: str) -> dict[str, Any]:
    """Return the resolved capsule block recorded inside a receipt."""
    return {**spec, "source_commit": source_commit}


def capsule_digest(resolved: dict[str, Any]) -> str:
    """Return the stable SHA-256 digest of a resolved capsule block."""
    return hashlib.sha256(_canonical(resolved).encode("utf-8")).hexdigest()


def collect_environment_class() -> dict[str, Any]:
    """Return a coarse, sanitized, host-identity-free environment class."""
    from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

    system = platform.system().strip().lower().split("-")[0]
    machine = platform.machine().strip().lower()
    architecture = ARCH_MAP.get(machine, machine if machine in {"x86_64", "aarch64"} else "other")
    return {
        "os_class": OS_MAP.get(system, "other"),
        "architecture_class": architecture,
        "python_implementation": platform.python_implementation(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "thread_env": dict(sorted(pin_thread_env_for_determinism().items())),
    }


def _execute_bounded_episode(spec: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

    pin_thread_env_for_determinism()
    try:
        from scripts.validation.run_per_context_determinism_smoke import run_single_episode_trace
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from run_per_context_determinism_smoke import run_single_episode_trace

    scenario = spec["scenario"]
    row, trace = run_single_episode_trace(
        scenario_path=scenario["path"],
        scenario_id=scenario["id"],
        schema_path=scenario["schema_path"],
        planner=spec["planner"],
        seed=spec["seed"],
        horizon=spec["horizon"],
    )
    return str(row.get("git_hash", "unknown")), row, trace


def _extract_execution_integrity(row: dict[str, Any]) -> dict[str, Any]:
    from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker

    metadata = _as_dict(row.get("algorithm_metadata"))
    model = _as_dict(row.get("pedestrian_model"))
    effective = _as_dict(_as_dict(row.get("integrity")).get("effective_view"))
    kinematics = _as_dict(metadata.get("planner_kinematics"))
    degraded = effective.get("degraded") is True
    reason = effective.get("degraded_reason")
    model_status = str(model.get("fallback_degraded_status", "unknown"))
    marker = runtime_fallback_or_degraded_marker(
        {
            "status": str(metadata.get("status", "unknown")),
            "execution_mode": str(
                metadata.get("execution_mode", kinematics.get("execution_mode", "unknown"))
            ),
            "fallback_used": model_status != "native",
            "fallback_count": 0 if model_status == "native" else 1,
            "degraded": degraded,
        }
    )
    markers = [f"{marker[0]}={marker[1]}"] if marker is not None else []
    markers += [f"pedestrian_model={model_status}"] if model_status != "native" else []
    markers += [f"degraded_reason={reason}"] if reason else []
    return {
        "fallback_degraded_status": "fallback" if markers else "native",
        "degraded": degraded,
        "degraded_reasons": [str(reason)] if reason else [],
        "fallback_markers": sorted(set(markers)),
    }


def _pedestrian_ids(steps: list[Any]) -> list[str]:
    pedestrians = _as_dict(steps[0]).get("pedestrians") if steps else []
    return [
        str(_as_dict(entry).get("actor_id", _as_dict(entry).get("id", "")))
        for entry in pedestrians or []
    ]


def _verify_structure_contract(spec: dict[str, Any], row: dict[str, Any], trace: Any) -> list[str]:
    expected = spec["expected_structure"]
    steps = _step_list(trace)
    violations: list[str] = []
    if not expected["min_step_count"] <= len(steps) <= expected["max_step_count"]:
        violations.append("step_count_out_of_contract")
    if str(row.get("status")) not in set(expected["allowed_row_statuses"]):
        violations.append("row_status_out_of_contract")
    if str(row.get("termination_reason")) not in set(expected["allowed_termination_reasons"]):
        violations.append("termination_reason_out_of_contract")
    prefix = expected["required_pedestrian_id_prefix"]
    if prefix and any(not item.startswith(prefix) for item in _pedestrian_ids(steps)):
        violations.append("pedestrian_id_prefix_out_of_contract")
    return violations


def _tolerance(value: Any, policy: dict[str, Any]) -> dict[str, Any]:
    number = _finite_number(value)
    if number is None:
        return _decl("unavailable", None, reason="numeric_value_not_finite")
    return _decl(
        "tolerance",
        number,
        abs_tol=float(policy["default_abs_tol"]),
        rel_tol=float(policy["default_rel_tol"]),
    )


def _build_fields(
    spec: dict[str, Any],
    row: dict[str, Any],
    trace: dict[str, Any],
    run_elapsed: float,
) -> dict[str, Any]:
    from robot_sf.benchmark.step_trace_comparator import canonical_step_trace_digest

    policy = spec["numeric_policy"]
    metadata = _as_dict(row.get("algorithm_metadata"))
    steps = _step_list(trace)
    robot = _as_dict(_as_dict(steps[-1]).get("robot")) if steps else {}
    coordinates = robot.get("position")
    coordinates = coordinates if isinstance(coordinates, list) else [None, None]
    metrics = _as_dict(row.get("metrics"))
    exact_fields = {
        "identity/source_commit": row.get("git_hash", "unknown"),
        "identity/scenario_id": row.get("scenario_id"),
        "identity/scenario_path": spec["scenario"]["path"],
        "identity/planner": metadata.get("algorithm"),
        "identity/seed": row.get("seed"),
        "identity/horizon": row.get("horizon"),
        "identity/episode_id": row.get("episode_id"),
        "identity/row_status": row.get("status"),
        "identity/termination_reason": row.get("termination_reason"),
        "identity/algo_status": metadata.get("status"),
        "structure/step_count": len(steps),
        "structure/dt": trace.get("dt"),
        "structure/trace_schema_version": trace.get("schema_version"),
        "checksums/config_hash": row.get("config_hash"),
    }
    fields = {path: _decl("exact", value) for path, value in exact_fields.items()}
    events = _as_dict(_as_dict(row.get("event_ledger")).get("exact_events"))
    fields["structure/pedestrian_ids"] = _decl("set", sorted(_pedestrian_ids(steps)))
    fields["structure/event_kinds"] = _decl(
        "set", sorted(str(key) for key, value in events.items() if value is True)
    )
    fields["structure/metric_keys"] = _decl("set", sorted(str(key) for key in metrics))
    fields["checksums/step_trace_sha256"] = _decl(
        "informational", canonical_step_trace_digest(trace)
    )
    fields.update(
        {
            "numeric/min_clearance_m": _tolerance(metrics.get("min_clearance"), policy),
            "numeric/avg_speed_m_s": _tolerance(metrics.get("avg_speed"), policy),
            "numeric/path_efficiency": _tolerance(metrics.get("path_efficiency"), policy),
            "numeric/initial_goal_distance_m": _tolerance(
                trace.get("initial_goal_distance_m"), policy
            ),
            "numeric/final_robot_x": _tolerance(coordinates[0], policy),
            "numeric/final_robot_y": _tolerance(coordinates[1], policy),
            "informational/run_wall_time_sec": _decl("informational", round(run_elapsed, 4)),
        }
    )
    return fields


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {key: _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def run_capsule(
    spec: dict[str, Any],
    host_alias: str,
) -> dict[str, Any]:
    """Execute the bounded capsule once and return one sanitized receipt."""
    _require(
        isinstance(host_alias, str) and HOST_LABEL_RE.fullmatch(host_alias) is not None,
        "host_alias must be a stable lowercase pseudonym",
    )
    _require(spec.get("schema_version") == CAPSULE_SCHEMA_VERSION, "capsule schema mismatch")
    execution: dict[str, Any] = {
        "status": "completed",
        "failure_reason": None,
        "fallback_degraded_status": "native",
        "degraded": False,
        "degraded_reasons": [],
        "fallback_markers": [],
    }
    row: dict[str, Any] | None = None
    trace: dict[str, Any] = {}
    observed_commit = "unknown"
    started = time.perf_counter()
    try:
        observed_commit, row, trace = _execute_bounded_episode(spec)
    except FAILURE_EXCEPTIONS as exc:
        execution["status"] = "failed"
        execution["failure_reason"] = f"{type(exc).__name__}: {exc}"
    run_elapsed = time.perf_counter() - started
    if row is None:
        fields = {
            "identity/scenario_id": _decl("exact", spec["scenario"]["id"]),
            "identity/scenario_path": _decl("exact", spec["scenario"]["path"]),
            "identity/planner": _decl("exact", spec["planner"]),
            "identity/seed": _decl("exact", spec["seed"]),
            "identity/horizon": _decl("exact", spec["horizon"]),
            "identity/source_commit": _decl("exact", "unknown"),
        }
    else:
        execution.update(_extract_execution_integrity(row))
        violations = _verify_structure_contract(spec, row, trace)
        if violations and execution["status"] == "completed":
            execution["status"] = "failed"
            execution["failure_reason"] = "structure_contract_violation: " + ", ".join(violations)
        elif execution["fallback_markers"] and execution["status"] == "completed":
            execution["status"] = "fallback"
        fields = _build_fields(spec, row, trace, run_elapsed)
    if spec["source_commit"] and str(spec["source_commit"]) != observed_commit:
        execution["status"] = "failed"
        execution["failure_reason"] = "source_commit_mismatch"
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "capsule_id": spec["capsule_id"],
        "capsule_digest": capsule_digest(resolve_capsule(spec, observed_commit)),
        "host_alias": host_alias,
        "environment_class": collect_environment_class(),
        "execution": execution,
        "fields": fields,
    }
    return _sanitize_value(receipt)


def _valid_receipt(receipt: Any) -> bool:
    if not isinstance(receipt, dict):
        return False
    alias = receipt.get("host_alias")
    return (
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and isinstance(alias, str)
        and HOST_LABEL_RE.fullmatch(alias) is not None
        and isinstance(receipt.get("fields"), dict)
        and _non_empty_str(receipt.get("capsule_digest"))
    )


def _issue(status: str, code: str, *, host: str | None = None, path: str | None = None, **detail):
    return {"status": status, "code": code, "host": host, "path": path, "detail": detail}


def _execution_problems(execution: Any) -> list[str]:
    if not isinstance(execution, dict):
        return ["execution_block_missing"]
    problems: list[str] = []
    status = str(execution.get("status", "")).strip().lower()
    if status != "completed":
        problems.append(f"execution_status_{status or 'missing'}")
    fallback_status = str(execution.get("fallback_degraded_status", "")).strip().lower()
    if fallback_status != "native":
        problems.append(f"fallback_degraded_status_{fallback_status or 'missing'}")
    if execution.get("degraded") is True:
        problems.append("degraded_execution")
    problems += [
        f"degraded_reason_{sanitize_text(str(item))}"
        for item in execution.get("degraded_reasons") or []
    ]
    problems += [
        f"fallback_marker_{sanitize_text(str(item))}"
        for item in execution.get("fallback_markers") or []
    ]
    if execution.get("failure_reason"):
        problems.append(f"failure_reason_{sanitize_text(str(execution['failure_reason']))}")
    return problems


def _environment_comparison(first: Any, second: Any, require_same: bool) -> dict[str, Any]:
    env_a = _as_dict(first.get("environment_class"))
    env_b = _as_dict(second.get("environment_class"))
    if not env_a or not env_b:
        return {
            "compatible": False,
            "differences": ["environment_class_unavailable"],
            "policy": "fail_closed",
        }
    differences = [
        key
        for key in sorted(set(env_a) | set(env_b))
        if _canonical(env_a.get(key)) != _canonical(env_b.get(key))
    ]
    return {
        "compatible": not (require_same and differences),
        "differences": differences,
        "policy": "required_identical" if require_same else "informational",
    }


def _tolerance_issue(path: str, decl_a: dict[str, Any], decl_b: dict[str, Any]) -> Any:
    abs_a, rel_a = _finite_number(decl_a.get("abs_tol")), _finite_number(decl_a.get("rel_tol"))
    abs_b, rel_b = _finite_number(decl_b.get("abs_tol")), _finite_number(decl_b.get("rel_tol"))
    if abs_a is None or rel_a is None or abs_b is None or rel_b is None:
        return _issue("unknown", "invalid_tolerance_policy", path=path)
    if abs_a != abs_b or rel_a != rel_b:
        return _issue("structural_mismatch", "tolerance_policy_mismatch", path=path)
    value_a, value_b = _finite_number(decl_a.get("value")), _finite_number(decl_b.get("value"))
    if value_a is None or value_b is None:
        return _issue("unknown", "non_finite_numeric_value", path=path)
    delta = abs(value_a - value_b)
    allowed = max(abs_a, rel_a * max(abs(value_a), abs(value_b)))
    if delta <= allowed:
        return None
    return _issue(
        "numeric_out_of_tolerance",
        "numeric_tolerance_exceeded",
        path=path,
        detail={"delta": delta, "allowed": allowed},
    )


def _compare_one_field(path: str, decl_a: Any, decl_b: Any) -> list[dict[str, Any]]:
    class_a, class_b = decl_a.get("class"), decl_b.get("class")
    if class_a not in FIELD_CLASSES or class_b not in FIELD_CLASSES:
        return [_issue("unknown", "invalid_field_declaration", path=path)]
    if "unavailable" in {class_a, class_b}:
        return [_issue("unknown", "field_unavailable", path=path)]
    if class_a != class_b:
        return [_issue("structural_mismatch", "field_class_mismatch", path=path)]
    class_name = class_a
    if class_name == "informational":
        return []
    if class_name in {"exact", "set"}:
        value_a, value_b = decl_a.get("value"), decl_b.get("value")
        if class_name == "set":
            if not isinstance(value_a, list) or not isinstance(value_b, list):
                return [_issue("unknown", "invalid_set_value", path=path)]
            value_a = sorted(str(item) for item in value_a)
            value_b = sorted(str(item) for item in value_b)
        if _canonical(value_a) == _canonical(value_b):
            return []
        code = "exact_value_mismatch" if class_name == "exact" else "set_value_mismatch"
        status = "identity_mismatch" if path.startswith("identity/") else "structural_mismatch"
        return [_issue(status, code, path=path)]
    issue = _tolerance_issue(path, decl_a, decl_b)
    return [] if issue is None else [issue]


def _compare_fields(first: Any, second: Any) -> list[dict[str, Any]]:
    fields_a, fields_b = first["fields"], second["fields"]
    issues: list[dict[str, Any]] = []
    for path in sorted(set(fields_a) | set(fields_b)):
        decl_a, decl_b = fields_a.get(path), fields_b.get(path)
        if decl_a is None or decl_b is None:
            declared = _as_dict(decl_a if decl_a is not None else decl_b)
            if declared.get("class") == "unavailable":
                issues.append(_issue("unknown", "field_unavailable", path=path))
            else:
                issues.append(_issue("structural_mismatch", "field_missing", path=path))
            continue
        issues += _compare_one_field(path, decl_a, decl_b)
    return issues


def _build_report(
    status: str,
    issues: list[dict[str, Any]],
    *,
    hosts: tuple[Any, Any],
    capsule_ref: dict[str, Any],
    environment: dict[str, Any],
    execution_integrity: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        issues,
        key=lambda item: _canonical([item["code"], item["host"], item["path"], item["detail"]]),
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "capsule_id": capsule_ref.get("capsule_id"),
        "capsule_digest": capsule_ref.get("capsule_digest"),
        "host_a": hosts[0],
        "host_b": hosts[1],
        "status": status,
        "reason_codes": sorted({str(issue["code"]) for issue in ordered}),
        "issues": ordered,
        "claim_boundary": CLAIM_BOUNDARY,
        "environment": environment,
        "execution_integrity": execution_integrity,
    }


def compare_receipts(
    receipt_a: Any,
    receipt_b: Any,
    *,
    require_same_environment_class: bool = False,
) -> dict[str, Any]:
    """Compare two sanitized receipts and return one deterministic report.

    Returns:
        Report payload with exactly one comparison status and stable reason codes.
    """
    empty_env = {"compatible": True, "differences": [], "policy": "skipped"}
    for label, receipt in (("host_a", receipt_a), ("host_b", receipt_b)):
        if not _valid_receipt(receipt):
            alias = receipt.get("host_alias") if isinstance(receipt, dict) else None
            issues = [_issue("unknown", "invalid_receipt", host=label)]
            return _build_report(
                "unknown",
                issues,
                hosts=(alias, None),
                capsule_ref={},
                environment=empty_env,
                execution_integrity={},
            )
    if receipt_a["host_alias"] == receipt_b["host_alias"]:
        alias = receipt_a["host_alias"]
        issues = [_issue("unknown", "duplicate_host_alias", host=alias)]
        return _build_report(
            "unknown",
            issues,
            hosts=(alias, alias),
            capsule_ref={},
            environment=empty_env,
            execution_integrity={},
        )
    first, second = sorted((receipt_a, receipt_b), key=lambda item: item["host_alias"])
    issues = [
        _issue("identity_mismatch", f"{key}_mismatch", path=key)
        for key in ("capsule_id", "capsule_digest")
        if _canonical(first.get(key)) != _canonical(second.get(key))
    ]
    execution_integrity: dict[str, Any] = {}
    for label, receipt in (("host_a", first), ("host_b", second)):
        problems = _execution_problems(receipt.get("execution"))
        execution_integrity[label] = {"problems": problems}
        issues += [
            _issue("environment_incompatible", "execution_integrity", host=label, detail=problem)
            for problem in problems
        ]
    environment = _environment_comparison(first, second, require_same_environment_class)
    if not environment["compatible"]:
        issues.append(
            _issue(
                "environment_incompatible",
                "environment_class_incompatible",
                detail={"differences": environment["differences"]},
            )
        )
    issues += _compare_fields(first, second)
    statuses = {issue["status"] for issue in issues}
    status = next((item for item in STATUS_PRECEDENCE if item in statuses), "conformant")
    return _build_report(
        status,
        issues,
        hosts=(first["host_alias"], second["host_alias"]),
        capsule_ref={
            "capsule_id": first.get("capsule_id"),
            "capsule_digest": first.get("capsule_digest"),
        },
        environment=environment,
        execution_integrity=execution_integrity,
    )


def _run_command(args: argparse.Namespace) -> int:
    try:
        spec = load_capsule_spec(args.config)
        receipt = run_capsule(spec, args.host_alias)
    except CapsuleContractError as exc:
        print(f"capsule contract error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = receipt["execution"]["status"]
    print(f"receipt written: {args.output} (execution status: {status})")
    return 0 if status == "completed" else 1


def _compare_command(args: argparse.Namespace) -> int:
    try:
        receipt_a = json.loads(args.receipt_a.read_text(encoding="utf-8"))
        receipt_b = json.loads(args.receipt_b.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "unknown",
            "reason_codes": ["malformed_receipt"],
            "detail": sanitize_text(str(exc)),
            "claim_boundary": CLAIM_BOUNDARY,
        }
    else:
        report = compare_receipts(
            receipt_a,
            receipt_b,
            require_same_environment_class=args.require_same_environment_class,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "conformant" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run the bounded capsule and write a receipt.")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--host-alias", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = sub.add_parser("compare", help="Compare two receipts.")
    compare_parser.add_argument("receipt_a", type=Path)
    compare_parser.add_argument("receipt_b", type=Path)
    compare_parser.add_argument("--json", action="store_true")
    compare_parser.add_argument("--require-same-environment-class", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the ``run`` and ``compare`` subcommands."""
    args = _parser().parse_args(argv)
    return _run_command(args) if args.command == "run" else _compare_command(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
