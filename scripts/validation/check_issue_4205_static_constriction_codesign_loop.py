#!/usr/bin/env python3
"""Validate issue #4205 static-constriction co-design-loop pre-registration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.cbf_safety_filter_runtime import (
    CBF_COLLISION_CONE_ARM,
    CBF_OFF_ARM,
)
from robot_sf.benchmark.cbf_safety_filter_runtime import (
    runtime_config_from_mapping as cbf_runtime_config_from_mapping,
)
from robot_sf.benchmark.safety_wrapper_runtime import (
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
)
from robot_sf.benchmark.safety_wrapper_runtime import (
    runtime_config_from_mapping as wrapper_runtime_config_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
REPORT_SCHEMA = "robot_sf.issue_4205_static_constriction_codesign_loop.check.v1"
SMOKE_MANIFEST_SCHEMA = "robot_sf.issue_4205.cpu_smoke_manifest.v1"
EVIDENCE_PACKET_SCHEMA = "robot_sf.issue_4205.pre_run_evidence_packet.v1"
HYDRATION_MANIFEST_SCHEMA = "robot_sf.issue_4205.frozen_ppo_checkpoint_hydration.v1"
EXPECTED_ARM_KEYS = (
    "ppo_frozen",
    "ppo_frozen_wrapper_on",
    "ppo_frozen_cbf_on",
)
EXPECTED_SCENARIOS = (
    "classic_bottleneck_low",
    "classic_head_on_corridor_low",
    "narrow_passage",
)
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_TRACE_FIELDS = {
    "low_progress_window",
    "recenter_activation_count",
    "distance_to_goal_delta",
    "local_minimum_indicator",
    "execution_mode",
    "row_status",
}
FORBIDDEN_TRANSIENT_KEYS = {"target_host", "packet_lineage", "queue_route", "submit_host"}
FAIL_CLOSED_ROW_STATUSES = {"failed", "fallback", "degraded", "not_available"}


class ContractError(ValueError):
    """Raised when the issue #4205 contract fails closed."""


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file as a mapping."""
    if not path.exists():
        raise ContractError(f"missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain YAML mapping")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON file as a mapping."""
    if not path.exists():
        raise ContractError(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain JSON mapping")
    return data


def _sha256_path(path: Path) -> str:
    """Return SHA-256 for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_file_sha256(value: object) -> str:
    """Return SHA-256 for a repo-relative file reference."""
    return _sha256_path(_repo_path(value))


def _repo_path(value: object) -> Path:
    """Resolve a repo-relative path and fail closed when it is missing."""
    if not isinstance(value, str) or not value:
        raise ContractError(f"expected non-empty repo-relative path, got {value!r}")
    path = REPO_ROOT / value
    if not path.exists():
        raise ContractError(f"referenced path does not exist: {value}")
    return path


def _find_suite(suite_config: dict[str, Any], suite_id: str) -> dict[str, Any]:
    """Return the named mechanism suite."""
    suites = suite_config.get("suites")
    if not isinstance(suites, list):
        raise ContractError("mechanism suite config must contain suites list")
    for suite in suites:
        if isinstance(suite, dict) and suite.get("suite_id") == suite_id:
            return suite
    raise ContractError(f"mechanism suite {suite_id!r} not found")


def _require_list(config: dict[str, Any], key: str) -> list[Any]:
    """Return a required list field."""
    value = config.get(key)
    if not isinstance(value, list):
        raise ContractError(f"{key} must be list")
    return value


def _find_forbidden_transient_keys(value: Any, *, prefix: str = "") -> list[str]:
    """Return tracked config keys reserved for private routing state."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRANSIENT_KEYS:
                found.append(path)
            found.extend(_find_forbidden_transient_keys(nested, prefix=path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_find_forbidden_transient_keys(nested, prefix=f"{prefix}[{index}]"))
    return found


def _validate_suite(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the static-deadlock mechanism suite binding."""
    mechanism = config.get("mechanism_suite")
    if not isinstance(mechanism, dict):
        raise ContractError("mechanism_suite must be mapping")
    if mechanism.get("suite_id") != "static_deadlock_recovery":
        raise ContractError("mechanism_suite.suite_id must static_deadlock_recovery")
    if mechanism.get("target_mechanism") != "static_deadlock_or_local_minimum":
        raise ContractError(
            "mechanism_suite.target_mechanism must static_deadlock_or_local_minimum"
        )
    suite_path = _repo_path(mechanism.get("path"))
    suite = _find_suite(_load_yaml(suite_path), "static_deadlock_recovery")

    scenarios = tuple(config.get("scenario_family", {}).get("scenario_ids", ()))
    if scenarios != EXPECTED_SCENARIOS:
        raise ContractError(f"scenario_family.scenario_ids must equal {list(EXPECTED_SCENARIOS)}")
    if tuple(suite.get("scenario_ids") or ()) != EXPECTED_SCENARIOS:
        raise ContractError("referenced static_deadlock_recovery suite scenario_ids drifted")

    seeds = tuple(int(seed) for seed in _require_list(config, "seeds"))
    if seeds != EXPECTED_SEEDS:
        raise ContractError(f"seeds must equal {list(EXPECTED_SEEDS)}")
    if tuple(int(seed) for seed in suite.get("seed_set") or ()) != EXPECTED_SEEDS:
        raise ContractError("referenced static_deadlock_recovery suite seed_set drifted")

    trace_fields = set(suite.get("required_trace_fields") or ())
    missing_trace = sorted(EXPECTED_TRACE_FIELDS - trace_fields)
    if missing_trace:
        raise ContractError(f"static_deadlock_recovery suite missing trace fields: {missing_trace}")
    return suite


def _validate_benchmark_contract(config: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Validate the paired benchmark-side no-submit contract."""
    benchmark_path = _repo_path(config.get("benchmark_contract"))
    benchmark = _load_yaml(benchmark_path)
    if benchmark.get("schema_version") != "robot_sf.issue_4205_static_constriction_benchmark.v1":
        raise ContractError("unexpected benchmark_contract schema_version")
    if benchmark.get("issue") != 4205:
        raise ContractError("benchmark_contract issue must be 4205")
    if benchmark.get("loop_id") != config.get("loop_id"):
        raise ContractError("benchmark_contract loop_id drifted")
    if benchmark.get("research_contract") != _repo_relative(DEFAULT_CONFIG):
        raise ContractError("benchmark_contract research_contract drifted")
    if tuple(benchmark.get("scenario_ids") or ()) != EXPECTED_SCENARIOS:
        raise ContractError("benchmark_contract scenario_ids drifted")
    if tuple(int(seed) for seed in benchmark.get("seeds") or ()) != EXPECTED_SEEDS:
        raise ContractError("benchmark_contract seeds drifted")
    if tuple(benchmark.get("arms") or ()) != EXPECTED_ARM_KEYS:
        raise ContractError("benchmark_contract arms drifted")

    authorization = benchmark.get("campaign_authorization")
    if not isinstance(authorization, dict):
        raise ContractError("benchmark_contract campaign_authorization must be a mapping")
    if authorization.get("compute_submit_authorized") is not False:
        raise ContractError("benchmark_contract compute_submit_authorized must stay false")

    row_status = benchmark.get("row_status_policy")
    if not isinstance(row_status, dict):
        raise ContractError("benchmark_contract row_status_policy must be a mapping")
    if row_status.get("fallback_rows_are_success_evidence") is not False:
        raise ContractError("fallback rows must not be success evidence")
    if row_status.get("degraded_rows_are_success_evidence") is not False:
        raise ContractError("degraded rows must not be success evidence")

    forbidden_keys = _find_forbidden_transient_keys(benchmark)
    if forbidden_keys:
        raise ContractError(f"benchmark_contract contains transient routing keys: {forbidden_keys}")
    return benchmark


def _validate_common(config: dict[str, Any]) -> None:  # noqa: C901, PLR0912
    """Validate top-level issue #4205 research-contract fields."""
    if config.get("schema_version") != "robot_sf.issue_4205_static_constriction_codesign_loop.v1":
        raise ContractError("unexpected schema_version")
    if config.get("issue") != 4205:
        raise ContractError("issue must be 4205")
    if config.get("loop_id") != "issue_4205_static_constriction_codesign_loop_v1":
        raise ContractError("loop_id drifted")
    if config.get("benchmark_evidence") is not False:
        raise ContractError("benchmark_evidence must stay false before private campaign")

    forbidden_keys = _find_forbidden_transient_keys(config)
    if forbidden_keys:
        raise ContractError(f"research contract contains transient routing keys: {forbidden_keys}")

    authorization = config.get("campaign_authorization")
    if not isinstance(authorization, dict):
        raise ContractError("campaign_authorization must be a mapping")
    if authorization.get("compute_submit_authorized") is not False:
        raise ContractError("compute_submit_authorized must stay false in public config")

    smoke = config.get("cpu_smoke")
    if not isinstance(smoke, dict):
        raise ContractError("cpu_smoke must be a mapping")
    _repo_path(smoke.get("matrix"))
    if tuple(int(seed) for seed in smoke.get("seeds") or ()) != (111,):
        raise ContractError("cpu_smoke.seeds must remain one-seed static-deadlock smoke [111]")

    lineage = config.get("frozen_ppo_lineage")
    if not isinstance(lineage, dict):
        raise ContractError("frozen_ppo_lineage must be a mapping")
    if lineage.get("algo") != "ppo":
        raise ContractError("frozen_ppo_lineage.algo must be ppo")
    algo_config_path = _repo_path(lineage.get("algo_config"))
    algo_config = _load_yaml(algo_config_path)
    model_id = algo_config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ContractError("frozen_ppo_lineage algo_config must declare model_id")
    if lineage.get("model_id") != model_id:
        raise ContractError("frozen_ppo_lineage.model_id must match algo_config model_id")
    algo_config_sha256 = _sha256_path(algo_config_path)
    if lineage.get("algo_config_sha256") != algo_config_sha256:
        raise ContractError("frozen_ppo_lineage.algo_config_sha256 must match algo_config")
    if lineage.get("hydration_manifest_schema") != HYDRATION_MANIFEST_SCHEMA:
        raise ContractError("frozen_ppo_lineage.hydration_manifest_schema drifted")
    required_fields = tuple(lineage.get("hydration_manifest_required_fields") or ())
    expected_fields = (
        "schema_version",
        "issue",
        "loop_id",
        "model_id",
        "algo_config",
        "algo_config_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "arms",
    )
    if required_fields != expected_fields:
        raise ContractError("frozen_ppo_lineage.hydration_manifest_required_fields drifted")
    if lineage.get("failure_job_id") != 13175:
        raise ContractError("frozen_ppo_lineage.failure_job_id must remain 13175")
    if lineage.get("model_preflight_required") is not True:
        raise ContractError("frozen_ppo_lineage.model_preflight_required must be true")
    if lineage.get("private_artifact_hydration_required") is not True:
        raise ContractError("frozen_ppo_lineage.private_artifact_hydration_required must be true")


def _validate_arms(config: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: C901
    """Validate the exact three mutually exclusive pre-registered arms."""
    lineage = config["frozen_ppo_lineage"]
    arms = _require_list(config, "arms")
    if tuple(arm.get("key") for arm in arms if isinstance(arm, dict)) != EXPECTED_ARM_KEYS:
        raise ContractError(f"arms must be exactly ordered {list(EXPECTED_ARM_KEYS)}")

    normalized: list[dict[str, Any]] = []
    for arm in arms:
        if not isinstance(arm, dict):
            raise ContractError("each arm must be a mapping")
        if arm.get("algo") != lineage["algo"]:
            raise ContractError(f"{arm.get('key')}: algo must match frozen_ppo_lineage")
        if arm.get("algo_config") != lineage["algo_config"]:
            raise ContractError(f"{arm.get('key')}: algo_config must match frozen_ppo_lineage")
        if arm.get("frozen_ppo_lineage_ref") != "frozen_ppo_lineage":
            raise ContractError(f"{arm.get('key')}: must reference frozen_ppo_lineage")

        wrapper_runtime = wrapper_runtime_config_from_mapping(arm.get("safety_wrapper"))
        cbf_runtime = cbf_runtime_config_from_mapping(arm.get("cbf_safety_filter"))
        if wrapper_runtime.enabled and cbf_runtime.enabled:
            raise ContractError(
                f"{arm['key']}: wrapper and CBF arms must remain mutually exclusive"
            )
        normalized.append(
            {
                "key": arm["key"],
                "role": arm.get("role"),
                "algo": arm["algo"],
                "algo_config": arm["algo_config"],
                "safety_wrapper": asdict(wrapper_runtime),
                "cbf_safety_filter": asdict(cbf_runtime),
            }
        )

    baseline, wrapper, cbf = normalized
    if (
        baseline["safety_wrapper"]["arm_key"] != WRAPPER_OFF_ARM
        or baseline["cbf_safety_filter"]["arm_key"] != CBF_OFF_ARM
    ):
        raise ContractError("ppo_frozen must disable wrapper and CBF")
    if (
        wrapper["safety_wrapper"]["arm_key"] != WRAPPER_ON_ARM
        or wrapper["cbf_safety_filter"]["arm_key"] != CBF_OFF_ARM
    ):
        raise ContractError("ppo_frozen_wrapper_on must enable only wrapper_on")
    if (
        cbf["safety_wrapper"]["arm_key"] != WRAPPER_OFF_ARM
        or cbf["cbf_safety_filter"]["arm_key"] != CBF_COLLISION_CONE_ARM
    ):
        raise ContractError("ppo_frozen_cbf_on must enable only cbf_collision_cone_on")
    return normalized


def _smoke_rows(config: dict[str, Any], arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the expected one-seed CPU smoke rows for each arm."""
    smoke = config["cpu_smoke"]
    scenario_id = EXPECTED_SCENARIOS[0]
    seed = int(smoke["seeds"][0])
    rows: list[dict[str, Any]] = []
    for arm in arms:
        wrapper = arm["safety_wrapper"]
        cbf = arm["cbf_safety_filter"]
        rows.append(
            {
                "arm_key": arm["key"],
                "scenario_id": scenario_id,
                "seed": seed,
                "episode_count": 1,
                "cpu_only": True,
                "row_status": "smoke_plumbing_only",
                "benchmark_evidence": False,
                "wrapper_intervention_rate": 0.0
                if not wrapper["enabled"]
                else "runtime_summary_required",
                "cbf_status_counts": {} if not cbf["enabled"] else "runtime_summary_required",
                "required_static_deadlock_trace_fields": sorted(EXPECTED_TRACE_FIELDS),
            }
        )
    return rows


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an issue #4205 pre-registration validation report."""
    _validate_common(config)
    suite = _validate_suite(config)
    benchmark = _validate_benchmark_contract(config)
    arms = _validate_arms(config)
    smoke_rows = _smoke_rows(config, arms)

    required_trace_fields = set(_require_list(config, "required_trace_fields"))
    missing_trace = sorted(
        (EXPECTED_TRACE_FIELDS | {"wrapper_intervention_rate", "cbf_status_counts"})
        - required_trace_fields
    )
    if missing_trace:
        raise ContractError(f"required_trace_fields missing: {missing_trace}")

    required_metrics = set(_require_list(config, "required_metrics"))
    if "deadlock_count" not in required_metrics:
        raise ContractError("required_metrics must include deadlock_count")

    digest = hashlib.sha256(
        (
            REPO_ROOT / "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
        ).read_bytes()
    ).hexdigest()
    lineage = config["frozen_ppo_lineage"]
    return {
        "schema_version": REPORT_SCHEMA,
        "ok": True,
        "issue": 4205,
        "loop_id": config["loop_id"],
        "config_sha256": digest,
        "suite_id": suite["suite_id"],
        "target_mechanism": suite["target_mechanism"],
        "scenario_ids": list(EXPECTED_SCENARIOS),
        "seeds": list(EXPECTED_SEEDS),
        "arm_keys": [arm["key"] for arm in arms],
        "frozen_ppo_algo_config": config["frozen_ppo_lineage"]["algo_config"],
        "frozen_ppo_checkpoint": {
            "model_id": lineage["model_id"],
            "algo_config": lineage["algo_config"],
            "algo_config_sha256": lineage["algo_config_sha256"],
            "hydration_manifest_schema": HYDRATION_MANIFEST_SCHEMA,
            "hydration_status": "blocked_missing_hydration_manifest",
            "blocker": "private checkpoint artifact must be hydrated before campaign authorization",
        },
        "benchmark_contract": benchmark["research_contract"],
        "compute_submit_authorized": False,
        "benchmark_evidence": False,
        "cpu_smoke": {
            "matrix": config["cpu_smoke"]["matrix"],
            "rows": smoke_rows,
        },
    }


def _resolve_hydrated_checkpoint_path(manifest_path: Path, value: object) -> Path:
    """Resolve private hydration checkpoint path relative to manifest location."""
    if not isinstance(value, str) or not value:
        raise ContractError("hydration checkpoint_path must be non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.exists() or not path.is_file():
        raise ContractError(f"blocked_missing_hydrated_checkpoint: {path}")
    return path


def _validate_checkpoint_hydration_manifest(  # noqa: C901, PLR0912
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate private frozen-PPO checkpoint hydration before campaign authorization."""
    expected = report["frozen_ppo_checkpoint"]
    if manifest.get("schema_version") != HYDRATION_MANIFEST_SCHEMA:
        raise ContractError(f"hydration schema_version must be {HYDRATION_MANIFEST_SCHEMA}")
    if manifest.get("issue") != 4205:
        raise ContractError("hydration issue must be 4205")
    if manifest.get("loop_id") != report["loop_id"]:
        raise ContractError("hydration loop_id drifted")
    for key in ("model_id", "algo_config", "algo_config_sha256"):
        if manifest.get(key) != expected[key]:
            raise ContractError(f"hydration {key} must match frozen_ppo_checkpoint")

    checkpoint_path = _resolve_hydrated_checkpoint_path(
        manifest_path, manifest.get("checkpoint_path")
    )
    observed_sha256 = _sha256_path(checkpoint_path)
    if manifest.get("checkpoint_sha256") != observed_sha256:
        raise ContractError("hydration checkpoint_sha256 must match hydrated checkpoint")

    arms = manifest.get("arms")
    if not isinstance(arms, list):
        raise ContractError("hydration arms must be a list")
    if [arm.get("key") for arm in arms if isinstance(arm, dict)] != list(EXPECTED_ARM_KEYS):
        raise ContractError(f"hydration arms must be ordered {list(EXPECTED_ARM_KEYS)}")

    report_rows = {row["arm_key"]: row for row in report["cpu_smoke"]["rows"]}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ContractError("each hydration arm must be a mapping")
        arm_key = arm["key"]
        for key in ("model_id", "algo_config", "checkpoint_sha256"):
            expected_value = observed_sha256 if key == "checkpoint_sha256" else expected[key]
            if arm.get(key) != expected_value:
                raise ContractError(
                    f"{arm_key}: hydration {key} must match frozen checkpoint lineage"
                )
        smoke_row = report_rows[arm_key]
        wrapper = arm.get("safety_wrapper")
        cbf = arm.get("cbf_safety_filter")
        if not isinstance(wrapper, dict) or not isinstance(cbf, dict):
            raise ContractError(f"{arm_key}: hydration must include wrapper and CBF metadata")
        wrapper_expected_enabled = arm_key == "ppo_frozen_wrapper_on"
        cbf_expected_enabled = arm_key == "ppo_frozen_cbf_on"
        if bool(wrapper.get("enabled")) != wrapper_expected_enabled:
            raise ContractError(f"{arm_key}: hydration wrapper metadata drifted")
        if bool(cbf.get("enabled")) != cbf_expected_enabled:
            raise ContractError(f"{arm_key}: hydration CBF metadata drifted")
        expected_wrapper_arm = WRAPPER_ON_ARM if wrapper_expected_enabled else WRAPPER_OFF_ARM
        expected_cbf_arm = CBF_COLLISION_CONE_ARM if cbf_expected_enabled else CBF_OFF_ARM
        if wrapper.get("arm_key") != expected_wrapper_arm:
            raise ContractError(f"{arm_key}: hydration wrapper arm_key drifted")
        if cbf.get("arm_key") != expected_cbf_arm:
            raise ContractError(f"{arm_key}: hydration CBF arm_key drifted")
        if smoke_row["arm_key"] != arm_key:
            raise ContractError(f"{arm_key}: hydration arm order drifted")

    return {
        "schema_version": HYDRATION_MANIFEST_SCHEMA,
        "status": "hydrated_checkpoint_ready",
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": observed_sha256,
        "model_id": expected["model_id"],
        "algo_config": expected["algo_config"],
        "algo_config_sha256": expected["algo_config_sha256"],
        "arms": [arm["key"] for arm in arms],
    }


def _validate_cpu_smoke_evidence(  # noqa: C901, PLR0912
    report: dict[str, Any], smoke_payload: dict[str, Any]
) -> dict[str, Any]:
    """Validate compact local CPU arm-smoke evidence without benchmark promotion."""
    if smoke_payload.get("schema_version") != SMOKE_MANIFEST_SCHEMA:
        raise ContractError(f"cpu smoke schema_version must be {SMOKE_MANIFEST_SCHEMA}")
    if smoke_payload.get("issue") != 4205:
        raise ContractError("cpu smoke issue must be 4205")
    if smoke_payload.get("loop_id") != report["loop_id"]:
        raise ContractError("cpu smoke loop_id drifted")
    if smoke_payload.get("benchmark_evidence") is not False:
        raise ContractError("cpu smoke benchmark_evidence must stay false")
    rows = smoke_payload.get("rows")
    if not isinstance(rows, list):
        raise ContractError("cpu smoke rows must be a list")
    if [row.get("arm_key") for row in rows if isinstance(row, dict)] != list(EXPECTED_ARM_KEYS):
        raise ContractError(f"cpu smoke rows must be ordered {list(EXPECTED_ARM_KEYS)}")

    checked_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ContractError("each cpu smoke row must be a mapping")
        arm_key = row["arm_key"]
        if row.get("scenario_id") != EXPECTED_SCENARIOS[0]:
            raise ContractError(f"{arm_key}: cpu smoke scenario must be {EXPECTED_SCENARIOS[0]}")
        if int(row.get("seed", -1)) != 111:
            raise ContractError(f"{arm_key}: cpu smoke seed must be 111")
        if row.get("cpu_only") is not True:
            raise ContractError(f"{arm_key}: cpu smoke must be marked cpu_only")
        if row.get("benchmark_evidence") is not False:
            raise ContractError(f"{arm_key}: cpu smoke must not be benchmark evidence")
        if row.get("row_status") in FAIL_CLOSED_ROW_STATUSES:
            raise ContractError(f"{arm_key}: cpu smoke row_status is fail-closed")
        trace_fields = set(row.get("emitted_trace_fields") or ())
        missing_trace = sorted(EXPECTED_TRACE_FIELDS - trace_fields)
        if missing_trace:
            raise ContractError(f"{arm_key}: cpu smoke missing trace fields {missing_trace}")

        wrapper_rate = row.get("wrapper_intervention_rate")
        cbf_counts = row.get("cbf_status_counts")
        if arm_key == "ppo_frozen":
            if wrapper_rate != 0.0:
                raise ContractError("ppo_frozen smoke must not report wrapper intervention")
            if cbf_counts != {}:
                raise ContractError("ppo_frozen smoke must not report CBF statuses")
        elif arm_key == "ppo_frozen_wrapper_on":
            if (
                isinstance(wrapper_rate, bool)
                or not isinstance(wrapper_rate, (int, float))
                or not math.isfinite(wrapper_rate)
            ):
                raise ContractError(
                    "ppo_frozen_wrapper_on smoke needs a finite numeric wrapper rate"
                )
            if cbf_counts != {}:
                raise ContractError("ppo_frozen_wrapper_on smoke must not report CBF statuses")
        elif arm_key == "ppo_frozen_cbf_on":
            if wrapper_rate != 0.0:
                raise ContractError("ppo_frozen_cbf_on smoke must not report wrapper intervention")
            if not isinstance(cbf_counts, dict) or not cbf_counts:
                raise ContractError("ppo_frozen_cbf_on smoke needs CBF status counts")
        checked_rows.append(row)

    return {
        "schema_version": SMOKE_MANIFEST_SCHEMA,
        "issue": 4205,
        "loop_id": report["loop_id"],
        "benchmark_evidence": False,
        "evidence_status": "cpu_arm_plumbing_smoke",
        "runtime_claim": "all pre-registered arms emitted required local smoke metadata",
        "rows": checked_rows,
    }


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text with a final newline and stable ``\\n`` line endings."""
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    """Serialize compact CSV text with stable ``\\n`` line endings."""
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return handle.getvalue()


def _write_pre_run_evidence_packet(
    *,
    evidence_dir: Path,
    report: dict[str, Any],
    smoke_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write compact pre-run evidence for review before campaign submission."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = smoke_manifest["rows"]
    metadata = {
        "schema_version": EVIDENCE_PACKET_SCHEMA,
        "issue": 4205,
        "loop_id": report["loop_id"],
        "evidence_status": "smoke evidence",
        "benchmark_evidence": False,
        "claim_boundary": (
            "CPU arm-plumbing smoke for one static-deadlock seed only; no full benchmark "
            "campaign, Slurm/GPU submission, retraining, or paper/dissertation claim."
        ),
        "scenario_id": EXPECTED_SCENARIOS[0],
        "seed": 111,
        "arms": list(EXPECTED_ARM_KEYS),
        "config_sha256": report["config_sha256"],
        "residual_blocker": (
            "Exact frozen PPO checkpoint hydration remains private-queue preflight before "
            "campaign execution."
        ),
    }
    pre_registration = {
        "schema_version": report["schema_version"],
        "research_contract": _repo_relative(DEFAULT_CONFIG),
        "benchmark_contract": report["benchmark_contract"],
        "scenario_ids": report["scenario_ids"],
        "seeds": report["seeds"],
        "arm_keys": report["arm_keys"],
        "compute_submit_authorized": False,
    }
    intervention_rows = [
        {
            "arm_key": row["arm_key"],
            "scenario_id": row["scenario_id"],
            "seed": row["seed"],
            "row_status": row["row_status"],
            "wrapper_intervention_rate": row["wrapper_intervention_rate"],
            "cbf_status_counts": json.dumps(row["cbf_status_counts"], sort_keys=True),
            "benchmark_evidence": row["benchmark_evidence"],
        }
        for row in rows
    ]
    failure_rows = [
        {
            "arm_key": row["arm_key"],
            "low_progress_window": "present",
            "recenter_activation_count": "present",
            "distance_to_goal_delta": "present",
            "local_minimum_indicator": "present",
            "row_status": row["row_status"],
        }
        for row in rows
    ]

    _write_text(
        evidence_dir / "README.md",
        "\n".join(
            [
                "# Issue #4205 Pre-Run CPU Smoke Evidence",
                "",
                "This packet records local CPU arm-plumbing smoke evidence before any campaign run.",
                "It is not benchmark evidence and does not promote mitigation, planner-superiority, paper, or dissertation claims.",
                "",
                "Files:",
                "- `metadata.json`: claim boundary, residual blocker, and config checksum.",
                "- `pre_registration.json`: compact copy of the checked contract identity.",
                "- `intervention_summary.csv`: per-arm wrapper/CBF smoke metadata.",
                "- `failure_mode_counts.csv`: required static-deadlock trace-field presence by arm.",
                "- `claim_boundary.md`: explicit out-of-scope boundary.",
                "- `SHA256SUMS`: checksums for this compact packet.",
            ]
        ),
    )
    _write_text(evidence_dir / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True))
    _write_text(
        evidence_dir / "pre_registration.json",
        json.dumps(pre_registration, indent=2, sort_keys=True),
    )
    _write_text(
        evidence_dir / "intervention_summary.csv",
        _csv_text(
            [
                "arm_key",
                "scenario_id",
                "seed",
                "row_status",
                "wrapper_intervention_rate",
                "cbf_status_counts",
                "benchmark_evidence",
            ],
            intervention_rows,
        ),
    )
    _write_text(
        evidence_dir / "failure_mode_counts.csv",
        _csv_text(
            [
                "arm_key",
                "low_progress_window",
                "recenter_activation_count",
                "distance_to_goal_delta",
                "local_minimum_indicator",
                "row_status",
            ],
            failure_rows,
        ),
    )
    _write_text(
        evidence_dir / "claim_boundary.md",
        "\n".join(
            [
                "# Claim Boundary",
                "",
                "Evidence status: smoke evidence.",
                "",
                "This packet only shows that the three pre-registered issue #4205 arms expose the expected local CPU smoke metadata for one static-deadlock seed.",
                "",
                "Out of scope: full benchmark campaign run, Slurm/GPU submission, retraining, broad mitigation claims, planner-superiority claims, and paper/dissertation claim edits.",
            ]
        ),
    )

    checksum_entries = []
    for path in sorted(p for p in evidence_dir.iterdir() if p.name != "SHA256SUMS"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Use repo-relative paths so the integrity checker resolves each entry to
        # this evidence packet rather than colliding with a same-named repo-root
        # file (e.g. a bare "README.md" would resolve to the top-level README).
        checksum_entries.append(f"{digest}  {_repo_relative(path)}")
    _write_text(evidence_dir / "SHA256SUMS", "\n".join(checksum_entries) + "\n")
    return {
        "path": _repo_relative(evidence_dir),
        "files": sorted(path.name for path in evidence_dir.iterdir()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for issue #4205 checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Issue #4205 research contract."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--smoke-out", help="Optional path for compact CPU smoke manifest.")
    parser.add_argument("--smoke-input", help="Optional local CPU smoke evidence JSON to verify.")
    parser.add_argument(
        "--hydration-manifest",
        help="Optional private frozen-PPO checkpoint hydration manifest.",
    )
    parser.add_argument(
        "--require-hydrated-checkpoint",
        action="store_true",
        help="Fail closed unless --hydration-manifest proves the checkpoint exists.",
    )
    parser.add_argument(
        "--evidence-dir", help="Optional compact pre-run evidence packet directory."
    )
    return parser.parse_args(argv)


def _emit(report: dict[str, Any], *, json_output: bool) -> None:
    """Emit validation report."""
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "issue #4205 pre-registration ok: "
        f"arms={len(report['arm_keys'])} scenarios={len(report['scenario_ids'])} "
        f"seeds={len(report['seeds'])} benchmark_evidence={report['benchmark_evidence']}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run issue #4205 pre-registration checker."""
    args = parse_args(argv)
    try:
        config = _load_yaml(Path(args.config))
        report = validate_config(config)
        smoke_manifest: dict[str, Any] | None = None
        if args.hydration_manifest:
            manifest_path = Path(args.hydration_manifest)
            report["frozen_ppo_checkpoint"] = _validate_checkpoint_hydration_manifest(
                report,
                _load_json(manifest_path),
                manifest_path=manifest_path,
            )
        elif args.require_hydrated_checkpoint:
            raise ContractError(
                "frozen PPO checkpoint hydration manifest required before campaign authorization"
            )
        if args.smoke_input:
            smoke_manifest = _validate_cpu_smoke_evidence(
                report, _load_json(Path(args.smoke_input))
            )
            report["cpu_smoke"]["evidence_status"] = smoke_manifest["evidence_status"]
            report["cpu_smoke"]["rows"] = smoke_manifest["rows"]

        if args.smoke_out:
            smoke_path = Path(args.smoke_out)
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            output_payload = smoke_manifest or {
                "schema_version": SMOKE_MANIFEST_SCHEMA,
                "issue": 4205,
                "loop_id": report["loop_id"],
                "benchmark_evidence": False,
                "rows": report["cpu_smoke"]["rows"],
            }
            smoke_path.write_text(
                json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report["cpu_smoke"]["manifest_path"] = str(smoke_path)

        if args.evidence_dir:
            if smoke_manifest is None:
                raise ContractError("--evidence-dir requires --smoke-input")
            report["evidence_packet"] = _write_pre_run_evidence_packet(
                evidence_dir=Path(args.evidence_dir),
                report=report,
                smoke_manifest=smoke_manifest,
            )

        _emit(report, json_output=args.json)
        return 0
    except (ContractError, ValueError, TypeError) as exc:
        error = {"schema_version": REPORT_SCHEMA, "ok": False, "issue": 4205, "error": str(exc)}
        if args.json:
            print(json.dumps(error, indent=2, sort_keys=True))
        else:
            print(f"issue #4205 pre-registration invalid: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
