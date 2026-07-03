#!/usr/bin/env python3
"""Validate issue #4205 static-constriction co-design-loop pre-registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
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


class ContractError(ValueError):
    """Raised when the pre-registration contract is malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ContractError(f"missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"{path} must contain a YAML mapping")
    return data


def _repo_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"expected non-empty repo-relative path, got {value!r}")
    path = REPO_ROOT / value
    if not path.exists():
        raise ContractError(f"referenced path does not exist: {value}")
    return path


def _find_suite(suite_config: dict[str, Any], suite_id: str) -> dict[str, Any]:
    suites = suite_config.get("suites")
    if not isinstance(suites, list):
        raise ContractError("mechanism suite config must define suites list")
    for suite in suites:
        if isinstance(suite, dict) and suite.get("suite_id") == suite_id:
            return suite
    raise ContractError(f"mechanism suite not found: {suite_id}")


def _require_list(config: dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ContractError(f"{key} must be a list")
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
    mechanism = config.get("mechanism_suite")
    if not isinstance(mechanism, dict):
        raise ContractError("mechanism_suite must be a mapping")
    if mechanism.get("suite_id") != "static_deadlock_recovery":
        raise ContractError("mechanism_suite.suite_id must be static_deadlock_recovery")
    if mechanism.get("target_mechanism") != "static_deadlock_or_local_minimum":
        raise ContractError(
            "mechanism_suite.target_mechanism must be static_deadlock_or_local_minimum"
        )
    suite_path = _repo_path(mechanism.get("path"))
    suite_config = _load_yaml(suite_path)
    suite = _find_suite(suite_config, "static_deadlock_recovery")
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
        raise ContractError("benchmark_contract schema_version drifted")
    if benchmark.get("issue") != 4205 or benchmark.get("loop_id") != config.get("loop_id"):
        raise ContractError("benchmark_contract issue or loop_id drifted")
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


def _validate_common(config: dict[str, Any]) -> None:  # noqa: C901
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
        raise ContractError("compute_submit_authorized must stay false in tracked pre-registration")
    smoke = config.get("cpu_smoke")
    if not isinstance(smoke, dict):
        raise ContractError("cpu_smoke must be a mapping")
    _repo_path(smoke.get("matrix"))
    if tuple(int(seed) for seed in smoke.get("seeds") or ()) != (111,):
        raise ContractError("cpu_smoke.seeds must remain the one-seed static-deadlock smoke [111]")
    lineage = config.get("frozen_ppo_lineage")
    if not isinstance(lineage, dict):
        raise ContractError("frozen_ppo_lineage must be a mapping")
    if lineage.get("algo") != "ppo":
        raise ContractError("frozen_ppo_lineage.algo must be ppo")
    _repo_path(lineage.get("algo_config"))
    if lineage.get("failure_job_id") != 13175:
        raise ContractError("frozen_ppo_lineage.failure_job_id must remain 13175")
    if lineage.get("model_preflight_required") is not True:
        raise ContractError("frozen_ppo_lineage.model_preflight_required must be true")


def _validate_arms(config: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: C901
    lineage = config["frozen_ppo_lineage"]
    arms = _require_list(config, "arms")
    if tuple(arm.get("key") for arm in arms if isinstance(arm, dict)) != EXPECTED_ARM_KEYS:
        raise ContractError(f"arms must be exactly ordered as {list(EXPECTED_ARM_KEYS)}")
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
    smoke = config["cpu_smoke"]
    scenario_id = EXPECTED_SCENARIOS[0]
    seed = int(smoke["seeds"][0])
    rows: list[dict[str, Any]] = []
    for arm in arms:
        wrapper = arm["safety_wrapper"]
        cbf = arm["cbf_safety_filter"]
        row = {
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
        rows.append(row)
    return rows


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a validation report for the issue #4205 pre-registration config."""
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
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
        "benchmark_contract": benchmark["research_contract"],
        "compute_submit_authorized": False,
        "benchmark_evidence": False,
        "cpu_smoke": {
            "matrix": config["cpu_smoke"]["matrix"],
            "rows": smoke_rows,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the issue #4205 checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Issue #4205 research contract."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--smoke-out", help="Optional path for the compact CPU smoke manifest.")
    return parser.parse_args(argv)


def _emit(report: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(
        "issue #4205 pre-registration ok: "
        f"arms={len(report['arm_keys'])} scenarios={len(report['scenario_ids'])} "
        f"seeds={len(report['seeds'])} benchmark_evidence=false"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the issue #4205 pre-registration checker."""
    args = parse_args(argv)
    try:
        config = _load_yaml(Path(args.config))
        report = validate_config(config)
        if args.smoke_out:
            smoke_path = Path(args.smoke_out)
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            smoke_path.write_text(
                json.dumps(
                    {
                        "schema_version": "robot_sf.issue_4205.cpu_smoke_manifest.v1",
                        "issue": 4205,
                        "loop_id": report["loop_id"],
                        "benchmark_evidence": False,
                        "rows": report["cpu_smoke"]["rows"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report["cpu_smoke"]["manifest_path"] = str(smoke_path)
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
