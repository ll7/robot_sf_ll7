#!/usr/bin/env python3
"""Validate issue #4206 trace-capable h600 re-run pre-registration contract.

Plain-language summary: issue #4206 mechanism cross-cut is blocked because
retained h600 runs predate the trace-capable exporter from issue #4301. This
checker validates the pre-registration contract and, optionally, the runnable
issue #4404 h600 config that implements it. It runs no simulation, submits no
campaign, and derives no mechanism label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
)
from robot_sf.benchmark.interaction_exposure import (
    INTERACTION_EXPOSURE_REQUIRED_FIELDS,
    INTERACTION_EXPOSURE_SCHEMA_VERSION,
)

CONFIG_SCHEMA_VERSION = "issue_4206_trace_capable_h600_rerun_preregistration.v1"
ISSUE = 4206
RUN_CONFIG_PATH = "configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml"
EXPECTED_SCENARIO_MATRIX = "configs/scenarios/classic_interactions_francis2023.yaml"
EXPECTED_SCENARIO_MATRIX_HASH = "c10df617a87c"


class RerunPreregistrationError(ValueError):
    """Raised when trace-capable h600 re-run pre-registration is invalid."""


def _require(condition: bool, message: str) -> None:
    """Raise ``RerunPreregistrationError`` with ``message`` when ``condition`` is false."""
    if not condition:
        raise RerunPreregistrationError(message)


def load_preregistration(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the trace-capable h600 re-run pre-registration config."""
    path = Path(config_path)
    _require(path.is_file(), f"pre-registration config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "pre-registration config must be a mapping")

    _require(
        payload.get("schema_version") == CONFIG_SCHEMA_VERSION,
        f"schema_version must be {CONFIG_SCHEMA_VERSION}",
    )
    _require(payload.get("issue") == ISSUE, f"issue must be {ISSUE}")
    _require(
        bool(str(payload.get("claim_boundary", "")).strip()),
        "claim_boundary required (pre-registration must state it is not evidence)",
    )

    _validate_provenance(payload.get("provenance"))
    _validate_required_outputs(payload.get("required_outputs"))
    _validate_trace_capture(payload.get("trace_capture"))
    _validate_planner_roster(payload.get("planner_roster"))
    _validate_seeds(payload.get("seeds"))
    _validate_fail_closed(payload.get("fail_closed_exclusions"))
    _validate_queue_plan(payload.get("queue_plan"))
    return payload


def validate_runnable_config_pair(
    preregistration: dict[str, Any],
    run_config_path: str | Path,
) -> dict[str, Any]:
    """Validate runnable h600 config preserves the pre-registration identity."""
    path = Path(run_config_path)
    _require(path.is_file(), f"runnable config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "runnable config must be a mapping")

    _require(
        payload.get("scenario_matrix") == EXPECTED_SCENARIO_MATRIX,
        f"scenario_matrix must be {EXPECTED_SCENARIO_MATRIX}",
    )
    prereg = payload.get("preregistration")
    _require(isinstance(prereg, dict), "runnable config preregistration block required")
    _require(
        prereg.get("implements_contract")
        == "configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml",
        "preregistration.implements_contract must point at the issue #4206 pre-registration",
    )
    _require(
        prereg.get("expected_scenario_matrix_hash") == EXPECTED_SCENARIO_MATRIX_HASH,
        f"expected_scenario_matrix_hash must be {EXPECTED_SCENARIO_MATRIX_HASH}",
    )
    _require(payload.get("horizon") == 600, "runnable config horizon must be 600")

    seed_policy = payload.get("seed_policy")
    _require(isinstance(seed_policy, dict), "runnable config seed_policy required")
    _require(seed_policy.get("mode") == "fixed-list", "seed_policy.mode must be fixed-list")
    _require(
        seed_policy.get("seeds") == preregistration["seeds"]["schedule"],
        "runnable config seeds must match pre-registration schedule",
    )

    trace_capture = preregistration["trace_capture"]
    for flag in ("record_planner_decision_trace", "record_simulation_step_trace"):
        _require(
            payload.get(flag) is True and trace_capture.get(flag) is True,
            f"runnable config and pre-registration must both set {flag}=true",
        )

    classes = preregistration["planner_roster"]["structural_classes"]
    expected_keys = [key for class_spec in classes.values() for key in class_spec["planner_keys"]]
    planners = payload.get("planners")
    _require(isinstance(planners, list), "runnable config planners must be a list")
    actual_keys = [planner.get("key") for planner in planners if isinstance(planner, dict)]
    _require(
        actual_keys == expected_keys,
        "runnable planner keys must match pre-registration order; "
        f"expected {expected_keys}, got {actual_keys}",
    )

    guarded = next(
        planner
        for planner in planners
        if isinstance(planner, dict) and planner.get("key") == "guarded_ppo"
    )
    _require(
        guarded.get("availability_gate") == "dependency_gated",
        "guarded_ppo must declare availability_gate=dependency_gated",
    )
    _require(
        bool(str(guarded.get("fail_closed_reason", "")).strip()),
        "guarded_ppo must declare fail_closed_reason",
    )

    return {
        "path": path.as_posix(),
        "planner_arm_count": len(actual_keys),
        "planner_keys": actual_keys,
        "seeds": list(seed_policy["seeds"]),
        "horizon": payload["horizon"],
        "trace_capture": {
            "record_planner_decision_trace": payload["record_planner_decision_trace"],
            "record_simulation_step_trace": payload["record_simulation_step_trace"],
        },
        "expected_scenario_matrix_hash": prereg["expected_scenario_matrix_hash"],
    }


def _validate_provenance(provenance: Any) -> None:
    """Validate provenance names predecessor runs and downstream consumer."""
    _require(isinstance(provenance, dict), "provenance block required")
    predecessors = provenance.get("predecessor_runs")
    _require(
        isinstance(predecessors, list) and len(predecessors) > 0,
        "provenance.predecessor_runs must list retained h600 runs being superseded",
    )
    for run in predecessors:
        _require(isinstance(run, dict), "each predecessor_runs entry must be a mapping")
        _require("job" in run, "predecessor run must name its job id")
        _require(
            bool(str(run.get("insufficient_reason", "")).strip()),
            f"predecessor run {run.get('job')} must state why insufficient",
        )
    consumer = provenance.get("downstream_consumer")
    _require(isinstance(consumer, dict), "provenance.downstream_consumer is required")
    _require(
        bool(str(consumer.get("builder", "")).strip()),
        "downstream_consumer.builder must name blocked cross-cut builder",
    )


def _validate_required_outputs(required_outputs: Any) -> None:
    """Cross-check declared required output fields against canonical schema owners."""
    _require(isinstance(required_outputs, dict), "required_outputs block required")
    mechanism = required_outputs.get("failure_mechanism")
    _require(isinstance(mechanism, dict), "required_outputs.failure_mechanism required")
    _require(
        mechanism.get("schema_version") == MECHANISM_SCHEMA_VERSION,
        f"failure_mechanism.schema_version must be {MECHANISM_SCHEMA_VERSION}",
    )
    declared_mech_fields = tuple(mechanism.get("required_fields") or ())
    _require(
        declared_mech_fields == REQUIRED_MECHANISM_FIELDS,
        "failure_mechanism.required_fields must match canonical "
        f"REQUIRED_MECHANISM_FIELDS {REQUIRED_MECHANISM_FIELDS}; got {declared_mech_fields}",
    )
    declared_modes = set(mechanism.get("trace_verified_evidence_modes") or ())
    _require(
        declared_modes == set(TRACE_VERIFIED_EVIDENCE_MODES),
        "failure_mechanism.trace_verified_evidence_modes must match canonical "
        f"TRACE_VERIFIED_EVIDENCE_MODES {sorted(TRACE_VERIFIED_EVIDENCE_MODES)}",
    )
    fraction = mechanism.get("min_trace_verified_labeled_fraction")
    _require(
        isinstance(fraction, (int, float)) and 0.0 < float(fraction) <= 1.0,
        "min_trace_verified_labeled_fraction must be in (0, 1] so an all-unknown "
        "re-run cannot pass successful outcome",
    )

    exposure = required_outputs.get("interaction_exposure")
    _require(isinstance(exposure, dict), "required_outputs.interaction_exposure required")
    _require(
        exposure.get("schema_version") == INTERACTION_EXPOSURE_SCHEMA_VERSION,
        f"interaction_exposure.schema_version must be {INTERACTION_EXPOSURE_SCHEMA_VERSION}",
    )
    declared_exp_fields = tuple(exposure.get("required_fields") or ())
    _require(
        declared_exp_fields == INTERACTION_EXPOSURE_REQUIRED_FIELDS,
        "interaction_exposure.required_fields must match canonical "
        f"INTERACTION_EXPOSURE_REQUIRED_FIELDS {INTERACTION_EXPOSURE_REQUIRED_FIELDS}; "
        f"got {declared_exp_fields}",
    )


def _validate_trace_capture(trace_capture: Any) -> None:
    """Validate trace capture is required and both trace record flags are on."""
    _require(isinstance(trace_capture, dict), "trace_capture block required")
    _require(
        trace_capture.get("required") is True,
        "trace_capture.required must be true (the whole point of the re-run)",
    )
    for flag in ("record_planner_decision_trace", "record_simulation_step_trace"):
        _require(
            trace_capture.get(flag) is True,
            f"trace_capture.{flag} must be true so mechanism labels are derivable",
        )


def _validate_planner_roster(planner_roster: Any) -> None:
    """Validate roster is non-empty and contains no duplicate planner keys."""
    _require(isinstance(planner_roster, dict), "planner_roster block required")
    classes = planner_roster.get("structural_classes")
    _require(
        isinstance(classes, dict) and len(classes) > 0,
        "planner_roster.structural_classes must declare at least one class",
    )
    seen: set[str] = set()
    total = 0
    for class_name, spec in classes.items():
        _require(isinstance(spec, dict), f"structural class {class_name} must be a mapping")
        keys = spec.get("planner_keys")
        _require(
            isinstance(keys, list) and len(keys) > 0,
            f"structural class {class_name} must list at least one planner_key",
        )
        for key in keys:
            _require(key not in seen, f"planner_key {key!r} declared in more than one class")
            seen.add(key)
            total += 1
    _require(total > 0, "planner_roster must declare at least one planner_key")


def _validate_seeds(seeds: Any) -> None:
    """Validate seed schedule is a non-empty unique integer list and locked."""
    _require(isinstance(seeds, dict), "seeds block required")
    schedule = seeds.get("schedule")
    _require(
        isinstance(schedule, list) and len(schedule) > 0,
        "seeds.schedule must be a non-empty list",
    )
    _require(
        all(isinstance(seed, int) for seed in schedule),
        "seeds.schedule must contain integers",
    )
    _require(len(set(schedule)) == len(schedule), "seeds.schedule must not contain duplicates")
    _require(
        seeds.get("seeds_locked") is True,
        "seeds.seeds_locked must be true so seed coverage cannot silently shrink",
    )


def _validate_fail_closed(fail_closed: Any) -> None:
    """Validate fail-closed exclusions forbid substitution and all-unknown output."""
    _require(isinstance(fail_closed, dict), "fail_closed_exclusions block required")
    _require(
        fail_closed.get("geometry_buckets_may_substitute_mechanism_labels") is False,
        "geometry_buckets_may_substitute_mechanism_labels must be false",
    )
    _require(
        fail_closed.get("all_not_derivable_output_is_success") is False,
        "all_not_derivable_output_is_success must be false (an all-unknown re-run failed)",
    )
    _require(
        fail_closed.get("fallback_or_degraded_rows_count_as_success") is False,
        "fallback_or_degraded_rows_count_as_success must be false",
    )
    _require(
        fail_closed.get("unknown_planner_keys_included_in_f_c4ii") is False,
        "unknown_planner_keys_included_in_f_c4ii must be false",
    )


def _validate_queue_plan(queue_plan: Any) -> None:
    """Validate queue plan does not submit in this PR and declares output root."""
    _require(isinstance(queue_plan, dict), "queue_plan block required")
    _require(
        queue_plan.get("submit_in_this_pr") is False,
        "queue_plan.submit_in_this_pr must be false (pre-registration only)",
    )
    _require(
        bool(str(queue_plan.get("output_root", "")).strip()),
        "queue_plan.output_root required",
    )


def build_dry_run_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize validated contract dry-run manifest without execution."""
    classes = payload["planner_roster"]["structural_classes"]
    roster = {name: list(spec["planner_keys"]) for name, spec in classes.items()}
    predecessors = [run.get("job") for run in payload["provenance"]["predecessor_runs"]]
    mechanism = payload["required_outputs"]["failure_mechanism"]
    return {
        "issue": ISSUE,
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": "preregistration_valid",
        "submits_campaign": False,
        "derives_mechanism_labels": False,
        "predecessor_jobs": predecessors,
        "planner_roster": roster,
        "planner_arm_count": sum(len(keys) for keys in roster.values()),
        "seeds": list(payload["seeds"]["schedule"]),
        "required_outputs": {
            "failure_mechanism_schema": mechanism["schema_version"],
            "interaction_exposure_schema": payload["required_outputs"]["interaction_exposure"][
                "schema_version"
            ],
            "min_trace_verified_labeled_fraction": mechanism["min_trace_verified_labeled_fraction"],
        },
        "downstream_consumer": payload["provenance"]["downstream_consumer"].get("builder"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the pre-registration checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml",
        help="Path to the trace-capable h600 re-run pre-registration config.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path to write a dry-run manifest JSON (no campaign submitted).",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="Optional runnable campaign config to cross-check against the pre-registration.",
    )
    args = parser.parse_args(argv)

    try:
        payload = load_preregistration(args.config)
        manifest = build_dry_run_manifest(payload)
        if args.run_config:
            manifest["runnable_config"] = validate_runnable_config_pair(payload, args.run_config)
    except RerunPreregistrationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    if args.manifest_out:
        out_path = Path(args.manifest_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote dry-run manifest: {out_path}")

    print(
        "PASS: trace-capable h600 re-run pre-registration valid "
        f"({manifest['planner_arm_count']} planner arms, {len(manifest['seeds'])} seeds, "
        "no campaign submitted)"
    )
    if args.run_config:
        run_manifest = manifest["runnable_config"]
        print(
            "PASS: runnable trace-capable h600 config valid "
            f"({run_manifest['planner_arm_count']} planner arms, "
            f"{len(run_manifest['seeds'])} seeds, trace capture on)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
