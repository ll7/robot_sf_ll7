#!/usr/bin/env python3
"""Validate the issue #4206 trace-capable h600 re-run pre-registration contract.

Plain-language summary: the issue #4206 mechanism cross-cut is blocked because
the retained h600 runs predate the trace-capable exporter (issue #4301), so their
failure-mechanism labels are all ``not_derivable`` and cannot be recovered by a
sidecar backfill (proved by PR #4341). The remaining CPU-dispatchable work is a
*contract* for the eventual trace-capable re-run. This checker enforces that
contract: it verifies the pre-registration config declares the required
``failure_mechanism_taxonomy.v1`` and ``interaction_exposure.v1`` outputs, the
trace-capture switches, a non-empty planner roster and seed schedule, provenance,
and the fail-closed exclusions -- and that its declared required-field lists stay
consistent with the canonical schema owners.

This is pre-registration validation only. It runs no simulation, submits no
campaign, and derives no mechanism label. It is fail-closed: any missing or
inconsistent contract element raises ``RerunPreregistrationError``.

Usage:
    uv run python scripts/validation/check_issue_4206_trace_capable_h600_rerun_preregistration.py \
        --config configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# Import the canonical schema owners so the contract's declared required fields
# are validated against the single source of truth rather than a local copy.
from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    REQUIRED_MECHANISM_FIELDS,
    TRACE_VERIFIED_EVIDENCE_MODES,
)
from robot_sf.benchmark.interaction_exposure import (
    INTERACTION_EXPOSURE_REQUIRED_FIELDS,
    INTERACTION_EXPOSURE_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

CONFIG_SCHEMA_VERSION = "issue_4206_trace_capable_h600_rerun_preregistration.v1"
ISSUE = 4206


class RerunPreregistrationError(ValueError):
    """Raised when the trace-capable h600 re-run pre-registration is invalid."""


def _require(condition: bool, message: str) -> None:
    """Raise ``RerunPreregistrationError`` with ``message`` when ``condition`` is false."""
    if not condition:
        raise RerunPreregistrationError(message)


def load_preregistration(config_path: str | Path) -> dict[str, Any]:
    """Load and structurally validate the re-run pre-registration config.

    Returns:
        The validated pre-registration payload.
    """
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
        "claim_boundary is required (pre-registration must state it is not evidence)",
    )

    _validate_provenance(payload.get("provenance"))
    _validate_required_outputs(payload.get("required_outputs"))
    _validate_trace_capture(payload.get("trace_capture"))
    _validate_planner_roster(payload.get("planner_roster"))
    _validate_seeds(payload.get("seeds"))
    _validate_fail_closed(payload.get("fail_closed_exclusions"))
    _validate_queue_plan(payload.get("queue_plan"))
    return payload


def _validate_provenance(provenance: Any) -> None:
    """Validate that provenance names predecessor runs and the downstream consumer."""
    _require(isinstance(provenance, dict), "provenance block is required")
    predecessors = provenance.get("predecessor_runs")
    _require(
        isinstance(predecessors, list) and len(predecessors) > 0,
        "provenance.predecessor_runs must list the retained h600 runs being superseded",
    )
    for run in predecessors:
        _require(isinstance(run, dict), "each predecessor_runs entry must be a mapping")
        _require("job" in run, "predecessor run must name its job id")
        _require(
            bool(str(run.get("insufficient_reason", "")).strip()),
            f"predecessor run {run.get('job')} must state why it is insufficient",
        )
    consumer = provenance.get("downstream_consumer")
    _require(isinstance(consumer, dict), "provenance.downstream_consumer is required")
    _require(
        bool(str(consumer.get("builder", "")).strip()),
        "downstream_consumer.builder must name the blocked cross-cut builder",
    )


def _validate_required_outputs(required_outputs: Any) -> None:
    """Cross-check declared required output fields against the canonical schema owners."""
    _require(isinstance(required_outputs, dict), "required_outputs block is required")

    mechanism = required_outputs.get("failure_mechanism")
    _require(isinstance(mechanism, dict), "required_outputs.failure_mechanism is required")
    _require(
        mechanism.get("schema_version") == MECHANISM_SCHEMA_VERSION,
        f"failure_mechanism.schema_version must be {MECHANISM_SCHEMA_VERSION}",
    )
    declared_mech_fields = tuple(mechanism.get("required_fields") or ())
    _require(
        declared_mech_fields == REQUIRED_MECHANISM_FIELDS,
        "failure_mechanism.required_fields must match the canonical "
        f"REQUIRED_MECHANISM_FIELDS {REQUIRED_MECHANISM_FIELDS}; got {declared_mech_fields}",
    )
    declared_modes = set(mechanism.get("trace_verified_evidence_modes") or ())
    _require(
        declared_modes == set(TRACE_VERIFIED_EVIDENCE_MODES),
        "failure_mechanism.trace_verified_evidence_modes must match the canonical "
        f"TRACE_VERIFIED_EVIDENCE_MODES {sorted(TRACE_VERIFIED_EVIDENCE_MODES)}",
    )
    fraction = mechanism.get("min_trace_verified_labeled_fraction")
    _require(
        isinstance(fraction, (int, float)) and 0.0 < float(fraction) <= 1.0,
        "min_trace_verified_labeled_fraction must be in (0, 1] so an all-unknown "
        "re-run cannot pass as a successful outcome",
    )

    exposure = required_outputs.get("interaction_exposure")
    _require(isinstance(exposure, dict), "required_outputs.interaction_exposure is required")
    _require(
        exposure.get("schema_version") == INTERACTION_EXPOSURE_SCHEMA_VERSION,
        f"interaction_exposure.schema_version must be {INTERACTION_EXPOSURE_SCHEMA_VERSION}",
    )
    declared_exp_fields = tuple(exposure.get("required_fields") or ())
    _require(
        declared_exp_fields == INTERACTION_EXPOSURE_REQUIRED_FIELDS,
        "interaction_exposure.required_fields must match the canonical "
        f"INTERACTION_EXPOSURE_REQUIRED_FIELDS {INTERACTION_EXPOSURE_REQUIRED_FIELDS}; "
        f"got {declared_exp_fields}",
    )


def _validate_trace_capture(trace_capture: Any) -> None:
    """Validate trace capture is required and both trace record flags are on."""
    _require(isinstance(trace_capture, dict), "trace_capture block is required")
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
    """Validate the roster is non-empty and has no duplicate planner keys."""
    _require(isinstance(planner_roster, dict), "planner_roster block is required")
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
            _require(key not in seen, f"planner_key {key!r} is declared in more than one class")
            seen.add(key)
            total += 1
    _require(total > 0, "planner_roster must declare at least one planner_key")


def _validate_seeds(seeds: Any) -> None:
    """Validate the seed schedule is a non-empty list of unique integers and is locked."""
    _require(isinstance(seeds, dict), "seeds block is required")
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
    """Validate the fail-closed exclusions forbid geometry substitution and all-unknown output."""
    _require(isinstance(fail_closed, dict), "fail_closed_exclusions block is required")
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
    """Validate the queue plan does not submit in this PR and declares an output root."""
    _require(isinstance(queue_plan, dict), "queue_plan block is required")
    _require(
        queue_plan.get("submit_in_this_pr") is False,
        "queue_plan.submit_in_this_pr must be false (pre-registration only)",
    )
    _require(
        bool(str(queue_plan.get("output_root", "")).strip()),
        "queue_plan.output_root is required",
    )


def build_dry_run_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the validated contract as a dry-run manifest (no execution).

    Returns:
        A compact manifest describing the re-run contract, for review artifacts.
    """
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
    """Run the pre-registration checker.

    Returns:
        Process exit code (0 on valid contract, 1 on failure).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml",
        help="Path to the trace-capable h600 re-run pre-registration config.",
    )
    parser.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path to write the dry-run manifest JSON (no campaign is submitted).",
    )
    args = parser.parse_args(argv)

    try:
        payload = load_preregistration(args.config)
    except RerunPreregistrationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    manifest = build_dry_run_manifest(payload)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
