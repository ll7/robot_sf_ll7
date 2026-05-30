"""Validate the policy_stack_v1 arbitration trace packet shape."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from robot_sf.planner.policy_stack_v1 import (
    ARBITRATION_TRACE_SCHEMA,
    PolicyStackV1Adapter,
    PolicyStackV1Config,
)

EXPECTED_INFERENCE_FEATURES = {
    "robot.position",
    "robot.heading",
    "goal.current",
    "goal.next",
    "pedestrians.positions",
}
EXPECTED_LEAKAGE_EXCLUSIONS = {
    "future_trajectory",
    "simulator_collision_label",
    "episode_success_label",
    "route_outcome_label",
    "benchmark_metric_rollups",
}

if TYPE_CHECKING:
    from collections.abc import Sequence


def _fixture_observation() -> dict[str, Any]:
    """Return a tiny inference-available observation fixture.

    Returns:
        Observation payload containing only fields allowed in the arbitration trace contract.
    """
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [1.0, 0.0]},
        "pedestrians": {"positions": []},
    }


def validate_packet(packet: dict[str, Any]) -> list[str]:
    """Validate the trace packet shape and return human-readable errors.

    Returns:
        List of validation errors. Empty means valid.
    """
    errors: list[str] = []
    if packet.get("schema_version") != ARBITRATION_TRACE_SCHEMA:
        errors.append("schema_version mismatch")
    if packet.get("training_enabled") is not False:
        errors.append("training_enabled must remain false for the trace preflight")
    if not packet.get("proposal_sources"):
        errors.append("proposal_sources must be non-empty")
    errors.extend(_validate_contract_sections(packet))
    errors.extend(_validate_trace_section(packet))
    return errors


def _validate_contract_sections(packet: dict[str, Any]) -> list[str]:
    """Validate packet contract sections.

    Returns:
        List of contract-section errors.
    """
    errors: list[str] = []
    command_contract = packet.get("command_contract")
    if not isinstance(command_contract, dict):
        errors.append("command_contract must be present")
    elif command_contract.get("action_space") != "unicycle_vw":
        errors.append("command_contract.action_space must be unicycle_vw")
    observation_contract = packet.get("observation_contract")
    if not isinstance(observation_contract, dict):
        errors.append("observation_contract must be present")
    else:
        features = set(observation_contract.get("inference_available_features", []))
        exclusions = set(observation_contract.get("leakage_exclusions", []))
        if features != EXPECTED_INFERENCE_FEATURES:
            errors.append("observation_contract.inference_available_features changed")
        if exclusions != EXPECTED_LEAKAGE_EXCLUSIONS:
            errors.append("observation_contract.leakage_exclusions changed")
    status_policy = packet.get("status_policy")
    if not isinstance(status_policy, dict):
        errors.append("status_policy must be present")
    elif "not_available" not in status_policy.get("not_available_statuses", []):
        errors.append("status_policy must identify not_available statuses")
    elif set(status_policy.get("non_executable_statuses", [])) != {
        "failed",
        "not_available",
        "rejected",
    }:
        errors.append("status_policy.non_executable_statuses changed")
    return errors


def _validate_trace_section(packet: dict[str, Any]) -> list[str]:
    """Validate packet trace section.

    Returns:
        List of trace-section errors.
    """
    errors: list[str] = []
    trace = packet.get("trace")
    if not isinstance(trace, dict):
        errors.append("trace must be present")
    elif not isinstance(trace.get("last_step"), dict):
        errors.append("trace.last_step must be present after one planner step")
    else:
        ranking = trace["last_step"].get("candidate_ranking")
        if not isinstance(ranking, list) or not ranking:
            errors.append("trace.last_step.candidate_ranking must be non-empty")
    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the trace packet validator CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the trace packet as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a tiny trace-packet smoke validation.

    Returns:
        Exit code 0 when valid, 2 when the packet shape is invalid.
    """
    args = build_arg_parser().parse_args(argv)
    stack = PolicyStackV1Adapter(
        config=PolicyStackV1Config(
            proposal_sources=("goal", "missing_optional"),
            optional_sources=("missing_optional",),
        )
    )
    stack.plan(_fixture_observation())
    packet = stack.arbitration_trace_packet()
    errors = validate_packet(packet)
    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    elif errors:
        print("\n".join(errors))
    else:
        print("OK policy_stack_v1 arbitration trace packet is valid")
    return 2 if errors else 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
