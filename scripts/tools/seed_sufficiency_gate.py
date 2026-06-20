#!/usr/bin/env python3
"""Decide S5/S10/S20 seed escalation from frozen sufficiency metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

SCHEDULES = ("s5", "s10", "s20")
BENCHMARK_VALID_ROW_STATUSES = frozenset({"native", "adapter"})
Decision = Literal["escalate", "stop_confirmed", "diagnostic_only"]


@dataclass(frozen=True, slots=True)
class SeedGateInput:
    """Minimal frozen inputs for seed-sufficiency scheduling."""

    schedule: str
    ci_half_width: float
    target_ci_half_width: float
    rank_flip_observed: bool = False
    heldout_delta_abs: float | None = None
    heldout_delta_threshold: float | None = None
    invalid_row_count: int = 0


@dataclass(frozen=True, slots=True)
class SeedGateDecision:
    """Decision emitted by the seed-sufficiency gate."""

    decision: Decision
    current_schedule: str
    next_schedule: str | None
    reason: str


def decide_seed_gate(inputs: SeedGateInput) -> SeedGateDecision:
    """Return whether to stop, escalate, or keep evidence diagnostic-only."""
    schedule = inputs.schedule.lower()
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, found {inputs.schedule!r}")
    if inputs.invalid_row_count > 0:
        return SeedGateDecision(
            decision="diagnostic_only",
            current_schedule=schedule,
            next_schedule=None,
            reason="invalid rows are present; do not escalate claim strength",
        )

    unstable = inputs.rank_flip_observed or inputs.ci_half_width > inputs.target_ci_half_width
    if (
        inputs.heldout_delta_abs is not None
        and inputs.heldout_delta_threshold is not None
        and inputs.heldout_delta_abs > inputs.heldout_delta_threshold
    ):
        unstable = True

    if not unstable:
        return SeedGateDecision(
            decision="stop_confirmed",
            current_schedule=schedule,
            next_schedule=None,
            reason="uncertainty and transfer checks satisfy the frozen stop rule",
        )

    index = SCHEDULES.index(schedule)
    if index == len(SCHEDULES) - 1:
        return SeedGateDecision(
            decision="diagnostic_only",
            current_schedule=schedule,
            next_schedule=None,
            reason="maximum S20 schedule remains unstable; keep evidence diagnostic-only",
        )
    next_schedule = SCHEDULES[index + 1]
    return SeedGateDecision(
        decision="escalate",
        current_schedule=schedule,
        next_schedule=next_schedule,
        reason=f"frozen sufficiency rule requires escalation to {next_schedule}",
    )


def seed_gate_payload_from_result_store(result_store: Path) -> dict[str, Any]:
    """Build a seed-sufficiency decision payload from a canonical result store."""
    summary = json.loads((result_store / "summary.json").read_text(encoding="utf-8"))
    analysis = json.loads((result_store / "analysis.json").read_text(encoding="utf-8"))
    gate_payload = analysis.get("seed_sufficiency_gate")
    if not isinstance(gate_payload, dict):
        raise ValueError("analysis.json seed_sufficiency_gate must be a mapping")

    gate_fields = {field.name for field in fields(SeedGateInput)}
    raw_input = {key: value for key, value in gate_payload.items() if key in gate_fields}
    raw_input["invalid_row_count"] = _invalid_row_count_from_summary(summary)
    decision = decide_seed_gate(SeedGateInput(**raw_input))
    return {
        "schema_version": "campaign-seed-sufficiency-schedule.v1",
        "source": "campaign_result_store.analysis_json",
        "input": raw_input,
        "decision": asdict(decision),
    }


def _invalid_row_count_from_summary(summary: dict[str, Any]) -> int:
    """Return result-store rows that cannot strengthen benchmark claims."""
    counts = summary.get("row_status_counts", {})
    if not isinstance(counts, dict):
        return 0
    return sum(
        int(count) for status, count in counts.items() if status not in BENCHMARK_VALID_ROW_STATUSES
    )


def _seed_gate_payload_from_input_json(input_json: Path) -> dict[str, Any]:
    """Build a seed-sufficiency decision payload from a frozen gate input JSON."""
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    gate_fields = {field.name for field in fields(SeedGateInput)}
    raw_input = {key: value for key, value in payload.items() if key in gate_fields}
    decision = decide_seed_gate(SeedGateInput(**raw_input))
    return {
        "schema_version": "seed-sufficiency-gate.v1",
        "source": "seed_gate_input_json",
        "input": raw_input,
        "decision": asdict(decision),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path, help="Seed gate input JSON.")
    source.add_argument("--result-store", type=Path, help="Canonical campaign result-store path.")
    parser.add_argument("--output-json", type=Path, help="Optional decision JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the seed-sufficiency gate from JSON."""
    args = _parse_args(argv)
    result = (
        seed_gate_payload_from_result_store(args.result_store)
        if args.result_store is not None
        else _seed_gate_payload_from_input_json(args.input_json)
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    decision = result["decision"]["decision"]
    return 0 if decision != "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
