#!/usr/bin/env python3
"""Decide S5/S10/S20 seed escalation from frozen sufficiency metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

SCHEDULES = ("s5", "s10", "s20")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Seed gate input JSON.")
    parser.add_argument("--output-json", type=Path, help="Optional decision JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the seed-sufficiency gate from JSON."""
    args = _parse_args(argv)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    gate_fields = {field.name for field in fields(SeedGateInput)}
    decision = decide_seed_gate(
        SeedGateInput(**{key: value for key, value in payload.items() if key in gate_fields})
    )
    result = asdict(decision)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if decision.decision != "diagnostic_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
