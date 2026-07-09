#!/usr/bin/env python3
"""Watch campaign live arm status and emit fail-closed cancellation decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_FAILURE_STATES = frozenset({"failed", "error", "contract_error"})
DEFAULT_EVENT_LOG = "arm_status.jsonl"
DEFAULT_LIVE_STATUS = "live_arm_status.json"
CANCEL_EXIT_CODE = 20


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


@dataclass(frozen=True, slots=True)
class ArmStatusDecision:
    """Single polling decision for a live campaign arm-status stream."""

    action: str
    reason: str
    failing_arms: list[str]
    failure_events: list[dict[str, Any]]
    live_status_path: str
    event_log_path: str
    scancel_command: list[str] | None
    executed: bool = False


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            event = json.loads(stripped)
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            events.append(event)
    return events


def _scancel_command(job_id: str | None) -> list[str] | None:
    if not job_id:
        return None
    return ["scancel", job_id]


def decide_arm_status_action(
    *,
    live_status_path: Path,
    event_log_path: Path,
    job_id: str | None = None,
    failure_states: set[str] | frozenset[str] = DEFAULT_FAILURE_STATES,
) -> ArmStatusDecision:
    """Return the private-ops action implied by status files without side effects."""

    normalized_failure_states = {state.lower() for state in failure_states}
    events = _load_events(event_log_path)
    failure_events = [
        event
        for event in events
        if str(event.get("state", "")).lower() in normalized_failure_states
    ]
    if failure_events:
        failing_arms = sorted({str(event.get("arm", "unknown")) for event in failure_events})
        return ArmStatusDecision(
            action="cancel",
            reason="fail_closed_arm_event",
            failing_arms=failing_arms,
            failure_events=failure_events,
            live_status_path=str(live_status_path),
            event_log_path=str(event_log_path),
            scancel_command=_scancel_command(job_id),
        )

    live_status = _load_json(live_status_path)
    if live_status is None:
        return ArmStatusDecision(
            action="wait",
            reason="live_status_missing",
            failing_arms=[],
            failure_events=[],
            live_status_path=str(live_status_path),
            event_log_path=str(event_log_path),
            scancel_command=None,
        )

    arms = live_status.get("arms")
    if not isinstance(arms, dict):
        raise ValueError(f"{live_status_path} must contain an object field named 'arms'")

    failing_from_snapshot: list[dict[str, Any]] = []
    for arm_key, arm_payload in arms.items():
        if not isinstance(arm_payload, dict):
            continue
        status = str(arm_payload.get("status", "")).lower()
        if status in normalized_failure_states:
            failing_from_snapshot.append(
                {
                    "arm": str(arm_key),
                    "state": status,
                    "phase": live_status.get("phase"),
                    "details": arm_payload,
                }
            )

    if failing_from_snapshot:
        return ArmStatusDecision(
            action="cancel",
            reason="fail_closed_live_status",
            failing_arms=sorted({event["arm"] for event in failing_from_snapshot}),
            failure_events=failing_from_snapshot,
            live_status_path=str(live_status_path),
            event_log_path=str(event_log_path),
            scancel_command=_scancel_command(job_id),
        )

    statuses = {
        str(arm_payload.get("status", "")).lower()
        for arm_payload in arms.values()
        if isinstance(arm_payload, dict)
    }
    action = "complete" if statuses == {"completed"} else "continue"
    return ArmStatusDecision(
        action=action,
        reason=f"no_fail_closed_arm_status:{','.join(sorted(statuses)) or 'empty'}",
        failing_arms=[],
        failure_events=[],
        live_status_path=str(live_status_path),
        event_log_path=str(event_log_path),
        scancel_command=None,
    )


def _paths_from_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output_root is not None:
        output_root = Path(args.output_root)
        live_status_path = output_root / DEFAULT_LIVE_STATUS
        event_log_path = output_root / DEFAULT_EVENT_LOG
    else:
        live_status_path = Path(args.live_status)
        event_log_path = Path(args.event_log)
    return live_status_path, event_log_path


def _execute_scancel(command: list[str] | None) -> None:
    if command is None:
        raise RuntimeError("--execute-scancel requires --job-id when cancellation is needed")
    subprocess.run(command, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse live arm-status watcher CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Poll live_arm_status.json and arm_status.jsonl, then emit the scancel decision "
            "needed by private ops. No cancellation is executed unless --execute-scancel is set."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--output-root", help="Campaign output directory containing live status files."
    )
    source.add_argument("--live-status", help="Path to live_arm_status.json.")
    parser.add_argument(
        "--event-log",
        help="Path to arm_status.jsonl. Required with --live-status; inferred with --output-root.",
    )
    parser.add_argument("--job-id", help="Slurm job id used to build a scancel command.")
    parser.add_argument(
        "--failure-state",
        action="append",
        default=[],
        help="Additional arm state that should trigger cancellation.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.0,
        help="Seconds between polls when --max-polls is greater than 1.",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=1,
        help="Maximum polls before returning the latest non-cancel decision.",
    )
    parser.add_argument(
        "--execute-scancel",
        action="store_true",
        help="Execute scancel when a fail-closed arm status is detected.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the decision as JSON.")
    args = parser.parse_args(argv)
    if args.live_status and not args.event_log:
        parser.error("--event-log is required when --live-status is used")
    if args.max_polls < 1:
        parser.error("--max-polls must be at least 1")
    if args.poll_interval_seconds < 0:
        parser.error("--poll-interval-seconds must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run one bounded live arm-status polling cycle."""

    args = parse_args(argv)
    live_status_path, event_log_path = _paths_from_args(args)
    failure_states = set(DEFAULT_FAILURE_STATES)
    failure_states.update(args.failure_state)

    decision: ArmStatusDecision | None = None
    for attempt in range(args.max_polls):
        decision = decide_arm_status_action(
            live_status_path=live_status_path,
            event_log_path=event_log_path,
            job_id=args.job_id,
            failure_states=failure_states,
        )
        if decision.action == "cancel" or attempt == args.max_polls - 1:
            break
        time.sleep(args.poll_interval_seconds)

    assert decision is not None
    if args.execute_scancel and decision.action == "cancel":
        _execute_scancel(decision.scancel_command)
        decision = ArmStatusDecision(**{**asdict(decision), "executed": True})

    if args.json:
        print(json.dumps(asdict(decision), indent=2, sort_keys=True))
    elif decision.action == "cancel":
        command = " ".join(decision.scancel_command or ["scancel", "<job-id-required>"])
        print(f"cancel recommended: {command}; failing_arms={','.join(decision.failing_arms)}")
    else:
        print(f"{decision.action}: {decision.reason}")

    return CANCEL_EXIT_CODE if decision.action == "cancel" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
