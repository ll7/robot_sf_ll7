#!/usr/bin/env python3
"""Write a bounded, credential-free receipt for interrupted PR readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pr_ready_termination.v1"
LANE_TERMINAL_SCHEMA_VERSION = "pr_ready_lane_terminal.v1"
MAX_TEXT_LENGTH = 200
READ_LIMIT_BYTES = 8192
CGROUP_ROOT = Path("/sys/fs/cgroup")
CHILD_REGISTRATION_STATES = frozenset({"not_started", "registering", "registered"})


def _bounded_text(value: object, default: str = "unknown") -> str:
    """Return one bounded line suitable for a diagnostic receipt."""
    if value is None:
        value = default
    text = " ".join(str(value).replace("\x00", " ").split())
    return (text or default)[:MAX_TEXT_LENGTH]


def _positive_int(value: object) -> int | None:
    """Parse a positive process identifier without raising in a signal path."""
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _identifier_is_absent(value: object) -> bool:
    """Distinguish an omitted identifier from a supplied invalid identifier."""
    return value is None or (isinstance(value, str) and not value.strip())


def _read_limited(path: Path) -> str | None:
    """Read only a small bounded prefix of a diagnostic pseudo-file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(READ_LIMIT_BYTES)
    except OSError:
        return None


def _read_value(path: Path, *, key: str | None = None) -> int | str | None:
    """Read one integer-like value from a cgroup file."""
    contents = _read_limited(path)
    if contents is None:
        return None
    candidate: str | None = None
    if key is None:
        candidate = contents.strip().splitlines()[0] if contents.strip() else None
    else:
        for line in contents.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == key:
                candidate = fields[1]
                break
    if candidate is None:
        return None
    if candidate == "max":
        return candidate
    try:
        parsed = int(candidate)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _memory_snapshot() -> dict[str, int | None]:
    """Return only total and currently available host memory in kilobytes."""
    contents = _read_limited(Path("/proc/meminfo"))
    values: dict[str, int | None] = {"total_kb": None, "available_kb": None}
    if contents is None:
        return values
    names = {"MemTotal:": "total_kb", "MemAvailable:": "available_kb"}
    for line in contents.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] not in names:
            continue
        try:
            value = int(fields[1])
        except ValueError:
            continue
        values[names[fields[0]]] = value if value >= 0 else None
    return values


def _cgroup_v2_root() -> Path | None:
    """Return the current cgroup-v2 directory without exposing its contents."""
    contents = _read_limited(Path("/proc/self/cgroup"))
    if contents is None:
        return None
    for line in contents.splitlines():
        if not line.startswith("0::"):
            continue
        relative = line.partition("::")[2].strip().lstrip("/")
        relative_path = Path(relative)
        if ".." in relative_path.parts:
            return None
        return CGROUP_ROOT / relative_path
    return None


def _resource_snapshot() -> dict[str, Any]:
    """Collect a fixed-size host and cgroup resource snapshot."""
    try:
        load_average = round(os.getloadavg()[0], 3)
    except (AttributeError, OSError):
        load_average = None
    cgroup_root = _cgroup_v2_root()
    cgroup: dict[str, Any] = {
        "version": "v2" if cgroup_root is not None else "unavailable",
        "memory_current_bytes": None,
        "memory_max_bytes": None,
        "cpu_usage_usec": None,
    }
    if cgroup_root is not None:
        cgroup.update(
            memory_current_bytes=_read_value(cgroup_root / "memory.current"),
            memory_max_bytes=_read_value(cgroup_root / "memory.max"),
            cpu_usage_usec=_read_value(cgroup_root / "cpu.stat", key="usage_usec"),
        )
    return {
        "host": {
            "cpu_count": os.cpu_count(),
            "load_average_1m": load_average,
        },
        "memory": _memory_snapshot(),
        "cgroup": cgroup,
    }


def _process_group_exists(process_group_id: int | None) -> bool | None:
    """Return whether a process group exists, or unknown when probing is unavailable."""
    if process_group_id is None:
        return None
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except (AttributeError, OSError):
        return None
    return True


def _parse_proc_stat_state(entry_path: str, process_group_id: int) -> str | None:
    """Return process state if entry belongs to process_group_id, else None."""
    try:
        with open(f"{entry_path}/stat", encoding="utf-8", errors="replace") as handle:
            content = handle.read(READ_LIMIT_BYTES)
    except OSError:
        return None
    rparen = content.rfind(")")
    if rparen == -1:
        return None
    fields = content[rparen + 1 :].split()
    if len(fields) < 3:
        return None
    try:
        if int(fields[2]) == process_group_id:
            return fields[0]
    except ValueError:
        return None
    return None


def _inspect_process_group_members(process_group_id: int) -> tuple[int, int] | None:
    """Return (live_count, zombie_count) for an existing group, or None if unreadable."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    try:
        entries = os.scandir(proc_root)
    except OSError:
        return None
    live_count = 0
    zombie_count = 0
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            state = _parse_proc_stat_state(entry.path, process_group_id)
            if state == "Z":
                zombie_count += 1
            elif state is not None:
                live_count += 1
    return live_count, zombie_count


def _process_group_liveness(process_group_id: int | None) -> str | None:
    """Return 'absent', 'zombie_only', 'live', or 'unknown' for a process group.

    - 'absent': the process group has no members in the system process table.
    - 'zombie_only': the process group exists, but every member is in zombie (Z) state.
    - 'live': the process group exists and has at least one live (non-zombie) member.
    - 'unknown': liveness cannot be verified (probing/platform unavailable or unreadable).
    """
    if process_group_id is None:
        return None
    exists = _process_group_exists(process_group_id)
    if exists is False:
        return "absent"
    if exists is None:
        return "unknown"

    counts = _inspect_process_group_members(process_group_id)
    if counts is not None:
        live_count, zombie_count = counts
        if live_count > 0:
            return "live"
        if zombie_count > 0:
            return "zombie_only"
        if _process_group_exists(process_group_id) is False:
            return "absent"

    return "unknown"


def _signal_details(signal_number: int) -> dict[str, int | str | None]:
    """Return a stable signal name and conventional shell status."""
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal_{signal_number}"
    exit_code = 128 + signal_number if 1 <= signal_number <= 127 else None
    return {"name": signal_name, "number": signal_number, "exit_code": exit_code}


def _source_sha(value: object) -> str | None:
    """Normalize a full Git object ID, leaving unavailable or invalid IDs null."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return (
        normalized
        if len(normalized) in (40, 64)
        and all(character in "0123456789abcdef" for character in normalized)
        else None
    )


def _repository_fingerprint(repo_root: str | Path) -> str | None:
    """Return an opaque fingerprint for the repository's common Git directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    common_dir = result.stdout.strip()
    if not common_dir:
        return None
    return hashlib.sha256(os.fsencode(os.path.realpath(common_dir))).hexdigest()


@dataclass(frozen=True, slots=True)
class LaneTerminalContext:
    """Observed process and timing values for one readiness lane."""

    lane: str
    repo_root: str | Path
    source_sha: object
    started_at_utc: object
    elapsed_seconds: object
    exit_status: object
    supervisor_pid: object = None
    child_pid: object = None
    child_process_group_id: object = None
    supervisor_signal_number: object = None


def _validate_lane_outcome(receipt: dict[str, Any]) -> None:
    outcome = receipt["outcome"]
    exit_status = receipt["exit_status"]
    supervisor_signal = receipt["signal"]["supervisor"]
    if supervisor_signal is not None and outcome != "terminated":
        raise ValueError("an observed supervisor signal requires terminated outcome")
    if supervisor_signal is None and exit_status is None and outcome != "unknown":
        raise ValueError("an unknown exit status requires unknown outcome")
    if supervisor_signal is None and exit_status == 0 and outcome != "completed":
        raise ValueError("zero exit status requires completed outcome")
    if supervisor_signal is None and exit_status not in (None, 0) and outcome != "failed":
        raise ValueError("nonzero exit status requires failed outcome")


def _validate_lane_signal(signal: object) -> None:
    if not isinstance(signal, dict) or set(signal) != {"supervisor", "child"}:
        raise ValueError("lane terminal receipt signal fields are invalid")
    supervisor_signal = signal["supervisor"]
    if supervisor_signal is not None and (
        not isinstance(supervisor_signal, dict)
        or not isinstance(supervisor_signal.get("name"), str)
        or not isinstance(supervisor_signal.get("number"), int)
        or supervisor_signal.get("sender_pid") is not None
    ):
        raise ValueError("observed supervisor signal must have an unknown sender")
    if signal["child"] is not None:
        raise ValueError("child signal cannot be inferred from its exit status")


def _validate_lane_repository(repository: object) -> None:
    if not isinstance(repository, dict):
        raise ValueError("lane terminal receipt repository identity is invalid")
    repository_id = repository.get("id_sha256")
    if repository_id is not None and (
        not isinstance(repository_id, str)
        or len(repository_id) != 64
        or any(character not in "0123456789abcdef" for character in repository_id)
    ):
        raise ValueError("lane terminal receipt repository identity is invalid")
    source_sha = repository.get("source_sha")
    if source_sha is not None and _source_sha(source_sha) != source_sha:
        raise ValueError("lane terminal receipt repository identity is invalid")


def _validate_lane_timing(timing: object) -> None:
    if not isinstance(timing, dict):
        raise ValueError("lane terminal receipt timing is invalid")
    elapsed = timing.get("elapsed_seconds")
    if elapsed is not None and (
        isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0
    ):
        raise ValueError("lane terminal receipt elapsed time must be null or nonnegative")


def validate_lane_terminal_receipt(receipt: object) -> None:
    """Reject malformed or contradictory per-lane terminal records."""
    if not isinstance(receipt, dict):
        raise ValueError("lane terminal receipt must be a JSON object")
    if receipt.get("schema") != LANE_TERMINAL_SCHEMA_VERSION:
        raise ValueError("lane terminal receipt schema is unsupported")
    if not isinstance(receipt.get("lane"), str) or not receipt["lane"].strip():
        raise ValueError("lane terminal receipt must identify a lane")
    if receipt.get("status") != "terminal":
        raise ValueError("lane terminal receipt status must be terminal")
    if receipt.get("outcome") not in {"completed", "failed", "terminated", "unknown"}:
        raise ValueError("lane terminal receipt outcome is invalid")
    exit_status = receipt.get("exit_status")
    if exit_status is not None and (
        isinstance(exit_status, bool)
        or not isinstance(exit_status, int)
        or not 0 <= exit_status <= 255
    ):
        raise ValueError("lane terminal receipt exit status must be null or 0..255")
    _validate_lane_signal(receipt.get("signal"))
    _validate_lane_outcome(receipt)
    _validate_lane_repository(receipt.get("repository"))
    _validate_lane_timing(receipt.get("timing"))
    if "command" in receipt or "arguments" in receipt or "environment" in receipt:
        raise ValueError("lane terminal receipt must not contain commands or environment")


def _optional_exit_status(value: object) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 255 else None


def _optional_elapsed_seconds(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and math.isfinite(parsed) else None


def _optional_signal_number(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if 1 <= parsed <= 127 else None


def build_lane_terminal_receipt(context: LaneTerminalContext) -> dict[str, Any]:
    """Build a credential-free terminal record for one started readiness lane."""
    status_value = _optional_exit_status(context.exit_status)
    elapsed_value = _optional_elapsed_seconds(context.elapsed_seconds)
    observed_signal_number = _optional_signal_number(context.supervisor_signal_number)
    supervisor_signal = (
        {**_signal_details(observed_signal_number), "sender_pid": None}
        if observed_signal_number is not None
        else None
    )
    if supervisor_signal is not None:
        outcome = "terminated"
    elif status_value is None:
        outcome = "unknown"
    elif status_value == 0:
        outcome = "completed"
    else:
        # In particular, status 143 alone does not identify a SIGTERM sender.
        outcome = "failed"
    receipt = {
        "schema": LANE_TERMINAL_SCHEMA_VERSION,
        "status": "terminal",
        "outcome": outcome,
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "lane": _bounded_text(context.lane),
        "repository": {
            "id_sha256": _repository_fingerprint(context.repo_root),
            "source_sha": _source_sha(context.source_sha),
        },
        "timing": {
            "started_at_utc": (
                _bounded_text(context.started_at_utc) if context.started_at_utc else None
            ),
            "ended_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_value,
        },
        "exit_status": status_value,
        "signal": {"supervisor": supervisor_signal, "child": None},
        "process": {
            "supervisor_pid": _positive_int(context.supervisor_pid),
            "child_pid": _positive_int(context.child_pid),
            "child_process_group_id": _positive_int(context.child_process_group_id),
        },
        "security": {"command_line_included": False, "environment_included": False},
    }
    validate_lane_terminal_receipt(receipt)
    return receipt


@dataclass(frozen=True, slots=True)
class TerminationContext:
    """Signal-path context supplied by the readiness shell wrapper."""

    signal_number: int
    phase: str
    lane: str
    last_progress: str
    last_progress_at_utc: str
    cleanup_status: str
    mode: str
    controller_pid: object = None
    child_pid: object = None
    child_process_group_id: object = None
    child_registration_state: object = "unknown"


def _registration_state(value: object) -> str:
    """Normalize the shell lifecycle state, rejecting unknown states fail-closed."""
    state = str(value) if value is not None else "unknown"
    return state if state in CHILD_REGISTRATION_STATES else "unknown"


def build_receipt(context: TerminationContext) -> dict[str, Any]:
    """Build the bounded receipt without collecting command or environment data."""
    child_pid_value = _positive_int(context.child_pid)
    process_group_value = _positive_int(context.child_process_group_id)
    child_pid_absent = _identifier_is_absent(context.child_pid)
    process_group_absent = _identifier_is_absent(context.child_process_group_id)
    registration_state = _registration_state(context.child_registration_state)
    process_group_exists = _process_group_exists(process_group_value)
    process_group_liveness = _process_group_liveness(process_group_value)
    direct_cleanup_verified_statuses = {
        "direct_process_already_exited_and_verified",
        "direct_process_killed_and_verified",
        "direct_process_terminated_and_verified",
    }
    process_group_cleanup_verified_statuses = {
        "process_group_already_exited_and_verified",
        "process_group_killed_and_verified",
        "process_group_terminated_and_verified",
    }
    cleanup_verified = False
    if context.cleanup_status == "no_child_active":
        cleanup_verified = (
            registration_state == "not_started" and child_pid_absent and process_group_absent
        )
    elif context.cleanup_status in process_group_cleanup_verified_statuses:
        cleanup_verified = (
            registration_state == "registered"
            and child_pid_value is not None
            and process_group_value is not None
            and process_group_liveness in ("absent", "zombie_only")
        )
    elif context.cleanup_status in direct_cleanup_verified_statuses:
        cleanup_verified = (
            registration_state == "registered"
            and child_pid_value is not None
            and (
                process_group_absent
                or (
                    process_group_value is not None
                    and process_group_liveness in ("absent", "zombie_only")
                )
            )
        )
    cleanup_status = _bounded_text(context.cleanup_status)
    if (
        context.cleanup_status in direct_cleanup_verified_statuses
        or context.cleanup_status in process_group_cleanup_verified_statuses
        or context.cleanup_status == "no_child_active"
    ) and not cleanup_verified:
        cleanup_status = "process_group_cleanup_unverified"
    return {
        "schema": SCHEMA_VERSION,
        "status": "terminated",
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": _bounded_text(context.mode),
        "phase": _bounded_text(context.phase),
        "lane": _bounded_text(context.lane),
        "signal": _signal_details(context.signal_number),
        "last_progress": {
            "message": _bounded_text(context.last_progress),
            "recorded_at_utc": _bounded_text(context.last_progress_at_utc),
        },
        "cleanup": {
            "status": cleanup_status,
            "verified": cleanup_verified,
        },
        "process": {
            "controller_pid": _positive_int(context.controller_pid),
            "child_pid": child_pid_value,
            "child_process_group_id": process_group_value,
            "child_process_group_exists": process_group_exists,
            "child_process_group_liveness": process_group_liveness,
            "child_registration_state": registration_state,
        },
        "resources": _resource_snapshot(),
        "security": {
            "command_line_included": False,
            "environment_included": False,
        },
    }


def write_receipt(receipt: dict[str, Any], output: str | Path) -> Path:
    """Write one receipt atomically without overwriting an existing file."""
    path = Path(os.path.abspath(output))
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite existing receipt: {path}")
    cursor = Path(path.anchor)
    for component in path.relative_to(cursor).parts[:-1]:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError("receipt path must not contain symlink components")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("receipt parent must not be a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError(f"refusing to overwrite existing receipt: {path}") from exc
        temporary.unlink()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _parser() -> argparse.ArgumentParser:
    """Build the receipt writer command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lane-terminal", action="store_true")
    parser.add_argument("--signal-number", type=int)
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--lane", default="none")
    parser.add_argument("--last-progress", default="unknown")
    parser.add_argument("--last-progress-at-utc", default="unknown")
    parser.add_argument("--cleanup-status", default="unknown")
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--controller-pid")
    parser.add_argument("--child-pid")
    parser.add_argument("--child-pgid")
    parser.add_argument("--child-registration-state", default="unknown")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--started-at-utc")
    parser.add_argument("--elapsed-seconds")
    parser.add_argument("--exit-status")
    parser.add_argument("--supervisor-signal-number", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a termination receipt and report its path."""
    args = _parser().parse_args(argv)
    if args.lane_terminal:
        if args.repo_root is None:
            print("ERROR: lane terminal receipt requires --repo-root", file=sys.stderr)
            return 2
        receipt = build_lane_terminal_receipt(
            LaneTerminalContext(
                lane=args.lane,
                repo_root=args.repo_root,
                source_sha=args.source_sha,
                started_at_utc=args.started_at_utc,
                elapsed_seconds=args.elapsed_seconds,
                exit_status=args.exit_status,
                supervisor_pid=args.controller_pid,
                child_pid=args.child_pid,
                child_process_group_id=args.child_pgid,
                supervisor_signal_number=args.supervisor_signal_number,
            )
        )
    else:
        if args.signal_number is None:
            print("ERROR: termination receipt requires --signal-number", file=sys.stderr)
            return 2
        receipt = build_receipt(
            TerminationContext(
                signal_number=args.signal_number,
                phase=args.phase,
                lane=args.lane,
                last_progress=args.last_progress,
                last_progress_at_utc=args.last_progress_at_utc,
                cleanup_status=args.cleanup_status,
                mode=args.mode,
                controller_pid=args.controller_pid,
                child_pid=args.child_pid,
                child_process_group_id=args.child_pgid,
                child_registration_state=args.child_registration_state,
            )
        )
    try:
        path = write_receipt(receipt, args.output)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot write PR readiness termination receipt: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
