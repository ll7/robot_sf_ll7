#!/usr/bin/env python3
"""Write a bounded, credential-free receipt for interrupted PR readiness."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pr_ready_termination.v1"
MAX_TEXT_LENGTH = 200
READ_LIMIT_BYTES = 8192
CGROUP_ROOT = Path("/sys/fs/cgroup")


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


def _signal_details(signal_number: int) -> dict[str, int | str | None]:
    """Return a stable signal name and conventional shell status."""
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"signal_{signal_number}"
    exit_code = 128 + signal_number if 1 <= signal_number <= 127 else None
    return {"name": signal_name, "number": signal_number, "exit_code": exit_code}


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


def build_receipt(context: TerminationContext) -> dict[str, Any]:
    """Build the bounded receipt without collecting command or environment data."""
    child_pid_value = _positive_int(context.child_pid)
    process_group_value = _positive_int(context.child_process_group_id)
    process_group_exists = _process_group_exists(process_group_value)
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
        cleanup_verified = process_group_value is None or process_group_exists is False
    elif context.cleanup_status in process_group_cleanup_verified_statuses:
        cleanup_verified = (
            child_pid_value is not None
            and process_group_value is not None
            and process_group_exists is False
        )
    elif context.cleanup_status in direct_cleanup_verified_statuses:
        cleanup_verified = child_pid_value is not None and (
            process_group_value is None or process_group_exists is False
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
    parser.add_argument("--signal-number", type=int, required=True)
    parser.add_argument("--phase", default="unknown")
    parser.add_argument("--lane", default="none")
    parser.add_argument("--last-progress", default="unknown")
    parser.add_argument("--last-progress-at-utc", default="unknown")
    parser.add_argument("--cleanup-status", default="unknown")
    parser.add_argument("--mode", default="unknown")
    parser.add_argument("--controller-pid")
    parser.add_argument("--child-pid")
    parser.add_argument("--child-pgid")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a termination receipt and report its path."""
    args = _parser().parse_args(argv)
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
