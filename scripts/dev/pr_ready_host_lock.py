#!/usr/bin/env python3
"""Host-wide kernel-backed admission for the canonical PR readiness wrapper.

The lock descriptor is opened and retained by ``pr_ready_check.sh``. This helper
uses that inherited descriptor to acquire or report on the lock; the kernel lock,
never the owner JSON file, is the admission authority.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "robot_sf.pr_ready_host_admission"
VERSION = 1
_HEAD_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_START_RE = re.compile(r"^linux-proc:([0-9a-f-]{36}):([0-9]+)$")


class AdmissionError(RuntimeError):
    """An ambiguous or unavailable host admission state must stop readiness."""


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _process_identity(pid: int) -> str:
    """Return a PID-reuse-resistant Linux process identity or fail closed."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        stat_end = stat_text.rfind(")")
        if stat_end < 0:
            raise ValueError("malformed proc stat")
        # The first token after the command is field 3 (state); starttime is 22.
        fields = stat_text[stat_end + 1 :].split()
        if len(fields) <= 19 or fields[0] == "Z":
            raise ValueError("process is absent, a zombie, or has incomplete proc stat")
        start_ticks = fields[19]
        if not start_ticks.isdecimal():
            raise ValueError("invalid proc start-time field")
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", boot_id):
            raise ValueError("invalid Linux boot identity")
    except (OSError, UnicodeError, ValueError) as exc:
        raise AdmissionError(
            f"cannot verify process-start identity for PID {pid}; refusing host admission ({exc})"
        ) from exc
    return f"linux-proc:{boot_id.lower()}:{start_ticks}"


def _validate_run_identity(pid: int, worktree: str, head: str) -> tuple[str, str]:
    if pid < 1:
        raise AdmissionError("invalid readiness process PID; refusing host admission")
    if not Path(worktree).is_absolute():
        raise AdmissionError("canonical worktree path must be absolute; refusing host admission")
    canonical_worktree = str(Path(worktree).resolve(strict=True))
    if canonical_worktree != worktree:
        raise AdmissionError("worktree path is not canonical; refusing host admission")
    if not _HEAD_RE.fullmatch(head):
        raise AdmissionError("exact HEAD must be a full Git object ID; refusing host admission")
    return _process_identity(pid), canonical_worktree


def _base_record(*, pid: int, identity: str, worktree: str, head: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "timestamp_utc": _timestamp(),
        "pid": pid,
        "process_start_identity": identity,
        "canonical_worktree_path": worktree,
        "head": head,
    }


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _append_event(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"PR_READY_HOST_ADMISSION_RECEIPT {payload.decode().rstrip()}", file=sys.stderr)


def _read_owner(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError(f"host admission owner record is unreadable; refusing to wait ({exc})")
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCHEMA
        or value.get("version") != VERSION
    ):
        raise AdmissionError("host admission owner record has an unknown schema; refusing to wait")
    status = value.get("status")
    if status == "released":
        return None
    required = (
        "pid",
        "process_start_identity",
        "canonical_worktree_path",
        "head",
        "timestamp_utc",
    )
    if status != "held" or any(key not in value for key in required):
        raise AdmissionError(
            "host admission owner record is incomplete or ambiguous; refusing to wait"
        )
    pid = value["pid"]
    identity = value["process_start_identity"]
    worktree = value["canonical_worktree_path"]
    head = value["head"]
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid < 1
        or not isinstance(identity, str)
        or not _START_RE.fullmatch(identity)
        or not isinstance(worktree, str)
        or not Path(worktree).is_absolute()
        or not isinstance(head, str)
        or not _HEAD_RE.fullmatch(head)
        or not isinstance(value["timestamp_utc"], str)
    ):
        raise AdmissionError(
            "host admission owner record has invalid identity fields; refusing to wait"
        )
    actual_identity = _process_identity(pid)
    if actual_identity != identity:
        raise AdmissionError(
            f"host admission owner PID {pid} has a different process-start identity; "
            "the record may refer to a reused PID, so readiness is refused"
        )
    return {key: value[key] for key in (*required, "status")}


def _event(
    receipt_path: Path,
    *,
    status: str,
    pid: int,
    identity: str,
    worktree: str,
    head: str,
    waiting_for: dict[str, Any] | None = None,
    detail: str | None = None,
) -> None:
    record = _base_record(pid=pid, identity=identity, worktree=worktree, head=head)
    record["status"] = status
    if waiting_for is not None:
        record["waiting_for"] = waiting_for
    if detail is not None:
        record["detail"] = detail
    _append_event(receipt_path, record)


class _WaitReceipts:
    """Track owner transitions and emit wait records without trusting the anchor."""

    def __init__(
        self,
        *,
        receipt_path: Path,
        pid: int,
        identity: str,
        worktree: str,
        head: str,
        metadata_grace: float,
    ) -> None:
        self.receipt_path = receipt_path
        self.pid = pid
        self.identity = identity
        self.worktree = worktree
        self.head = head
        self.metadata_grace = metadata_grace
        self.last_waiting_for: tuple[int, str, str, str] | None = None
        self.pending_since: float | None = None
        self.pending_event_written = False

    def record(self, owner: dict[str, Any] | None) -> None:
        """Write a wait receipt or fail when the current owner is ambiguous."""
        if owner is None:
            if self.pending_since is None:
                self.pending_since = time.monotonic()
            if not self.pending_event_written:
                _event(
                    self.receipt_path,
                    status="wait",
                    pid=self.pid,
                    identity=self.identity,
                    worktree=self.worktree,
                    head=self.head,
                    detail="owner_metadata_transition_pending",
                )
                self.pending_event_written = True
            if time.monotonic() - self.pending_since >= self.metadata_grace:
                raise AdmissionError(
                    "kernel host admission lock is busy but owner metadata is missing or released; "
                    "refusing to infer ownership"
                )
            return

        self.pending_since = None
        self.pending_event_written = False
        owner_key = (
            owner["pid"],
            owner["process_start_identity"],
            owner["canonical_worktree_path"],
            owner["head"],
        )
        if owner_key != self.last_waiting_for:
            _event(
                self.receipt_path,
                status="wait",
                pid=self.pid,
                identity=self.identity,
                worktree=self.worktree,
                head=self.head,
                waiting_for=owner,
            )
            self.last_waiting_for = owner_key


def _acquire(args: argparse.Namespace) -> int:
    owner_file = Path(args.owner_file)
    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity, worktree = _validate_run_identity(args.pid, args.worktree, args.head)
    receipt_path = (
        receipt_dir / f"{args.pid}.{identity.rsplit(':', 1)[-1]}.{uuid.uuid4().hex}.jsonl"
    )
    # Create the invocation record before contention so even a waiter/failure
    # has a durable, private machine-readable receipt.
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)

    wait_receipts = _WaitReceipts(
        receipt_path=receipt_path,
        pid=args.pid,
        identity=identity,
        worktree=worktree,
        head=args.head,
        metadata_grace=args.metadata_grace,
    )
    while True:
        try:
            fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise AdmissionError(
                    f"kernel host admission lock failed; refusing readiness ({exc})"
                )
        else:
            held = _base_record(pid=args.pid, identity=identity, worktree=worktree, head=args.head)
            held.update({"status": "held", "receipt_path": str(receipt_path)})
            try:
                _write_json_atomic(owner_file, held)
                _event(
                    receipt_path,
                    status="held",
                    pid=args.pid,
                    identity=identity,
                    worktree=worktree,
                    head=args.head,
                )
            except Exception:
                released = dict(held)
                released.update({"status": "released", "timestamp_utc": _timestamp()})
                try:
                    _write_json_atomic(owner_file, released)
                except OSError:
                    pass
                raise
            print(receipt_path)
            return 0

        owner = _read_owner(owner_file)
        # A released/missing record is only a brief publication transition.
        # The kernel lock remains authoritative; prolonged ambiguity fails closed.
        wait_receipts.record(owner)
        time.sleep(args.wait_interval)


def _release(args: argparse.Namespace) -> int:
    owner_file = Path(args.owner_file)
    try:
        with owner_file.open(encoding="utf-8") as handle:
            raw_owner = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A waiter or a helper interrupted before owner publication does not own
        # this slot. Bash closes its descriptor after this quiet no-op.
        return 0
    if (
        not isinstance(raw_owner, dict)
        or raw_owner.get("schema") != SCHEMA
        or raw_owner.get("version") != VERSION
        or raw_owner.get("status") != "held"
        or raw_owner.get("pid") != args.pid
        or raw_owner.get("canonical_worktree_path") != args.worktree
        or raw_owner.get("head") != args.head
    ):
        return 0

    identity, worktree = _validate_run_identity(args.pid, args.worktree, args.head)
    if raw_owner.get("process_start_identity") != identity:
        return 0
    current = _read_owner(owner_file)
    if current is None or (
        current["pid"],
        current["process_start_identity"],
        current["canonical_worktree_path"],
        current["head"],
    ) != (args.pid, identity, worktree, args.head):
        raise AdmissionError(
            "host admission owner record does not match this run; refusing to rewrite it"
        )
    try:
        fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise AdmissionError(
            f"this readiness process no longer holds its host lock ({exc})"
        ) from exc

    receipt_path_value = None
    try:
        with owner_file.open(encoding="utf-8") as handle:
            full_owner = json.load(handle)
        receipt_path_value = full_owner.get("receipt_path")
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    if not isinstance(receipt_path_value, str) or not Path(receipt_path_value).is_absolute():
        raise AdmissionError("host admission receipt path is missing; refusing silent release")
    receipt_path = Path(receipt_path_value)
    released = _base_record(pid=args.pid, identity=identity, worktree=worktree, head=args.head)
    released["status"] = "released"
    _append_event(receipt_path, released)
    released["receipt_path"] = str(receipt_path)
    _write_json_atomic(owner_file, released)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire", "release"):
        command = subparsers.add_parser(name)
        command.add_argument("--lock-fd", required=True, type=int)
        command.add_argument("--owner-file", required=True)
        command.add_argument("--pid", required=True, type=int)
        command.add_argument("--worktree", required=True)
        command.add_argument("--head", required=True)
        if name == "acquire":
            command.add_argument("--receipt-dir", required=True)
            command.add_argument("--wait-interval", type=float, default=0.1)
            command.add_argument("--metadata-grace", type=float, default=2.0)
    return parser


def main() -> int:
    """Run one host-admission operation and report fail-closed diagnostics."""
    args = _parser().parse_args()
    try:
        if args.command == "acquire":
            if args.wait_interval <= 0 or args.metadata_grace < 0:
                raise AdmissionError("host admission wait timing must be nonnegative")
            return _acquire(args)
        return _release(args)
    except (AdmissionError, OSError, ValueError) as exc:
        print(f"PR readiness host admission failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
