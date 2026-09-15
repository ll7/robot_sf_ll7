#!/usr/bin/env python3
"""Validate hash-bound terminal Slurm closeout receipts.

The scheduler is the authority for execution state.  An admission or queue
record is not allowed to remain the only source of truth after a terminal
readback.  This module owns the small, public validation contract consumed by
the release doctor; private-ops may produce the JSON without importing this
checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA_VERSION = "robot-sf-slurm-terminal-closeout.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
JOB_ID_RE = re.compile(r"^[0-9]+$")
EXIT_CODE_RE = re.compile(r"^[0-9]+:[0-9]+$")
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
ACTIVE_STATES = frozenset({"PENDING", "CONFIGURING", "RUNNING", "SUSPENDED", "COMPLETING"})
UNAVAILABLE_STATES = frozenset({"UNAVAILABLE", "UNKNOWN", "MISSING"})
SECRET_KEY_RE = re.compile(
    r"(?:token|secret|password|credential|private[_-]?key|authorization)", re.I
)


def _identity_digest(campaign_id: str, source_sha: str, job_id: str) -> str:
    """Return the deterministic identity digest bound into every receipt."""
    value = f"{campaign_id}\0{source_sha}\0{job_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _normalise_state(value: object) -> str:
    return str(value or "").strip().upper().split()[0] if str(value or "").strip() else ""


def _valid_sha(value: object, *, length: int = 64) -> bool:
    candidate = str(value or "")
    return bool((SHA256_RE if length == 64 else SHA1_RE).fullmatch(candidate))


def _contains_secret_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            SECRET_KEY_RE.search(str(key)) or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def validate_payload(  # noqa: C901, PLR0912, PLR0915
    payload: object,
    *,
    expected_campaign_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_job_id: str | None = None,
) -> list[str]:
    """Return sanitized validation problems for one closeout payload.

    The function never returns raw payload values in an error.  Callers can
    therefore safely expose the resulting messages in a release doctor report.
    """
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["closeout receipt is not a JSON object"]
    if payload.get("schema") != SCHEMA_VERSION:
        problems.append("closeout receipt schema is unsupported")
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"terminal", "unavailable"}:
        problems.append("closeout receipt status is not terminal or unavailable")
    if _contains_secret_key(payload):
        problems.append("closeout receipt contains a credential-shaped field")

    campaign_id = str(payload.get("campaign_id") or "").strip()
    source_sha = str(payload.get("source_sha") or "").strip().lower()
    job_id = str(payload.get("job_id") or "").strip()
    if not campaign_id:
        problems.append("closeout receipt campaign_id is missing")
    if not SHA1_RE.fullmatch(source_sha):
        problems.append("closeout receipt source_sha is not a 40-character commit SHA")
    if not JOB_ID_RE.fullmatch(job_id):
        problems.append("closeout receipt job_id is not numeric")
    if expected_campaign_id is not None and campaign_id != expected_campaign_id:
        problems.append("closeout receipt campaign_id does not match the expected campaign")
    if expected_source_sha is not None and source_sha != expected_source_sha.lower():
        problems.append("closeout receipt source_sha does not match the expected source")
    if expected_job_id is not None and job_id != str(expected_job_id):
        problems.append("closeout receipt job_id does not match the expected job")

    identity_digest = str(payload.get("identity_sha256") or "").lower()
    if not _valid_sha(identity_digest):
        problems.append("closeout receipt identity_sha256 is missing or malformed")
    elif campaign_id and SHA1_RE.fullmatch(source_sha) and JOB_ID_RE.fullmatch(job_id):
        if identity_digest != _identity_digest(campaign_id, source_sha, job_id):
            problems.append("closeout receipt identity digest does not match its identity")

    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        problems.append("closeout receipt scheduler section is missing")
        scheduler = {}
    state = _normalise_state(scheduler.get("state"))
    if status == "terminal" and state not in TERMINAL_STATES:
        problems.append("terminal closeout receipt does not contain a terminal scheduler state")
    if status == "unavailable" and state not in UNAVAILABLE_STATES:
        problems.append(
            "unavailable closeout receipt does not contain an unavailable scheduler state"
        )
    if status == "unavailable":
        problems.append("scheduler readback is unavailable; release admission is blocked")
    if state in ACTIVE_STATES:
        problems.append("closeout receipt still claims an active scheduler state")
    for field in ("exit_code", "derived_exit_code"):
        value = str(scheduler.get(field) or "").strip()
        if value and value.casefold() != "unavailable" and not EXIT_CODE_RE.fullmatch(value):
            problems.append(f"closeout receipt scheduler {field} is malformed")
    if state == "COMPLETED" and str(scheduler.get("exit_code") or "") != "0:0":
        problems.append("COMPLETED closeout receipt does not carry exit code 0:0")
    if state == "COMPLETED" and str(scheduler.get("derived_exit_code") or "") not in {"", "0:0"}:
        problems.append("COMPLETED closeout receipt has a non-zero derived exit code")
    if not str(scheduler.get("elapsed") or "").strip() and status == "terminal":
        problems.append("terminal closeout receipt elapsed time is missing")

    query = payload.get("query")
    if not isinstance(query, dict):
        problems.append("closeout receipt query section is missing")
        query = {}
    if not str(query.get("tool") or "").strip():
        problems.append("closeout receipt query tool is missing")
    if not str(query.get("tool_version") or "").strip():
        problems.append("closeout receipt query tool version is missing")
    if not str(query.get("queried_at") or "").strip():
        problems.append("closeout receipt query timestamp is missing")

    allocation = payload.get("allocation")
    if not isinstance(allocation, dict):
        problems.append("closeout receipt allocation contract is missing")
        allocation = {}
    for field in ("cluster", "partition", "cpus", "gpus", "mem_gb"):
        if str(allocation.get(field) or "").strip() == "":
            problems.append(f"closeout receipt allocation {field} is missing")

    reconciliation = payload.get("reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("status") != "reconciled":
        problems.append("closeout receipt does not prove terminal admission reconciliation")
    else:
        prior = _normalise_state(reconciliation.get("prior_scheduler_state"))
        if prior in TERMINAL_STATES and prior == state:
            # A repeated terminal readback is fine; this branch only documents
            # that a second query did not silently alter the scheduler verdict.
            pass

    output = payload.get("output")
    if not isinstance(output, dict):
        problems.append("closeout receipt output section is missing")
    else:
        digest = output.get("digest_sha256")
        if digest is not None and not _valid_sha(digest):
            problems.append("closeout receipt output digest is malformed")
        if status == "terminal" and digest is None:
            problems.append("terminal closeout receipt has no output digest")

    return list(dict.fromkeys(problems))


def validate_file(
    path: Path,
    *,
    expected_campaign_id: str | None = None,
    expected_source_sha: str | None = None,
    expected_job_id: str | None = None,
) -> list[str]:
    """Read and validate a receipt without exposing its contents."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["scheduler closeout receipt is missing"]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ["scheduler closeout receipt cannot be read"]
    return validate_payload(
        payload,
        expected_campaign_id=expected_campaign_id,
        expected_source_sha=expected_source_sha,
        expected_job_id=expected_job_id,
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expected-campaign-id")
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--expected-job-id")
    args = parser.parse_args()
    problems = validate_file(
        args.receipt,
        expected_campaign_id=args.expected_campaign_id,
        expected_source_sha=args.expected_source_sha,
        expected_job_id=args.expected_job_id,
    )
    print(json.dumps({"status": "pass" if not problems else "blocked", "problems": problems}))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(_main())
