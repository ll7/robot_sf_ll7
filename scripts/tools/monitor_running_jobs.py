#!/usr/bin/env python3
"""Bounded read-only running-job monitor over explicit job receipts (issue #8855).

Reduces supplied or polled sanitized scheduler observations for explicitly listed jobs into
one deterministic ``running_job_monitor.v1`` report. Job, source, and immutable
submission-receipt identity are verified before every state reduction; transitions, observation
timestamps, scheduler evidence digests, array summaries, and the first terminal state are
recorded. A terminal state produces a ``running_job_harvest_handoff.v1`` packet naming the
exact job, expected artifacts/rows, source/config packet, observed terminal state, and the next
canonical ``harvest_terminal_job`` command; it never claims result validity. A hard local
wall-clock limit keeps the monitor finite: on expiry it emits ``monitor_window_expired`` with
the current non-terminal state. The monitor never cancels, retries, submits, resubmits,
harvests, or mutates GitHub. CLI: ``--check --projection <json> --once --format json``
(deterministic, no query) or ``--check --projection <json> --state-query "<command with
{job_id}>" [--interval S] [--max-wall-seconds S]``. Exit codes: 0 report, 2 malformed,
3 monitor_unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tools.classify_scheduler_failure import sanitize_text  # noqa: E402
from scripts.tools.slurm_job_finalize import (  # noqa: E402
    INCOMPLETE_STATES,
    SUCCESS_STATES,
    UNAVAILABLE_STATES,
    normalize_job_state,
)
from scripts.validation.harvest_terminal_job import (  # noqa: E402
    REQUEST_SCHEMA as HARVEST_REQUEST_SCHEMA,
)

if TYPE_CHECKING:
    from collections.abc import Callable

PROJECTION_SCHEMA = "running_job_monitor_projection.v1"
REPORT_SCHEMA = "running_job_monitor.v1"
OBSERVATION_SCHEMA = "running_job_monitor_observation.v1"
HANDOFF_SCHEMA = "running_job_harvest_handoff.v1"
HARVEST_COMMAND_TEMPLATE = (
    "uv run python scripts/validation/harvest_terminal_job.py --check"
    " --request {request} --artifact-root {artifact_root} --format json"
)
CLAIM_BOUNDARY = (
    "Read-only monitor only: observed scheduler states, transitions, and evidence digests are "
    "recorded; scheduler completion is never artifact, benchmark, or scientific success, and "
    "no job is cancelled, retried, submitted, resubmitted, or harvested."
)
TERMINAL_STATES = frozenset(("completed", "failed", "cancelled", "timeout"))
_ACTIVE_RANK = {"pending": 0, "requeued": 1, "configuring": 2, "suspended": 3, "running": 4}
_TERMINAL_RANK = {"failed": 0, "timeout": 1, "cancelled": 2, "completed": 3}
_ALLOWED_NEXT = {
    "pending": frozenset(("pending", "configuring", "running", "suspended", "requeued")),
    "configuring": frozenset(("configuring", "running", "requeued")),
    "requeued": frozenset(("requeued", "pending", "configuring", "running")),
    "suspended": frozenset(("suspended", "running", "requeued")),
    "running": frozenset(("running", "requeued", "suspended")),
    "unavailable": frozenset(
        "pending configuring running suspended requeued completed failed cancelled timeout unavailable".split()
    ),
}
_FATAL_CODES = frozenset(
    ("job_identity_changed", "submission_identity_changed", "private_value_rejected")
)
_IDENTITY_KEYS = (
    "job_id",
    "issue",
    "owner",
    "campaign_id",
    "commit",
    "config_sha256",
    "receipt_sha256",
)
_JOB_KEYS = (
    *_IDENTITY_KEYS,
    "artifacts",
    "rows",
    "harvest_request",
    "harvest_artifact_root",
    "observations",
)
_OBSERVATION_KEYS = ("schema_version", "job_id", "observed_at", "state", "identity_sha256", "array")
EXIT_REPORT, EXIT_MALFORMED, EXIT_UNAVAILABLE = 0, 2, 3
MAX_WALL_SECONDS = 86400.0
DEFAULT_INTERVAL_SECONDS = 30.0
DEFAULT_MAX_WALL_SECONDS = 600.0
DEFAULT_QUERY_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 65536
_Problems = list[tuple[str, str, str]]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LOGICAL_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")
_ARRAY_KEY_RE = re.compile(r"^\d{1,9}$")
_PRIVATE_LOCATOR_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|@)")
_SECRET_RE = re.compile(r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_IDENTITY_PATTERNS = {
    "job_id": _ID_RE,
    "owner": _SLUG_RE,
    "campaign_id": _SLUG_RE,
    "commit": _COMMIT_RE,
    "config_sha256": _SHA256_RE,
    "receipt_sha256": _SHA256_RE,
}


class MonitorContractError(ValueError):
    """Raised when the projection cannot be read as a monitor contract at all."""


class ProjectedJob(NamedTuple):
    """Explicit job identity plus expected outputs, harvest packet, and observations."""

    identity: Mapping[str, Any]
    artifacts: tuple[str, ...]
    rows: tuple[str, ...]
    harvest_request: str
    harvest_artifact_root: str
    observations: tuple[Any, ...]


class Observation(NamedTuple):
    """One validated sanitized scheduler observation."""

    state: str
    observed_at: datetime
    evidence_sha256: str
    array_summary: dict[str, Any] | None


def _add(problems: _Problems, code: str, location: str, message: str) -> None:
    problems.append((code, location, message))


def _stable_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def identity_digest(identity: Mapping[str, Any]) -> str:
    """Return the checksum binding explicit job, source, and submission identity."""
    return _stable_digest({key: identity[key] for key in _IDENTITY_KEYS})


def _match(value: Any, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _private_findings(node: Any, location: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            findings.extend(_private_findings(value, f"{location}/{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            findings.extend(_private_findings(value, f"{location}[{index}]"))
    elif isinstance(node, str):
        if _PRIVATE_LOCATOR_RE.search(node) or _SECRET_RE.search(node):
            findings.append((location, "private or credential-like value"))
        elif _CONTROL_RE.search(node):
            findings.append((location, "control character in value"))
        elif ".." in node.split("/"):
            findings.append((location, "parent path segment"))
    return findings


def _canonical_state(value: Any) -> str | None:
    state = normalize_job_state(value) if isinstance(value, str) else ""
    if state == "COMPLETING":
        return "running"
    for name, members in (
        ("completed", SUCCESS_STATES),
        ("cancelled", {"CANCELLED", "PREEMPTED"}),
        ("failed", {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY"}),
        ("timeout", {"TIMEOUT"}),
        ("unavailable", UNAVAILABLE_STATES),
    ):
        if state in members:
            return name
    return state.lower() if state in INCOMPLETE_STATES else None


def _transition_problem(previous: str, current: str) -> str | None:
    if previous in (current, "unavailable") or current == "unavailable":
        return None
    if previous in TERMINAL_STATES:
        return "contradictory_states"
    if current in TERMINAL_STATES or current in _ALLOWED_NEXT[previous]:
        return None
    return "contradictory_states"


def _slug_list(value: Any, nonempty: bool) -> tuple[str, ...] | None:
    if not isinstance(value, list) or (nonempty and not value):
        return None
    entries = tuple(_match(item, _SLUG_RE) for item in value)
    if any(entry is None for entry in entries) or len(set(entries)) != len(entries):
        return None
    return entries  # type: ignore[return-value]


def _logical_path(value: Any) -> str | None:
    path = _match(value, _LOGICAL_PATH_RE)
    return path if path is not None and ".." not in path.split("/") else None


def _read_array(
    array: Any, aggregate_state: str, location: str
) -> tuple[dict[str, Any] | None, tuple[str, str, str] | None]:
    if not isinstance(array, Mapping) or not array:
        return None, ("malformed_response", location, "non-empty array mapping required")
    keys = list(array)
    if any(not isinstance(key, str) or not _ARRAY_KEY_RE.fullmatch(key) for key in keys):
        return None, ("malformed_response", location, "array indices must be simple numbers")
    states: dict[str, str] = {}
    for key in sorted(keys, key=int):
        state = _canonical_state(array[key])
        if state is None:
            return None, ("unsupported_scheduler_state", f"{location}/{key}", "unknown array state")
        states[key] = state
    counts = Counter(states.values())
    active = sorted(state for state in states.values() if state not in TERMINAL_STATES)
    if any(state != "unavailable" for state in active):
        derived = max((state for state in active if state != "unavailable"), key=_ACTIVE_RANK.get)
    elif active:
        derived = "unavailable"
    else:
        derived = min(counts, key=_TERMINAL_RANK.get)
    if derived != aggregate_state:
        return None, (
            "contradictory_states",
            location,
            "array summary contradicts the aggregate state",
        )
    summary = {
        "elements": len(states),
        "states": {name: counts[name] for name in sorted(counts)},
        "mixed": len(counts) > 1,
        "digest": _stable_digest(states),
    }
    return summary, None


def _check_identity(payload: Mapping[str, Any], job: ProjectedJob) -> tuple[str, str, str] | None:
    identity = job.identity
    if payload.get("schema_version") != OBSERVATION_SCHEMA:
        return "malformed_response", "observation/schema_version", "wrong schema"
    if payload.get("job_id") != identity["job_id"]:
        return "job_identity_changed", "observation/job_id", "job id differs"
    if payload.get("identity_sha256") != identity_digest(identity):
        return "submission_identity_changed", "observation/identity_sha256", "identity differs"
    return None


def _read_observation(payload: Any, job: ProjectedJob) -> tuple[Observation | None, _Problems]:
    findings = _private_findings(payload, "observation")
    if findings:
        return None, [("private_value_rejected", *findings[0])]
    if not isinstance(payload, Mapping):
        return None, [("malformed_response", "observation", "object required")]
    if set(payload) - set(_OBSERVATION_KEYS):
        return None, [("malformed_response", "observation", "unknown field")]
    problem = _check_identity(payload, job)
    if problem is not None:
        return None, [problem]
    observed_at = _parse_timestamp(payload.get("observed_at"))
    if observed_at is None:
        return None, [("malformed_response", "observation/observed_at", "timestamp required")]
    state = _canonical_state(payload.get("state"))
    if state is None:
        return None, [("unsupported_scheduler_state", "observation/state", "unknown state")]
    summary = None
    if payload.get("array") is not None:
        summary, problem = _read_array(payload["array"], state, "observation/array")
        if problem is not None:
            return None, [problem]
    evidence = _stable_digest(payload)
    return Observation(state, observed_at, evidence, summary), []


class JobMonitor:
    """Deterministic transition accumulator for one explicit job identity."""

    def __init__(self, job: ProjectedJob) -> None:
        """Initialize an empty monitor for one projected job."""
        self.job = job
        self.state: str | None = None
        self.first_terminal_state: str | None = None
        self.first_terminal_observed_at: str | None = None
        self.last_observed_at: datetime | None = None
        self.array_summary: dict[str, Any] | None = None
        self.transitions: list[dict[str, Any]] = []
        self.problems: _Problems = []
        self.observation_count = 0
        self.query_failure_count = 0
        self.identity_rejected = False

    @property
    def terminal(self) -> bool:
        """Whether a validated terminal scheduler state was observed."""
        return self.state in TERMINAL_STATES

    @property
    def monitor_status(self) -> str:
        """Return terminal, monitoring, or monitor_unavailable for this job."""
        if self.identity_rejected or self.observation_count == 0:
            return "monitor_unavailable"
        return "terminal" if self.terminal else "monitoring"

    def apply(self, payload: Any, *, source: str) -> None:
        """Verify identity and reduce one observation into a transition or a rejection."""
        observation, problems = _read_observation(payload, self.job)
        if observation is None:
            self.problems.extend(problems)
            self.identity_rejected |= any(code in _FATAL_CODES for code, _, _ in problems)
            return
        if self.last_observed_at is not None and observation.observed_at <= self.last_observed_at:
            _add(self.problems, "clock_regression", "observation/observed_at", "time regressed")
            return
        if self.state is not None:
            problem = _transition_problem(self.state, observation.state)
            if problem is not None:
                _add(
                    self.problems,
                    problem,
                    "observation/state",
                    f"{self.state} -> {observation.state}",
                )
                return
        previous = self.state
        self.state = observation.state
        self.array_summary = observation.array_summary or self.array_summary
        self.last_observed_at = observation.observed_at
        self.observation_count += 1
        self.transitions.append(
            {
                "from": previous,
                "to": observation.state,
                "observed_at": observation.observed_at.isoformat(),
                "evidence_sha256": observation.evidence_sha256,
                "source": source,
                "array_digest": (
                    observation.array_summary["digest"] if observation.array_summary else None
                ),
            }
        )
        if observation.state in TERMINAL_STATES and self.first_terminal_state is None:
            self.first_terminal_state = observation.state
            self.first_terminal_observed_at = observation.observed_at.isoformat()

    def _handoff(self) -> dict[str, Any]:
        identity = self.job.identity
        return {
            "schema_version": HANDOFF_SCHEMA,
            "job_id": identity["job_id"],
            "issue": identity["issue"],
            "campaign_id": identity["campaign_id"],
            "identity_sha256": identity_digest(identity),
            "source": {"commit": identity["commit"], "config_sha256": identity["config_sha256"]},
            "submission_receipt_sha256": identity["receipt_sha256"],
            "observed_terminal_state": self.first_terminal_state,
            "first_terminal_observed_at": self.first_terminal_observed_at,
            "expected_artifacts": list(self.job.artifacts),
            "expected_rows": list(self.job.rows),
            "harvest_request_schema": HARVEST_REQUEST_SCHEMA,
            "harvest_request": self.job.harvest_request,
            "harvest_artifact_root": self.job.harvest_artifact_root,
            "next_command": HARVEST_COMMAND_TEMPLATE.format(
                request=self.job.harvest_request, artifact_root=self.job.harvest_artifact_root
            ),
            "result_validity": "not_evaluated",
        }

    def as_record(self) -> dict[str, Any]:
        """Return the deterministic public record for this job."""
        identity = self.job.identity
        transitions = sorted(
            self.transitions, key=lambda item: (item["observed_at"], item["evidence_sha256"])
        )
        return {
            "job_id": identity["job_id"],
            "issue": identity["issue"],
            "campaign_id": identity["campaign_id"],
            "identity_sha256": identity_digest(identity),
            "source": {"commit": identity["commit"], "config_sha256": identity["config_sha256"]},
            "submission_receipt_sha256": identity["receipt_sha256"],
            "scheduler": {
                "state": self.state or "unavailable",
                "terminal": self.terminal,
                "first_terminal_state": self.first_terminal_state,
                "first_terminal_observed_at": self.first_terminal_observed_at,
            },
            "array_summary": self.array_summary,
            "transitions": transitions,
            "observation_count": self.observation_count,
            "query_failure_count": self.query_failure_count,
            "job_status": self.monitor_status,
            "handoff": self._handoff() if self.terminal else None,
            "problems": [
                {"code": code, "location": location, "message": message}
                for code, location, message in sorted(set(self.problems))
            ],
            "result_validity": "not_evaluated",
        }


def _parse_job(raw: Any, location: str, problems: _Problems) -> ProjectedJob | None:
    findings = (
        _private_findings(
            {key: value for key, value in raw.items() if key != "observations"}, location
        )
        if isinstance(raw, Mapping)
        else []
    )
    if findings:
        _add(problems, "private_value_rejected", *findings[0])
        return None
    if not isinstance(raw, Mapping):
        _add(problems, "malformed_projection", location, "job entry must be an object")
        return None
    if set(raw) - set(_JOB_KEYS):
        _add(problems, "malformed_projection", location, "unknown job field")
        return None
    identity = {key: _match(raw.get(key), pattern) for key, pattern in _IDENTITY_PATTERNS.items()}
    identity["issue"] = _positive_int(raw.get("issue"))
    if any(value is None for value in identity.values()):
        _add(problems, "invalid_job_identity", location, "identity fields required")
        return None
    artifacts = _slug_list(raw.get("artifacts"), nonempty=True)
    rows = _slug_list(raw.get("rows"), nonempty=False)
    request = _logical_path(raw.get("harvest_request"))
    root = _logical_path(raw.get("harvest_artifact_root"))
    observations = raw.get("observations", [])
    if (
        artifacts is None
        or rows is None
        or request is None
        or root is None
        or not isinstance(observations, list)
    ):
        _add(
            problems,
            "invalid_job_packet",
            location,
            "artifacts, rows, harvest packet, and observations are required",
        )
        return None
    return ProjectedJob(identity, artifacts, rows, request, root, tuple(observations))


def parse_projection(payload: Any) -> tuple[list[ProjectedJob], _Problems]:
    """Validate one projection document into explicit jobs plus fail-closed problems."""
    if not isinstance(payload, Mapping):
        raise MonitorContractError("projection must be a JSON object")
    if payload.get("schema_version") != PROJECTION_SCHEMA:
        raise MonitorContractError(f"projection schema must be {PROJECTION_SCHEMA}")
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise MonitorContractError("projection must list at least one explicit job")
    problems: _Problems = []
    jobs: list[ProjectedJob] = []
    seen: set[str] = set()
    for index, raw in enumerate(jobs_raw):
        job = _parse_job(raw, f"jobs[{index}]", problems)
        if job is None:
            continue
        job_id = job.identity["job_id"]
        if job_id in seen:
            _add(problems, "duplicate_job_identity", f"jobs[{index}]", "duplicate job id")
            continue
        seen.add(job_id)
        jobs.append(job)
    return jobs, problems


def _build_report(
    monitors: list[JobMonitor],
    projection_problems: _Problems,
    *,
    mode: str,
    interval: float | None,
    max_wall_seconds: float | None,
    expired: bool,
) -> dict[str, Any]:
    records = sorted((monitor.as_record() for monitor in monitors), key=lambda r: r["job_id"])
    problems: list[dict[str, Any]] = []
    for record in records:
        problems.extend({"job_id": record["job_id"], **problem} for problem in record["problems"])
    problems.extend(
        {"job_id": None, "code": code, "location": location, "message": message}
        for code, location, message in projection_problems
    )
    problems.sort(key=lambda item: (item["job_id"] or "", item["code"], item["location"]))
    if projection_problems or any(m.monitor_status == "monitor_unavailable" for m in monitors):
        status = "monitor_unavailable"
    elif all(m.terminal for m in monitors):
        status = "terminal"
    else:
        status = "monitor_window_expired" if expired else "monitoring"
    counts = Counter(record["job_status"] for record in records)
    return {
        "schema_version": REPORT_SCHEMA,
        "check_only": True,
        "monitor_mode": mode,
        "status": status,
        "window": (
            {"interval_seconds": interval, "max_wall_seconds": max_wall_seconds, "expired": expired}
            if mode == "state_query"
            else None
        ),
        "counts": {
            "jobs": len(records),
            "terminal": counts.get("terminal", 0),
            "monitoring": counts.get("monitoring", 0),
            "monitor_unavailable": counts.get("monitor_unavailable", 0),
        },
        "jobs": records,
        "problems": problems,
        "problem_count": len(problems),
        "reason_codes": sorted({problem["code"] for problem in problems}),
        "result_validity": "not_evaluated",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def monitor_once(projection_payload: Any) -> dict[str, Any]:
    """Reduce the observations embedded in one projection without any query."""
    jobs, problems = parse_projection(projection_payload)
    monitors = [JobMonitor(job) for job in jobs]
    for monitor in monitors:
        for payload in monitor.job.observations:
            monitor.apply(payload, source="projection")
    return _build_report(
        monitors, problems, mode="projection", interval=None, max_wall_seconds=None, expired=False
    )


def monitor_live(
    projection_payload: Any,
    query: Callable[[str], Any],
    *,
    interval: float,
    max_wall_seconds: float,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Poll explicit jobs until terminal or the hard wall-clock bound expires."""
    clock, sleeper = clock or time.monotonic, sleeper or time.sleep
    jobs, problems = parse_projection(projection_payload)
    monitors = [JobMonitor(job) for job in jobs]
    started = clock()
    expired = False
    while True:
        for monitor in monitors:
            if monitor.terminal or monitor.identity_rejected:
                continue
            try:
                payload: Any = query(monitor.job.identity["job_id"])
            except (OSError, ValueError, subprocess.SubprocessError):
                payload = None
            if payload is None:
                monitor.query_failure_count += 1
                _add(monitor.problems, "query_unavailable", "observation", "no evidence")
            else:
                monitor.apply(payload, source="state_query")
        if all(m.terminal or m.identity_rejected for m in monitors):
            break
        remaining = max_wall_seconds - (clock() - started)
        if remaining <= 0:
            expired = True
            break
        sleeper(min(interval, remaining))
    return _build_report(
        monitors,
        problems,
        mode="state_query",
        interval=interval,
        max_wall_seconds=max_wall_seconds,
        expired=expired,
    )


def render_report_json(report: Mapping[str, Any]) -> str:
    """Return byte-stable monitor JSON with sorted keys and a trailing newline."""
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_report_text(report: Mapping[str, Any]) -> str:
    """Return the concise deterministic human monitor summary."""
    counts = report["counts"]
    jobs = [
        f"{job['job_id']}:{job['scheduler']['state']}"
        f"(terminal={str(job['scheduler']['terminal']).lower()},"
        f"handoff={'yes' if job['handoff'] else 'no'})"
        for job in report["jobs"]
    ]
    return (
        f"status={report['status']} jobs={counts['jobs']} terminal={counts['terminal']} "
        f"unavailable={counts['monitor_unavailable']} | {'; '.join(jobs) or 'none'}\n"
        f"{report['claim_boundary']}\n"
    )


def _make_query(template: str, timeout: float) -> Callable[[str], Mapping[str, Any] | None]:
    argv = shlex.split(template)
    if not argv:
        raise MonitorContractError("--state-query must not be empty")

    def query(job_id: str) -> Mapping[str, Any] | None:
        command = [part.replace("{job_id}", job_id) for part in argv]
        try:
            completed = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0 or len(completed.stdout) > MAX_RESPONSE_BYTES:
            return None
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        return payload if isinstance(payload, Mapping) else None

    return query


def main(argv: list[str] | None = None) -> int:
    """Run the read-only monitor CLI and return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True, help="read-only report")
    parser.add_argument("--projection", type=Path, required=True, help="sanitized projection JSON")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="reduce embedded observations only")
    mode.add_argument("--state-query", help="read-only query command with {job_id}")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    parser.add_argument("--query-timeout", type=float, default=DEFAULT_QUERY_TIMEOUT_SECONDS)
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    if args.interval <= 0 or not 0 < args.max_wall_seconds <= MAX_WALL_SECONDS:
        print("error: --interval must be positive and --max-wall-seconds within (0, 86400]")
        return EXIT_MALFORMED
    try:
        payload = json.loads(args.projection.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise MonitorContractError("projection must be a JSON object")
        report = (
            monitor_once(payload)
            if args.once
            else monitor_live(
                payload,
                _make_query(args.state_query, args.query_timeout),
                interval=args.interval,
                max_wall_seconds=args.max_wall_seconds,
            )
        )
    except (OSError, ValueError) as exc:
        print(f"running job monitor: malformed input: {sanitize_text(str(exc))}", file=sys.stderr)
        return EXIT_MALFORMED
    rendered = render_report_json(report) if args.format == "json" else render_report_text(report)
    print(rendered, end="")
    return EXIT_UNAVAILABLE if report["status"] == "monitor_unavailable" else EXIT_REPORT


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
