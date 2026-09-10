#!/usr/bin/env python3
"""Fail-closed expiring-resource deadline feasibility check for campaign manifests (#8905).

A job can finish inside an expiring host, account, quota, licence, or data-route window and still
lose its outputs when retrieval, checksum verification, and preservation are not budgeted. This
tool reads a campaign manifest (JSON or YAML) and evaluates its optional ``expiring_resource``
contract block, computing exactly one verdict -- ``fits_conservative``, ``fits_expected``,
``too_late``, or ``unknown`` -- without guessing a scheduler start time.

Historical manifests without the block stay non-blocking (``contract_absent``). Naive or
incompatible timestamps, stale deadline evidence, negative/zero durable reserves, missing bases,
and wall-time contradictions fail closed. Reports echo only sanitized enums, slugs, timestamps,
durations, verdicts, and reason codes -- never locators, host names, mount paths, or routes.

CLI (check-only): ``--manifest PATH [--case NAME] --check [--json] [--as-of ISO]``; exit codes
0 fit/not applicable, 1 ``too_late``, 2 ``unknown``/unreadable. Fixture pack:
``tests/validation/fixtures/expiring_resource_feasibility/cases.json``; contract:
``docs/context/expiring_resource_deadlines.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

CONTRACT_SCHEMA = "expiring_resource_contract.v1"
REPORT_SCHEMA = "expiring_resource_feasibility_report.v1"
CASES_SCHEMA = "expiring_resource_feasibility_cases.v1"
DEFAULT_MAX_EVIDENCE_AGE_SECONDS = 7 * 86_400
DEADLINE_KINDS = ("known", "estimated", "unknown", "none")
QUEUE_START_STATUSES = ("not_queued", "queued", "running", "unknown")
RESOURCE_CLASSES = (
    "local slurm_cpu slurm_gpu carla_host multi_host host account cache licence data_route".split()
)
RETENTION_CLASSES = (
    "durable_required release_facing historical diagnostic superseded disposable".split()
)
DURABLE_RETENTION = frozenset({"durable_required", "release_facing"})
ADMISSION_POLICIES = ("block", "report")
POSITIVE_VERDICTS = frozenset({"fits_conservative", "fits_expected"})
MAX_UTC_OFFSET = timedelta(hours=14)
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
RESERVE_FIELDS = ("retrieval_seconds", "verification_seconds", "preservation_seconds")


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Deterministic public-safe deadline-feasibility result for one manifest."""

    manifest_id: str
    verdict: str
    applicable: bool
    admission_policy: str
    deadline_kind: str
    queue_start_status: str
    reason_codes: tuple[str, ...]
    deadline_utc: str | None = None
    as_of_utc: str | None = None
    latest_safe_submission_utc: str | None = None
    latest_safe_submission_expected_utc: str | None = None
    reserves_seconds: Mapping[str, int] | None = None

    @property
    def blocking(self) -> bool:
        """Return True when the declared policy blocks this contract."""
        return (
            self.applicable
            and self.admission_policy == "block"
            and self.verdict not in POSITIVE_VERDICTS
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the byte-stable report payload."""
        return {"schema": REPORT_SCHEMA, "blocking": self.blocking, **asdict(self)}


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _slug(value: object) -> str | None:
    return value if isinstance(value, str) and SLUG_RE.fullmatch(value) else None


def _enum(value: object, allowed: Sequence[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _render(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z") if moment else None


def _parse_timestamp(value: object) -> tuple[datetime | None, str | None]:
    """Parse one timezone-aware ISO 8601 timestamp; never guesses a zone."""
    raw = _text(value)
    if raw is None:
        return None, "missing_timestamp"
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None, "invalid_timestamp"
    if parsed.tzinfo is None:
        return None, "naive_timestamp"
    offset = parsed.utcoffset()
    if offset is None or offset % timedelta(minutes=1) or abs(offset) > MAX_UTC_OFFSET:
        return None, "incompatible_timezone"
    return parsed.astimezone(UTC), None


def _parse_deadline(  # noqa: C901 - explicit per-field deadline evidence validation
    raw: object, as_of: datetime | None, max_age: int, problems: list[str]
) -> tuple[datetime | None, str]:
    if not isinstance(raw, Mapping):
        problems.append("missing_deadline")
        return None, "unknown"
    kind = _enum(raw.get("kind"), DEADLINE_KINDS)
    if kind is None:
        problems.append("missing_deadline_kind")
        return None, "unknown"
    if kind in {"unknown", "none"}:
        return None, kind
    deadline, code = _parse_timestamp(raw.get("timestamp"))
    if deadline is None:
        problems.append(code or "missing_timestamp")
    if _slug(raw.get("source")) is None:
        problems.append("missing_deadline_source")
    freshness = _enum(raw.get("freshness"), ("fresh", "stale", "unknown"))
    if freshness is None:
        problems.append("missing_deadline_freshness")
    elif freshness != "fresh":
        problems.append(
            "stale_deadline_source" if freshness == "stale" else "unknown_deadline_source"
        )
    if raw.get("evidence_as_of") is not None:
        evidence, code = _parse_timestamp(raw["evidence_as_of"])
        if evidence is None:
            problems.append(code or "invalid_timestamp")
        elif as_of is not None and (as_of - evidence).total_seconds() > max_age:
            problems.append("expired_deadline_evidence")
    return deadline, kind


def _parse_runtime(raw: object, problems: list[str]) -> tuple[int, int] | None:
    if not isinstance(raw, Mapping):
        problems.append("missing_runtime")
        return None
    expected, conservative = (
        _int(raw.get("expected_seconds")),
        _int(raw.get("conservative_seconds")),
    )
    if expected is None or expected <= 0 or conservative is None or conservative <= 0:
        problems.append("invalid_runtime")
        return None
    if conservative < expected:
        problems.append("runtime_estimate_conflict")
    if _slug(raw.get("basis")) is None:
        problems.append("missing_runtime_basis")
    return expected, conservative


def _parse_reserves(
    raw: object, retention: str | None, problems: list[str]
) -> dict[str, int] | None:
    if not isinstance(raw, Mapping):
        problems.append("missing_reserves")
        return None
    values: dict[str, int] = {}
    for name in RESERVE_FIELDS:
        value = _int(raw.get(name))
        if value is None or value < 0:
            problems.append("invalid_reserve" if value is None else "negative_reserve")
            return None
        values[name] = value
    if retention in DURABLE_RETENTION and (
        values["retrieval_seconds"] == 0 or values["preservation_seconds"] == 0
    ):
        problems.append("zero_reserve")
    return values


def _check_output(
    raw: object, throughput: object, reserves: dict[str, int] | None, problems: list[str]
) -> None:
    if not isinstance(raw, Mapping):
        problems.append("missing_output_size")
        return
    size = _int(raw.get("estimate_bytes"))
    if size is None or size <= 0:
        problems.append("invalid_output_size")
        return
    if _slug(raw.get("basis")) is None:
        problems.append("missing_output_size_basis")
    if throughput is not None:
        valid = (
            not isinstance(throughput, bool)
            and isinstance(throughput, (int, float))
            and (not isinstance(throughput, float) or math.isfinite(throughput))
        )
        if not valid or throughput <= 0:
            problems.append("invalid_throughput")
        elif reserves is not None and reserves["retrieval_seconds"] * throughput < size:
            problems.append("retrieval_reserve_insufficient")


def _parse_queue_start(
    raw: object, as_of: datetime | None, problems: list[str]
) -> tuple[str, datetime | None]:
    if not isinstance(raw, Mapping):
        problems.append("missing_queue_start")
        return "unknown", None
    status = _enum(raw.get("status"), QUEUE_START_STATUSES)
    if status is None:
        problems.append("missing_queue_status")
        return "unknown", None
    if status in {"not_queued", "unknown"}:
        return status, None
    field = "start_timestamp" if status == "running" else "expected_start_timestamp"
    start, code = _parse_timestamp(raw.get(field))
    if start is None:
        problems.append(
            "queue_start_unknown" if status == "queued" else (code or "missing_start_timestamp")
        )
    elif status == "running" and as_of is not None and start > as_of:
        problems.append("queue_start_contradiction")
    return status, start


def _fit_verdict(
    deadline: datetime | None,
    queue_status: str,
    start: datetime | None,
    as_of: datetime | None,
    runtime: tuple[int, int] | None,
    reserves: dict[str, int] | None,
) -> str:
    if deadline is None or runtime is None or reserves is None:
        return "unknown"
    baseline = start if queue_status in {"running", "queued"} else as_of
    if baseline is None:
        return "unknown"
    total = sum(reserves.values())
    if baseline + timedelta(seconds=runtime[1] + total) <= deadline:
        return "fits_conservative"
    if baseline + timedelta(seconds=runtime[0] + total) <= deadline:
        return "fits_expected"
    return "too_late"


def evaluate_manifest(  # noqa: C901, PLR0912 - one bounded decision per manifest
    manifest: Mapping[str, Any],
    *,
    as_of: datetime | None = None,
    max_evidence_age_seconds: int = DEFAULT_MAX_EVIDENCE_AGE_SECONDS,
) -> FeasibilityReport:
    """Evaluate the optional ``expiring_resource`` contract and return one report."""
    manifest_id = _slug(manifest.get("manifest_id")) or "unidentified"
    contract = manifest.get("expiring_resource")
    if contract is None:
        return FeasibilityReport(
            manifest_id,
            "unknown",
            False,
            "report",
            "none",
            "unknown",
            ("contract_absent",),
            as_of_utc=_render(as_of),
        )
    problems: list[str] = []
    if not isinstance(contract, Mapping):
        problems.append("invalid_contract")
        contract = {}
    elif _text(contract.get("schema")) != CONTRACT_SCHEMA:
        problems.append("invalid_contract_schema")
    policy = _enum(contract.get("admission_policy"), ADMISSION_POLICIES) or "block"
    if _enum(contract.get("resource_class"), RESOURCE_CLASSES) is None:
        problems.append("unknown_resource_class")
    retention = _enum(contract.get("retention_class"), RETENTION_CLASSES)
    if retention is None:
        problems.append("unknown_retention_class")
    effective_as_of = as_of
    if effective_as_of is None and manifest.get("as_of") is not None:
        effective_as_of, code = _parse_timestamp(manifest.get("as_of"))
        if effective_as_of is None:
            problems.append(code or "missing_as_of")
    elif effective_as_of is None:
        # A historical generated_at timestamp is provenance, not the evaluation clock. Without
        # an explicit manifest/CLI as_of, real callers must evaluate against the present so an
        # expired deadline cannot appear feasible simply because the manifest is old.
        effective_as_of = datetime.now(UTC)
    deadline, kind = _parse_deadline(
        contract.get("deadline"), effective_as_of, max_evidence_age_seconds, problems
    )
    if kind == "unknown":
        problems.append("deadline_unknown")
    if kind == "none":
        return FeasibilityReport(
            manifest_id,
            "unknown",
            False,
            policy,
            "none",
            "unknown",
            ("no_declared_expiry",),
            as_of_utc=_render(effective_as_of),
        )
    runtime = _parse_runtime(contract.get("runtime"), problems)
    reserves = _parse_reserves(contract.get("reserves"), retention, problems)
    _check_output(
        contract.get("output"),
        contract.get("retrieval_throughput_bytes_per_second"),
        reserves,
        problems,
    )
    queue_status, start = _parse_queue_start(contract.get("queue_start"), effective_as_of, problems)
    latest_safe = latest_safe_expected = None
    if deadline is not None and runtime is not None and reserves is not None:
        total = sum(reserves.values())
        latest_safe = deadline - timedelta(seconds=runtime[1] + total)
        latest_safe_expected = deadline - timedelta(seconds=runtime[0] + total)
    declared = contract.get("latest_safe_submission")
    if declared is not None:
        declared_at, code = _parse_timestamp(declared)
        if declared_at is None:
            problems.append(code or "invalid_timestamp")
        elif latest_safe is not None and declared_at > latest_safe:
            problems.append("latest_safe_submission_conflict")
    if deadline is not None and effective_as_of is not None and deadline <= effective_as_of:
        problems.append("deadline_expired")
    if any(code != "deadline_expired" for code in problems):
        verdict = "unknown"
    elif "deadline_expired" in problems:
        verdict = "too_late"
    else:
        verdict = _fit_verdict(deadline, queue_status, start, effective_as_of, runtime, reserves)
    return FeasibilityReport(
        manifest_id,
        verdict,
        True,
        policy,
        kind,
        queue_status,
        tuple(sorted(set(problems))),
        _render(deadline),
        _render(effective_as_of),
        _render(latest_safe),
        _render(latest_safe_expected),
        reserves,
    )


def _load_document(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path.name}: manifest must be a mapping")
    return payload


def load_case_manifests(path: Path) -> dict[str, Mapping[str, Any]]:
    """Load a named manifest case pack, sorted by case name."""
    payload = _load_document(path)
    cases = payload.get("cases")
    if payload.get("schema") != CASES_SCHEMA or not isinstance(cases, Mapping):
        raise ValueError(f"{path.name}: expected schema {CASES_SCHEMA}")
    if not all(isinstance(case, Mapping) for case in cases.values()):
        raise ValueError(f"{path.name}: every case must be a mapping")
    return {str(name): case for name, case in sorted(cases.items())}


def _exit_code(report: FeasibilityReport) -> int:
    if not report.applicable or report.verdict in POSITIVE_VERDICTS:
        return 0
    return 1 if report.verdict == "too_late" else 2


def _iso_timestamp(raw: str) -> datetime:
    parsed, code = _parse_timestamp(raw)
    if parsed is None:
        raise argparse.ArgumentTypeError(code or "invalid timestamp")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", default=None, help="Case name in a case pack.")
    parser.add_argument("--as-of", type=_iso_timestamp, default=None)
    parser.add_argument("--check", action="store_true", help="Required check-only flag.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the feasibility check CLI and return a shell-friendly exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required; this checker never mutates state")
    try:
        payload = _load_document(args.manifest)
        if payload.get("schema") == CASES_SCHEMA:
            cases = load_case_manifests(args.manifest)
            if args.case is None:
                print("expiring-resource-feasibility: a case pack requires --case", file=sys.stderr)
                return 2
            if args.case not in cases:
                print(f"expiring-resource-feasibility: unknown case {args.case!r}", file=sys.stderr)
                return 2
            report = evaluate_manifest(cases[args.case], as_of=args.as_of)
        elif args.case is not None:
            print("expiring-resource-feasibility: --case requires a case pack", file=sys.stderr)
            return 2
        else:
            report = evaluate_manifest(payload, as_of=args.as_of)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"expiring-resource-feasibility: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        codes = ",".join(report.reason_codes) or "-"
        print(f"manifest={report.manifest_id} verdict={report.verdict} reason_codes={codes}")
    return _exit_code(report)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
