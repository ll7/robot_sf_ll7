#!/usr/bin/env python3
"""Fail-closed durable artifact locality audit for public pointer custody.

The sanitized locality packet (``durable_artifact_locator_projection.v1``) carries
a public ``references`` inventory and a locator-class ``artifacts`` projection. The
audit joins them by artifact ID, version, and digest and fails when an active
durable-required reference has no verified non-institutional locator, or a
release-facing reference lacks independent failure-domain copies.

The audit never reads private paths, authenticates, mutates artifact state, or emits
locator values. Historical inactive references stay recorded and never count as
active custody. A missing or schema-mismatched packet returns ``status="unknown"``
(exit 2). Motivating issue: #8907. Example:

    uv run python scripts/validation/check_durable_artifact_locality.py \
        --projection tests/validation/fixtures/durable_artifact_locality/compliant.json \
        --check --format json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "durable_artifact_locator_projection.v1"
REPORT_SCHEMA = "durable_artifact_locality_report.v1"
REFERENCE_FIELDS = (
    "reference_id artifact_id version digest consumer_path consumer_status retention_class"
).split()
ENTRY_FIELDS = ("artifact_id", "version", "digest")
LOCATOR_FIELDS = ("locator_class", "verification", "failure_domain_id")
LOCATOR_CLASSES = (
    "public_release cloud_durable personal_durable institutional_durable"
    " institutional_cache local_scratch unknown unavailable"
).split()
NON_INSTITUTIONAL = frozenset({"public_release", "cloud_durable", "personal_durable"})
INSTITUTIONAL = frozenset({"institutional_durable", "institutional_cache"})
NON_DURABLE = frozenset({"local_scratch", "unknown", "unavailable"})
VERIFICATIONS = ("verified", "unverified", "failed")
CONSUMER_STATUSES = ("active", "inactive")
RETENTION_CLASSES = ("durable_required", "release_facing", "historical")
CUSTODY_RETENTION = ("durable_required", "release_facing")
FORBIDDEN_LOCATOR_KEYS = frozenset(
    {"bucket", "endpoint", "host", "key", "locator", "locator_url", "mount", "path", "uri", "url"}
)
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MIN_RELEASE_COPIES = 2
MESSAGES = {
    "institutional_only": "active consumer has only institutional storage",
    "cache_only": "active consumer has only an institutional cache",
    "non_durable_custody": "only local scratch, unknown, or unavailable custody",
    "no_verified_locator": "no verified non-institutional locator",
    "missing_projection_row": "no projection row for artifact identity",
    "version_mismatch": "projection version differs from public reference",
    "digest_mismatch": "projection digest differs from public reference",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class Finding:
    """One public-safe finding with a stable reason code."""

    code: str
    reference_id: str | None
    artifact_id: str | None
    message: str


def _f(
    code: str, ref: str | None = None, artifact: str | None = None, message: str = ""
) -> Finding:
    return Finding(code, ref, artifact, message)


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _require(item: Mapping[str, Any], names: Sequence[str]) -> dict[str, str] | None:
    values = {name: _text(item.get(name)) for name in names}
    return {name: str(value) for name, value in values.items()} if all(values.values()) else None


def _positive_int(value: object, default: int, name: str, findings: list[Finding]) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if value is not None:
        findings.append(
            _f("invalid_projection_schema", message=f"{name} must be a positive integer")
        )
    return default


def _references(payload: Mapping[str, Any], findings: list[Finding]) -> list[dict[str, str]]:
    """Parse and validate the public reference inventory."""
    raw = payload.get("references")
    if not isinstance(raw, list):
        findings.append(_f("invalid_projection_schema", message="references must be a list"))
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        values = _require(item, REFERENCE_FIELDS) if isinstance(item, Mapping) else None
        if values is None:
            findings.append(_f("invalid_reference_entry", message=f"references[{index}] invalid"))
            continue
        digest, path = values["digest"], values["consumer_path"]
        if (
            _SHA256.fullmatch(digest) is None
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or "://" in path
            or values["consumer_status"] not in CONSUMER_STATUSES
            or values["retention_class"] not in RETENTION_CLASSES
        ):
            findings.append(
                _f(
                    "invalid_reference_entry",
                    values["reference_id"],
                    values["artifact_id"],
                    f"[{index}]",
                )
            )
            continue
        reference_id = values["reference_id"]
        if reference_id in seen:
            findings.append(
                _f("duplicate_reference_id", reference_id, values["artifact_id"], "dup")
            )
            continue
        seen.add(reference_id)
        out.append(values)
    return sorted(out, key=lambda reference: reference["reference_id"])


def _locator(item: object, index: int, findings: list[Finding]) -> dict[str, Any] | None:
    """Parse one sanitized locator-class record; values are never echoed."""
    if not isinstance(item, Mapping):
        findings.append(_f("invalid_projection_entry", message=f"locators[{index}] not a mapping"))
        return None
    private_keys = sorted(FORBIDDEN_LOCATOR_KEYS.intersection(item))
    if private_keys:
        findings.append(_f("private_locator_value_rejected", message=f"locators[{index}] rejected"))
        return None
    values = _require(item, LOCATOR_FIELDS)
    locator_class = (values or {}).get("locator_class")
    if locator_class is not None and locator_class not in LOCATOR_CLASSES:
        findings.append(_f("unknown_locator_class", message=f"locators[{index}] unknown class"))
        return None
    mutable = item.get("mutable_alias", False)
    verified_at = item.get("verified_at")
    invalid = (
        values is None
        or values["verification"] not in VERIFICATIONS
        or not isinstance(mutable, bool)
        or (values["verification"] == "verified" and _date(verified_at) is None)
    )
    if invalid:
        findings.append(_f("invalid_projection_entry", message=f"locators[{index}] invalid record"))
        return None
    return {**values, "verified_at": verified_at, "mutable_alias": mutable}


def _entries(payload: Mapping[str, Any], findings: list[Finding]) -> list[dict[str, Any]]:
    """Parse the sanitized locator projection rows."""
    raw = payload.get("artifacts")
    if not isinstance(raw, list):
        findings.append(_f("invalid_projection_schema", message="artifacts must be a list"))
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            findings.append(
                _f("invalid_projection_entry", message=f"artifacts[{index}] not a mapping")
            )
            continue
        values = _require(item, ENTRY_FIELDS)
        raw_locators = item.get("locators")
        raw_locators = raw_locators if isinstance(raw_locators, list) else []
        locators = []
        for locator_index, raw_locator in enumerate(raw_locators):
            if (found := _locator(raw_locator, locator_index, findings)) is not None:
                locators.append(found)
        if values is None or _SHA256.fullmatch(values["digest"]) is None:
            artifact_id = (values or {}).get("artifact_id")
            findings.append(_f("invalid_projection_entry", artifact=artifact_id))
            continue
        if not locators:
            findings.append(
                _f("empty_locator_set", artifact=values["artifact_id"], message="empty")
            )
            continue
        key = (values["artifact_id"], values["version"])
        if key in seen:
            findings.append(
                _f("duplicate_projection_row", artifact=values["artifact_id"], message="dup")
            )
            continue
        seen.add(key)
        out.append({**values, "locators": locators})
    return sorted(out, key=lambda entry: (entry["artifact_id"], entry["version"]))


def _failure_code(locators: Sequence[Mapping[str, Any]], *, specific: bool) -> str | None:
    """Return the artifact-level custody failure code when no locator is usable."""
    if specific:
        return None
    classes = [locator["locator_class"] for locator in locators]
    durable = any(name in NON_INSTITUTIONAL for name in classes)
    if any(name in INSTITUTIONAL for name in classes) and not durable:
        return "institutional_only" if "institutional_durable" in classes else "cache_only"
    if durable:
        return "no_verified_locator"
    return (
        "non_durable_custody"
        if any(name in NON_DURABLE for name in classes)
        else "no_verified_locator"
    )


def _outcome(reference: Mapping[str, str], status: str, copies: int = 0) -> dict[str, Any]:
    return {
        **reference,
        "outcome": status,
        "verified_non_institutional_copies": copies,
    }


def _evaluate(  # noqa: C901, PLR0912 - one bounded custody decision per reference
    reference: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: date,
    max_age_days: int,
    min_release_copies: int,
    findings: list[Finding],
) -> dict[str, Any]:
    """Join one public reference to its projection row and decide custody."""
    rid, aid = reference["reference_id"], reference["artifact_id"]
    if reference["consumer_status"] == "inactive":
        return _outcome(reference, "inactive")
    if reference["retention_class"] not in CUSTODY_RETENTION:
        return _outcome(reference, "not_required")
    exact = [
        row
        for row in rows
        if row["version"] == reference["version"] and row["digest"] == reference["digest"]
    ]
    if not exact:
        if not rows:
            code = "missing_projection_row"
        elif not any(row["version"] == reference["version"] for row in rows):
            code = "version_mismatch"
        else:
            code = "digest_mismatch"
        findings.append(_f(code, rid, aid, MESSAGES[code]))
        return _outcome(reference, "fail")
    usable: list[Mapping[str, Any]] = []
    specific = False
    for locator in exact[0]["locators"]:
        if locator["mutable_alias"]:
            specific = True
            findings.append(
                _f("mutable_alias", rid, aid, f"{locator['locator_class']} is an alias")
            )
        elif locator["verification"] != "verified":
            continue
        elif (as_of - _date(locator["verified_at"])).days > max_age_days:
            specific = True
            findings.append(
                _f("stale_verification", rid, aid, f"{locator['locator_class']} is stale")
            )
        elif locator["locator_class"] in NON_INSTITUTIONAL:
            usable.append(locator)
    required = 1 if reference["retention_class"] == "durable_required" else min_release_copies
    domains = {locator["failure_domain_id"] for locator in usable}
    if len(domains) >= required:
        if len(domains) == len(usable):
            return _outcome(reference, "pass", len(usable))
        findings.append(_f("same_failure_domain", rid, aid, "copies share a failure domain"))
        return _outcome(reference, "fail", len(usable))
    if usable:
        if len(domains) < len(usable):
            findings.append(_f("same_failure_domain", rid, aid, "copies share a failure domain"))
        findings.append(
            _f("insufficient_redundancy", rid, aid, f"{len(domains)} of {required} independent")
        )
        return _outcome(reference, "fail", len(usable))
    code = _failure_code(exact[0]["locators"], specific=specific)
    if code is not None:
        findings.append(_f(code, rid, aid, MESSAGES[code]))
    return _outcome(reference, "fail")


def _sorted_findings(findings: Sequence[Finding]) -> tuple[Finding, ...]:
    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.code,
                finding.reference_id or "",
                finding.artifact_id or "",
                finding.message,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class LocalityReport:
    """Deterministic public-safe locality audit report."""

    status: str
    as_of: str
    findings: tuple[Finding, ...]
    outcomes: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        """Return True when no active reference failed custody."""
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe report payload."""
        active = sum(1 for outcome in self.outcomes if outcome["consumer_status"] == "active")
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.code] = counts.get(finding.code, 0) + 1
        return {
            "schema": REPORT_SCHEMA,
            "status": self.status,
            "ok": self.ok,
            "as_of": self.as_of,
            "summary": {
                "reference_count": len(self.outcomes),
                "active_count": active,
                "inactive_count": len(self.outcomes) - active,
                "passing_count": sum(1 for o in self.outcomes if o["outcome"] == "pass"),
                "failing_count": sum(1 for o in self.outcomes if o["outcome"] == "fail"),
                "finding_counts": dict(sorted(counts.items())),
            },
            "findings": [asdict(finding) for finding in self.findings],
            "references": list(self.outcomes),
        }

    def render_json(self) -> str:
        """Return byte-stable report JSON with a trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def render_markdown(self) -> str:
        """Return a deterministic Markdown summary of the audit."""
        summary = self.to_dict()["summary"]
        lines = [
            "# Durable Artifact Locality Audit",
            "",
            f"- Schema: `{REPORT_SCHEMA}` | Status: `{self.status}` | As of: `{self.as_of}`",
            f"- References: {summary['reference_count']} (active: {summary['active_count']}, "
            f"inactive: {summary['inactive_count']}) | Passing: {summary['passing_count']} | "
            f"Failing: {summary['failing_count']}",
            "",
            "## Findings",
            "",
        ]
        if self.findings:
            lines.extend(["| Code | Reference | Artifact | Detail |", "| --- | --- | --- | --- |"])
            lines.extend(
                f"| `{f.code}` | `{f.reference_id or '-'}` | `{f.artifact_id or '-'}` | {f.message} |"
                for f in self.findings
            )
        else:
            lines.append("No findings.")
        lines.extend(
            [
                "",
                "## References",
                "",
                "| Reference | Artifact | Consumer | Status | Retention | Outcome | Copies |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| `{o['reference_id']}` | `{o['artifact_id']}` | `{o['consumer_path']}` |"
            f" {o['consumer_status']} | {o['retention_class']} | {o['outcome']} |"
            f" {o['verified_non_institutional_copies']} |"
            for o in self.outcomes
        )
        return "\n".join(lines) + "\n"


def audit_locality(projection_path: Path, *, as_of: date | None = None) -> LocalityReport:
    """Audit one sanitized locality packet and return its deterministic report."""
    path = Path(projection_path)
    findings: list[Finding] = []
    if not path.is_file():
        findings.append(_f("invalid_projection_schema", message=f"{path.name}: packet missing"))
        return LocalityReport("unknown", "unknown", _sorted_findings(findings), ())
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        findings.append(_f("invalid_projection_schema", message=f"{path.name}: unreadable JSON"))
        return LocalityReport("unknown", "unknown", _sorted_findings(findings), ())
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        findings.append(
            _f(
                "invalid_projection_schema",
                message=f"{path.name}: declared schema must be {SCHEMA}",
            )
        )
        return LocalityReport("unknown", "unknown", _sorted_findings(findings), ())
    generated_at = _date(payload.get("generated_at"))
    if generated_at is None:
        findings.append(_f("invalid_projection_schema", message="generated_at must be an ISO date"))
    max_age_days = _positive_int(
        payload.get("verification_max_age_days"),
        DEFAULT_MAX_AGE_DAYS,
        "verification_max_age_days",
        findings,
    )
    min_release_copies = _positive_int(
        payload.get("minimum_release_copies"),
        DEFAULT_MIN_RELEASE_COPIES,
        "minimum_release_copies",
        findings,
    )
    references = _references(payload, findings)
    entries = _entries(payload, findings)
    effective_as_of = as_of or generated_at
    if effective_as_of is None:
        return LocalityReport("unknown", "unknown", _sorted_findings(findings), ())
    rows_by_artifact: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        rows_by_artifact.setdefault(entry["artifact_id"], []).append(entry)
    outcomes = tuple(
        _evaluate(
            reference,
            rows_by_artifact.get(reference["artifact_id"], ()),
            as_of=effective_as_of,
            max_age_days=max_age_days,
            min_release_copies=min_release_copies,
            findings=findings,
        )
        for reference in references
    )
    return LocalityReport(
        "fail" if findings else "pass",
        effective_as_of.isoformat(),
        _sorted_findings(findings),
        tuple(sorted(outcomes, key=lambda outcome: outcome["reference_id"])),
    )


def _iso_date(text: str) -> date:
    parsed = _date(text)
    if parsed is None:
        raise argparse.ArgumentTypeError("expected an ISO date (YYYY-MM-DD)")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit sanitized durable-artifact locator projections against active public "
            "references. Fails when an active durable-required reference has no verified "
            "non-institutional locator or a release-facing reference lacks configured "
            "redundancy. Reads no private paths and never emits locator values."
        )
    )
    parser.add_argument("--projection", required=True, type=Path, help="Sanitized locality packet.")
    parser.add_argument(
        "--as-of",
        type=_iso_date,
        help="Audit date (YYYY-MM-DD); defaults to the packet generated_at.",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--check", action="store_true", help="Exit 1 when any finding is present.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the locality audit and return a shell-friendly exit code."""
    args = _build_parser().parse_args(argv)
    report = audit_locality(args.projection, as_of=args.as_of)
    rendered = report.render_markdown() if args.format == "markdown" else report.render_json()
    sys.stdout.write(rendered)
    if report.status == "unknown":
        return 2
    if report.ok or not args.check:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
