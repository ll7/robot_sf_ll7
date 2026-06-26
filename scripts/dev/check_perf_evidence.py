#!/usr/bin/env python3
"""Performance-PR evidence contract.

Fires only for ``perf``-typed conventional-commit changes (the #3611 failure
mode: a ``perf(planner): ...`` PR whose claimed speed-up targeted the wrong layer,
was never substantiated on the real campaign entry point, and had to be reverted
in #3613). When triggered, it requires the PR body to carry a ``Performance
Evidence`` section with concrete before/after runtime, a representative command,
a rollback/failure criterion, and — when the change claims caching — a
cache-hit/reuse counter.

The trigger (perf commit subjects) is read locally from ``git log`` so the check
runs the same way in ``pr_ready_check.sh`` and in CI, without needing the GitHub
``perf`` label. ``--commits-file`` injects subjects for tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts.dev.check_pr_followups import (
    _clean_value,
    _extract_section,
    _is_empty_or_option_placeholder,
    _value_after_label,
)

# Conventional-commit perf type: ``perf``, optional ``(scope)``, optional ``!``, then ``:``.
PERF_SUBJECT_RE = re.compile(r"^\s*perf(?:\([^)]*\))?!?:", re.IGNORECASE)
# Caching is inferred from the perf subject only (not the body, which mentions
# "cache"/"reuse" in its own field labels and would self-trigger).
CACHING_RE = re.compile(r"\b(?:cache|caching|cached|memoi[sz]e?d?|reuse)\b", re.IGNORECASE)

CORE_FIELDS = (
    "Baseline runtime",
    "Changed runtime",
    "Representative command",
    "Rollback or failure criterion",
)
CACHE_FIELD = "Cache-hit or reuse counter"
SECTION_HEADING = "Performance Evidence"
# A field is absent when it is empty/option-placeholder, or opens with an NA/none token —
# even with a trailing reason ("NA - measured later"), which does not satisfy the contract.
_NA_PREFIX_RE = re.compile(r"^(?:n/?a|none|no|not\s+applicable)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PerfEvidenceReport:
    """Compact performance-evidence contract report."""

    status: str
    source: str
    perf_subjects: tuple[str, ...]
    caching_claimed: bool
    missing_fields: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""


def perf_commit_subjects(subjects: tuple[str, ...]) -> tuple[str, ...]:
    """Return the subjects that are conventional-commit ``perf`` changes."""
    return tuple(subject for subject in subjects if PERF_SUBJECT_RE.match(subject))


def claims_caching(perf_subjects: tuple[str, ...]) -> bool:
    """Return whether any perf commit subject claims a caching/reuse speed-up."""
    return any(CACHING_RE.search(subject) for subject in perf_subjects)


def _field_missing(section: str, label: str) -> bool:
    """Return whether a required evidence field is empty, placeholder, or NA."""
    value = _value_after_label(section, label)
    if _is_empty_or_option_placeholder(value):
        return True
    return bool(_NA_PREFIX_RE.match(_clean_value(value)))


def analyze_perf_evidence(
    body: str,
    *,
    perf_subjects: tuple[str, ...],
    source: str,
) -> PerfEvidenceReport:
    """Return the performance-evidence contract report for a PR body."""
    triggers = perf_commit_subjects(perf_subjects)
    if not triggers:
        return PerfEvidenceReport(
            status="skipped",
            source=source,
            perf_subjects=(),
            caching_claimed=False,
            message="No perf-typed conventional-commit change; contract not required.",
        )

    caching = claims_caching(triggers)
    section = _extract_section(body, SECTION_HEADING)
    if not section:
        return PerfEvidenceReport(
            status="missing_perf_evidence",
            source=source,
            perf_subjects=triggers,
            caching_claimed=caching,
            missing_fields=CORE_FIELDS,
            message=(
                f"perf change requires a '## {SECTION_HEADING}' section with baseline/changed "
                "runtime, a representative command, and a rollback criterion."
            ),
        )

    missing = [label for label in CORE_FIELDS if _field_missing(section, label)]
    if missing:
        return PerfEvidenceReport(
            status="incomplete_perf_evidence",
            source=source,
            perf_subjects=triggers,
            caching_claimed=caching,
            missing_fields=tuple(missing),
            message=(
                "perf Performance Evidence section is missing concrete values for: "
                f"{', '.join(missing)}."
            ),
        )

    if caching and _field_missing(section, CACHE_FIELD):
        return PerfEvidenceReport(
            status="missing_cache_counter",
            source=source,
            perf_subjects=triggers,
            caching_claimed=True,
            missing_fields=(CACHE_FIELD,),
            message=(
                "perf change claims caching; provide a concrete "
                f"'{CACHE_FIELD}' (hit/miss or reuse count on the real entry point)."
            ),
        )

    return PerfEvidenceReport(
        status="ok",
        source=source,
        perf_subjects=triggers,
        caching_claimed=caching,
        message="Performance evidence contract satisfied.",
    )


def _read_commit_subjects(base_ref: str) -> tuple[str, ...]:
    """Return commit subjects in ``BASE_REF..HEAD`` (empty on git failure)."""
    try:
        result = subprocess.run(
            ["git", "log", "--no-merges", "--format=%s", f"{base_ref}..HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _load_commit_subjects(args: argparse.Namespace) -> tuple[str, ...]:
    if args.commits_file:
        return tuple(
            line.strip()
            for line in args.commits_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return _read_commit_subjects(args.base_ref)


def _load_body(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.body_file:
        return args.body_file.read_text(encoding="utf-8"), str(args.body_file)
    env_body = os.environ.get("PR_READY_PR_BODY_FILE")
    if env_body:
        path = Path(env_body)
        return path.read_text(encoding="utf-8"), str(path)
    return None, "none"


def _format_report(report: PerfEvidenceReport) -> str:
    subjects = "; ".join(report.perf_subjects) if report.perf_subjects else "none"
    missing = ", ".join(report.missing_fields) if report.missing_fields else "none"
    return (
        "PR perf-evidence check: "
        f"status={report.status}; source={report.source}; "
        f"perf_subjects={subjects!r}; caching_claimed={report.caching_claimed}; "
        f"missing_fields={missing}; {report.message}"
    )


def _emit_advisory_warning(message: str) -> None:
    print(f"::warning title=Performance evidence contract (advisory)::{message}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-file", type=Path, help="Markdown PR body to check.")
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("BASE_REF", "origin/main"),
        help="Base ref for the perf-commit scan (default: origin/main or $BASE_REF).",
    )
    parser.add_argument(
        "--commits-file",
        type=Path,
        help="Newline-delimited commit subjects (overrides git log; used by tests).",
    )
    parser.add_argument(
        "--require-body",
        action="store_true",
        help="Fail closed for a perf change when no PR body source is available.",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Report contract violations as warnings and exit 0 instead of failing.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


FAILING_STATUSES = {
    "missing_perf_evidence",
    "incomplete_perf_evidence",
    "missing_cache_counter",
    "missing_body",
}


def main(argv: list[str] | None = None) -> int:
    """Run the performance-evidence contract CLI."""
    args = _build_parser().parse_args(argv)
    advisory = args.advisory or os.environ.get("PR_READY_ADVISORY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    perf_subjects = _load_commit_subjects(args)
    triggers = perf_commit_subjects(perf_subjects)
    body, source = _load_body(args)

    if body is None:
        # No body to evaluate. Only a problem when there is a perf change and the
        # caller demands a body (final readiness).
        require = args.require_body and bool(triggers)
        report = PerfEvidenceReport(
            status="missing_body" if require else "skipped",
            source=source,
            perf_subjects=triggers,
            caching_claimed=claims_caching(triggers),
            message=(
                "perf change requires a PR body; provide --body-file or PR_READY_PR_BODY_FILE."
                if require
                else "No PR body source configured; perf-evidence contract not enforced."
            ),
        )
    else:
        report = analyze_perf_evidence(body, perf_subjects=perf_subjects, source=source)

    if args.json:
        print(json.dumps(report.__dict__, sort_keys=True))
    else:
        stream = sys.stderr if report.status in FAILING_STATUSES else sys.stdout
        print(_format_report(report), file=stream)

    if report.status in FAILING_STATUSES:
        if advisory:
            _emit_advisory_warning(report.message)
            return 0
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
