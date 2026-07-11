#!/usr/bin/env python3
"""Evidence-registry integrity downward ratchet (issue #5275).

This helper turns the report-mode evidence-registry linter
(``scripts/tools/lint_evidence_registry.py``) into a **monotone downward
ratchet** over the committed integrity findings, mirroring the
``scripts/dev/ty_advisory_ratchet.py`` precedent.

What this owns (issue #5275)
----------------------------
The strict-CI decision for the evidence-registry linter. PR #5280 made the
linter bundle-aware and classified all legacy findings via a report-mode
disposition packet. This ratchet is the *next* step: it makes the linter a
strict gate against **net-new** integrity regressions while explicitly
grandfathering the existing 359 legacy findings through a committed baseline.

The committed baseline **is** the "explicitly approved remaining exclusion
policy" the issue acceptance asks for: each grandfathered finding is enumerated
by path and code, and the disposition packet already documents its category,
status, and next remediation action.

Ratchet contract
----------------
Findings are keyed by ``(path, code)``:

* A **clean file** (absent from the baseline) that gains any finding -> FAIL.
  This is the primary value: no NEW integrity drift can land.
* A **tracked file** whose per-code count *increases* beyond its baseline -> FAIL.
* A **decrease** never fails; the helper prints a "ratchet opportunity" notice
  so the baseline can be refreshed to lock in the improvement
  (``--write-baseline``).
* A file whose findings are fully remediated disappears from the current report;
  ``--write-baseline`` drops it so the baseline only ever shrinks.

A brand-new evidence file that is integrity-clean never trips the gate (it has
no findings), so normal evidence growth is unrestricted; only new *findings*
are blocked.

Exit codes
----------
* ``0`` — ratchet holds (no new file regressed; no per-code count increased).
* ``1`` — a clean file regressed, or a tracked file's per-code count increased.
* ``2`` — the linter could not be run / produced unparseable output (infra error).

Usage
-----
::

    # Re-run the linter and check against the committed baseline (CI / local gate).
    uv run python scripts/dev/evidence_registry_ratchet.py --check

    # Refresh the baseline after intentionally reducing findings.
    uv run python scripts/dev/evidence_registry_ratchet.py --write-baseline

    # Parse a pre-rendered linter report (offline / test / no-network).
    uv run python scripts/dev/evidence_registry_ratchet.py --check \
        --report /tmp/lint_report.json

The committed baseline lives at
``scripts/validation/evidence_registry_baseline.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("scripts/validation/evidence_registry_baseline.json")
DEFAULT_LINTER = Path("scripts/tools/lint_evidence_registry.py")
DEFAULT_REGISTRY_ROOT = Path("docs/context/evidence")
DEFAULT_DISPOSITION = Path("docs/context/evidence/evidence_registry_dispositions.yaml")


def _repo_root() -> Path:
    """Return the current Git repository root."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Could not determine git repository root.")
    return Path(proc.stdout.strip())


def run_linter(repo_root: Path) -> dict[str, Any]:
    """Run the evidence-registry linter and return its parsed JSON report.

    The linter runs in report mode (no ``--strict``); the ratchet decides
    pass/fail. Raises ``RuntimeError`` on a non-zero linter exit (which would
    indicate an infra error, since report mode exits 0 even with findings) or
    on unparseable output.
    """
    linter = repo_root / DEFAULT_LINTER
    registry_root = repo_root / DEFAULT_REGISTRY_ROOT
    disposition = repo_root / DEFAULT_DISPOSITION
    cmd = [
        sys.executable,
        str(linter),
        "--repo-root",
        str(repo_root),
        "--registry-root",
        str(registry_root.relative_to(repo_root)),
        "--disposition-file",
        str(disposition.relative_to(repo_root)),
    ]
    try:
        proc = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"Could not invoke linter '{' '.join(cmd)}': {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"evidence-registry linter exited {proc.returncode} (report mode should "
            f"exit 0 even with findings).\nstderr:\n{proc.stderr[:2000]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse linter JSON output: {exc}\nstdout head:\n{proc.stdout[:1000]}"
        ) from exc


def load_report(path: Path) -> dict[str, Any]:
    """Load a pre-rendered linter JSON report from ``path``."""
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Aggregate linter findings into ``{path: {code: count}}``.

    ``path`` is the repository-relative evidence file a finding is attached to.
    Keying by ``(path, code)`` makes the ratchet granular enough to catch a
    single net-new finding in an otherwise-tracked file, while staying coarse
    enough that linter message-wording changes do not produce false regressions.
    """
    by_path: dict[str, dict[str, int]] = {}
    for finding in report.get("issues", []):
        path = finding.get("path", "<unknown>")
        code = finding.get("code", "<unknown>")
        by_path.setdefault(path, Counter())[code] += 1
    # Normalize Counters to plain int dicts with stable ordering.
    return {path: dict(sorted(codes.items())) for path, codes in sorted(by_path.items())}


def build_baseline_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build the versioned baseline JSON payload from a linter report."""
    findings_by_path = aggregate(report)
    by_code: Counter[str] = Counter()
    for codes in findings_by_path.values():
        for code, count in codes.items():
            by_code[code] += count
    summary = report.get("summary", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "linter": DEFAULT_LINTER.as_posix(),
        "description": (
            "Evidence-registry integrity downward ratchet (issue #5275). The "
            "ratchet gates on per-file, per-code finding counts: a clean file "
            "(absent here) that gains a finding fails, and a tracked file whose "
            "per-code count increases fails. The committed baseline is the "
            "explicitly-approved grandfathered exclusion policy for the 359 "
            "legacy findings classified by docs/context/evidence/"
            "evidence_registry_dispositions.yaml; remediate a category and "
            "refresh this baseline with "
            "`scripts/dev/evidence_registry_ratchet.py --write-baseline` to "
            "lock in the reduction. Promote the gate from advisory to blocking "
            "by removing continue-on-error in the ratchet CI workflow once the "
            "policy has settled."
        ),
        "summary": {
            "total_findings": int(summary.get("findings", 0)),
            "files_with_findings": len(findings_by_path),
            "by_code": dict(sorted(by_code.items())),
        },
        "findings_by_path": findings_by_path,
    }


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline {path} must be a JSON object, got {type(data).__name__}.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported baseline schema_version in {path}: "
            f"got {data.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    if not isinstance(data.get("findings_by_path"), dict):
        raise ValueError(f"Baseline {path} is missing a valid 'findings_by_path' mapping.")
    return data


def check_against_baseline(
    current: dict[str, dict[str, int]],
    baseline: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notices)`` for the downward ratchet.

    ``failures`` is non-empty -> the ratchet is broken (exit 1). ``notices`` are
    informational ratchet-opportunity hints (counts decreased) and are always
    advisory.
    """
    baseline_paths: dict[str, dict[str, int]] = {
        str(path): {str(code): int(count) for code, count in codes.items()}
        for path, codes in baseline.get("findings_by_path", {}).items()
    }

    failures: list[str] = []
    notices: list[str] = []

    all_paths = sorted(set(current) | set(baseline_paths))
    for path in all_paths:
        base_codes = baseline_paths.get(path, {})
        cur_codes = current.get(path, {})
        all_codes = sorted(set(base_codes) | set(cur_codes))
        path_regressions: list[str] = []
        for code in all_codes:
            base_n = base_codes.get(code, 0)
            cur_n = cur_codes.get(code, 0)
            if cur_n > base_n:
                if base_n == 0:
                    path_regressions.append(f"{code} went from 0 to {cur_n}")
                else:
                    path_regressions.append(f"{code} increased from {base_n} to {cur_n}")
            elif cur_n < base_n:
                notices.append(
                    f"ratchet opportunity: '{path}' {code} dropped from "
                    f"{base_n} to {cur_n}; refresh the baseline to lock in "
                    f"the improvement."
                )
        if path_regressions:
            if not base_codes:
                failures.append(
                    f"clean file regressed: '{path}' is not in the baseline and "
                    f"now has findings ({'; '.join(path_regressions)})."
                )
            else:
                failures.append(
                    f"file '{path}' finding count increased ({'; '.join(path_regressions)})."
                )

    return failures, notices


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write stable, reviewable, sort-keyed JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Run the ratchet gate.")
    mode.add_argument(
        "--write-baseline",
        action="store_true",
        help="Recompute findings and (re)write the baseline file.",
    )
    mode.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Print the aggregate (per-file, per-code counts) without a baseline.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--root", type=Path, default=None, help="Repository root (defaults to git toplevel)."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to a pre-rendered linter JSON report. When set, the linter is "
            "NOT re-run; the report is parsed instead (offline / test mode)."
        ),
    )
    return parser.parse_args(argv)


def _gather_report(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Resolve the linter report either by running the linter or parsing --report."""
    if args.report is not None:
        return load_report(args.report)
    return run_linter(repo_root)


def _print_aggregate(report: dict[str, Any]) -> None:
    """Print the per-file aggregate report for ``--aggregate-only``."""
    findings_by_path = aggregate(report)
    total = sum(sum(codes.values()) for codes in findings_by_path.values())
    summary = report.get("summary", {})
    print(
        f"evidence-registry ratchet: findings={total} files={len(findings_by_path)} "
        f"(linter summary findings={summary.get('findings', 0)})"
    )
    for path, codes in findings_by_path.items():
        rendered = ", ".join(f"{code}={count}" for code, count in codes.items())
        print(f"  {sum(codes.values()):5d}  {path}  ({rendered})")


def _report_check(
    report: dict[str, Any],
    baseline: dict[str, Any],
    failures: list[str],
    notices: list[str],
) -> int:
    """Print the ``--check`` ratchet result and return the exit code."""
    baseline_total = int(baseline.get("summary", {}).get("total_findings", 0))
    summary = report.get("summary", {})
    print(
        f"evidence-registry ratchet: findings={summary.get('findings', 0)} "
        f"(baseline={baseline_total})."
    )
    for notice in notices:
        print(f"NOTICE: {notice}")
    if failures:
        print(
            "\nevidence-registry ratchet FAILED (net-new integrity findings):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRemediate the new findings, or refresh the baseline with "
            "`scripts/dev/evidence_registry_ratchet.py --write-baseline` if the "
            "increase is intentional and reviewed.",
            file=sys.stderr,
        )
        return 1
    print("evidence-registry ratchet passed: no net-new findings; clean files stayed clean.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ratchet gate, baseline refresh, or aggregate report."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = args.root.resolve() if args.root is not None else _repo_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline

    try:
        report = _gather_report(args, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: could not obtain linter report: {exc}", file=sys.stderr)
        return 2

    if args.aggregate_only:
        _print_aggregate(report)
        return 0

    payload = build_baseline_payload(report)

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, payload)
        print(
            f"Wrote evidence-registry baseline to {baseline_path}: "
            f"{payload['summary']['total_findings']} findings across "
            f"{payload['summary']['files_with_findings']} files."
        )
        return 0

    # --check
    if not baseline_path.exists():
        print(
            f"ERROR: baseline not found at {baseline_path}. "
            f"Generate it with --write-baseline first.",
            file=sys.stderr,
        )
        return 2
    baseline = load_baseline(baseline_path)
    failures, notices = check_against_baseline(aggregate(report), baseline)
    return _report_check(report, baseline, failures, notices)


if __name__ == "__main__":
    raise SystemExit(main())
