#!/usr/bin/env python3
# evidence-writer-exempt: ratchet gate tooling; the local write_json is a definition-only
# forwarding helper that persists the monotone baseline next to the ratchet, and the
# docs/context/evidence tree is a read-only input.
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

A brand-new evidence file that is integrity-clean never trips the *findings*
gate (it has no findings), so integrity growth is unrestricted. However, the
baseline also records an ``evidence_tree`` manifest (issue #6839): adding or
removing any file under ``docs/context/evidence/`` without refreshing the
baseline fails the gate on the causing PR, so a stale hand-committed baseline
can never redden main after merge. The fix is one command,
``--write-baseline``, which the failure message names.

Exit codes
----------
* ``0`` — ratchet holds (no new file regressed; no per-code count increased).
* ``1`` — a clean file regressed, or a tracked file's per-code count increased.
* ``2`` — the linter could not be run / produced unparseable output (infra error).

``--companion-delta`` reports the review-companion delta for the committed baseline
(issue #7467): it returns ``0`` when the review companion already covers every new
baseline path and per-code increase (and carries no stale entries), ``1`` when new or
stale companion entries need a disposition, and ``2`` when a baseline, review
companion, or prior baseline cannot be loaded. It is read-only: it never edits the
review companion.

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

    # Surface the review-companion delta for the committed baseline (issue #7467).
    # Prints a paste-able template for any missing/stale reviewed_files or
    # reviewed_baseline_increases entries and exits 1 when dispositions are missing;
    # exits 0 when the companion already covers the baseline delta. Read-only.
    uv run python scripts/dev/evidence_registry_ratchet.py --companion-delta

The committed baseline lives at
``scripts/validation/evidence_registry_baseline.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import yaml

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("scripts/validation/evidence_registry_baseline.json")
DEFAULT_REVIEW_COMPANION = Path("scripts/validation/evidence_registry_baseline_review.yaml")
DEFAULT_LINTER = Path("scripts/tools/lint_evidence_registry.py")
DEFAULT_REGISTRY_ROOT = Path("docs/context/evidence")
DEFAULT_DISPOSITION = Path("docs/context/evidence/evidence_registry_dispositions.yaml")
# Fallback anchor when the review companion does not yet record a prior baseline
# commit (issue #7467); the committed review companion currently carries this value.
FALLBACK_PRIOR_BASELINE_COMMIT = "9fa96c01bf1c8152459f5fa8c481e938fb1e6725"
REVIEW_SCHEMA_VERSION = "evidence_registry_baseline_review.v1"
VALID_DISPOSITIONS = ("baseline", "remediate")


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


def _validate_report(data: Any, source: str) -> dict[str, Any]:
    """Validate the linter report schema before the ratchet consumes it."""
    if not isinstance(data, dict):
        raise RuntimeError(f"Report {source} must be a dictionary, got {type(data).__name__}")

    issues = data.get("issues")
    if not isinstance(issues, list):
        raise RuntimeError(f"Report {source} has an invalid or missing 'issues' list")
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise RuntimeError(f"Report {source} has a non-mapping issue at index {index}")
        for field in ("path", "code"):
            value = issue.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"Report {source} has an invalid issue {field!r} at index {index}"
                )

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"Report {source} has an invalid or missing 'summary' mapping")
    findings = summary.get("findings")
    if isinstance(findings, bool) or not isinstance(findings, int) or findings < 0:
        raise RuntimeError(
            f"Report {source} has an invalid 'summary.findings'; expected a non-negative integer"
        )
    if findings != len(issues):
        raise RuntimeError(
            f"Report {source} has inconsistent findings metadata: "
            f"summary.findings={findings}, but issues contains {len(issues)} entries"
        )
    return data


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
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse linter JSON output: {exc}\nstdout head:\n{proc.stdout[:1000]}"
        ) from exc
    return _validate_report(data, f"linter output from '{linter}'")


def load_report(path: Path) -> dict[str, Any]:
    """Load a pre-rendered linter JSON report from ``path``.

    Raises ``RuntimeError`` on a missing or malformed file so the CLI maps a bad
    ``--report`` to the infra-error exit code (2) instead of an uncaught traceback.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read report file '{path}': {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse report JSON '{path}': {exc}") from exc
    return _validate_report(data, f"JSON '{path}'")


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


def evidence_tree_manifest(registry_root: Path) -> dict[str, Any]:
    """Return a compact, deterministic manifest of the audited evidence files.

    The manifest is keyed on the *sorted set of every regular,
    registry-relative file* under the evidence root. It is path-based rather
    than content-based on purpose: adding or removing any evidence file changes
    the manifest (so a baseline refresh is required and stays a pure function
    of the evidence tree), while editing an existing file's contents is left to
    the findings ratchet, which already catches net-new integrity findings. This
    closes the issue #6839 gap: a PR that adds or removes any file under
    ``docs/context/evidence/`` without refreshing the baseline now fails its own
    ratchet gate pre-merge instead of reddening main after merge.
    """
    root = registry_root.resolve()
    files: list[str] = []
    if root.exists():
        files = sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
        )
    # Hash a structured JSON array rather than a newline-delimited string. A bare join
    # aliases the distinct path sets {"a", "b"} and {"a\nb"}; JSON escaping keeps
    # each path boundary unambiguous even when a filename contains a newline.
    serialized = json.dumps(files, ensure_ascii=True, separators=(",", ":"))
    return {
        "count": len(files),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def check_evidence_tree_manifest(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return ``(failures, notices)`` for the evidence-tree manifest ratchet.

    When the committed baseline carries an ``evidence_tree`` manifest, the live
    evidence tree must match it exactly: any add/remove/rename of an evidence
    file without a baseline refresh is a hard failure, so the drift is
    attributed to the causing PR pre-merge (issue #6839). A baseline that
    predates manifest tracking (no ``evidence_tree`` field) is advisory -- it
    cannot enforce the manifest, but it does not weaken the findings ratchet.
    """
    baseline_tree = baseline.get("evidence_tree")
    if not isinstance(baseline_tree, dict) or "sha256" not in baseline_tree:
        return [], [
            "baseline has no evidence_tree manifest; regenerate it with "
            "`scripts/dev/evidence_registry_ratchet.py --write-baseline` to "
            "enable add/remove drift detection for evidence files."
        ]
    if baseline_tree.get("count") != current.get("count") or baseline_tree.get(
        "sha256"
    ) != current.get("sha256"):
        return [
            (
                "evidence tree changed without a matching baseline refresh: "
                f"{current.get('count')} evidence files now vs "
                f"{baseline_tree.get('count')} in the baseline. A PR that adds "
                "or removes files under docs/context/evidence/ must refresh "
                "scripts/validation/evidence_registry_baseline.json in the same "
                "change so the baseline stays a machine-generated function of "
                "the evidence tree (issue #6839)."
            )
        ], []
    return [], []


def build_baseline_payload(
    report: dict[str, Any],
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Build the versioned baseline JSON payload from a linter report.

    When ``registry_root`` is provided, the payload records an ``evidence_tree``
    manifest of the audited evidence files so the ratchet can fail a PR that
    adds or removes evidence files without refreshing the baseline (issue #6839).
    """
    findings_by_path = aggregate(report)
    by_code: Counter[str] = Counter()
    for codes in findings_by_path.values():
        for code, count in codes.items():
            by_code[code] += count
    summary = report.get("summary", {})
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "linter": DEFAULT_LINTER.as_posix(),
        "description": (
            "Evidence-registry integrity downward ratchet (issue #5275). The "
            "ratchet gates on per-file, per-code finding counts: a clean file "
            "(absent here) that gains a finding fails, and a tracked file whose "
            "per-code count increases fails. The committed baseline is the "
            "explicitly-approved grandfathered exclusion policy for legacy "
            "findings classified by docs/context/evidence/"
            "evidence_registry_dispositions.yaml. Remediate a category and "
            "refresh this baseline with "
            "`scripts/dev/evidence_registry_ratchet.py --write-baseline` to "
            "lock in the reduction; the blocking evidence-registry workflow "
            "enforces the ratchet on its configured paths. The payload also "
            "carries an evidence_tree manifest (issue #6839): adding or removing "
            "any file under docs/context/evidence/ without refreshing this "
            "baseline fails the ratchet on the causing PR, so a stale baseline "
            "can never redden main after merge."
        ),
        "summary": {
            "total_findings": int(summary.get("findings", 0)),
            "files_with_findings": len(findings_by_path),
            "by_code": dict(sorted(by_code.items())),
        },
        "findings_by_path": findings_by_path,
    }
    if registry_root is not None:
        payload["evidence_tree"] = evidence_tree_manifest(registry_root)
    return payload


def _validate_finding_counts(path: Path, findings_by_path: dict[Any, Any]) -> None:
    """Validate the nested finding-count mapping consumed by the ratchet."""
    for evidence_path, codes in findings_by_path.items():
        if not isinstance(evidence_path, str) or not evidence_path:
            raise ValueError(f"Baseline {path} has an invalid findings path key.")
        if not isinstance(codes, dict):
            raise ValueError(
                f"Baseline {path} has an invalid finding-code mapping for '{evidence_path}'."
            )
        for code, count in codes.items():
            if not isinstance(code, str) or not code:
                raise ValueError(
                    f"Baseline {path} has an invalid finding code for '{evidence_path}'."
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"Baseline {path} has an invalid finding count for '{evidence_path}'/{code}."
                )


def _validate_baseline_metadata(path: Path, data: dict[str, Any]) -> None:
    """Validate optional metadata mappings that the ratchet reads directly."""
    summary = data.get("summary")
    if summary is not None and not isinstance(summary, dict):
        raise ValueError(f"Baseline {path} has an invalid 'summary' mapping.")
    if isinstance(summary, dict) and "total_findings" in summary:
        total_findings = summary["total_findings"]
        if (
            isinstance(total_findings, bool)
            or not isinstance(total_findings, int)
            or total_findings < 0
        ):
            raise ValueError(
                f"Baseline {path} has an invalid 'summary.total_findings'; "
                "expected a non-negative integer."
            )
        actual_total = sum(sum(codes.values()) for codes in data["findings_by_path"].values())
        if total_findings != actual_total:
            raise ValueError(
                f"Baseline {path} has inconsistent 'summary.total_findings': "
                f"summary.total_findings={total_findings}, but findings_by_path contains "
                f"{actual_total} findings."
            )

    evidence_tree = data.get("evidence_tree")
    if evidence_tree is None:
        return
    if not isinstance(evidence_tree, dict):
        raise ValueError(f"Baseline {path} has an invalid 'evidence_tree' mapping.")
    count = evidence_tree.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"Baseline {path} has an invalid evidence_tree count.")
    if not isinstance(evidence_tree.get("sha256"), str) or not evidence_tree["sha256"]:
        raise ValueError(f"Baseline {path} has an invalid evidence_tree sha256.")


def load_baseline(path: Path) -> dict[str, Any]:
    """Load and validate the structure consumed by the ratchet checks."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read baseline {path}: {exc}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Baseline {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Baseline {path} must be a JSON object, got {type(data).__name__}.")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported baseline schema_version in {path}: "
            f"got {data.get('schema_version')}, expected {SCHEMA_VERSION}"
        )
    findings_by_path = data.get("findings_by_path")
    if not isinstance(findings_by_path, dict):
        raise ValueError(f"Baseline {path} is missing a valid 'findings_by_path' mapping.")
    _validate_finding_counts(path, findings_by_path)
    _validate_baseline_metadata(path, data)
    return data


# --- issue #7467: review-companion delta ------------------------------------------


def _review_entries(review: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return the validated list of ``reviewed_files`` / ``reviewed_baseline_increases``.

    The review companion is schema-versioned (``evidence_registry_baseline_review.v1``);
    a foreign schema is a hard error so a drift in the human contract surfaces loudly
    instead of being silently misread.
    """
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported review companion schema_version: "
            f"got {review.get('schema_version')!r}, expected {REVIEW_SCHEMA_VERSION!r}"
        )
    entries = review.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Review companion '{key}' must be a list, got {type(entries).__name__}.")
    return entries


def _reviewed_paths(review: dict[str, Any]) -> set[str]:
    """Return the set of paths already dispositioned under ``reviewed_files``."""
    reviewed = set()
    for entry in _review_entries(review, "reviewed_files"):
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("Review companion reviewed_files entries must name a 'path'.")
        reviewed.add(path)
    return reviewed


def _declared_increases(review: dict[str, Any]) -> set[tuple[str, str]]:
    """Return the declared ``(path, code)`` set from ``reviewed_baseline_increases``."""
    declared: set[tuple[str, str]] = set()
    for entry in _review_entries(review, "reviewed_baseline_increases"):
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(
                "Review companion reviewed_baseline_increases entries must name a 'path'."
            )
        codes = entry.get("codes")
        if not isinstance(codes, list):
            raise ValueError(
                f"Review companion reviewed_baseline_increases entry '{path}' must list 'codes'."
            )
        for code in codes:
            if not isinstance(code, str) or not code:
                raise ValueError(
                    f"Review companion reviewed_baseline_increases entry '{path}' "
                    "has an invalid code."
                )
            declared.add((path, code))
    return declared


def _findings_by_path(baseline: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Return the baseline ``findings_by_path`` normalized to plain ints."""
    return {
        str(path): {str(code): int(count) for code, count in codes.items()}
        for path, codes in baseline.get("findings_by_path", {}).items()
    }


def companion_delta(
    baseline: dict[str, Any],
    review: dict[str, Any],
    prior_baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compute the review-companion delta for ``baseline`` against ``prior_baseline``.

    Mirrors the decomposition enforced by
    ``tests/dev/test_evidence_registry_ratchet.py::test_review_companion_covers_every_post_5317_baseline_file``
    so a baseline refresh surfaces exactly what the human review companion must
    disposition (issue #7467):

    * ``missing_reviewed_files`` — baseline paths absent from the prior baseline and
      not yet listed in ``reviewed_files``; each needs a companion disposition.
    * ``missing_reviewed_files_codes`` — ``{path: [codes]}`` for the missing
      ``reviewed_files`` paths, so the rendered template lists the exact finding codes
      a maintainer must disposition.
    * ``missing_reviewed_increases`` — ``(path, code)`` pairs where a file already in
      the prior baseline has a higher per-code count in ``baseline`` and the pair is
      not yet declared in ``reviewed_baseline_increases``.
    * ``stale_reviewed_files`` — ``reviewed_files`` paths no longer present in
      ``baseline``; the companion should drop them to stay honest.
    * ``stale_reviewed_increases`` — declared ``(path, code)`` increases that are no
      longer a current increase; the companion should drop them.

    ``empty`` is True when every delta category is empty (the companion fully covers
    the baseline delta). The function is pure: it never reads or writes files.
    """
    current = _findings_by_path(baseline)
    prior = _findings_by_path(prior_baseline)
    reviewed = _reviewed_paths(review)
    declared = _declared_increases(review)

    current_paths = set(current)
    prior_paths = set(prior)
    missing_reviewed_files = sorted(current_paths - prior_paths - reviewed)
    missing_reviewed_increases = sorted(
        {
            (path, code)
            for path, codes in current.items()
            for code, count in codes.items()
            if path in prior_paths and count > prior[path].get(code, 0)
        }
        - declared
    )
    stale_reviewed_files = sorted(reviewed - current_paths)
    stale_reviewed_increases = sorted(
        {
            (path, code)
            for path, code in declared
            if path not in current_paths
            or path not in prior_paths
            or current[path].get(code, 0) <= prior[path].get(code, 0)
        }
    )
    return {
        "missing_reviewed_files": missing_reviewed_files,
        "missing_reviewed_files_codes": {
            path: sorted(current[path]) for path in missing_reviewed_files
        },
        "missing_reviewed_increases": missing_reviewed_increases,
        "stale_reviewed_files": stale_reviewed_files,
        "stale_reviewed_increases": stale_reviewed_increases,
        "empty": not (
            missing_reviewed_files
            or missing_reviewed_increases
            or stale_reviewed_files
            or stale_reviewed_increases
        ),
    }


def load_review_companion(path: Path) -> dict[str, Any]:
    """Load and validate the human review companion YAML.

    Raises ``ValueError`` on a missing, unparseable, or foreign-schema companion so
    the CLI maps the problem to the documented infra-error exit code instead of an
    uncaught traceback.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read review companion {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Review companion {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Review companion {path} must be a YAML mapping, got {type(data).__name__}."
        )
    if data.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported review companion schema_version in {path}: "
            f"got {data.get('schema_version')!r}, expected {REVIEW_SCHEMA_VERSION!r}"
        )
    # Validate the sections the delta computation reads directly, plus the
    # disposition contract ({baseline, remediate}) so a drift in the human
    # companion surfaces loudly instead of silently weakening the gate.
    _reviewed_paths(data)
    _declared_increases(data)
    for key in ("reviewed_files", "reviewed_baseline_increases"):
        for entry in _review_entries(data, key):
            disposition = entry.get("disposition")
            if disposition not in VALID_DISPOSITIONS:
                raise ValueError(
                    f"Review companion {path} entry '{entry.get('path')}' has an invalid "
                    f"disposition {disposition!r}; expected one of {VALID_DISPOSITIONS}."
                )
    return data


def load_prior_baseline_from_git(repo_root: Path, commit: str) -> dict[str, Any]:
    """Load the prior ratchet baseline at ``commit`` from the local git history.

    Raises ``ValueError`` (never a traceback) when the commit does not exist or does
    not contain the baseline file, so ``--companion-delta`` degrades to a documented
    error the maintainer can act on.
    """
    proc = subprocess.run(
        [
            "git",
            "show",
            f"{commit}:scripts/validation/evidence_registry_baseline.json",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Could not load the prior baseline at commit {commit}: git show exited "
            f"{proc.returncode}. Ensure the commit exists in the local history.\n"
            f"git stderr: {proc.stderr.strip()[:500]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Prior baseline at commit {commit} is not valid JSON: {exc}") from exc


def _render_code_list(codes: list[str]) -> str:
    """Render a sorted code list as a YAML inline list (deterministic)."""
    if not codes:
        return "[]"
    return "[" + ", ".join(codes) + "]"


def render_companion_template(delta: dict[str, Any]) -> str:
    """Render a deterministic, paste-able YAML template for ``delta``.

    The output is emitted by ``--companion-delta`` so a maintainer can append the
    missing ``reviewed_files`` / ``reviewed_baseline_increases`` entries to the review
    companion verbatim. Entries carry a placeholder ``baseline`` disposition and a
    placeholder reason; the ordering is deterministic (sorted paths / path-then-code)
    so repeated runs produce byte-identical templates. Stale entries are listed under
    a comment block rather than in the paste-able payload. The rendered YAML must
    round-trip through ``yaml.safe_load``.
    """
    missing_files = delta.get("missing_reviewed_files", [])
    missing_files_codes = delta.get("missing_reviewed_files_codes", {})
    missing_increases = delta.get("missing_reviewed_increases", [])
    stale_files = delta.get("stale_reviewed_files", [])
    stale_increases = delta.get("stale_reviewed_increases", [])
    lines = [
        "# Evidence-registry review-companion delta (issue #7467)",
        "# Generated by scripts/dev/evidence_registry_ratchet.py --companion-delta",
        "# Paste the entries below into scripts/validation/evidence_registry_baseline_review.yaml.",
        "# A disposition is exactly one of: baseline | remediate. Replace the placeholder",
        "# reason with a short justification before committing.",
        "",
    ]
    if missing_files:
        lines.extend(
            [
                "reviewed_files:",
                *[
                    (
                        f"  - path: {path}\n"
                        f"    codes: {_render_code_list(missing_files_codes.get(path, []))}\n"
                        "    disposition: baseline\n"
                        "    reason: >-\n"
                        "      PLACEHOLDER: review this newly baselined file and record its\n"
                        "      remediate-or-baseline disposition (issue #7467)."
                    )
                    for path in missing_files
                ],
            ]
        )
    else:
        lines.append("reviewed_files: []")
    lines.append("")
    if missing_increases:
        lines.extend(
            [
                "reviewed_baseline_increases:",
                *[
                    (
                        f"  - path: {path}\n"
                        f"    codes: {_render_code_list([code])}\n"
                        "    disposition: baseline\n"
                        "    reason: >-\n"
                        "      PLACEHOLDER: review this per-code baseline increase and record\n"
                        "      its remediate-or-baseline disposition (issue #7467)."
                    )
                    for path, code in missing_increases
                ],
            ]
        )
    else:
        lines.append("reviewed_baseline_increases: []")
    lines.append("")
    if stale_files or stale_increases:
        lines.append("# Stale companion entries (no longer in the baseline):")
        for path in stale_files:
            lines.append(f"#   - reviewed_files path: {path}")
        for path, code in stale_increases:
            lines.append(f"#   - reviewed_baseline_increases path/code: {path} :: {code}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _report_companion_delta(delta: dict[str, Any]) -> int:
    """Print the ``--companion-delta`` report and return the exit code."""
    missing_files = delta.get("missing_reviewed_files", [])
    missing_increases = delta.get("missing_reviewed_increases", [])
    stale_files = delta.get("stale_reviewed_files", [])
    stale_increases = delta.get("stale_reviewed_increases", [])
    if delta.get("empty", False):
        print(
            "evidence-registry companion delta is empty: the review companion already "
            "covers the committed baseline (no new paths, no per-code increases, no "
            "stale entries)."
        )
        return 0
    print(
        "evidence-registry companion delta: the review companion does not yet cover "
        "the committed baseline.",
        file=sys.stderr,
    )
    if missing_files:
        print("\nreviewed_files entries required:", file=sys.stderr)
        for path in missing_files:
            print(f"  - {path}", file=sys.stderr)
    if missing_increases:
        print("\nreviewed_baseline_increases entries required:", file=sys.stderr)
        for path, code in missing_increases:
            print(f"  - {path} :: {code}", file=sys.stderr)
    if stale_files:
        print("\nstale reviewed_files entries (no longer in the baseline):", file=sys.stderr)
        for path in stale_files:
            print(f"  - {path}", file=sys.stderr)
    if stale_increases:
        print("\nstale reviewed_baseline_increases entries:", file=sys.stderr)
        for path, code in stale_increases:
            print(f"  - {path} :: {code}", file=sys.stderr)
    print(
        "\nRefresh scripts/validation/evidence_registry_baseline.json with --write-baseline,"
        "\nthen paste the rendered template below into the review companion",
        "\n(scripts/validation/evidence_registry_baseline_review.yaml) with real",
        "\nremediating-or-baseline dispositions before the full pr_ready_check.sh run.",
        file=sys.stderr,
    )
    return 1


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
    mode.add_argument(
        "--companion-delta",
        action="store_true",
        help=(
            "Report the review-companion delta for the committed baseline: every new "
            "baseline path and per-code increase still needing a companion disposition, "
            "plus stale companion entries. Prints a deterministic YAML template to paste "
            "into the review companion. Read-only; exits 1 when dispositions are missing."
        ),
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--root", type=Path, default=None, help="Repository root (defaults to git toplevel)."
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=DEFAULT_REVIEW_COMPANION,
        help="Path to the review companion YAML (default: the committed companion).",
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
        print("\nevidence-registry ratchet FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nFix: regenerate the baseline from a full-history checkout with the "
            "single command\n"
            "    uv run python scripts/dev/evidence_registry_ratchet.py --write-baseline\n"
            "then commit scripts/validation/evidence_registry_baseline.json in this PR. "
            "The baseline is derived data -- never hand-edit it (issue #6839).",
            file=sys.stderr,
        )
        return 1
    print("evidence-registry ratchet passed: no net-new findings; clean files stayed clean.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ratchet gate, baseline refresh, aggregate report, or companion-delta."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = args.root.resolve() if args.root is not None else _repo_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else repo_root / args.baseline

    if args.companion_delta:
        return _run_companion_delta(repo_root, baseline_path, args.review)

    try:
        report = _gather_report(args, repo_root)
    except RuntimeError as exc:
        print(f"ERROR: could not obtain linter report: {exc}", file=sys.stderr)
        return 2

    if args.aggregate_only:
        _print_aggregate(report)
        return 0

    registry_root = repo_root / DEFAULT_REGISTRY_ROOT

    if args.write_baseline:
        payload = build_baseline_payload(report, registry_root)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(baseline_path, payload)
        manifest = payload["evidence_tree"]
        print(
            f"Wrote evidence-registry baseline to {baseline_path}: "
            f"{payload['summary']['total_findings']} findings across "
            f"{payload['summary']['files_with_findings']} files; "
            f"evidence_tree manifest tracks {manifest['count']} evidence files."
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
    try:
        baseline = load_baseline(baseline_path)
    except ValueError as exc:
        print(f"ERROR: could not load baseline: {exc}", file=sys.stderr)
        return 2
    findings_failures, notices = check_against_baseline(aggregate(report), baseline)
    manifest_failures, manifest_notices = check_evidence_tree_manifest(
        evidence_tree_manifest(registry_root), baseline
    )
    failures = [*findings_failures, *manifest_failures]
    notices = [*notices, *manifest_notices]
    return _report_check(report, baseline, failures, notices)


def _run_companion_delta(repo_root: Path, baseline_path: Path, review_arg: Path) -> int:
    """Run the read-only ``--companion-delta`` report for the committed baseline.

    Loads the committed baseline, the human review companion, and the prior baseline
    (via ``git show`` at the companion's ``prior_baseline_commit``), computes the
    decomposition the companion contract enforces, and prints either a success line
    (delta empty) or a deterministic YAML template listing exactly the missing and
    stale companion entries. The companion file is never written.
    """
    review_path = review_arg if review_arg.is_absolute() else repo_root / review_arg
    try:
        baseline = load_baseline(baseline_path)
        review = load_review_companion(review_path)
    except ValueError as exc:
        print(f"ERROR: could not load baseline or review companion: {exc}", file=sys.stderr)
        return 2
    prior_commit = str(review.get("prior_baseline_commit") or FALLBACK_PRIOR_BASELINE_COMMIT)
    try:
        prior_baseline = load_prior_baseline_from_git(repo_root, prior_commit)
    except ValueError as exc:
        print(f"ERROR: could not load prior baseline: {exc}", file=sys.stderr)
        return 2

    delta = companion_delta(baseline, review, prior_baseline)
    if delta.get("empty", False):
        return _report_companion_delta(delta)
    code = _report_companion_delta(delta)
    print(render_companion_template(delta))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
