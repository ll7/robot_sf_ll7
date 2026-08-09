#!/usr/bin/env python3
# evidence-writer-exempt: baseline reconciliation tooling; the docs/context/evidence tree is
# a read-only input and the baseline is derived data this helper regenerates.
"""Mechanically resolve a stale evidence-registry baseline (issue #6839, Option 1 complement).

The baseline at ``scripts/validation/evidence_registry_baseline.json`` is derived data:
a pure function of the evidence tree (per-file findings + an ``evidence_tree`` manifest).
The blocking ratchet gate (``evidence_registry_ratchet.py --check``) now fails a PR that
adds or removes files under ``docs/context/evidence/`` without refreshing the baseline, so
the drift is attributed to the causing PR (Option 2). This helper makes the *fix*
mechanical (Option 1): it regenerates the baseline for the author in one step.

It regenerates ONLY when it is safe -- when the live per-file findings reconcile with the
committed baseline (no net-new integrity regressions). Net-new findings require human
review and an explicit disposition recorded in
``evidence_registry_baseline_review.yaml``, so this helper refuses to silently grandfather
them and points at the manual review path instead. It never hand-edits the baseline and
always finishes by re-running the fail-closed ``--check``.

This mirrors the generated-file resolver pattern of
``scripts/dev/resolve_context_catalog_conflict.py`` (PR #6788), but for a fully-regenerated
file the union resolution is simply "regenerate from the current tree".

Usage::

    # Regenerate the baseline if its evidence_tree manifest is stale and findings match.
    uv run python scripts/dev/resolve_evidence_registry_baseline.py

    # Offline / test mode: parse a pre-rendered linter report instead of re-running it.
    uv run python scripts/dev/resolve_evidence_registry_baseline.py --report /tmp/report.json

Exit codes
----------
* ``0`` -- baseline regenerated (or already current) and the fail-closed check holds.
* ``1`` -- refused: net-new findings need review, or the post-regeneration check failed.
* ``2`` -- infra error (report could not be obtained, or baseline missing/malformed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
RATCHET_PATH = REPO_ROOT / "scripts" / "dev" / "evidence_registry_ratchet.py"
DEFAULT_BASELINE = REPO_ROOT / "scripts" / "validation" / "evidence_registry_baseline.json"
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")
REGENERATE_COMMAND = "scripts/dev/evidence_registry_ratchet.py --write-baseline"


def _load_ratchet_module(ratchet_path: Path):
    """Import the ratchet helper as a source module (it lives under scripts/dev)."""
    spec = importlib.util.spec_from_file_location("evidence_registry_ratchet", ratchet_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_has_conflict_markers(text: str) -> bool:
    """Return whether ``text`` contains unresolved Git conflict markers."""
    return any(marker in text for marker in CONFLICT_MARKERS)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Path to a pre-rendered linter JSON report. When set, the linter is NOT "
            "re-run; the report is parsed instead (offline / test mode)."
        ),
    )
    return parser.parse_args(argv)


def _load_resolvable_baseline(baseline_path: Path) -> tuple[dict | None, int]:
    """Load the baseline for resolution, returning ``(baseline, exit_code)``.

    ``exit_code`` is 0 with a loaded baseline, or a non-zero infra/refuse code with the
    diagnostic already printed. A conflicted or malformed baseline is not auto-resolved.
    """
    if not baseline_path.is_file():
        print(
            f"ERROR: baseline not found at {baseline_path}. Generate it first with "
            f"`{REGENERATE_COMMAND}`.",
            file=sys.stderr,
        )
        return None, 2
    raw = baseline_path.read_text(encoding="utf-8")
    if baseline_has_conflict_markers(raw):
        # A conflicted baseline may hide net-new findings from either side, so we do not
        # auto-resolve it. Point at the explicit regenerate command (still mechanical).
        print(
            f"resolve: {baseline_path} has unresolved Git conflict markers. A conflict may "
            "combine net-new findings from either side, so this helper will not auto-resolve "
            f"it. After reviewing the merge, regenerate from the merged tree with "
            f"`{REGENERATE_COMMAND}` and record any new file's disposition in "
            "scripts/validation/evidence_registry_baseline_review.yaml.",
            file=sys.stderr,
        )
        return None, 1
    try:
        return json.loads(raw), 0
    except json.JSONDecodeError as exc:
        print(f"ERROR: baseline {baseline_path} is not valid JSON: {exc}", file=sys.stderr)
        return None, 2


def _gather_report(ratchet, args: argparse.Namespace, repo_root: Path) -> tuple[dict | None, int]:
    """Return ``(report, exit_code)``, running the linter or parsing ``--report``."""
    try:
        if args.report is not None:
            return ratchet.load_report(args.report), 0
        return ratchet.run_linter(repo_root), 0
    except RuntimeError as exc:
        print(
            f"ERROR: could not obtain the evidence-registry linter report: {exc}",
            file=sys.stderr,
        )
        return None, 2


def _regenerate_and_verify(ratchet, registry_root: Path, baseline_path: Path, report: dict) -> int:
    """Regenerate the baseline from ``report`` and re-run the fail-closed check."""
    payload = ratchet.build_baseline_payload(report, registry_root)
    ratchet.write_json(baseline_path, payload)
    print(
        f"resolve: regenerated baseline {baseline_path}: "
        f"{payload['summary']['total_findings']} findings across "
        f"{payload['summary']['files_with_findings']} files; "
        f"evidence_tree manifest now tracks {payload['evidence_tree']['count']} evidence files."
    )
    reloaded = ratchet.load_baseline(baseline_path)
    check_failures, _ = ratchet.check_against_baseline(ratchet.aggregate(report), reloaded)
    manifest_recheck, _ = ratchet.check_evidence_tree_manifest(
        ratchet.evidence_tree_manifest(registry_root), reloaded
    )
    remaining = [*check_failures, *manifest_recheck]
    if remaining:
        print("resolve: post-regeneration --check still fails:\n", file=sys.stderr)
        for failure in remaining:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("resolve: baseline reconciled and the fail-closed --check holds.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve a stale baseline when it is safe; refuse when findings need review."""
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    repo_root = args.root.resolve()
    baseline_path = args.baseline.resolve()
    ratchet = _load_ratchet_module(RATCHET_PATH)
    registry_root = repo_root / ratchet.DEFAULT_REGISTRY_ROOT

    baseline, code = _load_resolvable_baseline(baseline_path)
    if code != 0:
        return code
    report, code = _gather_report(ratchet, args, repo_root)
    if code != 0:
        return code

    # Safety gate: never silently grandfather net-new integrity findings.
    findings_failures, _ = ratchet.check_against_baseline(ratchet.aggregate(report), baseline)
    if findings_failures:
        print(
            "resolve: REFUSING to auto-regenerate -- the live scan reports net-new integrity "
            "findings that require human review and an explicit disposition:\n",
            file=sys.stderr,
        )
        for failure in findings_failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nRemediate the findings, then refresh the baseline with review via "
            f"`{REGENERATE_COMMAND}` and record the new file's disposition in "
            "scripts/validation/evidence_registry_baseline_review.yaml.",
            file=sys.stderr,
        )
        return 1

    manifest_failures, _ = ratchet.check_evidence_tree_manifest(
        ratchet.evidence_tree_manifest(registry_root), baseline
    )
    if not manifest_failures:
        print(
            "resolve: baseline is already current (findings and evidence_tree manifest match); "
            "nothing to regenerate."
        )
        return 0

    # Safe to regenerate: findings reconcile, only the evidence_tree manifest is stale.
    return _regenerate_and_verify(ratchet, registry_root, baseline_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
