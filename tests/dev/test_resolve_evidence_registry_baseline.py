"""Tests for the evidence-registry baseline resolver (issue #6839, Option 1 complement).

The resolver regenerates a stale baseline mechanically -- but ONLY when the live per-file
findings reconcile with the committed baseline, so it can never silently grandfather a
net-new integrity regression. These tests exercise the safety gate, the safe-regenerate
path, the conflict-marker guard, and the missing/malformed-baseline error paths using a
synthetic ``--report`` (no live linter scan) so the suite stays fast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESOLVER = ROOT / "scripts" / "dev" / "resolve_evidence_registry_baseline.py"
RATCHET = ROOT / "scripts" / "dev" / "evidence_registry_ratchet.py"


def _write_report(repo: Path, findings: list[dict]) -> Path:
    """Write a synthetic linter report under ``repo`` and return its path."""
    report_path = repo / "report.json"
    report_path.write_text(
        json.dumps({"summary": {"findings": len(findings)}, "issues": findings}),
        encoding="utf-8",
    )
    return report_path


def _seed_baseline(repo: Path, report_path: Path, baseline: Path) -> None:
    """Generate a baseline (with the evidence_tree manifest) for the current tree."""
    proc = subprocess.run(
        [
            sys.executable,
            str(RATCHET),
            "--write-baseline",
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def _run_resolver(repo: Path, report_path: Path, baseline: Path) -> subprocess.CompletedProcess:
    """Run the resolver against a synthetic report and temp baseline."""
    return subprocess.run(
        [
            sys.executable,
            str(RESOLVER),
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )


def test_resolver_regenerates_stale_manifest_when_findings_reconcile(tmp_path: Path) -> None:
    """A stale evidence_tree manifest with matching findings is regenerated (safe)."""
    repo = tmp_path
    evidence = repo / "docs" / "context" / "evidence" / "issue_6839_bundle"
    evidence.mkdir(parents=True)
    (evidence / "a.json").write_text("{}", encoding="utf-8")
    report_path = _write_report(repo, [])
    baseline = repo / "baseline.json"
    _seed_baseline(repo, report_path, baseline)
    assert (
        "evidence_tree manifest tracks 1 evidence files"
        in subprocess.run(
            [
                sys.executable,
                str(RATCHET),
                "--write-baseline",
                "--report",
                str(report_path),
                "--baseline",
                str(baseline),
                "--root",
                str(repo),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        ).stdout
    )

    # PR adds a NEW clean evidence file: findings still reconcile, manifest is stale.
    (evidence / "b.json").write_text("{}", encoding="utf-8")
    result = _run_resolver(repo, report_path, baseline)
    assert result.returncode == 0, result.stderr
    assert "resolve: regenerated baseline" in result.stdout
    assert "evidence_tree manifest now tracks 2 evidence files" in result.stdout

    # The regenerated baseline must pass the fail-closed --check.
    check = subprocess.run(
        [
            sys.executable,
            str(RATCHET),
            "--check",
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert check.returncode == 0, check.stderr


def test_resolver_refuses_to_grandfather_net_new_findings(tmp_path: Path) -> None:
    """Net-new integrity findings require review; the resolver must refuse (fail-closed)."""
    repo = tmp_path
    evidence = repo / "docs" / "context" / "evidence" / "bundle"
    evidence.mkdir(parents=True)
    (evidence / "a.json").write_text("{}", encoding="utf-8")
    clean_report = _write_report(repo, [])
    baseline = repo / "baseline.json"
    _seed_baseline(repo, clean_report, baseline)

    # A follow-up scan now reports a net-new finding in a clean file.
    drift_report = _write_report(
        repo,
        [{"path": "docs/context/evidence/bundle/a.json", "code": "missing_commit", "message": "x"}],
    )
    result = _run_resolver(repo, drift_report, baseline)
    assert result.returncode == 1, (
        "resolver must REFUSE when the live scan reports net-new findings"
    )
    assert "REFUSING to auto-regenerate" in result.stderr
    # And it must NOT have rewritten the baseline.
    committed = json.loads(baseline.read_text(encoding="utf-8"))
    assert committed["evidence_tree"]["count"] == 1


def test_resolver_refuses_on_conflict_markers(tmp_path: Path) -> None:
    """A conflicted baseline may hide findings from either side; refuse with guidance."""
    repo = tmp_path
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        "<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> branch\n",
        encoding="utf-8",
    )
    report_path = _write_report(repo, [])
    result = _run_resolver(repo, report_path, baseline)
    assert result.returncode == 1
    assert "conflict markers" in result.stderr
    assert "evidence_registry_ratchet.py --write-baseline" in result.stderr


def test_resolver_reports_already_current_baseline(tmp_path: Path) -> None:
    """An up-to-date baseline is left untouched (nothing to regenerate)."""
    repo = tmp_path
    evidence = repo / "docs" / "context" / "evidence" / "bundle"
    evidence.mkdir(parents=True)
    (evidence / "a.json").write_text("{}", encoding="utf-8")
    report_path = _write_report(repo, [])
    baseline = repo / "baseline.json"
    _seed_baseline(repo, report_path, baseline)

    result = _run_resolver(repo, report_path, baseline)
    assert result.returncode == 0, result.stderr
    assert "already current" in result.stdout


def test_resolver_errors_on_missing_baseline(tmp_path: Path) -> None:
    """A missing baseline is an infra error (exit 2), not a silent regenerate."""
    repo = tmp_path
    report_path = _write_report(repo, [])
    result = _run_resolver(repo, report_path, tmp_path / "missing.json")
    assert result.returncode == 2
    assert "baseline not found" in result.stderr


def test_resolver_errors_on_malformed_baseline(tmp_path: Path) -> None:
    """A malformed (non-JSON, non-conflict) baseline is an infra error."""
    repo = tmp_path
    baseline = tmp_path / "baseline.json"
    baseline.write_text("not json at all {{{", encoding="utf-8")
    report_path = _write_report(repo, [])
    result = _run_resolver(repo, report_path, baseline)
    assert result.returncode == 2
    assert "not valid JSON" in result.stderr
