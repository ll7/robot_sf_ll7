"""Tests for the evidence-registry integrity downward ratchet (issue #5275).

These tests cover the pure ratchet logic directly (per-file/per-code aggregation, the
downward-ratchet gate) and the CLI end-to-end via ``--report`` so no live linter
invocation is required for the unit suite.

Issue #5952 acceptance (clean-main baseline drift): a regression check asserts that the
committed baseline at ``scripts/validation/evidence_registry_baseline.json`` passes
``--check`` against the *live* registry on a clean checkout, and that the committed
baseline reproduces from ``--write-baseline`` (i.e. the counts are machine-generated, not
hand-edited). A second guard asserts that every evidence file added since the #5275/#5317
baseline carries an explicit remediate-or-baseline disposition in the review companion, so
the downward ratchet cannot silently re-drift by grandfathering unreviewed files.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "evidence_registry_ratchet.py"
BASELINE = ROOT / "scripts" / "validation" / "evidence_registry_baseline.json"
REVIEW = ROOT / "scripts" / "validation" / "evidence_registry_baseline_review.yaml"
STRICT_POLICY = ROOT / "docs" / "context" / "evidence" / "evidence_registry_strict_ci_policy.yaml"
LINTER = ROOT / "scripts" / "tools" / "lint_evidence_registry.py"
PRIOR_BASELINE_COMMIT = "9fa96c01bf1c8152459f5fa8c481e938fb1e6725"

# Import the helper as a source module (it lives under scripts/dev, not a package).
_spec = importlib.util.spec_from_file_location("evidence_registry_ratchet", SCRIPT)
assert _spec is not None and _spec.loader is not None
ratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ratchet)


@pytest.fixture(scope="module")
def live_lint_report() -> dict[str, object]:
    """Run the evidence-registry linter once per module and cache its report-mode JSON.

    The report-mode linter scan is the expensive part (~12 s over the full evidence tree);
    the reproducibility, drift, and review-coverage guards all consume the same report, so it
    runs once instead of once per test.
    """
    return ratchet.run_linter(ROOT)


@pytest.fixture(scope="module")
def live_strict_report() -> dict[str, object]:
    """Run the strict linter once per module and cache its strict-mode JSON."""
    proc = subprocess.run(
        [
            sys.executable,
            str(LINTER),
            "--strict",
            "--strict-exclusion-policy",
            str(STRICT_POLICY),
            "--repo-root",
            str(ROOT),
            "--registry-root",
            "docs/context/evidence",
            "--disposition-file",
            "docs/context/evidence/evidence_registry_dispositions.yaml",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 0, (
        f"strict evidence-registry linter failed to run:\n{proc.stdout}\n{proc.stderr}"
    )
    return json.loads(proc.stdout)


def _issue(path: str, code: str) -> dict[str, str]:
    """Build one synthetic linter finding for deterministic ratchet tests."""
    return {"path": path, "code": code, "message": f"{code} on {path}"}


def _report(*findings: dict[str, str]) -> dict[str, object]:
    """Wrap synthetic findings in a minimal linter report envelope."""
    return {"summary": {"findings": len(findings)}, "issues": list(findings)}


# --- pure ratchet logic -----------------------------------------------------------


def test_aggregate_groups_by_path_and_code() -> None:
    """Findings aggregate to {path: {code: count}} with stable ordering."""
    report = _report(
        _issue("a.json", "missing_commit"),
        _issue("a.json", "missing_commit"),
        _issue("a.json", "dangling_commit"),
        _issue("b.json", "missing_commit"),
    )
    assert ratchet.aggregate(report) == {
        "a.json": {"dangling_commit": 1, "missing_commit": 2},
        "b.json": {"missing_commit": 1},
    }


def test_aggregate_tolerates_missing_fields() -> None:
    """A finding missing path/code falls back to a sentinel rather than crashing."""
    report = _report({"path": "a.json"}, {"code": "missing_commit"}, {})
    assert ratchet.aggregate(report) == {
        "<unknown>": {"<unknown>": 1, "missing_commit": 1},
        "a.json": {"<unknown>": 1},
    }


def test_ratchet_passes_when_counts_unchanged() -> None:
    """Equal current and baseline counts hold the gate."""
    current = {"a.json": {"missing_commit": 2}}
    baseline = {"findings_by_path": {"a.json": {"missing_commit": 2}}}
    failures, notices = ratchet.check_against_baseline(current, baseline)
    assert failures == []
    assert notices == []


def test_ratchet_fails_on_clean_file_regression() -> None:
    """A clean file (absent from baseline) that gains any finding fails."""
    current = {"new.json": {"missing_commit": 1}}
    baseline = {"findings_by_path": {}}
    failures, notices = ratchet.check_against_baseline(current, baseline)
    assert len(failures) == 1
    assert "new.json" in failures[0]
    assert "clean file regressed" in failures[0]
    assert notices == []


def test_ratchet_fails_on_tracked_file_per_code_increase() -> None:
    """A tracked file whose per-code count increases fails."""
    current = {"a.json": {"missing_commit": 3}}
    baseline = {"findings_by_path": {"a.json": {"missing_commit": 2}}}
    failures, _ = ratchet.check_against_baseline(current, baseline)
    assert len(failures) == 1
    assert "increased from 2 to 3" in failures[0]


def test_ratchet_passes_on_decrease_and_emits_refresh_notice() -> None:
    """A decrease never fails; it emits an advisory ratchet-opportunity notice."""
    current = {"a.json": {"missing_commit": 1}}
    baseline = {"findings_by_path": {"a.json": {"missing_commit": 3}}}
    failures, notices = ratchet.check_against_baseline(current, baseline)
    assert failures == []
    assert len(notices) == 1
    assert "dropped from 3 to 1" in notices[0]


def test_ratchet_fully_remediated_file_disappears_from_current() -> None:
    """A fully remediated file is absent from current; that is never a failure."""
    current: dict[str, dict[str, int]] = {}
    baseline = {"findings_by_path": {"a.json": {"missing_commit": 2}}}
    failures, _ = ratchet.check_against_baseline(current, baseline)
    assert failures == []


def test_build_baseline_payload_round_trips_through_check() -> None:
    """A freshly built baseline from a report must reproduce and pass --check."""
    report = _report(_issue("a.json", "missing_commit"), _issue("b.json", "dangling_commit"))
    payload = ratchet.build_baseline_payload(report)
    assert payload["summary"]["total_findings"] == 2
    assert payload["summary"]["files_with_findings"] == 2
    failures, _ = ratchet.check_against_baseline(ratchet.aggregate(report), payload)
    assert failures == []


def test_load_baseline_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A baseline with the wrong schema_version fails closed."""
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"schema_version": 999, "findings_by_path": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        ratchet.load_baseline(bad)


def test_load_baseline_rejects_non_dict_findings(tmp_path: Path) -> None:
    """A baseline whose findings_by_path is not a mapping fails closed."""
    bad = tmp_path / "baseline.json"
    bad.write_text(json.dumps({"schema_version": 1, "findings_by_path": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="findings_by_path"):
        ratchet.load_baseline(bad)


# --- CLI end-to-end (no live linter; uses --report) --------------------------------


def _run_cli(tmp_path: Path, report: object, *args: str) -> subprocess.CompletedProcess:
    """Run the ratchet CLI against a pre-rendered report, returning the process result."""
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(tmp_path),
        ]
        + list(args),
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return proc


def test_cli_write_then_check_roundtrip(tmp_path: Path) -> None:
    """A baseline written by --write-baseline must reproduce and pass --check."""
    report = _report(
        _issue("a.json", "missing_commit"),
        _issue("a.json", "missing_commit"),
        _issue("b.json", "dangling_commit"),
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    write = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--write-baseline",
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert write.returncode == 0, write.stderr
    assert baseline.exists()

    check = _run_cli(tmp_path, report)
    assert check.returncode == 0, check.stderr
    assert "ratchet passed" in check.stdout


def test_cli_check_fails_on_clean_file_regression(tmp_path: Path) -> None:
    """A net-new finding in a clean file fails the gate even with a baseline present."""
    # Baseline knows only a.json; current report adds a finding in clean file b.json.
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"schema_version": 1, "findings_by_path": {"a.json": {"missing_commit": 1}}}),
        encoding="utf-8",
    )
    report = _report(_issue("a.json", "missing_commit"), _issue("b.json", "missing_commit"))
    check = _run_cli(tmp_path, report)
    assert check.returncode == 1
    assert "clean file regressed" in check.stderr
    assert "b.json" in check.stderr


def test_cli_check_fails_on_per_code_increase(tmp_path: Path) -> None:
    """A tracked file whose per-code count increased fails the gate."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"schema_version": 1, "findings_by_path": {"a.json": {"missing_commit": 1}}}),
        encoding="utf-8",
    )
    report = _report(_issue("a.json", "missing_commit"), _issue("a.json", "missing_commit"))
    check = _run_cli(tmp_path, report)
    assert check.returncode == 1
    assert "finding count increased" in check.stderr


def test_cli_check_reports_infra_error_when_report_missing(tmp_path: Path) -> None:
    """A missing --report path is an infra error (exit 2), not a ratchet failure."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"schema_version": 1, "findings_by_path": {}}), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--report",
            str(tmp_path / "nope.json"),
            "--baseline",
            str(baseline),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 2


@pytest.mark.parametrize(
    "report",
    [[], None, {"issues": None}, {"issues": [None]}, {"summary": []}],
)
def test_cli_check_reports_infra_error_for_malformed_report(tmp_path: Path, report: object) -> None:
    """Malformed report JSON fails with the documented infra-error exit code."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"schema_version": 1, "findings_by_path": {}}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--report",
            str(report_path),
            "--baseline",
            str(baseline),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 2


# --- issue #5952 acceptance: clean-main baseline drift guard -----------------------


def test_committed_baseline_exists_and_is_valid() -> None:
    """The committed baseline exists and loads under the current schema."""
    assert BASELINE.is_file(), "evidence-registry baseline is missing; run --write-baseline."
    data = ratchet.load_baseline(BASELINE)
    assert data["schema_version"] == ratchet.SCHEMA_VERSION
    assert isinstance(data["findings_by_path"], dict)
    assert data["summary"]["total_findings"] > 0


def test_committed_baseline_reproduces_from_write_baseline(
    live_lint_report: dict[str, object],
) -> None:
    """The committed baseline must match `--write-baseline` output byte-for-byte.

    This is the reproducibility half of the #5952 guard: the committed counts must be
    machine-generated, so that a regenerated baseline is a true refresh and not a silent
    hand-edit. If this fails, the baseline was edited by hand or the linter output changed
    without a refresh; regenerate with `evidence_registry_ratchet.py --write-baseline`.
    """
    regenerated = ratchet.build_baseline_payload(live_lint_report)
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    # Compare the machine-generated fields (generated_at is a timestamp by design).
    for key in ("findings_by_path", "summary", "linter", "schema_version"):
        assert committed.get(key) == regenerated.get(key), (
            f"evidence-registry baseline field '{key}' does not reproduce from "
            "--write-baseline; regenerate with `evidence_registry_ratchet.py --write-baseline`."
        )


@pytest.mark.slow
def test_committed_baseline_passes_live_check_on_clean_main(
    live_lint_report: dict[str, object],
) -> None:
    """The live downward ratchet passes against the committed baseline (clean-main guard).

    This is the drift guard the #5275/#5317 baseline was missing: on a clean checkout the
    committed baseline must reconcile with the *current* tracked evidence files. If a PR
    merges new evidence files without refreshing the baseline, this check fails. It consumes
    the cached linter report (the scan is shared across the module) and exercises the
    production check gate directly.
    """
    baseline = ratchet.load_baseline(BASELINE)
    failures, _ = ratchet.check_against_baseline(ratchet.aggregate(live_lint_report), baseline)
    assert failures == [], (
        "evidence-registry ratchet does not pass on clean main; the committed baseline has "
        "drifted from the tracked evidence files:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nRefresh with `evidence_registry_ratchet.py --write-baseline` and record the "
        "per-file disposition in evidence_registry_baseline_review.yaml."
    )


def test_review_companion_covers_every_post_5317_baseline_file() -> None:
    """Every file added since the #5275/#5317 baseline has an explicit disposition (#5952 DoD).

    The downward ratchet only stops drift if newly-baselined files are deliberate. This guard
    fails if the committed baseline grew beyond the documented prior boundary without a matching
    explicit per-file disposition in the review companion. It is self-contained: the prior
    boundary file count and the refresh delta are both recorded in the review companion, so the
    invariant is ``len(baseline_files) == prior_count + len(reviewed_files)`` together with
    ``reviewed_files ⊆ baseline_files`` and a valid disposition on every reviewed entry. Future
    net-new drift is additionally caught by the live check above.
    """
    assert REVIEW.is_file(), "evidence_registry_baseline_review.yaml is missing."
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    reviewed_entries = review.get("reviewed_files", [])
    reviewed = {entry["path"] for entry in reviewed_entries}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_files = set(baseline["findings_by_path"])

    # (1) Load the actual prior baseline, rather than trusting only a self-reported count.
    assert review.get("prior_baseline_commit") == PRIOR_BASELINE_COMMIT
    prior_proc = subprocess.run(
        [
            "git",
            "show",
            f"{PRIOR_BASELINE_COMMIT}:scripts/validation/evidence_registry_baseline.json",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert prior_proc.returncode == 0, prior_proc.stderr
    prior_baseline = json.loads(prior_proc.stdout)
    prior_files = set(prior_baseline["findings_by_path"])

    # (2) The committed baseline must decompose as actual prior files + reviewed delta.
    prior_count = int(review["prior_baseline_files_with_findings"])
    assert len(prior_files) == prior_count
    assert baseline_files - prior_files == reviewed, (
        "The committed baseline contains a file not present in the anchored prior baseline "
        "without an explicit disposition in evidence_registry_baseline_review.yaml. Add a "
        "reviewed_files entry naming the new path and its remediate-or-baseline disposition."
    )

    # (3) Every reviewed entry must still be in the baseline (review stays honest).
    stale_review = reviewed - baseline_files
    assert not stale_review, (
        "evidence_registry_baseline_review.yaml lists files no longer in the baseline: "
        f"{sorted(stale_review)}"
    )

    # (4) Every reviewed entry must carry a valid disposition and list its codes.
    valid_dispositions = {"baseline", "remediate"}
    for entry in reviewed_entries:
        assert entry.get("disposition") in valid_dispositions, (
            f"reviewed file {entry.get('path')} lacks a valid disposition "
            f"(got {entry.get('disposition')!r})"
        )
        assert isinstance(entry.get("codes"), list) and entry["codes"], (
            f"reviewed file {entry.get('path')} must list its finding codes"
        )


def test_strict_ci_policy_has_zero_active_findings_on_clean_main(
    live_strict_report: dict[str, object],
) -> None:
    """No active strict-linter finding is introduced (issue #5952 DoD #3).

    Consumes the cached strict linter report and asserts zero active findings, so a net-new
    code that is not in the exclusion policy cannot hide behind the refreshed baseline.
    """
    assert live_strict_report["summary"]["findings"] == 0
    assert live_strict_report["issues"] == []
