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

# evidence-writer-exempt: these tests intentionally write synthetic evidence-tree and
# baseline JSON fixtures only under pytest tmp_path to exercise checker diagnostics;
# the fixtures must keep exact raw bytes (including malformed/conflict content).

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import warnings
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


def test_build_review_companion_template_is_deterministic() -> None:
    """The review delta exposes every new/increased finding without approving it."""
    previous = {
        "schema_version": 1,
        "generated_at": "2026-08-18T06:00:00+00:00",
        "summary": {"total_findings": 1, "files_with_findings": 1},
        "findings_by_path": {"old.json": {"missing_commit": 1}},
        "evidence_tree": {"count": 1, "sha256": "old-tree"},
    }
    candidate = {
        "schema_version": 1,
        "generated_at": "2026-08-18T06:01:00+00:00",
        "summary": {"total_findings": 4, "files_with_findings": 2},
        "findings_by_path": {
            "old.json": {"missing_commit": 2, "dangling_commit": 1},
            "new.json": {"missing_config_path": 1},
        },
        "evidence_tree": {"count": 2, "sha256": "new-tree"},
    }

    template = ratchet.build_review_companion_template(previous, candidate)

    assert template == ratchet.build_review_companion_template(previous, candidate)
    assert template["schema_version"] == "evidence_registry_baseline_review_delta.v1"
    assert template["status"] == "needs_human_review"
    assert template["reviewed_files"] == [
        {
            "path": "new.json",
            "codes": ["missing_config_path"],
            "current_counts": {"missing_config_path": 1},
            "disposition": "TODO",
            "reason": "TODO: choose baseline or remediate and explain why.",
        }
    ]
    assert template["reviewed_baseline_increases"] == [
        {
            "path": "old.json",
            "codes": ["dangling_commit", "missing_commit"],
            "previous_counts": {"dangling_commit": 0, "missing_commit": 1},
            "current_counts": {"dangling_commit": 1, "missing_commit": 2},
            "disposition": "TODO",
            "reason": "TODO: choose baseline or remediate and explain why.",
        }
    ]
    assert template["evidence_tree_delta"]["changed"] is True
    assert template["prior_baseline"]["digest"] == ratchet._payload_digest(previous)
    assert template["candidate_baseline"]["digest"] == ratchet._payload_digest(candidate)


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


def test_load_baseline_rejects_inconsistent_summary_total(tmp_path: Path) -> None:
    """A baseline total must agree with the nested per-file finding counts."""
    bad = tmp_path / "baseline.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"total_findings": 2},
                "findings_by_path": {"a.json": {"missing_commit": 1}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inconsistent 'summary.total_findings'"):
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


def test_cli_write_can_emit_review_companion_template(tmp_path: Path) -> None:
    """A refresh can emit a human-review delta before replacing the machine baseline."""
    baseline = tmp_path / "baseline.json"
    prior_report_path = tmp_path / "prior-report.json"
    prior_report_path.write_text(
        json.dumps(_report(_issue("old.json", "missing_commit"))), encoding="utf-8"
    )
    initial = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--write-baseline",
            "--report",
            str(prior_report_path),
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
    assert initial.returncode == 0, initial.stderr

    current_report_path = tmp_path / "current-report.json"
    current_report_path.write_text(
        json.dumps(
            _report(
                _issue("old.json", "missing_commit"),
                _issue("old.json", "missing_commit"),
                _issue("new.json", "dangling_commit"),
            )
        ),
        encoding="utf-8",
    )
    template_path = tmp_path / "review-delta.yaml"
    refresh = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--write-baseline",
            "--review-template",
            str(template_path),
            "--report",
            str(current_report_path),
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

    assert refresh.returncode == 0, refresh.stderr
    review = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert review["status"] == "needs_human_review"
    assert [entry["path"] for entry in review["reviewed_files"]] == ["new.json"]
    assert [entry["path"] for entry in review["reviewed_baseline_increases"]] == ["old.json"]
    assert ratchet.load_baseline(baseline)["summary"]["total_findings"] == 3


def test_cli_rejects_review_template_without_baseline_refresh(tmp_path: Path) -> None:
    """The review-template output cannot be requested on a read-only check."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--review-template", str(tmp_path / "delta.yaml")],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert proc.returncode == 2
    assert "requires --write-baseline" in proc.stderr


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
    [
        [],
        None,
        {},
        {"issues": None},
        {"issues": [None]},
        {"summary": []},
        {"summary": {"findings": "not-a-number"}, "issues": []},
        {"summary": {"findings": 1}, "issues": []},
        {"summary": {"findings": 1}, "issues": [{"path": [], "code": "missing_commit"}]},
    ],
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


def test_run_linter_reports_malformed_output_as_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live linter path uses the same fail-closed report validation as --report."""
    completed = subprocess.CompletedProcess(
        args=["fake-linter"], returncode=0, stdout=json.dumps({}), stderr=""
    )
    monkeypatch.setattr(ratchet.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="invalid or missing 'issues'"):
        ratchet.run_linter(tmp_path)


def test_cli_check_reports_infra_error_for_malformed_baseline_summary(tmp_path: Path) -> None:
    """A non-numeric baseline total returns the infra code without a traceback."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {"total_findings": "not-a-number"},
                "findings_by_path": {},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"summary": {"findings": 0}, "issues": []}), encoding="utf-8")

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
    assert "summary.total_findings" in proc.stderr
    assert "Traceback" not in proc.stderr


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
    """ADVISORY (non-blocking): committed baseline reproduces from ``--write-baseline``.

    Historically this asserted the committed baseline matched ``--write-baseline`` output
    byte-for-byte, proving the counts were machine-generated rather than hand-edited. In
    practice it became a self-inflicted treadmill: the ``dangling_commit`` counts depend on
    which commits are present in the checkout, and the blocking ``fast-feedback`` CI job uses a
    shallow checkout (``actions/checkout`` default ``fetch-depth: 1``) while the committed
    baseline is generated with full history. Every merged PR that shifts the shallow graft
    point re-drifts the counts, reddening ``main`` and jamming the whole merge sweep.

    Maintainer ruling (2026-07-17, tracked in issue #5991; origin guard #5952): development
    speed outweighs this one reproducibility check, so on drift it is ADVISORY -- it emits a
    loud warning and ``xfail``s instead of failing the build, so it can never jam ``main``.

    The genuinely-protective ratchet behaviour is unchanged and stays fail-closed: net-new
    integrity findings are still caught by
    ``test_committed_baseline_passes_live_check_on_clean_main`` and the CLI ``--check`` gate
    (evidence-registry-ratchet workflow, ``fetch-depth: 0``), structural tamper by
    ``test_committed_baseline_exists_and_is_valid``, and net-new strict findings by
    ``test_strict_ci_policy_has_zero_active_findings_on_clean_main``. Only the byte-for-byte
    reproduce assertion is downgraded.
    """
    regenerated = ratchet.build_baseline_payload(live_lint_report, ROOT / "docs/context/evidence")
    committed = json.loads(BASELINE.read_text(encoding="utf-8"))
    # Compare the machine-generated fields (generated_at is a timestamp by design).
    # evidence_tree is filesystem-derived (not git-state-derived), so it is stable
    # across checkouts and is part of the machine-reproducible contract (issue #6839):
    # the committed baseline must be exactly what --write-baseline regenerates.
    drifted = [
        key
        for key in (
            "findings_by_path",
            "summary",
            "linter",
            "schema_version",
            "evidence_tree",
        )
        if committed.get(key) != regenerated.get(key)
    ]
    if drifted:
        message = (
            "ADVISORY (non-blocking, maintainer ruling 2026-07-17, issue #5991): the committed "
            f"evidence-registry baseline no longer reproduces from --write-baseline; drifted "
            f"fields: {drifted}. This is the environment-sensitive reproducibility treadmill "
            "(dangling_commit counts depend on checkout git-state); it is intentionally NOT a "
            "hard failure so it can never jam main. Refresh with "
            "`scripts/dev/evidence_registry_ratchet.py --write-baseline` when convenient. "
            "Net-new integrity regressions remain fail-closed via the live ratchet check."
        )
        warnings.warn(message, stacklevel=2)
        pytest.xfail(message)


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
    """Every baseline delta has an explicit machine-checkable disposition (#5952 DoD).

    Newly baselined files are listed under ``reviewed_files``. Per-code increases on files that
    were already in the anchored baseline are listed under ``reviewed_baseline_increases`` so a
    baseline refresh cannot silently grandfather a regression. Future net-new drift is additionally
    caught by the live check above.
    """
    assert REVIEW.is_file(), "evidence_registry_baseline_review.yaml is missing."
    review = yaml.safe_load(REVIEW.read_text(encoding="utf-8"))
    reviewed_entries = review.get("reviewed_files", [])
    increase_entries = review.get("reviewed_baseline_increases", [])
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
    prior_findings = prior_baseline["findings_by_path"]
    current_findings = baseline["findings_by_path"]

    # (2) The committed baseline must decompose as actual prior files + reviewed delta.
    prior_count = int(review["prior_baseline_files_with_findings"])
    assert len(prior_files) == prior_count
    assert baseline_files - prior_files == reviewed, (
        "The committed baseline contains a file not present in the anchored prior baseline "
        "without an explicit disposition in evidence_registry_baseline_review.yaml. Add a "
        "reviewed_files entry naming the new path and its remediate-or-baseline disposition."
    )

    expected_increases = {
        (path, code)
        for path, current_codes in current_findings.items()
        for code, current_count in current_codes.items()
        if path in prior_findings and current_count > prior_findings[path].get(code, 0)
    }
    declared_increases = [
        (entry.get("path"), code) for entry in increase_entries for code in entry.get("codes", [])
    ]
    assert len(declared_increases) == len(set(declared_increases)), (
        "reviewed_baseline_increases contains duplicate path/code rows."
    )
    assert set(declared_increases) == expected_increases, (
        "Every per-code baseline increase must have an explicit reviewed_baseline_increases "
        "entry, and stale entries must be removed."
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

    # (5) Per-code increases must be existing files with a valid explicit disposition.
    for entry in increase_entries:
        path = entry.get("path")
        assert path in prior_files and path in baseline_files, (
            f"reviewed baseline increase {path!r} must refer to an existing baseline file"
        )
        assert entry.get("disposition") in valid_dispositions, (
            f"reviewed baseline increase {path} lacks a valid disposition "
            f"(got {entry.get('disposition')!r})"
        )
        assert isinstance(entry.get("codes"), list) and entry["codes"], (
            f"reviewed baseline increase {path} must list its finding codes"
        )
        for code in entry["codes"]:
            assert current_findings[path].get(code, 0) > prior_findings[path].get(code, 0), (
                f"reviewed baseline increase {path}::{code} is not a current per-code increase"
            )
        assert isinstance(entry.get("reason"), str) and entry["reason"].strip(), (
            f"reviewed baseline increase {path} must explain its disposition"
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


# --- issue #6839: evidence-tree manifest ratchet (fail on the PR, not on main) --------


def test_evidence_tree_manifest_is_path_based_and_stable(tmp_path: Path) -> None:
    """The manifest keys on every sorted evidence-file path, not file contents."""
    root = tmp_path / "evidence"
    (root / "bundle_a").mkdir(parents=True)
    (root / "bundle_a" / "a.json").write_text('{"x": 1}', encoding="utf-8")
    (root / "bundle_b").mkdir(parents=True)
    (root / "bundle_b" / "b.md").write_text("# b\n", encoding="utf-8")
    before = ratchet.evidence_tree_manifest(root)
    assert before["count"] == 2

    # Editing an existing file's contents must NOT change the manifest: that case
    # is owned by the findings ratchet, which already catches net-new findings.
    (root / "bundle_a" / "a.json").write_text('{"x": 2}', encoding="utf-8")
    assert ratchet.evidence_tree_manifest(root) == before

    # Adding or removing a file changes the manifest (so a refresh is required).
    (root / "bundle_c.jsonl").write_text('{"event": "c"}\n', encoding="utf-8")
    after_add = ratchet.evidence_tree_manifest(root)
    assert after_add["count"] == 3
    assert after_add["sha256"] != before["sha256"]

    (root / "bundle_c.jsonl").unlink()
    assert ratchet.evidence_tree_manifest(root) == before


def test_evidence_tree_manifest_uses_unambiguous_path_serialization(tmp_path: Path) -> None:
    """Distinct path sets cannot collide merely because a path contains a newline."""
    two_files = tmp_path / "two_files"
    two_files.mkdir()
    (two_files / "a").write_text("", encoding="utf-8")
    (two_files / "b").write_text("", encoding="utf-8")

    one_file = tmp_path / "one_file"
    one_file.mkdir()
    (one_file / "a\nb").write_text("", encoding="utf-8")

    two_manifest = ratchet.evidence_tree_manifest(two_files)
    one_manifest = ratchet.evidence_tree_manifest(one_file)

    assert two_manifest["count"] == 2
    assert one_manifest["count"] == 1
    assert two_manifest["sha256"] != one_manifest["sha256"]


def test_evidence_tree_manifest_handles_missing_root(tmp_path: Path) -> None:
    """A missing registry root yields an empty, stable manifest."""
    manifest = ratchet.evidence_tree_manifest(tmp_path / "does_not_exist")
    assert manifest["count"] == 0
    assert manifest["sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_check_evidence_tree_manifest_fails_when_tree_grew() -> None:
    """A baseline manifest that no longer matches the live tree fails (issue #6839)."""
    current = {"count": 2, "sha256": "same", "file_suffixes": []}
    baseline = {"evidence_tree": {"count": 1, "sha256": "same", "file_suffixes": []}}
    failures, notices = ratchet.check_evidence_tree_manifest(current, baseline)
    assert len(failures) == 1
    assert "evidence tree changed without a matching baseline refresh" in failures[0]
    assert "2 evidence files now vs 1 in the baseline" in failures[0]
    assert notices == []


def test_check_evidence_tree_manifest_passes_when_manifest_matches() -> None:
    """A live manifest that matches the baseline manifest holds the gate."""
    current = {"count": 5, "sha256": "same", "file_suffixes": []}
    baseline = {"evidence_tree": {"count": 5, "sha256": "same", "file_suffixes": []}}
    failures, notices = ratchet.check_evidence_tree_manifest(current, baseline)
    assert failures == []
    assert notices == []


def test_check_evidence_tree_manifest_advisory_when_baseline_lacks_manifest() -> None:
    """A baseline predating manifest tracking is advisory, not a hard failure."""
    current = {"count": 5, "sha256": "live", "file_suffixes": []}
    baseline: dict[str, object] = {"findings_by_path": {}}
    failures, notices = ratchet.check_evidence_tree_manifest(current, baseline)
    assert failures == []
    assert len(notices) == 1
    assert "evidence_tree manifest" in notices[0]


def test_committed_baseline_evidence_tree_manifest_matches_live_tree() -> None:
    """The committed baseline's evidence_tree manifest matches the live evidence tree.

    Non-slow (no linter scan): enumerates the real evidence files (~1900) and
    compares to the committed baseline manifest. This is the fast pre-merge
    signal for issue #6839 -- it runs in the required fast-feedback gate on PRs,
    so a PR that adds or removes evidence files without refreshing the baseline
    fails here, on the PR, instead of reddening main after merge via the slow
    findings check that only runs post-merge.
    """
    baseline = ratchet.load_baseline(BASELINE)
    expected = baseline.get("evidence_tree")
    assert isinstance(expected, dict) and expected.get("sha256"), (
        "committed baseline is missing the evidence_tree manifest; regenerate with "
        "`scripts/dev/evidence_registry_ratchet.py --write-baseline`."
    )
    live = ratchet.evidence_tree_manifest(ROOT / "docs/context/evidence")
    assert live["sha256"] == expected["sha256"], (
        "committed baseline evidence_tree manifest does not match the live evidence "
        "tree: a PR added or removed files under docs/context/evidence/ without "
        "refreshing scripts/validation/evidence_registry_baseline.json. Regenerate "
        "with `scripts/dev/evidence_registry_ratchet.py --write-baseline`."
    )
    assert live["count"] == expected["count"]


def test_pr_adding_evidence_files_without_baseline_refresh_fails_pre_merge(
    tmp_path: Path,
) -> None:
    """Regression for issue #6733 / #6839: added evidence files, untouched baseline.

    Reproduces the #6733 shape exactly: a PR adds a file under
    docs/context/evidence/ and leaves scripts/validation/evidence_registry_baseline.json
    untouched. The ratchet ``--check`` must exit 1 (fail pre-merge, not after merge
    on clean main), name the regeneration command, and recover after a single
    ``--write-baseline`` refresh. Uses a synthetic ``--report`` so it is fast
    (non-slow) and runs in the required PR gate.
    """
    repo = tmp_path
    evidence = repo / "docs" / "context" / "evidence" / "issue_6733_reexport"
    evidence.mkdir(parents=True)
    (evidence / "campaign.json").write_text('{"campaign_id": "issue_6733"}', encoding="utf-8")
    report = {"summary": {"findings": 0}, "issues": []}
    report_path = repo / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline = repo / "baseline.json"

    write0 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert write0.returncode == 0, write0.stderr
    assert "evidence_tree manifest tracks 1 evidence files" in write0.stdout

    # PR #6733 shape: add a NEW evidence file, leave the baseline untouched.
    (evidence / "run_meta.txt").write_text("campaign_id: issue_6733\n", encoding="utf-8")
    check_drift = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert check_drift.returncode == 1, (
        "ratchet must FAIL pre-merge when evidence files are added without a "
        f"baseline refresh.\nstdout:\n{check_drift.stdout}\nstderr:\n{check_drift.stderr}"
    )
    assert "evidence tree changed without a matching baseline refresh" in check_drift.stderr
    assert "scripts/dev/evidence_registry_ratchet.py --write-baseline" in check_drift.stderr

    # The one-command fix: refresh the baseline, then --check passes again.
    refresh = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert refresh.returncode == 0, refresh.stderr
    assert "evidence_tree manifest tracks 2 evidence files" in refresh.stdout

    check_ok = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert check_ok.returncode == 0, check_ok.stderr
    assert "ratchet passed" in check_ok.stdout


def test_pr_removing_evidence_files_without_baseline_refresh_fails_pre_merge(
    tmp_path: Path,
) -> None:
    """Removing an evidence file without a baseline refresh also fails pre-merge.

    The issue #6839 acceptance criterion covers both adds and removes; this locks
    the remove direction so a stale baseline cannot slip through either way.
    """
    repo = tmp_path
    evidence = repo / "docs" / "context" / "evidence" / "bundle"
    evidence.mkdir(parents=True)
    (evidence / "keep.json").write_text("{}", encoding="utf-8")
    (evidence / "drop.json").write_text("{}", encoding="utf-8")
    report = {"summary": {"findings": 0}, "issues": []}
    report_path = repo / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline = repo / "baseline.json"

    write0 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert write0.returncode == 0, write0.stderr

    # Remove an evidence file, leave the baseline untouched.
    (evidence / "drop.json").unlink()
    check_drift = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    assert check_drift.returncode == 1, (
        "ratchet must FAIL pre-merge when an evidence file is removed without a "
        f"baseline refresh.\nstderr:\n{check_drift.stderr}"
    )
    assert "evidence tree changed without a matching baseline refresh" in check_drift.stderr
