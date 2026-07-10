"""Tests for the ty advisory diagnostic baseline + per-module downward ratchet (#5004).

These tests cover the pure ratchet logic directly (classification, module keying,
the downward-ratchet gate) and the CLI end-to-end via ``--ty-output`` so no live
``ty`` invocation is required in the unit suite. They also assert the acceptance
criteria: a committed baseline exists, the worked-example module was driven to
zero, and the ratchet fails on a net-new finding in a clean module.

Issue #5070: the live ``ty`` per-module counts are host-dependent (they depend
on each host's dependency-resolution state), so baseline *reproduction* is checked
against a committed deterministic fixture
(``scripts/validation/ty_advisory_findings_fixture.json``), NOT a live ty run.
The live scan remains available as a separate opt-in advisory diagnostic.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "dev" / "ty_advisory_ratchet.py"
BASELINE = ROOT / "scripts" / "validation" / "ty_advisory_baseline.json"
# Deterministic, host-independent raw-findings fixture reconstructed from the
# committed baseline. The baseline-reproduction test parses THIS file, never a
# live ty run, so reproduction holds on every clean worktree (issue #5070).
FIXTURE = ROOT / "scripts" / "validation" / "ty_advisory_findings_fixture.json"

# Import the helper as a source module (it lives under scripts/dev, not a package).
_spec = importlib.util.spec_from_file_location("ty_advisory_ratchet", SCRIPT)
assert _spec is not None and _spec.loader is not None
tyratchet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tyratchet)


def _finding(
    path: str,
    line: int = 1,
    *,
    check_name: str = "invalid-argument-type",
    description: str = "invalid-argument-type: Argument is incorrect",
    severity: str = "major",
) -> dict:
    """Build one synthetic ty gitlab-JSON finding for deterministic tests."""
    return {
        "check_name": check_name,
        "description": description,
        "severity": severity,
        "fingerprint": f"{path}:{line}:{check_name}",
        "location": {
            "path": path,
            "positions": {"begin": {"line": line, "column": 1}},
        },
    }


# --------------------------------------------------------------------------- #
# classify_finding
# --------------------------------------------------------------------------- #


def test_classify_general_finding() -> None:
    """A normal type rule is part of the general (ratcheted) bucket."""
    assert (
        tyratchet.classify_finding(
            _finding("robot_sf/nav/x.py", check_name="invalid-argument-type")
        )
        == "general"
    )


def test_classify_optional_import_whole_module_resolution_is_excluded() -> None:
    """'Cannot resolve imported module' is the optional-import category (excluded)."""
    finding = _finding(
        "robot_sf/planner/ompl.py",
        check_name="unresolved-import",
        description="unresolved-import: Cannot resolve imported module `ompl`",
    )
    assert tyratchet.classify_finding(finding) == "optional_import"


def test_classify_member_resolution_is_kept_in_general() -> None:
    """First-party member-resolution errors are real bugs, not optional imports."""
    finding = _finding(
        "robot_sf/benchmark/_preflight.py",
        check_name="unresolved-import",
        description="unresolved-import: Module `robot_sf.benchmark.camera_ready_campaign` has no member `_x`",
    )
    assert tyratchet.classify_finding(finding) == "general"


# --------------------------------------------------------------------------- #
# module_of
# --------------------------------------------------------------------------- #


def test_module_of_robot_sf_subpackage() -> None:
    """robot_sf groups two levels deep (robot_sf/<subpkg>)."""
    assert tyratchet.module_of("robot_sf/data/external/ind.py") == "robot_sf/data"
    assert tyratchet.module_of("robot_sf/benchmark/cli.py") == "robot_sf/benchmark"


def test_module_of_top_level_dir() -> None:
    """Non-robot_sf paths use the single top-level segment."""
    assert tyratchet.module_of("scripts/dev/foo.py") == "scripts"
    assert tyratchet.module_of("examples/plot/x.py") == "examples"


# --------------------------------------------------------------------------- #
# check_against_baseline (the downward ratchet gate)
# --------------------------------------------------------------------------- #


def test_ratchet_passes_when_counts_unchanged() -> None:
    """No per-module change -> pass, no failures."""
    findings = [_finding("robot_sf/nav/a.py"), _finding("robot_sf/nav/b.py")]
    baseline = {"modules": {"robot_sf/nav": {"general": 2, "total": 2}}}
    failures, notices = tyratchet.check_against_baseline(findings, baseline)
    assert failures == []
    assert notices == []


def test_ratchet_fails_on_clean_module_regression() -> None:
    """A clean module gaining any finding fails (net-new in clean territory)."""
    findings = [_finding("robot_sf/data/external/ind.py")]
    baseline = {"modules": {}}  # robot_sf/data is clean / absent
    failures, _notices = tyratchet.check_against_baseline(findings, baseline)
    assert any("clean module regressed" in m and "robot_sf/data" in m for m in failures)


def test_ratchet_fails_on_tracked_module_increase() -> None:
    """A tracked module's count increasing violates the downward ratchet."""
    findings = [
        _finding("robot_sf/nav/a.py"),
        _finding("robot_sf/nav/b.py"),
        _finding("robot_sf/nav/c.py"),
    ]
    baseline = {"modules": {"robot_sf/nav": {"general": 2, "total": 2}}}
    failures, _notices = tyratchet.check_against_baseline(findings, baseline)
    assert any("increased from 2 to 3" in m for m in failures)


def test_ratchet_passes_on_decrease_and_emits_refresh_notice() -> None:
    """A decrease passes and hints at refreshing the baseline (downward motion)."""
    findings = [_finding("robot_sf/nav/a.py")]
    baseline = {"modules": {"robot_sf/nav": {"general": 2, "total": 2}}}
    failures, notices = tyratchet.check_against_baseline(findings, baseline)
    assert failures == []
    assert any("ratchet opportunity" in n and "robot_sf/nav" in n for n in notices)


def test_ratchet_ignores_optional_import_findings_in_gate() -> None:
    """Optional-import findings do not count toward the general-bucket gate."""
    findings = [
        _finding(
            "robot_sf/nav/a.py",
            check_name="unresolved-import",
            description="unresolved-import: Cannot resolve imported module `ompl`",
        )
    ]
    baseline = {"modules": {}}  # robot_sf/nav treated as clean
    failures, _notices = tyratchet.check_against_baseline(findings, baseline)
    # Optional-import finding must NOT trip the clean-module gate.
    assert failures == []


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #


def test_aggregate_splits_optional_and_general_buckets() -> None:
    """Optional-import findings land in the excluded bucket; others in general."""
    findings = [
        _finding("robot_sf/nav/a.py"),
        _finding(
            "robot_sf/nav/b.py",
            check_name="unresolved-import",
            description="unresolved-import: Cannot resolve imported module `rvo2`",
        ),
        _finding("scripts/x.py"),
    ]
    agg = tyratchet.aggregate(findings)
    assert agg["general_total"] == 2
    assert agg["optional_import_total"] == 1
    assert agg["total"] == 3
    nav = agg["modules"]["robot_sf/nav"]
    assert nav == {"general": 1, "optional_import_excluded": 1, "total": 2}


# --------------------------------------------------------------------------- #
# build_baseline_payload
# --------------------------------------------------------------------------- #


def test_baseline_payload_records_exclusion_and_sibling_issues() -> None:
    """The baseline must document the optional-import exclusion and sibling owners."""
    payload = tyratchet.build_baseline_payload(
        [_finding("robot_sf/nav/a.py")], ty_version="ty 0.0.99"
    )
    assert payload["schema_version"] == tyratchet.SCHEMA_VERSION
    assert payload["ty_version"] == "ty 0.0.99"
    assert "#4990" in payload["exclusion"]["owned_by_other_issues"]
    assert "#4995" in payload["exclusion"]["owned_by_other_issues"]
    assert "#4988" in payload["exclusion"]["owned_by_other_issues"]
    assert "Cannot resolve imported module" in payload["exclusion"]["rule"]


# --------------------------------------------------------------------------- #
# CLI end-to-end via --ty-output (no live ty run)
# --------------------------------------------------------------------------- #


def _run_cli(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the ratchet helper in ``repo`` with the given args."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_write_then_check_roundtrip(tmp_path: Path) -> None:
    """A generated baseline from a ty report must reproduce and pass (--check)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    report = repo / "report.json"
    report.write_text(json.dumps([_finding("robot_sf/nav/a.py")]) + "\n", encoding="utf-8")
    baseline_rel = Path("scripts/validation/ty_advisory_baseline.json")

    write_res = _run_cli(
        repo,
        "--write-baseline",
        "--ty-output",
        str(report),
        "--baseline",
        str(baseline_rel),
    )
    assert write_res.returncode == 0, write_res.stderr
    assert (repo / baseline_rel).exists()

    check_res = _run_cli(
        repo, "--check", "--ty-output", str(report), "--baseline", str(baseline_rel)
    )
    assert check_res.returncode == 0, check_res.stderr
    assert "ty advisory ratchet passed" in check_res.stdout


def test_cli_check_fails_on_clean_module_regression(tmp_path: Path) -> None:
    """A clean baseline + a ty report with a new finding fails with exit 1."""
    repo = tmp_path / "repo"
    repo.mkdir()
    empty_report = repo / "empty.json"
    empty_report.write_text("[]\n", encoding="utf-8")
    baseline_rel = Path("b.json")
    _run_cli(
        repo,
        "--write-baseline",
        "--ty-output",
        str(empty_report),
        "--baseline",
        str(baseline_rel),
    )

    dirty_report = repo / "dirty.json"
    dirty_report.write_text(json.dumps([_finding("robot_sf/nav/a.py")]) + "\n", encoding="utf-8")
    res = _run_cli(
        repo, "--check", "--ty-output", str(dirty_report), "--baseline", str(baseline_rel)
    )
    assert res.returncode == 1, res.stdout + res.stderr
    assert "clean module regressed" in res.stderr
    assert "robot_sf/nav" in res.stderr


# --------------------------------------------------------------------------- #
# Acceptance criteria against the committed baseline
# --------------------------------------------------------------------------- #


def test_committed_baseline_exists_and_is_valid() -> None:
    """Acceptance: a checked-in per-module ty baseline exists and parses."""
    assert BASELINE.exists(), "ty advisory baseline is missing from the repo"
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["schema_version"] == tyratchet.SCHEMA_VERSION
    assert data.get("modules"), "baseline must contain a non-empty modules mapping"
    assert "general_findings" in data["summary"]


def test_worked_example_module_robot_sf_data_is_driven_to_zero() -> None:
    """Acceptance: the worked-example module was cleared and removed from baseline."""
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert "robot_sf/data" not in data["modules"], (
        "robot_sf/data should have been driven to zero and removed from the ty "
        "baseline as the #5004 worked example"
    )


def test_committed_baseline_reproduces_from_fixture() -> None:
    """Acceptance (#5070): the committed baseline reproduces deterministically.

    The live ``ty`` per-module counts are host-dependent (they depend on each
    host's dependency-resolution state), which made the previous live-gate
    reproduction test fail on clean worktrees that had nothing to do with the
    branch under test. Reproduction is now checked against the committed
    deterministic fixture reconstructed from the baseline, so it holds on every
    clean worktree regardless of host state. The live scan stays available as a
    separate opt-in advisory diagnostic (see ``test_live_ty_advisory_scan``).
    """
    assert FIXTURE.exists(), (
        "ty advisory findings fixture is missing; regenerate with "
        "`uv run python scripts/dev/ty_advisory_ratchet.py --emit-baseline-fixture`"
    )
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--ty-output",
            str(FIXTURE),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, (
        "ty advisory ratchet did not reproduce from the committed fixture:\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    assert "ty advisory ratchet passed" in res.stdout


def test_committed_baseline_fixture_is_in_sync() -> None:
    """Guard (#5070): the committed fixture must match the committed baseline.

    If the baseline is refreshed without regenerating the fixture (or vice
    versa), the deterministic reproduction test would silently drift. This test
    regenerates the fixture from the baseline in memory and asserts it matches
    the checked-in fixture byte-for-byte, so a stale fixture fails loudly.
    """
    assert BASELINE.exists() and FIXTURE.exists()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    regenerated = tyratchet.materialize_findings_from_baseline(baseline)
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert regenerated == committed, (
        "ty advisory findings fixture is out of sync with the baseline; "
        "regenerate with `uv run python scripts/dev/ty_advisory_ratchet.py "
        "--emit-baseline-fixture`"
    )


def test_materialize_findings_reproduces_baseline_counts() -> None:
    """Unit (#5070): materialized findings aggregate back to the baseline counts."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    findings = tyratchet.materialize_findings_from_baseline(baseline)
    agg = tyratchet.aggregate(findings)
    # Per-module general/optional/total counts must round-trip exactly.
    for mod, counts in baseline["modules"].items():
        assert agg["modules"][mod] == {
            "general": int(counts["general"]),
            "optional_import_excluded": int(counts.get("optional_import_excluded", 0)),
            "total": int(counts["total"]),
        }, f"module {mod} did not round-trip through materialize->aggregate"
    assert agg["general_total"] == baseline["summary"]["general_findings"]
    assert agg["total"] == baseline["summary"]["total_findings"]


def test_live_ty_advisory_scan() -> None:
    """Advisory diagnostic (#5070): the live ``ty`` scan, opt-in and non-gating.

    The live per-module counts are host-dependent, so they are NOT a PR gate
    (see ``test_committed_baseline_reproduces_from_fixture`` for the
    deterministic contract). This test keeps the live scan available as a
    separate advisory diagnostic: it is skipped unless ``TY_ADVISORY_LIVE=1``
    is set, skipped when ``uvx``/``ty`` is unavailable, and when it runs it
    only asserts the helper executed and emitted a ratchet summary line (it
    intentionally does NOT assert pass/fail, so host drift never blocks the
    suite). Run it locally to see the live vs baseline delta for your host.
    """
    import os
    import shutil

    if os.environ.get("TY_ADVISORY_LIVE") != "1":
        pytest.skip("live ty advisory scan is opt-in; set TY_ADVISORY_LIVE=1 to run")
    if shutil.which("uvx") is None:
        pytest.skip("uvx not available; cannot run live ty check")
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Advisory: assert the diagnostic RAN and reported a ratchet summary, but do
    # NOT assert returncode == 0 — host drift may legitimately make the live
    # ratchet fail without indicating a branch regression (#5070).
    assert "ty advisory ratchet" in res.stdout, (
        f"live ty advisory scan produced no ratchet summary:\n{res.stdout}\n{res.stderr}"
    )


# --------------------------------------------------------------------------- #
# CI wiring (mirror the broad-exception ratchet wiring test)
# --------------------------------------------------------------------------- #


def test_ratchet_helper_is_registered_in_repo() -> None:
    """The ratchet helper exists at the documented path."""
    assert SCRIPT.exists(), f"ty ratchet helper missing at {SCRIPT}"


def test_aggregate_tolerates_null_location() -> None:
    """A finding with an explicit null location must not crash aggregation (gate #5052 fix)."""
    finding = {"check_name": "invalid-argument-type", "location": None}
    agg = tyratchet.aggregate([finding])
    # The finding is bucketed under the '<unknown>' path module, not an AttributeError.
    assert agg["modules"]["<unknown>"]["total"] == 1


def test_load_baseline_rejects_non_object_json(tmp_path: Path) -> None:
    """A baseline whose top-level JSON is not an object must fail closed (gate #5052 fix)."""
    bad = tmp_path / "baseline.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        tyratchet.load_baseline(bad)


def test_load_baseline_rejects_non_dict_modules(tmp_path: Path) -> None:
    """A baseline whose 'modules' is not a mapping must fail closed (gate #5052 fix)."""
    bad = tmp_path / "baseline.json"
    bad.write_text(
        json.dumps({"schema_version": tyratchet.SCHEMA_VERSION, "modules": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="valid 'modules' mapping"):
        tyratchet.load_baseline(bad)
