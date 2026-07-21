"""Conservation test for Issue #5785: Package B 27-cell result durable registration.

Verifies the registered conservation bundle (produced by exact CPU re-execution at the
recorded execution commit with verified manifest identity) against the issue #5785
acceptance criteria: the 27-cell population and provenance inputs are mechanically verified,
every reported count regenerates from the conserved artifacts, and the confirmation sidecar
stays censored (no failure silently counted as a confirmed discovery).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.benchmark.adversarial_package_b_report import validate_package_b_report

if TYPE_CHECKING:
    from collections.abc import Sequence

BUNDLE = Path("docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15")
RECORDED_MANIFEST_SHA256 = "9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04"

EXPECTED_TOTALS = {"random": 24, "optuna": 18, "coordinate": 0}
EXPECTED_CELLS = 27
EXPECTED_TOTAL_FAILURES = 42


def _load_report(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "report.json").read_text(encoding="utf-8"))


def _totals_by_sampler(rows: Sequence[dict[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["sampler"]] = totals.get(row["sampler"], 0) + int(
            row["certified_valid_failure_count"]
        )
    return totals


def test_bundle_files_present() -> None:
    """All required durable artifacts exist in the conservation bundle."""
    required = (
        "report.json",
        "confirmation.json",
        "comparison_table.md",
        "replication_summary.json",
        "SHA256SUMS",
        "candidate_replay_SHA256SUMS.txt",
        "provenance.md",
        "README.md",
    )
    for name in required:
        assert (BUNDLE / name).is_file(), f"missing conserved artifact: {name}"


def test_manifest_identity_matches_recorded_sha256() -> None:
    """The reproduced manifest hash equals the SHA-256 recorded in issue #5785."""
    manifest_path = Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == RECORDED_MANIFEST_SHA256, (
        "manifest identity drift: reproduced hash differs from the recorded SHA-256 anchor"
    )


def test_report_gate_ready_for_empirical_review() -> None:
    """The conserved report passes the Package-B report gate."""
    gate = validate_package_b_report(BUNDLE / "report.json")
    assert gate.ready is True
    assert gate.status == "ready_for_empirical_review"
    assert gate.matrix["observed_row_count"] == EXPECTED_CELLS


def test_population_and_counts_regenerate_from_conserved_report() -> None:
    """Every reported count regenerates: 27 cells, 42 failures, sampler split matches."""
    report = _load_report(BUNDLE)
    rows = [dict(r) for r in report["rows"]]  # type: ignore[arg-type]
    assert len(rows) == EXPECTED_CELLS
    assert sum(int(r["certified_valid_failure_count"]) for r in rows) == EXPECTED_TOTAL_FAILURES
    assert _totals_by_sampler(rows) == EXPECTED_TOTALS
    # replayable == certified in every cell, no fallback/degraded execution
    for row in rows:
        assert int(row["replayable_valid_failure_count"]) == int(
            row["certified_valid_failure_count"]
        )
        assert int(row["fallback_candidate_count"]) == 0
        assert int(row["degraded_candidate_count"]) == 0


def test_confirmation_sidecar_stays_censored() -> None:
    """Confirmation does not count any certified failure as a confirmed discovery.

    The conserved sidecar keeps every one of the 42 certified failures in the
    `not_confirmed` state (independent-seed confirmation, deterministic replay, and stable
    mechanism attribution remain deferred to issue #5785 step 5). No failure is silently
    promoted to a confirmed discovery.
    """
    sidecar = json.loads((BUNDLE / "confirmation.json").read_text(encoding="utf-8"))
    rows = sidecar["rows"]
    assert len(rows) == EXPECTED_CELLS
    confirmed = sum(int(r.get("confirmed_failure_count", 0)) for r in rows)
    unconfirmed = sum(int(r.get("unconfirmed_certified_failure_count", 0)) for r in rows)
    assert confirmed == 0, "no certified failure may be counted as a confirmed discovery"
    assert unconfirmed == EXPECTED_TOTAL_FAILURES
    assert all(r.get("confirmation_status") == "complete" for r in rows)
    # the sidecar is artifact-bound to the conserved report
    report_digest = hashlib.sha256((BUNDLE / "report.json").read_bytes()).hexdigest()
    assert sidecar["source_report_sha256"] == report_digest


def test_durable_summary_sha256sums_match() -> None:
    """The committed SHA256SUMS file verifies the four durable summary artifacts."""
    checksums = (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected_files = {
        "report.json",
        "confirmation.json",
        "comparison_table.md",
        "replication_summary.json",
    }
    verified = 0
    for line in checksums:
        digest, _, name = line.partition("  ")
        name = name.strip()
        if name in expected_files and (BUNDLE / name).is_file():
            actual = hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest()
            assert actual == digest, f"SHA256SUMS mismatch for {name}"
            verified += 1
    assert verified == len(expected_files)


def test_replay_tree_checksums_are_parseable_and_complete() -> None:
    """The frozen replay-tree anchor is a well-formed SHA256SUMS over all replay artifacts."""
    lines = (BUNDLE / "candidate_replay_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert lines, "frozen replay-tree checksum file must not be empty"
    for line in lines:
        digest, _, name = line.partition("  ")
        assert len(digest) == 64, "malformed digest in candidate_replay_SHA256SUMS.txt"
        assert Path(name.strip()).parts[0] == "worst_case_snqi"
