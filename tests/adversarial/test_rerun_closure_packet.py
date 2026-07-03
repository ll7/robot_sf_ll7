"""Tests for the issue #3275 consolidated rerun closure packet.

The packet is a consolidation layer over the canonical pair gate
(``classify_failure_archive_rerun_readiness``). These tests lock the
aggregation contract: disposition mapping, consolidated blockers, fail-closed
behavior on missing/malformed real archives, and deterministic next-action
guidance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.adversarial.disjoint_evaluation import archive_sha256
from robot_sf.benchmark.failure_archive_rerun_closure import (
    CLAIM_BOUNDARY,
    DIAGNOSTIC_ONLY_DISPOSITION,
    FAIL_CLOSED_BLOCKED,
    READY_FOR_RERUN,
    SCHEMA_VERSION,
    build_rerun_closure_packet,
)
from scripts.adversarial.produce_rerun_closure_packet import main as closure_cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _entry(
    archive_id: str,
    *,
    family: str,
    seed: int,
    certified: bool = True,
) -> dict:
    """Build a minimal certified failure-archive entry fixture."""

    entry = {
        "archive_id": archive_id,
        "cluster_key": {
            "policy": "goal",
            "scenario_template": family,
            "primary_failure": "collision",
            "termination_reason": "collision",
        },
        "candidate": {"scenario_seed": seed},
        "failure_attribution": {
            "primary_failure": "collision",
            "details": {"termination_reason": "collision"},
        },
    }
    if certified:
        entry["certification_metadata"] = {
            "status": "passed",
            "source": "unit-test scenario_cert.v1 fixture",
        }
    return entry


def _archive(path: Path, entries: list[dict], *, source_manifests: list[str] | None = None) -> Path:
    """Write an adversarial failure-archive fixture and return its path."""

    payload = {
        "schema_version": "adversarial_failure_archive.v1",
        "config": {
            "source_manifests": source_manifests
            if source_manifests is not None
            else [f"search-manifests/{path.stem}.json"],
        },
        "entries": entries,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _archive_hash(path: Path) -> str:
    """Return the deterministic archive SHA-256 for a fixture archive file."""

    return archive_sha256(json.loads(path.read_text(encoding="utf-8")))


def _null_test_prerequisites(source_archive: Path, rerun_archive: Path) -> dict:
    """Return complete null-test prerequisite metadata bound to the archive pair."""

    return {
        "null_tests_reject_null": True,
        "split_policy": "scenario-family-disjoint",
        "source_archive_sha256": _archive_hash(source_archive),
        "rerun_archive_sha256": _archive_hash(rerun_archive),
        "shuffled_outcome_null_test": {"status": "complete", "p_value": 0.01},
        "ranking_permutation_test": {"status": "complete", "p_value": 0.02},
    }


def _disjoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Return a genuinely disjoint, certified source/rerun archive pair."""

    source = _archive(
        tmp_path / "source.json",
        [
            _entry("source_0000", family="family_a", seed=1),
            _entry("source_0001", family="family_b", seed=2),
        ],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [
            _entry("rerun_0000", family="family_c", seed=101),
            _entry("rerun_0001", family="family_d", seed=102),
        ],
    )
    return source, rerun


def test_disjoint_certified_pair_is_ready_for_rerun(tmp_path: Path) -> None:
    """A disjoint, certified, split-able pair is ready for the rerun."""

    source, rerun = _disjoint_pair(tmp_path)

    packet = build_rerun_closure_packet(
        source, rerun, null_test_prerequisites=_null_test_prerequisites(source, rerun)
    )

    assert packet.disposition == READY_FOR_RERUN
    assert packet.ready is True
    assert packet.consolidated_blockers == []
    assert "run_proposal_vs_random_issue_2921.py" in packet.next_empirical_action
    payload = packet.to_payload()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["claim_boundary"] == CLAIM_BOUNDARY
    # The packet embeds the full pair verdict for auditability.
    assert payload["pair_readiness"]["status"] == "ready"


def test_archive_id_overlap_fails_closed_with_leakage_action(tmp_path: Path) -> None:
    """Shared archive IDs block the rerun and select the leakage next-action."""

    source = _archive(
        tmp_path / "source.json",
        [
            _entry("shared_0000", family="family_a", seed=1),
            _entry("source_0001", family="family_b", seed=2),
        ],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [
            _entry("shared_0000", family="family_c", seed=101),
            _entry("rerun_0001", family="family_d", seed=102),
        ],
    )

    packet = build_rerun_closure_packet(
        source, rerun, null_test_prerequisites=_null_test_prerequisites(source, rerun)
    )

    assert packet.disposition == FAIL_CLOSED_BLOCKED
    assert packet.ready is False
    assert any(b.startswith("archive_id_overlap") for b in packet.consolidated_blockers)
    assert "disjoint rerun archive" in packet.next_empirical_action


def test_missing_certification_selects_certification_action(tmp_path: Path) -> None:
    """Uncertified entries block the rerun and select the certification action."""

    source = _archive(
        tmp_path / "source.json",
        [
            _entry("source_0000", family="family_a", seed=1, certified=False),
            _entry("source_0001", family="family_b", seed=2, certified=False),
        ],
    )
    rerun = _archive(
        tmp_path / "rerun.json",
        [
            _entry("rerun_0000", family="family_c", seed=101, certified=False),
            _entry("rerun_0001", family="family_d", seed=102, certified=False),
        ],
    )

    packet = build_rerun_closure_packet(
        source, rerun, null_test_prerequisites=_null_test_prerequisites(source, rerun)
    )

    assert packet.disposition == FAIL_CLOSED_BLOCKED
    assert any("certification" in b for b in packet.consolidated_blockers)
    assert "certify_adversarial_candidate_batch.py" in packet.next_empirical_action


def test_missing_archive_input_fails_closed_not_synthetic(tmp_path: Path) -> None:
    """An absent source archive fails closed rather than fabricating input."""

    _, rerun = _disjoint_pair(tmp_path)
    missing = tmp_path / "does_not_exist.json"

    # No null-test prerequisites are supplied: the missing archive itself must
    # dominate the fail-closed verdict and next-action guidance.
    packet = build_rerun_closure_packet(missing, rerun)

    assert packet.disposition == FAIL_CLOSED_BLOCKED
    # The absent source archive is reported by the pair gate as a blocked side.
    assert any(b.startswith("source_archive_blocked") for b in packet.consolidated_blockers)
    assert "real populated failure archive" in packet.next_empirical_action


def test_diagnostic_only_rerun_output_caps_disposition(tmp_path: Path) -> None:
    """A diagnostic-only rerun output caps an otherwise-ready pair."""

    source, rerun = _disjoint_pair(tmp_path)
    rerun_output = tmp_path / "rerun_report.json"
    rerun_output.write_text(
        json.dumps({"result_classification": "diagnostic_only"}), encoding="utf-8"
    )

    packet = build_rerun_closure_packet(
        source,
        rerun,
        rerun_output=rerun_output,
        null_test_prerequisites=_null_test_prerequisites(source, rerun),
    )

    assert packet.disposition == DIAGNOSTIC_ONLY_DISPOSITION
    assert packet.ready is False
    assert "diagnostic-only" in packet.next_empirical_action


def test_missing_null_test_prerequisites_fail_closed(tmp_path: Path) -> None:
    """Absent null-test prerequisites block an otherwise-disjoint pair."""

    source, rerun = _disjoint_pair(tmp_path)

    packet = build_rerun_closure_packet(source, rerun)

    assert packet.disposition == FAIL_CLOSED_BLOCKED
    assert any("null_test" in b for b in packet.consolidated_blockers)
    assert "null-test prerequisite report" in packet.next_empirical_action


def test_cli_exit_codes_match_disposition(tmp_path: Path) -> None:
    """The CLI maps ready/blocked dispositions to fail-closed exit codes."""

    source, rerun = _disjoint_pair(tmp_path)
    null_path = tmp_path / "nulls.json"
    null_path.write_text(json.dumps(_null_test_prerequisites(source, rerun)), encoding="utf-8")
    ready_output = tmp_path / "packet_ready.json"

    ready_code = closure_cli_main(
        [
            "--source-archive",
            str(source),
            "--rerun-archive",
            str(rerun),
            "--null-test-prerequisites",
            str(null_path),
            "--output",
            str(ready_output),
        ]
    )
    assert ready_code == 0
    written = json.loads(ready_output.read_text(encoding="utf-8"))
    assert written["disposition"] == READY_FOR_RERUN

    # Missing null-test prerequisites must fail closed with exit code 3.
    blocked_code = closure_cli_main(
        ["--source-archive", str(source), "--rerun-archive", str(rerun)]
    )
    assert blocked_code == 3
