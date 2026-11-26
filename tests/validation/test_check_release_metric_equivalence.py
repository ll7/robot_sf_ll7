"""Focused fail-closed tests for the 0.0.8 predecessor-metric equivalence gate."""

from __future__ import annotations

import hashlib
import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.analysis.compare_release_007_008 import (
    ATTRIBUTION_SCHEMA,
    RECEIPT_SCHEMA,
    _finding,
    compare_with_attribution,
    load_attribution_ledger,
)
from scripts.validation.check_release_metric_equivalence import (
    _classify_status,
    _read_archive,
    _read_archive_manifest,
    _read_candidate,
    _read_candidate_manifest,
    _read_scientific_candidate_manifest,
    compare,
    scan_robot_force_metrics,
    scientific_manifest_differences,
)

OLD_SHA = "a" * 40
NEW_SHA = "b" * 40
KEY = ("goal", "doorway", 111)


def _manifest(source: str) -> dict[str, object]:
    return {
        "provenance": {"source_sha": source},
        "matrix": {"expected_episode_cells": 1, "horizon_steps": 600},
        "scenario": {"matrix_sha256": "scenario-sha"},
        "seed_policy": {"resolved_seeds": [111]},
        "planners": {"config_identities": [{"key": "goal", "sha256": None}]},
        "kinematics": {"matrix": ["differential_drive"]},
        "metrics": {"snqi_weights_sha256": "v1-sha"},
    }


def _row(source: str, *, metric: float = 0.25) -> dict[str, object]:
    return {
        "scenario_id": "doorway",
        "seed": 111,
        "git_hash": source,
        "result_provenance": {"repo_commit": source},
        "metrics": {"snqi": metric, "nested": {"values": [1.0, 2.0]}},
        "metric_values": {"snqi": metric},
        "outcome": {"route_complete": True, "collision_event": False},
        "status": "success",
        "steps": 25,
    }


def _write_archive(path: Path, rows: list[dict[str, object]]) -> None:
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    info = tarfile.TarInfo("bundle/payload/runs/goal__differential_drive/episodes.jsonl")
    info.size = len(data)
    manifest_data = json.dumps(_manifest(OLD_SHA)).encode()
    manifest_info = tarfile.TarInfo("bundle/payload/release/release_manifest.resolved.json")
    manifest_info.size = len(manifest_data)
    with tarfile.open(path, "w:gz") as handle:
        handle.addfile(info, BytesIO(data))
        handle.addfile(manifest_info, BytesIO(manifest_data))


def _write_candidate(root: Path, rows: list[dict[str, object]]) -> None:
    path = root / "runs/goal__differential_drive/episodes.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = root / "release/release_manifest.resolved.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(_manifest(NEW_SHA)), encoding="utf-8")


def test_added_metric_preserves_every_old_field(tmp_path: Path) -> None:
    archive = tmp_path / "baseline.tar.gz"
    _write_archive(archive, [_row(OLD_SHA)])
    successor = _row(NEW_SHA, metric=0.25 + 5e-13)
    successor["metrics"]["robot_force_impulse_total"] = 1.0
    _write_candidate(tmp_path / "candidate", [successor])

    report = compare(
        _read_archive(archive, OLD_SHA), _read_candidate(tmp_path / "candidate", NEW_SHA)
    )

    assert report["status"] == "pass"
    assert report["paired_rows"] == 1


def test_changed_or_missing_old_metric_is_a_release_stop() -> None:
    old = {KEY: {"metrics": {"snqi": 0.25, "nested": {"value": 2.0}}}}
    new = {KEY: {"metrics": {"snqi": 0.26, "robot_force_peak": 1.0}}}

    report = compare(old, new)

    assert report["status"] == "mismatch"
    assert report["mismatches"][0]["paths"] == [
        "episode.metrics.nested",
        "episode.metrics.snqi",
    ]


def test_missing_or_extra_episode_identity_is_reported() -> None:
    report = compare({KEY: {"metrics": {}}}, {("orca", "doorway", 111): {"metrics": {}}})
    assert report["mismatch_episodes"] == 2
    assert {tuple(item["paths"]) for item in report["mismatches"]} == {
        ("missing_identity",),
        ("unexpected_identity",),
    }


@pytest.mark.parametrize(
    "candidate_key",
    [
        ("goal", "other_scenario", 111),
        ("goal", "doorway", 112),
    ],
)
def test_diagnostic_status_rejects_equal_count_identity_substitution(candidate_key) -> None:
    report = compare({KEY: {"metrics": {}}}, {candidate_key: {"metrics": {}}})
    assert report["baseline_rows"] == report["candidate_rows"] == 1
    assert (
        _classify_status(
            report,
            expected_rows=1,
            manifest_differences=[],
        )
        == "identity_count_mismatch"
    )


def test_diagnostic_status_keeps_force_failure_ahead_of_manifest_mismatch() -> None:
    report = {
        "status": "mismatch",
        "baseline_rows": 1,
        "candidate_rows": 1,
        "mismatches": [],
    }
    status = _classify_status(
        report,
        expected_rows=1,
        manifest_differences=["planners.config_identities[0].sha256"],
        force_report={"status": "invalid_robot_force_metrics"},
    )
    assert status == "invalid_robot_force_metrics"


@pytest.mark.parametrize(
    "manifest_path",
    [
        "matrix.expected_episode_cells",
        "matrix.horizon_steps",
        "seed_policy.resolved_seeds[0]",
        "kinematics.matrix[0]",
        "metrics.snqi_weights_sha256",
    ],
)
def test_diagnostic_status_rejects_structural_manifest_changes(manifest_path: str) -> None:
    report = {"status": "mismatch", "baseline_rows": 1, "candidate_rows": 1, "mismatches": []}
    assert (
        _classify_status(
            report,
            expected_rows=1,
            manifest_differences=[manifest_path],
        )
        == "scientific_manifest_mismatch"
    )


@pytest.mark.parametrize(
    "manifest_path",
    [
        "planners.config_identities[0].sha256",
        "scenario.matrix_sha256",
    ],
)
def test_diagnostic_status_allows_source_bound_scenario_or_planner_change_for_stage3(
    manifest_path: str,
) -> None:
    report = {"status": "mismatch", "baseline_rows": 1, "candidate_rows": 1, "mismatches": []}
    assert (
        _classify_status(
            report,
            expected_rows=1,
            manifest_differences=[manifest_path],
        )
        == "mismatch"
    )


def test_legacy_nan_sentinel_must_remain_nan() -> None:
    old = {KEY: {"metrics": {"min_separation_corrupted_m": float("nan")}}}
    same = {KEY: {"metrics": {"min_separation_corrupted_m": float("nan")}}}
    changed = {KEY: {"metrics": {"min_separation_corrupted_m": 0.0}}}

    assert compare(old, same)["status"] == "pass"
    assert compare(old, changed)["mismatches"][0]["paths"] == [
        "episode.metrics.min_separation_corrupted_m"
    ]


def test_archive_source_or_duplicate_identity_fails_closed(tmp_path: Path) -> None:
    archive = tmp_path / "baseline.tar.gz"
    _write_archive(archive, [_row(OLD_SHA), _row(OLD_SHA)])
    with pytest.raises(ValueError, match="duplicate archive identity"):
        _read_archive(archive, OLD_SHA)
    with pytest.raises(ValueError, match="archive source mismatch"):
        _read_archive(archive, NEW_SHA)


def test_frozen_manifest_contract_allows_new_v2_assets_but_rejects_roster_change(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "baseline.tar.gz"
    _write_archive(archive, [_row(OLD_SHA)])
    candidate_root = tmp_path / "candidate"
    _write_candidate(candidate_root, [_row(NEW_SHA)])
    baseline = _read_archive_manifest(archive, OLD_SHA)
    candidate = _read_candidate_manifest(candidate_root, NEW_SHA)
    for role in ("weights", "anchors", "family"):
        candidate["metrics"][f"snqi_v2_{role}_path"] = f"configs/{role}.json"
        candidate["metrics"][f"snqi_v2_{role}_sha256"] = "b" * 64
    assert scientific_manifest_differences(baseline, candidate) == []
    candidate["planners"]["config_identities"][0]["key"] = "orca"
    assert scientific_manifest_differences(baseline, candidate) == [
        "planners.config_identities[0].key"
    ]


def test_scientific_candidate_rejects_publication_and_changed_rows(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    _write_candidate(root, [_row(NEW_SHA)])
    raw_path = root / "runs/goal__differential_drive/episodes.jsonl"
    identity = {
        "schema_version": "benchmark-scientific-candidate.v1",
        "source_sha": NEW_SHA,
        "baseline_archive_sha256": "a" * 64,
        "raw_episode_sha256": {
            raw_path.relative_to(root).as_posix(): hashlib.sha256(raw_path.read_bytes()).hexdigest()
        },
        "scientific_manifest": _manifest(NEW_SHA),
    }
    candidate_path = root / "release/scientific_candidate.json"
    candidate_path.write_text(json.dumps(identity), encoding="utf-8")
    assert _read_scientific_candidate_manifest(root, NEW_SHA, "a" * 64) == _manifest(NEW_SHA)
    with pytest.raises(ValueError, match="predecessor archive checksum mismatch"):
        _read_scientific_candidate_manifest(root, NEW_SHA, "b" * 64)

    identity["scientific_manifest"]["provenance"]["doi"] = "{{version_doi}}"
    candidate_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="publication fields"):
        _read_scientific_candidate_manifest(root, NEW_SHA)

    del identity["scientific_manifest"]["provenance"]["doi"]
    candidate_path.write_text(json.dumps(identity), encoding="utf-8")
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw episode checksums mismatch"):
        _read_scientific_candidate_manifest(root, NEW_SHA)


def test_successor_manifest_rejects_unrelated_additions() -> None:
    """Only v2 asset declarations may extend the predecessor's scientific manifest."""
    baseline = _manifest(OLD_SHA)
    candidate = _manifest(NEW_SHA)
    candidate["matrix"]["extra_policy"] = "unreviewed"
    candidate["planners"]["config_identities"][0]["fallback_allowed"] = True
    candidate["metrics"]["snqi_v2_spec_sha256"] = "invented"

    assert scientific_manifest_differences(baseline, candidate) == [
        "matrix.extra_policy",
        "metrics.snqi_v2_spec_sha256",
        "planners.config_identities[0].fallback_allowed",
    ]


def test_resolved_manifest_source_must_match_episode_source(tmp_path: Path) -> None:
    archive = tmp_path / "baseline.tar.gz"
    _write_archive(archive, [_row(OLD_SHA)])
    with pytest.raises(ValueError, match="manifest source mismatch"):
        _read_archive_manifest(archive, NEW_SHA)


def _force_row(*, count: int) -> dict[str, object]:
    row = _row(NEW_SHA)
    row["metrics"].update(
        {
            "robot_force_impulse_total": float(count),
            "robot_force_peak": float(count),
            "robot_force_time_above_ref_s": 0.0,
            "robot_force_exposed_ped_count": count,
            "robot_force_impulse_per_exposed_ped": 1.0 if count else None,
            "robot_force_mean_active": 1.0 if count else None,
        }
    )
    return row


def test_force_gate_accepts_zero_exposure_nulls_and_absent_pp_variant(tmp_path: Path) -> None:
    zero = _force_row(count=0)
    zero["metrics"]["robot_force_exposed_ped_count"] = 0.0
    exposed = _force_row(count=1)
    exposed["metrics"]["robot_force_exposed_ped_count"] = 1.0
    exposed["seed"] = 112
    _write_candidate(tmp_path, [zero, exposed])

    report = scan_robot_force_metrics(tmp_path, NEW_SHA)

    assert report["status"] == "pass"
    assert report["checked_rows"] == 2
    assert report["zero_exposure_rows"] == 1
    assert report["pp_equivalent_absent_rows"] == 2


def test_force_gate_rejects_nonfinite_and_partial_variant(tmp_path: Path) -> None:
    row = _force_row(count=0)
    row["metrics"]["robot_force_peak"] = float("nan")
    row["metrics"]["robot_force_mean_active"] = 0.0
    row["metrics"]["robot_force_pp_equiv_peak"] = 1.0
    _write_candidate(tmp_path, [row])

    report = scan_robot_force_metrics(tmp_path, NEW_SHA)

    assert report["status"] == "invalid_robot_force_metrics"
    assert report["failed_rows"] == 1
    assert report["examples"][0]["fields"] == [
        "robot_force_mean_active",
        "robot_force_peak",
        "robot_force_pp_equiv_incomplete",
    ]


def test_force_gate_rejects_fractional_exposed_count(tmp_path: Path) -> None:
    row = _force_row(count=1)
    row["metrics"]["robot_force_exposed_ped_count"] = 1.5
    _write_candidate(tmp_path, [row])

    report = scan_robot_force_metrics(tmp_path, NEW_SHA)

    assert report["status"] == "invalid_robot_force_metrics"
    assert report["examples"][0]["fields"] == ["robot_force_exposed_ped_count"]


def test_changed_row_requires_intact_per_finding_causal_receipt(tmp_path: Path) -> None:
    """Changed fields pass Stage 3 only with an intact accepted receipt."""
    old = {
        KEY: {
            "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
            "metrics": {"snqi": 1.0},
            "metric_values": {},
            "status": "ok",
            "steps": 10,
            "algorithm_metadata": {
                "algorithm": "goal",
                "planner_kinematics": {"execution_mode": "native"},
            },
            "integrity": {"contradictions": []},
        }
    }
    new = {
        KEY: {
            "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
            "metrics": {"snqi": 2.0},
            "metric_values": {},
            "status": "ok",
            "steps": 10,
            "algorithm_metadata": {
                "algorithm": "goal",
                "planner_kinematics": {"execution_mode": "native"},
            },
            "integrity": {"contradictions": []},
        }
    }
    finding = _finding("goal", "doorway", 111, "metrics.snqi", 1.0, 2.0, False)
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"reviewed": true}\n', encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": RECEIPT_SCHEMA,
                "change_id": "planner-correction",
                "finding_ids": [finding["finding_id"]],
                "mechanism": "synthetic planner correction",
                "evidence_path": evidence.name,
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "review": {
                    "decision": "accepted",
                    "reviewer": "synthetic-reviewer",
                    "reviewed_at_utc": "2026-09-29T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": ATTRIBUTION_SCHEMA,
                "candidate_source_sha": "a" * 40,
                "entries": [
                    {
                        "finding_id": finding["finding_id"],
                        "change_id": "planner-correction",
                        "receipt_path": receipt.name,
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    identity = {"source_sha": "a" * 40, "versioned_changes": [{"id": "planner-correction"}]}
    ledger, anomalies, _ = load_attribution_ledger(ledger_path, identity)
    summary = compare_with_attribution(
        old,
        new,
        [{"old_key": "goal", "new_key": "goal", "implementation_replaced": False}],
        ledger,
        scenarios=("doorway",),
        seeds=(111,),
        emit=lambda _: None,
    )
    assert anomalies == []
    assert summary["comparison_passed"] is True

    evidence.write_text('{"reviewed": false}\n', encoding="utf-8")
    _, tampered, _ = load_attribution_ledger(ledger_path, identity)
    assert [item["kind"] for item in tampered] == ["invalid_causal_receipt"]

    empty_summary = compare_with_attribution(
        old,
        new,
        [{"old_key": "goal", "new_key": "goal", "implementation_replaced": False}],
        {},
        scenarios=("doorway",),
        seeds=(111,),
        emit=lambda _: None,
    )
    assert empty_summary["unexplained_findings"]
    assert empty_summary["comparison_passed"] is False
