"""Focused fail-closed tests for the 0.0.8 predecessor-metric equivalence gate."""

from __future__ import annotations

import json
import tarfile
from io import BytesIO
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validation.check_release_metric_equivalence import (
    _read_archive,
    _read_archive_manifest,
    _read_candidate,
    _read_candidate_manifest,
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
    candidate["metrics"]["snqi_v2_spec_sha256"] = "v2-sha"
    assert scientific_manifest_differences(baseline, candidate) == []
    candidate["planners"]["config_identities"][0]["key"] = "orca"
    assert scientific_manifest_differences(baseline, candidate) == [
        "planners.config_identities[0].key"
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
    exposed = _force_row(count=1)
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
