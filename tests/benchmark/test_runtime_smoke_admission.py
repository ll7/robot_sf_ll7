"""Tests for exact-source runtime-smoke release admission."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.runtime_smoke_admission import (
    RUNTIME_SMOKE_CONFIG,
    RUNTIME_SMOKE_MANIFEST,
    RUNTIME_SMOKE_RELEASE_ID,
    RuntimeSmokeAdmissionError,
    validate_runtime_smoke_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    planners = tuple(f"planner-{index}" for index in range(14))
    manifest = tmp_path / RUNTIME_SMOKE_MANIFEST
    config = tmp_path / RUNTIME_SMOKE_CONFIG
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("manifest\n", encoding="utf-8")
    config.write_text("config\n", encoding="utf-8")
    root = tmp_path / "output" / "benchmarks" / "camera_ready" / "smoke"
    result_path = root / "release" / "release_result.json"
    finished = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    result = {
        "campaign_id": "smoke",
        "benchmark_release": {
            "release_id": RUNTIME_SMOKE_RELEASE_ID,
            "manifest_path": RUNTIME_SMOKE_MANIFEST.as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "canonical_campaign_config": RUNTIME_SMOKE_CONFIG.as_posix(),
            "canonical_campaign_config_sha256": sha256_file(config),
        },
        "resolved_manifest": {"planners": {"keys": list(planners)}},
        "total_runs": 14,
        "successful_runs": 14,
        "total_episodes": 14,
        "non_success_runs": 0,
        "accepted_unavailable_runs": 0,
        "unexpected_failed_runs": 0,
        "row_status_summary": {
            "successful_evidence_rows": 14,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        },
        "campaign_integrity": {"status": "valid", "checked_arm_count": 14},
        "checkpoint_staging_receipt": {"submit_safe": True},
        "release_benchmark_success": True,
        "release_status": "ok",
        "release_exit_code": 0,
    }
    _write_json(result_path, result)
    _write_json(
        root / "run_meta.json",
        {"repo": {"commit": "a" * 40}, "campaign_id": "smoke", "finished_at_utc": finished},
    )
    _write_json(
        root / "reports" / "campaign_summary.json", {"campaign": {"benchmark_success": True}}
    )
    return result_path, planners


def test_runtime_smoke_admits_exact_source_complete_roster(tmp_path: Path) -> None:
    result, planners = _fixture(tmp_path)

    admitted = validate_runtime_smoke_result(
        result,
        repo_root=tmp_path,
        expected_source_commit="a" * 40,
        expected_planner_keys=planners,
    )

    assert admitted["status"] == "admitted"
    assert admitted["planner_arms"] == 14
    assert admitted["fallback_or_degraded_rows"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(total_runs=13), "total_runs mismatch"),
        (
            lambda payload: payload["row_status_summary"].update(fallback_or_degraded_rows=1),
            "fallback_or_degraded_rows mismatch",
        ),
        (
            lambda payload: payload.update(release_benchmark_success=False),
            "release success mismatch",
        ),
    ],
)
def test_runtime_smoke_rejects_incomplete_or_fallback_result(
    tmp_path: Path, mutation, message: str
) -> None:
    result, planners = _fixture(tmp_path)
    payload = json.loads(result.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(result, payload)

    with pytest.raises(RuntimeSmokeAdmissionError, match=message):
        validate_runtime_smoke_result(
            result,
            repo_root=tmp_path,
            expected_source_commit="a" * 40,
            expected_planner_keys=planners,
        )


def test_runtime_smoke_rejects_different_source_commit(tmp_path: Path) -> None:
    result, planners = _fixture(tmp_path)

    with pytest.raises(RuntimeSmokeAdmissionError, match="source commit mismatch"):
        validate_runtime_smoke_result(
            result,
            repo_root=tmp_path,
            expected_source_commit="b" * 40,
            expected_planner_keys=planners,
        )
