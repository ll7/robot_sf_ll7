"""Focused integration and edge-case tests for issue #5351 release analysis pipeline."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from robot_sf.benchmark.hierarchical_paired_release_analysis import (
    CLAIM_GATE_BLOCKED_REVIEW_PENDING,
)
from robot_sf.benchmark.hierarchical_paired_release_inputs import (
    load_hierarchical_paired_release_input_manifest,
)
from scripts.analysis.run_hierarchical_paired_release_analysis_issue_5351 import (
    EXPECTED_PUBLICATION_COMMIT,
    EXPECTED_RELEASE_TAG,
    EXPECTED_TOTAL_EPISODES,
    ReleaseAnalysisPipelineError,
    adapt_record_to_typed_ledger,
    find_or_download_bundle,
    hydrate_and_adapt_release_bundle,
    main,
    sha256_file,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = (
    _REPO_ROOT / "configs/benchmarks/releases/hierarchical_paired_release_analysis_issue_5351.yaml"
)
_EVIDENCE_DIR = _REPO_ROOT / "docs/context/evidence/issue_5351_hierarchical_paired_release_analysis"


def _make_mock_tar(tmp_path: Path, arm_data: dict[str, list[dict]]) -> Path:
    """Helper to create a synthetic tar.gz release bundle for testing."""
    tar_path = tmp_path / "mock_release_bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for arm_dir, records in arm_data.items():
            jsonl_lines = [json.dumps(r) for r in records]
            buf = "\n".join(jsonl_lines).encode("utf-8") + b"\n"
            tarinfo = tarfile.TarInfo(
                name=f"paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_post1_corrected_publication_bundle/payload/runs/{arm_dir}/episodes.jsonl"
            )
            tarinfo.size = len(buf)
            tar.addfile(tarinfo, io.BytesIO(buf))
    return tar_path


def test_manifest_contract_version_and_sha256() -> None:
    """The checked-in manifest must reference release 0.0.3.post1 and valid rows."""
    manifest = load_hierarchical_paired_release_input_manifest(_MANIFEST_PATH)
    successor = manifest["successor_release"]
    assert successor["release_tag"] == EXPECTED_RELEASE_TAG
    assert successor["commit"] == EXPECTED_PUBLICATION_COMMIT

    rows_path = _REPO_ROOT / successor["typed_ledger_rows"]
    assert rows_path.is_file()
    assert rows_path.stat().st_size > 0
    actual_sha256 = sha256_file(rows_path)
    assert actual_sha256 == successor["typed_ledger_rows_sha256"]


def test_pipeline_fails_on_archive_digest_mismatch(tmp_path: Path) -> None:
    """An archive whose SHA-256 does not match EXPECTED_BUNDLE_SHA256 must fail closed."""
    fake_tar = tmp_path / "invalid_sha256.tar.gz"
    fake_tar.write_bytes(b"invalid_tar_bytes_data")

    with pytest.raises(ReleaseAnalysisPipelineError, match="SHA-256 digest mismatch"):
        find_or_download_bundle(fake_tar, repo_root=tmp_path)


def test_pipeline_fails_on_incomplete_arm_count(tmp_path: Path) -> None:
    """An archive missing any of the 14 required arms must fail closed."""
    mock_data = {
        "goal__differential_drive": [
            {
                "scenario_id": "scn_1",
                "seed": 1,
                "steps": 10,
                "outcome": {"route_complete": True},
                "scenario_params": {"metadata": {"archetype": "bottleneck"}},
            }
        ]
    }
    tar_path = _make_mock_tar(tmp_path, mock_data)

    with pytest.raises(ReleaseAnalysisPipelineError, match="Expected 14 arms in archive"):
        hydrate_and_adapt_release_bundle(tar_path)


def test_pipeline_fails_on_duplicate_cell(tmp_path: Path) -> None:
    """Duplicate (scenario_id, seed, planner) cells within an arm must fail closed."""
    all_14_arms = [
        "goal",
        "social_force",
        "orca",
        "ppo",
        "socnav_sampling",
        "sacadrl",
        "scenario_adaptive_hybrid_orca_v1",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "guarded_ppo",
        "predictive_mppi",
        "risk_dwa",
        "prediction_planner",
    ]
    mock_data = {}
    for arm in all_14_arms:
        arm_dir = f"{arm}__differential_drive"
        records = [
            {
                "scenario_id": f"scn_{i // 30}",
                "seed": i % 30,
                "steps": 10,
                "outcome": {"route_complete": True},
            }
            for i in range(1440)
        ]
        if arm == "goal":
            # Duplicate first cell
            records[1] = dict(records[0])
        mock_data[arm_dir] = records

    tar_path = _make_mock_tar(tmp_path, mock_data)

    with pytest.raises(ReleaseAnalysisPipelineError, match="Duplicate cell found in archive"):
        hydrate_and_adapt_release_bundle(tar_path)


def test_adapt_record_to_typed_ledger_preserves_semantics() -> None:
    """Record adaptation extracts exact and surrogate events without altering metrics."""
    record = {
        "scenario_id": "classic_doorway_medium",
        "seed": 202,
        "steps": 150,
        "scenario_params": {"run_dt": 0.1, "metadata": {"archetype": "doorway"}},
        "outcome": {"collision_event": True, "route_complete": False, "timeout_event": False},
        "metrics": {"near_misses": 3, "socnavbench_path_length": 15.5},
    }
    row, archetype = adapt_record_to_typed_ledger(record, planner_name="orca")

    assert row["schema_version"] == "EpisodeEventLedger.v2"
    assert row["scenario_id"] == "classic_doorway_medium"
    assert row["seed"] == 202
    assert row["planner"] == "orca"
    assert row["exact_events"]["collision"] is True
    assert row["exact_events"]["goal_reached"] is False
    assert row["surrogate_events"]["near_miss"] is True
    assert row["provenance"]["completion_time"] == pytest.approx(15.0)
    assert row["provenance"]["exposure"]["distance"] == pytest.approx(15.5)
    assert archetype == "doorway"


def test_seeded_determinism(tmp_path: Path) -> None:
    """Two isolated runs produce identical registered evidence bytes."""
    isolated_root = tmp_path / "repo"
    isolated_manifest = isolated_root / _MANIFEST_PATH.relative_to(_REPO_ROOT)
    isolated_evidence = isolated_root / _EVIDENCE_DIR.relative_to(_REPO_ROOT)
    isolated_manifest.parent.mkdir(parents=True)
    shutil.copyfile(_MANIFEST_PATH, isolated_manifest)

    bundle = find_or_download_bundle(None, repo_root=_REPO_ROOT)
    args = [
        "--repo-root",
        str(isolated_root),
        "--bundle-tar",
        str(bundle),
    ]
    assert main(args) == 0

    artifact_names = (
        "README.md",
        "hierarchical_paired_release_analysis_report.json",
        "successor_rows.jsonl",
        "successor_rows.jsonl.review.json",
    )
    first_digests = {name: sha256_file(isolated_evidence / name) for name in artifact_names}
    first_digests["manifest"] = sha256_file(isolated_manifest)

    report_data = json.loads(
        (isolated_evidence / "hierarchical_paired_release_analysis_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report_data["claim_gate"]["status"] == CLAIM_GATE_BLOCKED_REVIEW_PENDING
    assert report_data["issue"] == 5351
    assert report_data["evidence_status"] == "not_benchmark_evidence"
    readme = (isolated_evidence / "README.md").read_text(encoding="utf-8")
    assert "](successor_rows.jsonl)" in readme

    assert main(args) == 0
    second_digests = {name: sha256_file(isolated_evidence / name) for name in artifact_names}
    second_digests["manifest"] = sha256_file(isolated_manifest)
    assert second_digests == first_digests


def test_clean_checkout_hydration_and_execution_against_release_0_0_3_post1() -> None:
    """Pipeline execution against release 0.0.3.post1 registers all 20,160 rows."""
    report_path = _EVIDENCE_DIR / "hierarchical_paired_release_analysis_report.json"
    readme_path = _EVIDENCE_DIR / "README.md"
    rows_path = _EVIDENCE_DIR / "successor_rows.jsonl"

    assert report_path.is_file()
    assert readme_path.is_file()
    assert rows_path.is_file()

    lines = [line for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == EXPECTED_TOTAL_EPISODES

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "hierarchical_paired_release_analysis_report.v1"
    assert report["issue"] == 5351
    assert report["analysis_executed"] is True
    assert report["claim_gate"]["status"] == CLAIM_GATE_BLOCKED_REVIEW_PENDING

    protocol_conformance = report["protocol_conformance"]
    assert len(protocol_conformance) == 8
    for item in protocol_conformance:
        assert item["status"] == "delivered_analysis_pending_human_review"
