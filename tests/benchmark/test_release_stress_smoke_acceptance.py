"""Tests for diagnostic stress-smoke acceptance and source-provenance validators."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from robot_sf.benchmark import release_acceptance
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.release_acceptance import (
    validate_diagnostic_stress_smoke_acceptance,
    validate_diagnostic_stress_smoke_source_provenance,
)
from robot_sf.benchmark.release_protocol import (
    STRESS_SMOKE_EXPECTED_KINEMATICS,
    STRESS_SMOKE_EXPECTED_PLANNER_ARMS,
    STRESS_SMOKE_EXPECTED_SCENARIO_IDS,
    STRESS_SMOKE_EXPECTED_SEED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STRESS_CONFIG = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml"
)
_SOURCE_SHA = "a" * 40
_PLANNER_KEYS = tuple(f"planner_{index:02d}" for index in range(STRESS_SMOKE_EXPECTED_PLANNER_ARMS))


def _stress_manifest() -> SimpleNamespace:
    """Return a fixed stress-smoke release manifest with the repo's pinned hashes."""
    return SimpleNamespace(
        release_kind="benchmark-stress-smoke",
        schema_version="hybrid-release-stress-smoke.v1",
        planner_keys=_PLANNER_KEYS,
        expected_kinematics_matrix=(STRESS_SMOKE_EXPECTED_KINEMATICS,),
        scenario_matrix_sha256="5eba52486609b45960469ff13f4e2c2530c54aa7c93bf4690f4e3e6706e63f9b",
        campaign_config_sha256="b046b4298be25c9144c91db218749d3a08310aeabc54d8e0fd7f0f46cd52ae49",
        canonical_campaign_config_path=str(STRESS_CONFIG),
        seed_sets_sha256="",
        route_certification_sha256="",
        snqi_weights_sha256="",
        snqi_baseline_sha256="",
    )


def _write_stress_campaign(tmp_path: Path) -> Path:
    """Write a minimal campaign summary and episode rows for the fixed contract."""
    campaign_root = tmp_path / "campaign"
    reports = campaign_root / "reports"
    reports.mkdir(parents=True)
    runs = []
    for planner_key in _PLANNER_KEYS:
        arm_dir = campaign_root / "runs" / f"{planner_key}__{STRESS_SMOKE_EXPECTED_KINEMATICS}"
        arm_dir.mkdir(parents=True)
        rows = []
        for scenario_id in STRESS_SMOKE_EXPECTED_SCENARIO_IDS:
            rows.append(
                {
                    "episode_id": f"{planner_key}-{scenario_id}-{STRESS_SMOKE_EXPECTED_SEED}",
                    "scenario_id": scenario_id,
                    "seed": STRESS_SMOKE_EXPECTED_SEED,
                    "source_commit": _SOURCE_SHA,
                    "event_ledger": {"software_commit": _SOURCE_SHA},
                    "success": True,
                    "status": "benchmark_success",
                }
            )
        (arm_dir / "episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        runs.append(
            {
                "episodes_path": f"runs/{planner_key}__{STRESS_SMOKE_EXPECTED_KINEMATICS}/episodes.jsonl",
                "planner": {
                    "key": planner_key,
                    "kinematics": STRESS_SMOKE_EXPECTED_KINEMATICS,
                },
            }
        )
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "benchmark_success": True,
                    "status": "benchmark_success",
                    "evidence_status": "valid",
                    "campaign_execution_status": "completed",
                    "git_hash": _SOURCE_SHA,
                },
                "git": {"commit": _SOURCE_SHA},
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )
    (campaign_root / "manifest.json").write_text(
        json.dumps({"git_hash": _SOURCE_SHA}), encoding="utf-8"
    )
    (campaign_root / "run_meta.json").write_text(
        json.dumps({"repo": {"commit": _SOURCE_SHA}}), encoding="utf-8"
    )
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps({"git": {"commit": _SOURCE_SHA}}), encoding="utf-8"
    )
    return campaign_root


# --- source provenance ------------------------------------------------------


def test_source_provenance_accepts_exact_commit(tmp_path: Path) -> None:
    """All campaign metadata and rows naming the exact source SHA pass."""
    campaign_root = _write_stress_campaign(tmp_path)
    report = validate_diagnostic_stress_smoke_source_provenance(
        campaign_root, expected_source_commit=_SOURCE_SHA
    )
    assert report["status"] == "valid", report
    assert not report["blockers"]


def test_source_provenance_rejects_short_expected_commit(tmp_path: Path) -> None:
    """A non-40-hex expected commit is a blocker."""
    campaign_root = _write_stress_campaign(tmp_path)
    report = validate_diagnostic_stress_smoke_source_provenance(
        campaign_root, expected_source_commit="abc123"
    )
    assert report["status"] == "invalid"
    assert any("exact 40-character SHA" in b for b in report["blockers"])


def test_source_provenance_rejects_missing_row_commit(tmp_path: Path) -> None:
    """A row without a source commit is a blocker."""
    campaign_root = _write_stress_campaign(tmp_path)
    rows_path = (
        campaign_root
        / "runs"
        / f"{_PLANNER_KEYS[0]}__{STRESS_SMOKE_EXPECTED_KINEMATICS}"
        / "episodes.jsonl"
    )
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    rows[0].pop("source_commit")
    rows[0].pop("event_ledger")
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = validate_diagnostic_stress_smoke_source_provenance(
        campaign_root, expected_source_commit=_SOURCE_SHA
    )
    assert report["status"] == "invalid"
    assert any("source_commit" in b or "missing" in b for b in report["blockers"])


# --- stress smoke acceptance ------------------------------------------------


def test_stress_smoke_acceptance_not_applicable_for_other_kind(tmp_path: Path) -> None:
    """A non-stress-smoke manifest is not applicable, never a release grant."""
    manifest = SimpleNamespace(release_kind="benchmark-data-release")
    report = validate_diagnostic_stress_smoke_acceptance(
        tmp_path, manifest=manifest, campaign_config=None, expected_source_commit=_SOURCE_SHA
    )
    assert report["status"] == "not_applicable"
    assert report["diagnostic_success"] is False


def test_stress_smoke_acceptance_invalid_without_summary(tmp_path: Path) -> None:
    """A missing campaign summary fails closed as invalid."""
    campaign_config = load_campaign_config(STRESS_CONFIG)
    report = validate_diagnostic_stress_smoke_acceptance(
        tmp_path,
        manifest=_stress_manifest(),
        campaign_config=campaign_config,
        expected_source_commit=_SOURCE_SHA,
    )
    assert report["status"] == "invalid"
    assert report["diagnostic_success"] is False
    assert report["blockers"]


def test_stress_helpers_cover_leaf_contracts() -> None:
    """Leaf helpers behave deterministically on representative inputs."""
    assert release_acceptance._explicit_success(True) is True
    assert release_acceptance._explicit_success("yes") is False
    assert release_acceptance._status_is("benchmark_success", {"benchmark_success", "ok"}) is True
    assert release_acceptance._status_is("failed", {"benchmark_success", "ok"}) is False
    assert release_acceptance._path_has_symlink_component(Path("/a/b/c")) is False
    assert release_acceptance._nested_value({"a": {"b": 1}}, "a", "b") == 1
    assert release_acceptance._nested_value({"a": {}}, "a", "b") is release_acceptance._MISSING
