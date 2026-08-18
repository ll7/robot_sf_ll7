"""Contract tests for the existing-data trace dossier exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import robot_sf.benchmark.candidate_trace_resolution as resolution_module
from robot_sf.benchmark.candidate_trace_resolution import (
    CampaignResultStore,
    resolve_episode_source,
)
from scripts.tools.campaign_result_store import write_result_store
from scripts.tools.export_trace_dossier import TraceDossierExportError, export_trace_dossier

_ROOT = Path(__file__).resolve().parents[2]
_RELEASE = _ROOT / "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
_REAL_RELEASE = _ROOT / "configs/benchmarks/releases/issue_7086_trace_dossier_diagnostic_v0_1.yaml"
_TRACE_FIXTURE = (
    _ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)
_REAL_TRACE_SERIES = (
    _ROOT
    / "docs/context/evidence/issue_4848_group_crossing_exemplars_2026-07/goal/"
    / "classic_group_crossing_medium_seed22_median/trace_series.json"
)


def _write_typed_source(tmp_path: Path) -> Path:
    """Write a typed trace fixture with the smoke-release tuple identity."""
    payload = json.loads(_TRACE_FIXTURE.read_text(encoding="utf-8"))
    payload["source"].update(
        {
            "scenario_id": "francis2023_blind_corner",
            "seed": 111,
            "planner_id": "goal",
            "episode_id": "fixture_episode_001",
        }
    )
    path = tmp_path / "fixture_episode_001.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_store(tmp_path: Path, source: Path) -> Path:
    """Write the minimum valid campaign result store for one source artifact."""
    store = tmp_path / "campaign-store"
    write_result_store(
        store,
        [
            {
                "run_id": "run-fixture",
                "episode_id": "fixture_episode_001",
                "planner": "goal",
                "scenario_id": "francis2023_blind_corner",
                "scenario_family": "classic",
                "seed": 111,
                "row_status": "native",
                "artifact_uri": str(source),
                "artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        study_id="trace-dossier-fixture",
        command="fixture",
        source_commit="fixture-commit",
    )
    return store


def _write_jsonl_source(tmp_path: Path) -> Path:
    """Write a minimal existing JSONL recording for the conversion path."""
    records = [
        {
            "event": "step",
            "step_idx": 0,
            "timestamp": 0.0,
            "episode_id": "fixture_episode_001",
            "scenario_id": "francis2023_blind_corner",
            "seed": 111,
            "scenario_params": {"algo": "goal"},
            "state": {
                "robot_pose": [[0.0, 0.0], 0.0],
                "pedestrian_positions": [],
            },
        },
        {
            "event": "step",
            "step_idx": 1,
            "timestamp": 0.1,
            "episode_id": "fixture_episode_001",
            "scenario_id": "francis2023_blind_corner",
            "seed": 111,
            "scenario_params": {"algo": "goal"},
            "state": {
                "robot_pose": [[0.1, 0.0], 0.0],
                "pedestrian_positions": [],
            },
        },
    ]
    path = tmp_path / "fixture_episode_001.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _write_real_trace_store(tmp_path: Path) -> Path:
    """Bind the retained #4848 trace-series exemplar to its campaign identity."""
    store = tmp_path / "real-campaign-store"
    write_result_store(
        store,
        [
            {
                "run_id": "run-13334",
                "episode_id": "classic_group_crossing_medium--22--605d6793ad25c1f5",
                "planner": "goal",
                "scenario_id": "classic_group_crossing_medium",
                "scenario_family": "group_crossing",
                "seed": 22,
                "row_status": "native",
                "artifact_uri": str(_REAL_TRACE_SERIES),
                "artifact_sha256": hashlib.sha256(_REAL_TRACE_SERIES.read_bytes()).hexdigest(),
            }
        ],
        study_id="issue4206_trace_capable_h600_rerun_20260704",
        command="retained #4848 exemplar fixture",
        source_commit="0b0214ced856eac77fa9a4c15b02921eabab1661",
    )
    return store


def test_export_trace_dossier_writes_trace_manifest_and_checksums(tmp_path: Path) -> None:
    """A release-pinned existing trace becomes a deterministic dossier bundle."""
    source = _write_typed_source(tmp_path)
    store = _write_store(tmp_path, source)
    output = tmp_path / "dossier"

    manifest = export_trace_dossier(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=output,
    )

    assert manifest["schema_version"] == "trace_dossier_export_manifest.v1"
    assert manifest["release"]["release_tag"] == "paper-benchmark-smoke-v0.1.0"
    assert manifest["identity"]["planner_id"] == "goal"
    assert manifest["artifacts"]["trace"]["schema_version"] == "simulation_trace_export.v1"
    exported = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    assert exported["source"]["scenario_id"] == "francis2023_blind_corner"
    assert exported["source"]["planner_id"] == "goal"
    checksums = (output / "SHA256SUMS").read_text(encoding="utf-8")
    assert "trace.json" in checksums
    assert "manifest.json" in checksums
    assert "normalization_receipt.json" in checksums


def test_export_trace_dossier_converts_existing_jsonl_recording(tmp_path: Path) -> None:
    """The production conversion path emits a schema-valid per-step trace."""
    source = _write_jsonl_source(tmp_path)
    store = _write_store(tmp_path, source)

    export_trace_dossier(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=tmp_path / "dossier",
    )

    exported = json.loads((tmp_path / "dossier" / "trace.json").read_text(encoding="utf-8"))
    assert exported["schema_version"] == "simulation_trace_export.v1"
    assert len(exported["frames"]) == 2
    assert exported["source"]["episode_id"] == "fixture_episode_001"


def test_export_trace_dossier_converts_retained_real_trace_series(tmp_path: Path) -> None:
    """A retained #4848 trace-series exemplar becomes a provenance-bound typed trace."""
    store = _write_real_trace_store(tmp_path)
    output = tmp_path / "real-dossier"

    manifest = export_trace_dossier(
        scenario_id="classic_group_crossing_medium",
        planner_id="goal",
        seed=22,
        release_manifest_path=_REAL_RELEASE,
        campaign_store_dir=store,
        output_dir=output,
    )

    exported = json.loads((output / "trace.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "normalization_receipt.json").read_text(encoding="utf-8"))
    assert manifest["release"]["release_id"] == "issue_7086_trace_dossier_diagnostic_v0_1"
    assert exported["source"] == {
        "episode_id": "classic_group_crossing_medium--22--605d6793ad25c1f5",
        "generated_by": ("scripts/tools/export_trace_dossier.py; source=trace_series.json"),
        "planner_id": "goal",
        "scenario_id": "classic_group_crossing_medium",
        "seed": 22,
    }
    assert len(exported["frames"]) == 167
    assert exported["frames"][0]["pedestrians"][0]["id"] == "0"
    assert exported["frames"][0]["planner"]["rl"]["reward"] == pytest.approx(0.0008005650800357953)
    assert receipt["provenance"]["source_schema_version"] == "issue-4848-trace-series.v1"
    assert receipt["provenance"]["transformation"] == "trace_series_to_simulation_trace_export"
    assert receipt["provenance"]["trace_series_campaign_job"] == "13334"


def test_export_trace_dossier_fails_closed_when_source_is_missing(tmp_path: Path) -> None:
    """Missing source artifacts must not produce a partial or synthetic trace."""
    store = tmp_path / "campaign-store"
    write_result_store(
        store,
        [
            {
                "run_id": "run-missing",
                "episode_id": "missing_episode",
                "planner": "goal",
                "scenario_id": "francis2023_blind_corner",
                "scenario_family": "classic",
                "seed": 111,
                "row_status": "native",
                "artifact_uri": str(tmp_path / "missing.json"),
                "artifact_sha256": "a" * 64,
            }
        ],
        study_id="trace-dossier-missing",
        command="fixture",
    )

    with pytest.raises(TraceDossierExportError, match="trace-missing"):
        export_trace_dossier(
            scenario_id="francis2023_blind_corner",
            planner_id="goal",
            seed=111,
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "dossier",
        )


def _resolved_row(source: Path, *, status: str = "native") -> dict[str, object]:
    """Build a direct resolver row for fail-closed branch tests."""
    return {
        "run_id": "run-fixture",
        "episode_id": "fixture_episode_001",
        "planner": "goal",
        "scenario_id": "francis2023_blind_corner",
        "scenario_family": "classic",
        "seed": 111,
        "row_status": status,
        "artifact_uri": str(source),
        "artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def test_resolve_episode_source_reports_fail_closed_provenance_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every malformed tuple/source state remains explicit and non-resolved."""
    source = _write_typed_source(tmp_path)

    missing_store = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "missing-store",
    )
    assert missing_store["resolution_status"] == "provenance-incomplete"

    rows = {"one": _resolved_row(source)}
    monkeypatch.setattr(
        resolution_module,
        "load_campaign_result_store",
        lambda _path: CampaignResultStore("study", rows),
    )
    invalid_tuple = resolve_episode_source(
        scenario_id="",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
    )
    assert invalid_tuple["reason_code"] == "invalid_requested_tuple"

    absent_row = resolve_episode_source(
        scenario_id="other-scenario",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
    )
    assert absent_row["reason_code"] == "campaign_row_not_found"

    rows.clear()
    rows["bad-status"] = _resolved_row(source, status="fallback")
    unsupported = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
    )
    assert unsupported["resolution_status"] == "provenance-incomplete"
    assert unsupported["reason_code"] == "unsupported_campaign_row_status:fallback"


def test_resolve_episode_source_rejects_duplicate_campaign_episode_identity(
    tmp_path: Path,
) -> None:
    """Duplicate source rows must not be collapsed into an arbitrary episode."""
    source = _write_typed_source(tmp_path)
    row = _resolved_row(source)
    duplicate = dict(row)
    duplicate["run_id"] = "run-other"
    store = _write_store(tmp_path, source)

    write_result_store(
        store,
        [row, duplicate],
        study_id="trace-dossier-duplicate",
        command="fixture",
    )

    result = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=store,
    )

    assert result["resolution_status"] == "provenance-incomplete"
    assert result["reason_code"].startswith(
        "campaign_store_unreadable:campaign result store has duplicate episode identity:"
    )


def test_resolve_episode_source_rejects_bad_hash_and_ambiguous_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hash mismatch and duplicate local artifact matches never resolve silently."""
    source = _write_typed_source(tmp_path)
    bad_hash = _resolved_row(source)
    bad_hash["artifact_sha256"] = "b" * 64
    monkeypatch.setattr(
        resolution_module,
        "load_campaign_result_store",
        lambda _path: CampaignResultStore("study", {"one": bad_hash}),
    )
    mismatch = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
    )
    assert mismatch["resolution_status"] == "provenance-incomplete"
    assert mismatch["reason_code"].startswith("campaign_row_artifact_sha256_mismatch")

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "episode.json"
    second = second_root / "episode.json"
    first.write_bytes(source.read_bytes())
    second.write_bytes(source.read_bytes())
    ambiguous_row = _resolved_row(source)
    ambiguous_row["artifact_uri"] = "episode.json"
    ambiguous_row["artifact_sha256"] = hashlib.sha256(first.read_bytes()).hexdigest()
    monkeypatch.setattr(
        resolution_module,
        "load_campaign_result_store",
        lambda _path: CampaignResultStore("study", {"one": ambiguous_row}),
    )
    ambiguous = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
        trace_search_roots=(first_root, second_root),
    )
    assert ambiguous["resolution_status"] == "provenance-incomplete"
    assert ambiguous["reason_code"].startswith("ambiguous_trace_artifact:")


def test_resolve_episode_source_rejects_incomplete_row_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing row fields are provenance-incomplete rather than inferred."""
    source = _write_typed_source(tmp_path)
    row = _resolved_row(source)
    row["artifact_uri"] = None
    monkeypatch.setattr(
        resolution_module,
        "load_campaign_result_store",
        lambda _path: CampaignResultStore("study", {"one": row}),
    )
    result = resolve_episode_source(
        scenario_id="francis2023_blind_corner",
        planner_id="goal",
        seed=111,
        campaign_store_dir=tmp_path / "unused",
    )
    assert result["resolution_status"] == "provenance-incomplete"
    assert result["reason_code"] == "campaign_row_missing_artifact_provenance"


def test_export_trace_dossier_rejects_output_source_overlap(tmp_path: Path) -> None:
    """The output bundle must not overwrite a source artifact in place."""
    source = tmp_path / "trace.json"
    source.write_bytes(_write_typed_source(tmp_path).read_bytes())
    store = _write_store(tmp_path, source)

    with pytest.raises(TraceDossierExportError, match="overlaps the existing source artifact"):
        export_trace_dossier(
            scenario_id="francis2023_blind_corner",
            planner_id="goal",
            seed=111,
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path,
        )
