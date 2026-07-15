"""Focused contract tests for the #5756 request-to-figure pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.candidate_trace_resolution import (
    CandidateTraceResolutionError,
    load_episode_mapping,
    load_episode_requests,
    resolve_episode_requests,
    validate_candidate_trace_resolution,
)
from robot_sf.benchmark.trace_scene_figure import load_episode_from_trace_export

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)


def _request_manifest(**request_overrides: object) -> dict[str, object]:
    request = {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": "111",
        "episode_id": "fixture_episode_001",
        **request_overrides,
    }
    return {
        "schema_version": "issue_5446_trace_reexport_list.v1",
        "n_tuples": 1,
        "tuples": [request],
    }


def _mapping_row(**overrides: object) -> dict[str, object]:
    return {
        "scenario_id": "classic_bottleneck_medium",
        "planner": "hybrid_rule_v0_minimal",
        "seed": 111,
        "episode_id": "fixture_episode_001",
        "outcome": "success",
        "trace_artifact_uri": str(TRACE),
        **overrides,
    }


def test_request_resolution_validates_identity_outcome_and_trace(tmp_path: Path) -> None:
    """A concrete request resolves only after all provenance gates pass."""
    request_path = tmp_path / "requests.json"
    mapping_path = tmp_path / "mapping.json"
    request_path.write_text(json.dumps(_request_manifest()), encoding="utf-8")
    mapping_path.write_text(json.dumps({"rows": [_mapping_row()]}), encoding="utf-8")

    request_manifest, normalized = load_episode_requests(request_path)
    assert normalized[0]["seed"] == 111
    mapping = load_episode_mapping(mapping_path)
    result = resolve_episode_requests(request_manifest, mapping, trace_search_roots=[])

    assert result["summary"] == {
        "n_candidates": 1,
        "n_resolved": 1,
        "n_trace_missing": 0,
        "n_schema_mismatch": 0,
        "n_provenance_incomplete": 0,
    }
    assert result["rows"][0]["reason_code"] == "trace_schema_valid:outcome=success"
    assert validate_candidate_trace_resolution(result)["ok"]


@pytest.mark.parametrize(
    ("request_overrides", "mapping_overrides", "status", "reason"),
    [
        (
            {"seed": 112, "episode_id": "missing"},
            {},
            "provenance-incomplete",
            "missing_episode_mapping",
        ),
        (
            {},
            {"scenario_id": "classic_doorway_medium"},
            "provenance-incomplete",
            "mapping_identity_mismatch:scenario_id",
        ),
        ({"expected_outcome": "collision_event"}, {}, "provenance-incomplete", "outcome_mismatch"),
        (
            {},
            {"trace_artifact_uri": "/not/a/trace.json"},
            "trace-missing",
            "trace_artifact_not_found",
        ),
    ],
)
def test_request_resolution_fails_closed(
    tmp_path: Path,
    request_overrides: dict[str, object],
    mapping_overrides: dict[str, object],
    status: str,
    reason: str,
) -> None:
    """Missing mapping, identity drift, outcome drift, and trace absence are explicit."""
    request_manifest = _request_manifest(**request_overrides)
    mapping = load_episode_mapping(
        _write_json(tmp_path / "mapping.json", {"rows": [_mapping_row(**mapping_overrides)]})
    )
    result = resolve_episode_requests(request_manifest, mapping)
    row = result["rows"][0]
    assert row["resolution_status"] == status
    assert row["reason_code"].startswith(reason)


def test_duplicate_request_tuple_is_rejected(tmp_path: Path) -> None:
    """The 90-request contract cannot silently resolve a duplicate tuple twice."""
    payload = _request_manifest()
    payload["n_tuples"] = 2
    payload["tuples"] = [payload["tuples"][0], dict(payload["tuples"][0])]  # type: ignore[index]
    path = _write_json(tmp_path / "requests.json", payload)
    with pytest.raises(CandidateTraceResolutionError, match="duplicate episode request tuple"):
        load_episode_requests(path)


def test_trace_export_adapter_provides_renderer_episode() -> None:
    """The typed trace export becomes the renderer's existing derived-series contract."""
    episode = load_episode_from_trace_export(TRACE, outcome="success")
    assert episode.metadata["episode_id"] == "fixture_episode_001"
    assert episode.metadata["episode_status"] == "success"
    assert episode.steps == (0, 1)
    assert episode.nearest_pedestrian_id == ("ped_1", "ped_1")
    assert episode.min_robot_ped_distance_m[0] > episode.min_robot_ped_distance_m[1]


def test_release_episode_id_alias_joins_to_rerun_episode(tmp_path: Path) -> None:
    """A rerun may assign a new id while retaining the release request id."""
    request_manifest = _request_manifest()
    rerun_trace = json.loads(TRACE.read_text(encoding="utf-8"))
    rerun_trace["source"]["episode_id"] = "rerun_episode_001"
    rerun_trace_path = _write_json(tmp_path / "rerun_trace.json", rerun_trace)
    mapping = load_episode_mapping(
        _write_json(
            tmp_path / "mapping.json",
            {
                "rows": [
                    _mapping_row(
                        episode_id="rerun_episode_001",
                        release_episode_id="fixture_episode_001",
                        trace_artifact_uri=str(rerun_trace_path),
                    )
                ]
            },
        )
    )
    result = resolve_episode_requests(request_manifest, mapping)
    assert result["summary"]["n_resolved"] == 1
    assert result["rows"][0]["episode_id"] == "rerun_episode_001"


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
