"""Focused tests for the SREV-07 review-storyboard component (issue #9276)."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_storyboard import (
    CANONICAL_SCORE_ADAPTER_VERSION,
    COMPONENT_ID,
    descriptor,
    result_document,
    run,
)

FIXTURE_DIR = "tests/fixtures/scenario_review/review_storyboard"

BUNDLE = {
    "schema_version": "review-bundle.v1",
    "bundle_id": "t-bundle",
    "episodes": [
        {
            "episode_id": "ep-1",
            "references": [
                {
                    "artifact_id": "a1",
                    "uri": "episode-1.json",
                    "format": "episode-json",
                    "schema": "s",
                    "sha256": "0" * 64,
                    "source_commit": "0" * 40,
                    "units": "u",
                    "coordinate_frame": "map",
                }
            ],
        },
        {
            "episode_id": "ep-2",
            "references": [
                {
                    "artifact_id": "a2",
                    "uri": "episode-2.json",
                    "format": "episode-json",
                    "schema": "s",
                    "sha256": "0" * 64,
                    "source_commit": "0" * 40,
                    "units": "u",
                    "coordinate_frame": "map",
                }
            ],
        },
    ],
}

SCORES = {
    "schema_version": "srev07-exemplar-scores.v1",
    "scores": {"ep-1": 0.5, "ep-2": 0.5},
}

EVENTS = {"intervals": [{"interval_id": "ev-1", "start_s": 0.0, "end_s": 0.5}]}


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic fixture JSON."""
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    """Return one fixture file's raw-byte digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_digest(path: Path) -> str:
    """Return the logical digest used by override provenance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stage(tmp_path: Path) -> None:
    """Stage integrity-bound episode, bundle, score, and event fixtures."""
    bundle = copy.deepcopy(BUNDLE)
    for episode in bundle["episodes"]:
        episode_id = episode["episode_id"]
        episode_path = tmp_path / f"episode-{episode_id.removeprefix('ep-')}.json"
        _write_json(
            episode_path,
            {
                "artifact_id": episode["references"][0]["artifact_id"],
                "episode_id": episode_id,
                "evidence_boundary": "diagnostic_only",
            },
        )
        reference = episode["references"][0]
        reference["uri"] = episode_path.name
        reference["sha256"] = _sha256(episode_path)
    _write_json(tmp_path / "bundle.json", bundle)
    _write_json(tmp_path / "scores.json", SCORES)
    _write_json(tmp_path / "events.json", EVENTS)


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict[str, Any] | None = None,
    output: str = "out",
    sources: list[dict[str, Any]] | None = None,
    base_path: Path | None = None,
) -> Any:
    config: dict[str, Any] = {"source_duration_s": 10.0}
    config.update(config_extra or {})
    source_list = copy.deepcopy(
        sources
        if sources is not None
        else [
            {"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"},
            {"artifact_id": "scores", "uri": "scores.json", "format": "exemplar-scores"},
        ]
    )
    if base_path is not None:
        for source in source_list:
            source_path = base_path / source["uri"]
            if not source.get("sha256") and source_path.is_file():
                source["sha256"] = _sha256(source_path)
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": source_list,
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def _spec(tmp_path: Path, output: str = "out") -> dict[str, Any]:
    return json.loads((tmp_path / output / "storyboard-spec.json").read_text())


def _source_list_with_events() -> list[dict[str, str]]:
    return [
        {"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"},
        {"artifact_id": "scores", "uri": "scores.json", "format": "exemplar-scores"},
        {"artifact_id": "events", "uri": "events.json", "format": "event-index"},
    ]


def test_success_ranks_with_stable_tie_break_and_versioned_result(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "bundle.json").read_bytes()
    result = run(_request(required=("exemplar-scores",), base_path=tmp_path), base=tmp_path)
    assert result.status == "complete"
    assert result.schema_version == "component-result.v1"
    assert result.reason == ""
    assert (tmp_path / "bundle.json").read_bytes() == source_before
    spec = _spec(tmp_path)
    ranking = spec["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-1", "ep-2"]
    assert [r["source_artifact_ids"] for r in ranking] == [["a1"], ["a2"]]
    assert spec["sources"] == [{"artifact_id": "a1"}, {"artifact_id": "a2"}]
    assert spec["annotations"][0]["tie_break_method"].startswith("score-descending")
    assert all(r["score"] == 0.5 for r in ranking)
    assert len(result.artifacts) == 3
    component_result_from_dict(result_document(result))


def test_deterministic_repeat_runs_match_bytes(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a", base_path=tmp_path), base=tmp_path)
    second = run(_request(output="out-b", base_path=tmp_path), base=tmp_path)
    assert first.status == second.status == "complete"
    assert (tmp_path / "out-a" / "storyboard-spec.json").read_bytes() == (
        tmp_path / "out-b" / "storyboard-spec.json"
    ).read_bytes()
    assert result_document(first)["provenance"]["emitted_artifacts"]


def test_pin_override_survives_regeneration(tmp_path: Path) -> None:
    _stage(tmp_path)
    actual = _bundle_digest(tmp_path / "bundle.json")
    result = run(
        _request(
            config_extra={
                "overrides": {
                    "pin": ["ep-2"],
                    "exclude": [],
                    "source_digest": actual,
                }
            },
            base_path=tmp_path,
        ),
        base=tmp_path,
    )
    assert result.status == "complete"
    ranking = _spec(tmp_path)["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-2", "ep-1"]
    assert ranking[0]["pinned"] is True


def test_stale_override_source_ignores_overrides(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(
        _request(
            config_extra={"overrides": {"pin": ["ep-2"], "exclude": [], "source_digest": "0" * 64}},
            base_path=tmp_path,
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "stale_override_source" in result.reason
    ranking = _spec(tmp_path)["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-1", "ep-2"]


def test_exclude_override_removes_candidate(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(
        _request(
            config_extra={
                "overrides": {
                    "pin": [],
                    "exclude": ["ep-2"],
                    "source_digest": _bundle_digest(tmp_path / "bundle.json"),
                }
            },
            base_path=tmp_path,
        ),
        base=tmp_path,
    )
    assert result.status == "complete"
    ranking = _spec(tmp_path)["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-1"]


def test_corrupt_source_is_failed_without_result_artifacts(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "bundle.json").write_text("{not json", encoding="utf-8")
    result = run(_request(base_path=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert result.artifacts == []
    assert "source_unreadable" in result.reason


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("rvo2-binary",), base_path=tmp_path), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(
        _request(config_extra={"min_component_version": "2.0.0"}, base_path=tmp_path),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(base_path=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope", base_path=tmp_path), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_is_a_valid_versioned_envelope() -> None:
    info = descriptor()
    assert info["schema_version"] == "component-descriptor.v1"
    assert info["component_id"] == COMPONENT_ID
    assert list(info["required_capabilities"]) == ["review-bundle"]
    assert "exemplar-scores" in info["optional_capabilities"]
    assert "missing-capability-report.v1" in info["output_types"]
    component_descriptor_from_dict(info)


def test_module_touches_no_simulator_paths() -> None:
    """The adapter must stay observational: no sim/planner imports."""
    tree = ast.parse(
        (
            Path(__file__).resolve().parents[2] / "robot_sf/analysis_workbench/review_storyboard.py"
        ).read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    imported -= {"robot_sf", "__future__", "typing"}
    assert not (imported & {"sim", "planner", "training", "torch"}), imported


def test_cli_produces_versioned_result_and_bound_fixture_sources() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / FIXTURE_DIR
    assert (fixture / "request.json").exists()
    output_rel = "output/scenario_review/srev-07-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.analysis_workbench.review_storyboard",
                "--input",
                str(fixture / "request.json"),
                "--config",
                str(fixture / "config.json"),
                "--output",
                output_rel,
            ],
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        payload = json.loads(completed.stdout)
        component_result_from_dict(payload)
        assert payload["schema_version"] == "component-result.v1"
        assert payload["status"] == "complete"
        assert {item["artifact_id"] for item in payload["artifacts"]} == {
            "storyboard-spec.json",
            "override-provenance.json",
            "missing-capability-report.json",
        }
        spec = json.loads((repo / output_rel / "storyboard-spec.json").read_text())
        assert [s["artifact_id"] for s in spec["sources"]] == ["a3", "a1", "a2"]
        assert spec["annotations"][0]["event_intervals"][0]["interval_id"] == "ev-1"
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)


def test_selected_sources_and_all_sidecars_are_hash_bound(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(base_path=tmp_path), base=tmp_path)
    assert result.status == "complete"
    emitted = result.provenance["emitted_artifacts"]
    assert emitted == list(result.artifacts)
    for artifact in emitted:
        assert _sha256(tmp_path / artifact["uri"]) == artifact["sha256"]
    assert result.provenance["selected_source_artifact_ids"] == ["a1", "a2"]
    assert {record["artifact_id"] for record in result.provenance["source_provenance"]} >= {
        "bundle",
        "scores",
        "a1",
        "a2",
    }


def test_wrong_declared_source_digest_fails_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    sources = [
        {
            "artifact_id": "bundle",
            "uri": "bundle.json",
            "format": "review-bundle",
            "sha256": "0" * 64,
        },
        {
            "artifact_id": "scores",
            "uri": "scores.json",
            "format": "exemplar-scores",
        },
    ]
    result = run(_request(sources=sources, base_path=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_digest_mismatch" in result.reason
    assert not (tmp_path / "out").exists()


def test_symlink_source_escape_fails_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / f"storyboard-outside-{tmp_path.name}.json"
    _write_json(outside, BUNDLE)
    link = tmp_path / "bundle-link.json"
    try:
        os.symlink(outside, link)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    sources = [
        {
            "artifact_id": "bundle",
            "uri": link.name,
            "format": "review-bundle",
            "sha256": _sha256(outside),
        },
    ]
    result = run(_request(sources=sources, base_path=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_uri_unsafe" in result.reason
    assert not (tmp_path / "out").exists()


def test_required_scores_are_reported_missing_without_zero_fallback(tmp_path: Path) -> None:
    _stage(tmp_path)
    sources = [{"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"}]
    result = run(
        _request(required=("exemplar-scores",), sources=sources, base_path=tmp_path),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert result.artifacts == []
    capability = json.loads(
        (tmp_path / "out" / "missing-capability-report.json").read_text(encoding="utf-8")
    )
    assert capability["missing_capabilities"] == ["exemplar-scores"]
    ranking = _spec(tmp_path)["annotations"][0]["ranking"]
    assert all(item["score"] is None for item in ranking)
    assert all(item["score_source"] == "unavailable" for item in ranking)


def test_canonical_exemplar_interest_shape_uses_explicit_adapter(tmp_path: Path) -> None:
    _stage(tmp_path)
    canonical = {
        "roots": ["diagnostic-fixture"],
        "weights": {},
        "episodes": [
            {"episode_id": "ep-1", "composite_score": 0.1},
            {"episode_id": "ep-2", "composite_score": 0.9},
        ],
        "comparison_pairs": [],
    }
    _write_json(tmp_path / "canonical-scores.json", canonical)
    sources = [
        {"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"},
        {
            "artifact_id": "scores",
            "uri": "canonical-scores.json",
            "format": "exemplar-scores",
        },
    ]
    result = run(
        _request(required=("exemplar-scores",), sources=sources, base_path=tmp_path),
        base=tmp_path,
    )
    assert result.status == "complete"
    assert result.provenance["score_input_format"] == CANONICAL_SCORE_ADAPTER_VERSION
    assert result.provenance["score_adapter_version"] == CANONICAL_SCORE_ADAPTER_VERSION
    assert [item["episode_id"] for item in _spec(tmp_path)["annotations"][0]["ranking"]] == [
        "ep-2",
        "ep-1",
    ]


def test_unversioned_score_map_is_not_silently_defaulted(tmp_path: Path) -> None:
    _stage(tmp_path)
    _write_json(tmp_path / "legacy-scores.json", {"scores": {"ep-1": 0.9, "ep-2": 0.1}})
    sources = [
        {"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"},
        {"artifact_id": "scores", "uri": "legacy-scores.json", "format": "exemplar-scores"},
    ]
    result = run(
        _request(required=("exemplar-scores",), sources=sources, base_path=tmp_path),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "exemplar_scores_unversioned_or_malformed" in result.reason
    assert all(item["score"] is None for item in _spec(tmp_path)["annotations"][0]["ranking"])


def test_duplicate_episode_ids_fail_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    bundle = json.loads((tmp_path / "bundle.json").read_text(encoding="utf-8"))
    bundle["episodes"][1]["episode_id"] = bundle["episodes"][0]["episode_id"]
    _write_json(tmp_path / "bundle.json", bundle)
    result = run(_request(base_path=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "duplicate_episode_id:ep-1" in result.reason
    assert not (tmp_path / "out").exists()


def test_event_interval_identity_is_preserved_in_annotation(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(
        _request(
            required=("event-index",),
            sources=_source_list_with_events(),
            base_path=tmp_path,
        ),
        base=tmp_path,
    )
    assert result.status == "complete"
    spec = _spec(tmp_path)
    assert spec["source_intervals"] == [
        {"start_s": 0.0, "end_s": 0.5, "include_terminal_frame": False}
    ]
    assert spec["annotations"][0]["event_intervals"] == [
        {"start_s": 0.0, "end_s": 0.5, "source_row": 0, "interval_id": "ev-1"}
    ]
