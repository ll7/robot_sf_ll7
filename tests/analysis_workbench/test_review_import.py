"""Focused tests for the SREV-02 review-import component (issue #9271)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from robot_sf.analysis_workbench.review_contracts import (
    component_result_from_dict,
    review_bundle_from_dict,
)
from robot_sf.analysis_workbench.review_import import (
    COMPONENT_ID,
    descriptor,
    run,
)

EPISODE_JSONL = (
    '{"episode_id": "ep-001", "units": "seconds/metres/radians", '
    '"actor_ids": ["robot"], "evidence_status": "fixture"}\n'
    '{"episode_id": "ep-002"}\n'
)
ANALYSIS_TRACE = '{"episode_id": "an-001"}\n'
SIM_TRACE = '{"episode_id": "sim-001"}\n'
DOSSIER = '{"episode_id": "dos-001"}\n'
VIDEO_META = '{"episode_id": "vid-001", "codec": "none"}\n'

SOURCES = {
    "episodes": ("episode.jsonl", "episode-jsonl", EPISODE_JSONL),
    "analysis": ("analysis.json", "analysis-trace", ANALYSIS_TRACE),
    "simtrace": ("sim.json", "simulation-trace", SIM_TRACE),
    "dossier": ("dossier.json", "trace-dossier", DOSSIER),
    "video": ("video.json", "video-metadata", VIDEO_META),
}


def _stage(tmp_path: Path) -> dict[str, str]:
    """Write canned sources; return artifact_id -> sha256 of staged bytes."""
    import hashlib

    digests: dict[str, str] = {}
    for artifact_id, (name, _family, text) in SOURCES.items():
        target = tmp_path / name
        target.write_text(text, encoding="utf-8")
        digests[artifact_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digests


def _request(
    tmp_path: Path,
    digests: dict[str, str],
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
) -> dict:
    from robot_sf.analysis_workbench.review_import import (
        component_request_from_dict,
    )

    sources = [
        {"artifact_id": artifact_id, "uri": name, "format": family}
        for artifact_id, (name, family, _text) in SOURCES.items()
    ]
    config: dict = {
        "source_metadata": {
            artifact_id: {"sha256": digests[artifact_id], "source_commit": "b" * 40}
            for artifact_id in SOURCES
        }
    }
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": sources,
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def test_success_indexes_all_families_without_rewriting_sources(tmp_path: Path) -> None:
    before = {name: text for name, _f, text in SOURCES.values()}
    digests = _stage(tmp_path)
    result = run(_request(tmp_path, digests), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    # Sources are byte-identical after the run: index by reference only.
    for name, text in before.items():
        assert (tmp_path / name).read_text(encoding="utf-8") == text
    bundle_path = tmp_path / "out" / "review-bundle.json"
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle = review_bundle_from_dict(bundle_payload)
    assert {episode["episode_id"] for episode in bundle.episodes} == {
        "ep-001",
        "ep-002",
        "an-001",
        "sim-001",
        "dos-001",
        "vid-001",
    }
    # Envelope round-trips through the shared result contract.
    envelope = {
        "schema_version": "component-result.v1",
        "request_id": result.request_id,
        "component_id": result.component_id,
        "status": result.status,
        "artifacts": [dict(a) for a in result.artifacts],
        "diagnostics": [dict(d) for d in result.diagnostics],
        "provenance": dict(result.provenance),
        "reason": result.reason,
    }
    assert component_result_from_dict(envelope).status == "complete"
    assert len(result.artifacts) == 1


def test_deterministic_repeat_runs_match_digest(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    first = run(_request(tmp_path, digests, output="out-a"), base=tmp_path)
    second = run(_request(tmp_path, digests, output="out-b"), base=tmp_path)
    assert first.status == second.status == "complete"
    assert first.provenance["bundle_digest"] == second.provenance["bundle_digest"]


def test_corrupt_json_line_yields_partial(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    (tmp_path / "episode.jsonl").write_text(EPISODE_JSONL + "{not json}\n", encoding="utf-8")
    result = run(_request(tmp_path, digests), base=tmp_path)
    assert result.status == "partial"
    assert "corrupt_json_line" in result.reason
    assert result.artifacts == ()


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    result = run(_request(tmp_path, digests, required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    result = run(
        _request(tmp_path, digests, config_extra={"min_component_version": "2.0.0"}),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(tmp_path, digests), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_duplicate_episode_ids_yield_partial(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    (tmp_path / "analysis.json").write_text('{"episode_id": "ep-001"}\n', encoding="utf-8")
    result = run(_request(tmp_path, digests), base=tmp_path)
    assert result.status == "partial"
    assert "duplicate_episode_id:ep-001" in result.reason


def test_stale_digest_yields_partial(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    digests["episodes"] = "0" * 64
    result = run(_request(tmp_path, digests), base=tmp_path)
    assert result.status == "partial"
    assert "stale_digest" in result.reason


def test_video_only_source_imports_without_telemetry_claim(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "video.json"
    target.write_text(VIDEO_META, encoding="utf-8")
    digest = hashlib.sha256(VIDEO_META.encode("utf-8")).hexdigest()
    from robot_sf.analysis_workbench.review_import import (
        component_request_from_dict,
    )

    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "v",
            "component_id": COMPONENT_ID,
            "sources": [{"artifact_id": "video", "uri": "video.json", "format": "video-metadata"}],
            "output_directory": "out",
            "config": {"source_metadata": {"video": {"sha256": digest, "source_commit": "b" * 40}}},
            "required_capabilities": [],
        }
    )
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert any("video_source_without_telemetry" in item["code"] for item in result.diagnostics)


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    digests = _stage(tmp_path)
    result = run(_request(tmp_path, digests, component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_five_families() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert len(info["required_capabilities"]) == 5
    assert "presentation-manifest" in info["optional_capabilities"]


def test_cli_produces_bundle_from_fixture_request(tmp_path: Path) -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/fixtures/scenario_review/review_import"
    assert (fixture / "request.json").exists()
    output_rel = "output/scenario_review/srev-02-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.analysis_workbench.review_import",
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
        bundle = json.loads((repo / output_rel / "review-bundle.json").read_text())
        assert review_bundle_from_dict(bundle).bundle_id == "srev02-smoke-bundle"
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
