"""Focused tests for the SREV-07 review-storyboard component (issue #9276)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.analysis_workbench.review_storyboard import (
    COMPONENT_ID,
    descriptor,
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
                    "uri": "x",
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
                    "uri": "x",
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

SCORES = {"scores": {"ep-1": 0.5, "ep-2": 0.5}}


def _stage(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_text(json.dumps(BUNDLE), encoding="utf-8")
    (tmp_path / "scores.json").write_text(json.dumps(SCORES), encoding="utf-8")


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
) -> dict:
    from robot_sf.analysis_workbench.review_storyboard import (
        component_request_from_dict,
    )

    config: dict = {"source_duration_s": 10.0}
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": (
            sources
            if sources is not None
            else [
                {"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"},
                {"artifact_id": "scores", "uri": "scores.json", "format": "exemplar-scores"},
            ]
        ),
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def _spec(tmp_path: Path, output: str = "out") -> dict:
    return json.loads((tmp_path / output / "storyboard-spec.json").read_text())


def test_success_ranks_with_stable_tie_break(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "bundle.json").read_bytes()
    result = run(_request(required=("exemplar-scores",)), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "bundle.json").read_bytes() == source_before
    spec = _spec(tmp_path)
    ranking = spec["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-1", "ep-2"]
    assert spec["annotations"][0]["tie_break_method"].startswith("score-descending")
    assert all(r["score"] == 0.5 for r in ranking)
    assert len(result.artifacts) == 1


def test_deterministic_repeat_runs_match_bytes(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a"), base=tmp_path)
    second = run(_request(output="out-b"), base=tmp_path)
    assert first.status == second.status == "complete"
    assert (tmp_path / "out-a" / "storyboard-spec.json").read_bytes() == (
        tmp_path / "out-b" / "storyboard-spec.json"
    ).read_bytes()


def test_pin_override_survives_regeneration(tmp_path: Path) -> None:
    import hashlib

    _stage(tmp_path)
    actual = hashlib.sha256(
        json.dumps(
            json.loads((tmp_path / "bundle.json").read_text()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    result = run(
        _request(
            config_extra={
                "overrides": {
                    "pin": ["ep-2"],
                    "exclude": [],
                    "source_digest": actual,
                }
            }
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
            config_extra={"overrides": {"pin": ["ep-2"], "exclude": [], "source_digest": "0" * 64}}
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
        _request(config_extra={"overrides": {"pin": [], "exclude": ["ep-2"]}}),
        base=tmp_path,
    )
    assert result.status == "complete"
    ranking = _spec(tmp_path)["annotations"][0]["ranking"]
    assert [r["episode_id"] for r in ranking] == ["ep-1"]


def test_corrupt_source_is_failed_or_partial(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_text("{not json", encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status in ("partial", "failed")
    assert result.artifacts == ()


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"min_component_version": "2.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert list(info["required_capabilities"]) == ["review-bundle"]
    assert "exemplar-scores" in info["optional_capabilities"]


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


def test_cli_produces_spec_from_fixture_request() -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/fixtures/scenario_review/review_storyboard"
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
        assert payload["status"] == "complete"
        spec = json.loads((repo / output_rel / "storyboard-spec.json").read_text())
        assert [s["artifact_id"] for s in spec["sources"]] == ["ep-3", "ep-1", "ep-2"]
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
