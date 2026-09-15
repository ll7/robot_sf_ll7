"""Focused tests for the SREV-04 review-package component (issue #9273)."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.analysis_workbench.review_contracts import (
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_package import (
    COMPONENT_ID,
    descriptor,
    run,
)

COMMIT = "e" * 40


def _payload(path: Path, text: str) -> dict:
    path.write_text(text, encoding="utf-8")
    return {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_commit": COMMIT,
    }


def _stage(tmp_path: Path) -> dict:
    """Stage a two-episode bundle; return its request payload dict."""
    first = _payload(tmp_path / "a.json", '{"episode_id": "ep-1"}\n')
    second = _payload(tmp_path / "b.json", '{"episode_id": "ep-2"}\n')

    def ref(artifact_id: str, name: str, integrity: dict) -> dict:
        return {
            "artifact_id": artifact_id,
            "uri": name,
            "format": "episode-json",
            "schema": "episode.fixture",
            "sha256": integrity["sha256"],
            "source_commit": integrity["source_commit"],
            "units": "seconds/metres/radians",
            "coordinate_frame": "map",
        }

    bundle = {
        "schema_version": "review-bundle.v1",
        "bundle_id": "t-bundle",
        "episodes": [
            {"episode_id": "ep-1", "references": [ref("a1", "a.json", first)]},
            {"episode_id": "ep-2", "references": [ref("b1", "b.json", second)]},
        ],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    return {
        "schema_version": "component-request.v1",
        "request_id": "t",
        "component_id": COMPONENT_ID,
        "sources": [{"artifact_id": "bundle", "uri": "bundle.json", "format": "review-bundle"}],
        "output_directory": "out",
        "config": {},
        "required_capabilities": [],
    }


def _request(tmp_path: Path, payload: dict):
    from robot_sf.analysis_workbench.review_package import (
        component_request_from_dict,
    )

    return component_request_from_dict(payload)


def test_success_relocates_verified_payloads(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    before_a = (tmp_path / "a.json").read_bytes()
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "a.json").read_bytes() == before_a
    assert (tmp_path / "out" / "package" / "a.json").read_bytes() == before_a
    assert (tmp_path / "out" / "package" / "b.json").read_bytes() == b'{"episode_id": "ep-2"}\n'
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["schema_version"] == "review-package.v1"
    assert len(manifest["entries"]) == 2
    assert all(len(e["sha256"]) == 64 for e in manifest["entries"])
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


def test_deterministic_repeat_runs_match_manifest(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    first = run(_request(tmp_path, dict(payload, output_directory="out-a")), base=tmp_path)
    second = run(_request(tmp_path, dict(payload, output_directory="out-b")), base=tmp_path)
    assert first.status == second.status == "complete"
    a = (tmp_path / "out-a" / "manifest.json").read_bytes()
    b = (tmp_path / "out-b" / "manifest.json").read_bytes()
    assert json.loads(a)["entries"] == json.loads(b)["entries"]


def test_tampered_payload_is_partial_and_unstaged(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    (tmp_path / "a.json").write_text('{"episode_id": "tampered"}\n', encoding="utf-8")
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "partial"
    assert "tampered_payload" in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out" / "package" / "a.json").exists()


def test_missing_payload_is_partial(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    (tmp_path / "b.json").unlink()
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "partial"
    assert "source_unreadable" in result.reason


def test_symlink_source_is_refused(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    (tmp_path / "a.json").unlink()
    (tmp_path / "a.json").symlink_to(tmp_path / "b.json")
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "partial"
    assert "symlink_refused" in result.reason
    assert not (tmp_path / "out" / "package" / "a.json").is_symlink()


def test_video_payload_is_never_staged(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    (tmp_path / "clip.mp4").write_text("fixture-marker-not-a-video", encoding="utf-8")
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    bundle["episodes"].append(
        {
            "episode_id": "ep-v",
            "references": [
                {
                    "artifact_id": "v1",
                    "uri": "clip.mp4",
                    "format": "video-mp4",
                    "schema": "video.fixture",
                    "sha256": hashlib.sha256(b"fixture-marker-not-a-video").hexdigest(),
                    "source_commit": COMMIT,
                    "units": "seconds/metres/radians",
                    "coordinate_frame": "map",
                }
            ],
        }
    )
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "partial"
    assert "video_not_staged" in result.reason
    assert not (tmp_path / "out" / "package" / "clip.mp4").exists()


def test_dry_run_lists_operations_without_writing(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    payload["config"] = {"dry_run": True}
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "complete"
    assert result.provenance["operations"] != []
    assert not (tmp_path / "out").exists()


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    payload["required_capabilities"] = ["rvo2-binary"]
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    payload["config"] = {"min_component_version": "2.0.0"}
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    payload["component_id"] = "srev99-nope"
    result = run(_request(tmp_path, payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert list(info["required_capabilities"]) == ["review-bundle"]


def test_module_touches_no_simulator_paths() -> None:
    """The adapter must stay a file-level packager: no sim/planner imports."""
    tree = ast.parse(
        (
            Path(__file__).resolve().parents[2] / "robot_sf/analysis_workbench/review_package.py"
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


def test_cli_produces_package_from_fixture_request() -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/fixtures/scenario_review/review_package"
    assert (fixture / "request.json").exists()
    output_rel = "output/scenario_review/srev-04-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.analysis_workbench.review_package",
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
        manifest = json.loads((repo / output_rel / "manifest.json").read_text())
        assert len(manifest["entries"]) == 3
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
