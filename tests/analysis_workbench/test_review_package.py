"""Focused tests for the SREV-04 review-package component (issue #9273)."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import review_package
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


def _reference(
    artifact_id: str,
    name: str,
    integrity: dict,
    *,
    format_name: str = "episode-json",
    config_identity: str = "",
) -> dict:
    """Build one validated bundle reference for a fixture payload."""
    return {
        "artifact_id": artifact_id,
        "uri": name,
        "format": format_name,
        "schema": "episode.fixture",
        "sha256": integrity["sha256"],
        "source_commit": integrity["source_commit"],
        "config_identity": config_identity,
        "units": "seconds/metres/radians",
        "coordinate_frame": "map",
    }


def _stage(tmp_path: Path, *, config_identity: str = "") -> dict:
    """Stage a two-episode bundle; return its request payload dict."""
    first = _payload(tmp_path / "a.json", '{"episode_id": "ep-1"}\n')
    second = _payload(tmp_path / "b.json", '{"episode_id": "ep-2"}\n')

    bundle = {
        "schema_version": "review-bundle.v1",
        "bundle_id": "t-bundle",
        "episodes": [
            {
                "episode_id": "ep-1",
                "references": [_reference("a1", "a.json", first, config_identity=config_identity)],
            },
            {
                "episode_id": "ep-2",
                "references": [_reference("b1", "b.json", second, config_identity=config_identity)],
            },
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
    report_path = tmp_path / "out" / "verification-report.json"
    report = json.loads(report_path.read_text())
    assert manifest["schema_version"] == "review-package.v1"
    assert len(manifest["entries"]) == 2
    assert all(len(e["sha256"]) == 64 for e in manifest["entries"])
    assert report == {
        "diagnostics": [],
        "schema_version": "package-verification-report.v1",
        "verified_entries": 2,
        "videos_not_staged": [],
        "videos_staged": [],
    }
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
    artifacts = {str(item["artifact_id"]): item for item in result.artifacts}
    assert set(artifacts) == {
        "manifest.json",
        "verification-report.json",
        "package/a.json",
        "package/b.json",
    }
    for item in artifacts.values():
        assert (
            item["sha256"] == hashlib.sha256((tmp_path / str(item["uri"])).read_bytes()).hexdigest()
        )


def test_declared_source_provenance_is_preserved_in_manifest_and_result(
    tmp_path: Path,
) -> None:
    payload = _stage(tmp_path, config_identity="fixture-config-v1")
    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "complete"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    sources = manifest["source_provenance"]
    assert [source["artifact_id"] for source in sources] == ["a1", "b1"]
    assert all(source["source_commit"] == COMMIT for source in sources)
    assert all(source["config_identity"] == "fixture-config-v1" for source in sources)
    assert all(
        entry["source_commit"] == COMMIT and entry["config_identity"] == "fixture-config-v1"
        for entry in manifest["entries"]
    )
    assert result.provenance["source_provenance"] == sources
    assert result.provenance["source_commits"] == [COMMIT]


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


def test_declared_video_format_is_independent_of_uri_suffix(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    declared_video = _payload(tmp_path / "declared-video.json", '{"fixture": "video"}\n')
    non_video_mp4 = _payload(tmp_path / "episode.mp4", '{"fixture": "episode"}\n')
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    bundle["episodes"].append(
        {
            "episode_id": "ep-format-boundary",
            "references": [
                _reference(
                    "v-json",
                    "declared-video.json",
                    declared_video,
                    format_name="video-metadata",
                ),
                _reference("episode-mp4", "episode.mp4", non_video_mp4),
            ],
        }
    )
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "partial"
    assert "v-json: video_not_staged" in result.reason
    assert not (tmp_path / "out" / "package" / "declared-video.json").exists()
    assert (tmp_path / "out" / "package" / "episode.mp4").read_bytes() == (
        tmp_path / "episode.mp4"
    ).read_bytes()
    report = json.loads((tmp_path / "out" / "verification-report.json").read_text())
    assert [source["artifact_id"] for source in report["videos_not_staged"]] == ["v-json"]


def test_parent_symlink_escape_is_refused(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-source-outside"
    outside.mkdir()
    (outside / "a.json").write_bytes((tmp_path / "a.json").read_bytes())
    (tmp_path / "source-parent").symlink_to(outside, target_is_directory=True)
    bundle = json.loads((tmp_path / "bundle.json").read_text())
    bundle["episodes"][0]["references"][0]["uri"] = "source-parent/a.json"
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")

    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "partial"
    assert "a1: reference_not_local" in result.reason
    assert not (tmp_path / "out" / "package" / "a.json").exists()


def test_bundle_symlink_escape_is_refused(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-bundle-outside"
    outside.mkdir()
    outside_bundle = outside / "bundle.json"
    outside_bundle.write_bytes((tmp_path / "bundle.json").read_bytes())
    (tmp_path / "bundle-link.json").symlink_to(outside_bundle)
    payload["sources"][0]["uri"] = "bundle-link.json"

    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "partial"
    assert "bundle: reference_not_local" in result.reason
    assert not (tmp_path / "out" / "package").exists()


def test_output_symlink_escape_is_rejected(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-output-outside"
    outside.mkdir()
    (tmp_path / "output-link").symlink_to(outside, target_is_directory=True)
    payload["output_directory"] = "output-link/escaped"

    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "failed"
    assert "unsafe_output_path" in result.reason
    assert result.artifacts == ()
    assert not (outside / "escaped").exists()


def test_absolute_package_dirname_is_rejected(tmp_path: Path) -> None:
    payload = _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-package-outside"
    payload["config"] = {"package_dirname": str(outside)}

    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "failed"
    assert "invalid_config" in result.reason
    assert result.artifacts == ()
    assert not outside.exists()


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


def test_staging_filesystem_failure_returns_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _stage(tmp_path)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("fixture staging failure")

    monkeypatch.setattr(review_package.shutil, "copyfileobj", fail_copy)
    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == "staging_failed"
    assert result.artifacts == ()
    assert result.status != "complete"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(OSError("fixture publication filesystem failure"), id="filesystem"),
        pytest.param(ValueError("fixture publication value failure"), id="value"),
    ],
)
def test_publication_operation_failure_returns_failed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    payload = _stage(tmp_path)
    original_write = review_package._write_json

    def fail_report(path: Path, document: object) -> str:
        if path.name == "verification-report.json":
            raise failure
        return original_write(path, document)

    monkeypatch.setattr(review_package, "_write_json", fail_report)
    result = run(_request(tmp_path, payload), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == "publication_failed"
    assert result.artifacts == ()
    assert not (tmp_path / "out" / "verification-report.json").exists()


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
        artifact_ids = {str(item["artifact_id"]) for item in payload["artifacts"]}
        assert artifact_ids == {
            "manifest.json",
            "verification-report.json",
            "package/tests/fixtures/scenario_review/review_package/payload-a.json",
            "package/tests/fixtures/scenario_review/review_package/payload-b.json",
        }
        for item in payload["artifacts"]:
            assert item["sha256"] == hashlib.sha256((repo / item["uri"]).read_bytes()).hexdigest()
        manifest = json.loads((repo / output_rel / "manifest.json").read_text())
        assert len(manifest["entries"]) == 3
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
