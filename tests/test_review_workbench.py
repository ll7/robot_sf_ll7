"""Smoke-tier contract tests for the SREV-15 review workbench (issue #9284)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.render import review_workbench

_EPISODE_REF = {
    "artifact_id": "trace-0000",
    "uri": "trace-0000.json",
    "format": "simulation_trace_export.v1",
    "schema": "simulation-trace-export.v1",
    "sha256": "a" * 64,
    "source_commit": "b" * 40,
    "units": "seconds/metres/radians",
    "coordinate_frame": "map",
}


def _bundle(
    tmp_path: Path,
    *,
    bundle_id: str = "bundle-0",
    episode_id: str = "ep-0",
    reference_uri: str = "trace-0000.json",
    declared_sha256: str | None = None,
) -> dict[str, object]:
    """Build a review bundle whose reference points at an on-disk artifact."""
    if (tmp_path / reference_uri).exists():
        digest = hashlib.sha256((tmp_path / reference_uri).read_bytes()).hexdigest()
    else:
        digest = "a" * 64
    reference = {
        **_EPISODE_REF,
        "uri": reference_uri,
        "sha256": digest if declared_sha256 is None else declared_sha256,
    }
    return {
        "schema_version": "review-bundle.v1",
        "bundle_id": bundle_id,
        "episodes": [{"episode_id": episode_id, "references": [reference]}],
    }


def _spec(**presentation: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": "visualization-spec.v1",
        "spec_id": "spec-0",
        "sources": [{"artifact_id": "trace-0000"}],
    }
    if presentation:
        document["presentation"] = presentation
    return document


def _write(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(
    tmp_path: Path,
    *,
    sources: list[dict[str, object]],
    config: dict[str, object] | None = None,
    required_capabilities: list[str] | None = None,
    output_directory: str = "out",
) -> review_workbench.ComponentRequest:
    payload: dict[str, object] = {
        "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
        "request_id": "req-0",
        "component_id": review_workbench.COMPONENT_ID,
        "sources": sources,
        "config": config or {},
        "output_directory": output_directory,
    }
    if required_capabilities is not None:
        payload["required_capabilities"] = required_capabilities
    return component_request_from_dict(payload)


def test_descriptor_round_trips_through_the_v1_schema() -> None:
    """The shipped descriptor validates against component-descriptor.v1."""
    document = review_workbench.descriptor_document()

    descriptor = component_descriptor_from_dict(document)

    assert descriptor.component_id == review_workbench.COMPONENT_ID
    assert descriptor.supported_input_versions == (COMPONENT_REQUEST_SCHEMA_VERSION,)
    assert set(descriptor.required_capabilities) == set(review_workbench.REQUIRED_CAPABILITIES)
    assert set(descriptor.optional_capabilities) == set(review_workbench.OPTIONAL_CAPABILITIES)
    assert "review-workbench.v1" in descriptor.output_types


def test_run_writes_offline_workbench_with_episode_and_provenance_index(tmp_path: Path) -> None:
    """A bundle-only request completes and writes an offline, network-free view."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "complete"
    assert result.diagnostics == ()
    by_id = {artifact["artifact_id"]: artifact for artifact in result.artifacts}
    html = (tmp_path / "out" / "review-workbench.v1.html").read_text(encoding="utf-8")
    document = json.loads((tmp_path / "out" / "review-workbench.v1.json").read_text("utf-8"))
    assert "http://" not in html and "https://" not in html
    assert document["time_base"] == {
        "authority": "simulation_telemetry",
        "video_alignment": "unavailable",
    }
    assert document["episodes"] == [{"bundle": "bundle", "episode_id": "ep-0"}]
    assert document["artifacts"][0]["integrity"] == "unverifiable"
    assert document["verified_artifacts"][0]["integrity"] == "match"
    assert document["verified_artifacts"][0]["admission"] == "not_evaluated"
    assert result.provenance["admission"] == "not_evaluated"
    assert result.provenance["source_integrity"] == {"bundle": "unverifiable"}
    assert result.provenance["verified_artifact_integrity"] == {"trace-0000": "match"}
    for artifact in by_id.values():
        assert (
            artifact["sha256"]
            == hashlib.sha256((tmp_path / "out" / artifact["artifact_id"]).read_bytes()).hexdigest()
        )


def test_canonical_launch_mounts_diagnostic_audit_extension_and_browser_runtime(
    tmp_path: Path,
) -> None:
    """SREV-15 owns the offline launch while mounting the BA-06 fixture controls."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "complete"
    output = tmp_path / "out"
    document = json.loads((output / "review-workbench.v1.json").read_text(encoding="utf-8"))
    extension = document["extensions"]["audit_workbench"]
    assert extension["slot"] == "panels"
    assert extension["mode"] == "diagnostic_fixture"
    assert extension["evidence_status"] == "diagnostic_only"
    assert extension["native"] is False
    assert extension["native_or_live_claim"] is False
    assert extension["model"]["service"]["id"] == "ba06-fixture-facade"
    assert [case["episode_id"] for case in extension["model"]["queue"]] == [
        "fixture-normal-control",
        "fixture-missing-media",
    ]
    assert extension["model"]["queue"][1]["media"]["reason"] == "recording_not_present"

    html = (output / "review-workbench.v1.html").read_text(encoding="utf-8")
    assert "data-extension-slot" in html
    assert "components/audit_workbench/audit_workbench.js" in html
    assert "http://" not in html and "https://" not in html
    expected_assets = {
        "components/audit_workbench/audit_workbench.css",
        "components/audit_workbench/audit_workbench.js",
        "components/review_editor/review_editor.js",
        "components/review_panels/review_panels.js",
    }
    emitted = {artifact["artifact_id"] for artifact in result.artifacts}
    assert expected_assets <= emitted
    assert all((output / artifact).is_file() for artifact in expected_assets)

    runtime = Path(__file__).parent / "render" / "review_workbench_runtime.mjs"
    completed = subprocess.run(
        ["node", str(runtime), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "review_workbench_runtime: ok"


def test_missing_source_is_partial_with_a_reason(tmp_path: Path) -> None:
    """A declared but absent source degrades to partial instead of aborting."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {"artifact_id": "spec", "uri": "absent.json", "format": review_workbench.SPEC_FORMAT},
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "partial"
    assert sorted(diagnostic["reason_code"] for diagnostic in result.diagnostics) == [
        "source_missing"
    ]
    assert (tmp_path / "out" / "review-workbench.v1.json").is_file()


def test_integrity_mismatch_is_reported_without_blocking_or_admitting(tmp_path: Path) -> None:
    """A digest mismatch is provenance, not admission: it is reported and never granted."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path, declared_sha256="0" * 64))
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    document = json.loads((tmp_path / "out" / "review-workbench.v1.json").read_text("utf-8"))
    assert result.status == "partial"
    assert [diagnostic["reason_code"] for diagnostic in result.diagnostics] == [
        "source_integrity_mismatch"
    ]
    assert document["verified_artifacts"][0]["integrity"] == "mismatch"
    assert document["verified_artifacts"][0]["admission"] == "not_evaluated"
    assert result.provenance["verified_artifact_integrity"] == {"trace-0000": "mismatch"}


def test_video_without_timestamp_map_is_unavailable_and_not_guessed(tmp_path: Path) -> None:
    """A video stream without an explicit timestamp map never gets a guessed plan."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(tmp_path / "video.mp4", "not-a-real-video")
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {"artifact_id": "video", "uri": "video.mp4", "format": "video-mp4.v1"},
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "partial"
    assert [diagnostic["reason_code"] for diagnostic in result.diagnostics] == [
        "presentation_timestamp_map_missing"
    ]
    assert not (tmp_path / "out" / "presentation-plan.v1.json").exists()


def test_presentation_plan_records_declared_map_and_preset(tmp_path: Path) -> None:
    """A declared timestamp map yields an explicit plan with the requested preset."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(tmp_path / "video.mp4", "not-a-real-video")
    _write(
        tmp_path / "spec.json",
        _spec(
            width=640,
            height=360,
            fps=25.0,
            speed=0.5,
            cuts=[{"at_s": 1.0}],
            presentation_timestamp_map={"0": 0.0, "1": 0.04},
        ),
    )
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {"artifact_id": "spec", "uri": "spec.json", "format": review_workbench.SPEC_FORMAT},
            {"artifact_id": "video", "uri": "video.mp4", "format": "video-mp4.v1"},
        ],
        config={"preset": "test"},
    )

    result = review_workbench.run(request, base=tmp_path)

    plan = json.loads((tmp_path / "out" / "presentation-plan.v1.json").read_text("utf-8"))
    assert result.status == "complete"
    assert plan["presentation"] == {
        "width": 640,
        "height": 360,
        "fps": 25.0,
        "speed": 0.5,
        "preset": "test",
    }
    assert plan["recorded_edits"] == {"cuts": [{"at_s": 1.0}]}
    assert plan["timestamp_map"] == {"0": 0.0, "1": 0.04}


def test_request_without_a_bundle_is_unavailable(tmp_path: Path) -> None:
    """A request that declares no review bundle cannot produce a workbench."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "spec.json", _spec())
    request = _request(
        tmp_path,
        sources=[
            {"artifact_id": "spec", "uri": "spec.json", "format": review_workbench.SPEC_FORMAT}
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "unavailable"
    assert "no review bundle" in result.reason


def test_unsupported_capability_and_component_are_unavailable(tmp_path: Path) -> None:
    """Unknown capabilities and foreign component ids fail closed."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    capability_request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
        required_capabilities=["ai-narration"],
    )
    foreign_request = component_request_from_dict(
        {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": "req-1",
            "component_id": "srev01-inspect",
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "x.v1"}],
            "output_directory": "out",
        }
    )

    capability_result = review_workbench.run(capability_request, base=tmp_path)
    foreign_result = review_workbench.run(foreign_request, base=tmp_path)

    assert capability_result.status == "unavailable"
    assert "missing capabilities" in capability_result.reason
    assert foreign_result.status == "unavailable"
    assert "unsupported component" in foreign_result.reason


def test_output_collision_fails_closed(tmp_path: Path) -> None:
    """An existing output directory is never overwritten."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("previous run", encoding="utf-8")
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "failed"
    assert "output collision" in result.reason
    assert (tmp_path / "out" / "keep.txt").read_text(encoding="utf-8") == "previous run"
    assert not (tmp_path / "out" / "review-workbench.v1.html").exists()


def test_cli_descriptor_and_complete_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI exposes the descriptor and returns 0 for a complete run."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(
        tmp_path / "request.json",
        {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": "req-cli",
            "component_id": review_workbench.COMPONENT_ID,
            "sources": [
                {
                    "artifact_id": "bundle",
                    "uri": "bundle.json",
                    "format": review_workbench.BUNDLE_FORMAT,
                }
            ],
        },
    )

    assert review_workbench.main(["--descriptor"]) == 0
    assert json.loads(capsys.readouterr().out)["component_id"] == review_workbench.COMPONENT_ID

    rc = review_workbench.main(
        [
            "--input",
            str(tmp_path / "request.json"),
            "--output",
            "cli-out",
            "--base",
            str(tmp_path),
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert printed["status"] == "complete"
    assert (tmp_path / "cli-out" / "review-workbench.v1.html").is_file()


def test_cli_partial_run_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A partial run is reported as such with exit code 2."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(
        tmp_path / "request.json",
        {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": "req-cli-partial",
            "component_id": review_workbench.COMPONENT_ID,
            "sources": [
                {
                    "artifact_id": "bundle",
                    "uri": "bundle.json",
                    "format": review_workbench.BUNDLE_FORMAT,
                },
                {"artifact_id": "gone", "uri": "gone.json", "format": review_workbench.SPEC_FORMAT},
            ],
        },
    )

    rc = review_workbench.main(
        [
            "--input",
            str(tmp_path / "request.json"),
            "--output",
            "cli-out",
            "--base",
            str(tmp_path),
        ]
    )

    assert rc == 2
    assert json.loads(capsys.readouterr().out)["status"] == "partial"


def test_recording_scene_export_reuses_the_canonical_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A required scene export delegates to robot_sf.render.threejs_viewer."""
    from robot_sf.render import threejs_viewer

    calls: list[tuple[Path, Path]] = []

    class _Result:
        def __init__(self, output_dir: Path) -> None:
            self.output_dir = output_dir
            self.html_path = output_dir / "index.html"
            self.scene_path = output_dir / "scene.json"

    def _fake_export(recording_path: Path, output_dir: Path) -> _Result:
        calls.append((Path(recording_path), Path(output_dir)))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "index.html").write_text("<html></html>", encoding="utf-8")
        (output_dir / "scene.json").write_text("{}", encoding="utf-8")
        return _Result(output_dir)

    monkeypatch.setattr(threejs_viewer, "export_threejs_viewer", _fake_export)
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(tmp_path / "recording.jsonl", "")
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {
                "artifact_id": "recording",
                "uri": "recording.jsonl",
                "format": "recording-jsonl.v1",
            },
        ],
        required_capabilities=["threejs-scene"],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "complete"
    assert calls == [(tmp_path / "recording.jsonl", tmp_path / "out" / "threejs" / "recording")]
    exported = {artifact["artifact_id"] for artifact in result.artifacts}
    assert "threejs/recording/index.html" in exported
    assert "threejs/recording/scene.json" in exported


def test_source_outside_the_base_directory_is_refused(tmp_path: Path) -> None:
    """A programmatically built request cannot make the component read outside its base.

    The contract loader already rejects traversal in request URIs, so this exercises the
    component's own containment guard against a request object built directly.
    """
    from robot_sf.analysis_workbench.review_contracts import ComponentRequest, SourceRef

    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    request = ComponentRequest(
        request_id="req-escape",
        component_id=review_workbench.COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="bundle", uri="bundle.json", format=review_workbench.BUNDLE_FORMAT
            ),
            SourceRef(
                artifact_id="escape", uri="../escape.json", format=review_workbench.SPEC_FORMAT
            ),
        ),
        output_directory="out",
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "partial"
    assert [diagnostic["reason_code"] for diagnostic in result.diagnostics] == [
        "source_outside_base"
    ]


def test_invalid_bundle_payload_is_reported_as_a_contract_failure(tmp_path: Path) -> None:
    """A bundle that violates the v1 schema is an invalid source, not a crash."""
    _write(tmp_path / "bundle.json", {"schema_version": "review-bundle.v1", "bundle_id": "b"})
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            }
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "unavailable"
    assert result.diagnostics[0]["reason_code"] == "source_contract_invalid"
    assert not (tmp_path / "out").exists()


def test_invalid_visualization_spec_degrades_to_partial(tmp_path: Path) -> None:
    """A malformed spec is reported while the bundle view is still produced."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(tmp_path / "spec.json", {"schema_version": "visualization-spec.v1"})
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {"artifact_id": "spec", "uri": "spec.json", "format": review_workbench.SPEC_FORMAT},
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    assert result.status == "partial"
    assert [diagnostic["reason_code"] for diagnostic in result.diagnostics] == [
        "source_contract_invalid"
    ]
    assert (tmp_path / "out" / "review-workbench.v1.json").is_file()


def test_recording_without_the_scene_capability_is_not_exported(tmp_path: Path) -> None:
    """A recording source stays inert unless the request requires the scene export."""
    _write(tmp_path / "trace-0000.json", {"step": 0})
    _write(tmp_path / "bundle.json", _bundle(tmp_path))
    _write(tmp_path / "recording.jsonl", "")
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "bundle",
                "uri": "bundle.json",
                "format": review_workbench.BUNDLE_FORMAT,
            },
            {"artifact_id": "recording", "uri": "recording.jsonl", "format": "recording-jsonl.v1"},
        ],
    )

    result = review_workbench.run(request, base=tmp_path)

    document = json.loads((tmp_path / "out" / "review-workbench.v1.json").read_text("utf-8"))
    assert result.status == "complete"
    assert document["artifacts"][1]["scene_export"] == "not_requested"
    assert not (tmp_path / "out" / "threejs").exists()


def test_cli_failed_request_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A failed request is reported with exit code 1."""
    (tmp_path / "out").mkdir()
    _write(
        tmp_path / "request.json",
        {
            "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
            "request_id": "req-cli-failed",
            "component_id": review_workbench.COMPONENT_ID,
            "sources": [{"artifact_id": "a", "uri": "a.json", "format": "x.v1"}],
        },
    )

    rc = review_workbench.main(
        ["--input", str(tmp_path / "request.json"), "--output", "out", "--base", str(tmp_path)]
    )

    assert rc == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
