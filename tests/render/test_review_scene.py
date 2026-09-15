"""Focused tests for the SREV-09 review-scene component (issue #9278)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    SourceRef,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.render import review_scene
from robot_sf.render.review_scene import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    descriptor,
    run,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "scenario_review" / "review_scene"


def _scene(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "frame_indices": [0, 1, 2],
        "formats": ["svg", "png", "pdf"],
        "figure": {"width_in": 3.2, "height_in": 2.4, "dpi": 80},
    }
    doc.update(overrides)
    return doc


def _request_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "component-request.v1",
        "request_id": "srev09-test",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "trace-0002",
                "uri": "trace-export.json",
                "format": "simulation_trace_export.v1",
            }
        ],
        "config": {"scene": _scene()},
        "output_directory": "out",
    }
    doc.update(overrides)
    return doc


def _typed_request(
    doc: dict[str, Any], *, output_directory: str | None = None, uri: str | None = None
) -> ComponentRequest:
    """Build a direct API request for path and malformed-input coverage."""
    sources = []
    for index, source in enumerate(doc["sources"]):
        source = dict(source)
        if index == 0 and uri is not None:
            source["uri"] = uri
        sources.append(
            SourceRef(
                artifact_id=source["artifact_id"],
                uri=source["uri"],
                format=source["format"],
            )
        )
    return ComponentRequest(
        request_id=doc["request_id"],
        component_id=doc["component_id"],
        sources=tuple(sources),
        output_directory=output_directory or doc["output_directory"],
        config=dict(doc.get("config", {})),
        required_capabilities=tuple(doc.get("required_capabilities", [])),
    )


def _run_request(doc: dict[str, Any], tmp_path: Path, output: str = "out") -> tuple[Any, Path]:
    request = component_request_from_dict({**doc, "output_directory": output})
    result = run(request, base=tmp_path, source_base=FIXTURES)
    return result, tmp_path / output


def test_fixture_smoke_renders_numbered_scenes_and_source_map(tmp_path: Path) -> None:
    """The checked-in fixture must yield scene files plus a truthful map."""
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    request = component_request_from_dict({**payload, "output_directory": "smoke"})
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "complete", result.reason
    assert [artifact["artifact_id"] for artifact in result.artifacts] == [
        "trace-0002-scene_000000.svg",
        "trace-0002-scene_000000.png",
        "trace-0002-scene_000000.pdf",
        "trace-0002-scene_000001.svg",
        "trace-0002-scene_000001.png",
        "trace-0002-scene_000001.pdf",
        "trace-0002-scene_000002.svg",
        "trace-0002-scene_000002.png",
        "trace-0002-scene_000002.pdf",
        "frame-source-map.json",
        "component-descriptor.json",
    ]
    source_map = json.loads((tmp_path / "smoke" / "frame-source-map.json").read_text())
    assert source_map["evidence_boundary"] == "analysis_workbench_only"
    assert source_map["diagnostic_only"] is True
    assert source_map["admission"] == "not_evaluated"
    assert source_map["claim_boundary"] == "diagnostic_only; not benchmark evidence"
    assert [row["step"] for row in source_map["frames"]] == [0, 1, 2]
    assert [row["time_s"] for row in source_map["frames"]] == [0.0, 0.5, 1.0]
    assert source_map["frames"][0]["robot"]["position_m"] == [0.0, 0.0]
    assert source_map["frames"][0]["robot"]["radius_defaulted"] is True
    source_record = source_map["provenance"]["source_artifacts"][0]
    assert (
        source_record["observed_sha256"]
        == hashlib.sha256((FIXTURES / "trace-export.json").read_bytes()).hexdigest()
    )
    assert source_record["integrity_status"] == "observed-only"
    assert source_record["identity_status"] == "embedded-observed"
    assert source_record["trace"]["evidence_boundary"] == "analysis_workbench_only"
    assert result.provenance["source_artifacts"] == [source_record]
    assert source_map["sourcemap_sha256"] == result.provenance["sourcemap_sha256"]
    assert result.provenance["evidence_boundary"] == "analysis_workbench_only"
    assert result.provenance["diagnostic_only"] is True
    assert result.provenance["admission"] == "not_evaluated"
    assert result.provenance["claim_boundary"] == "diagnostic_only; not benchmark evidence"
    assert descriptor()["component_id"] == COMPONENT_ID


def test_svg_carries_fixture_coordinates_and_png_has_expected_size(
    tmp_path: Path,
) -> None:
    """Vector output must contain source numbers; raster size follows the preset."""
    result, out_dir = _run_request(_request_doc(), tmp_path)

    assert result.status == "complete", result.reason
    svg = (out_dir / "trace-0002-scene_000001.svg").read_text(encoding="utf-8")
    assert "step 1 t=0.5s" in svg
    assert svg.count("<path") > 10
    with Image.open(out_dir / "trace-0002-scene_000001.png") as image:
        assert image.size == (256, 192)


def test_repeated_runs_compare_equal_artifact_bytes(tmp_path: Path) -> None:
    """Deterministic fixture runs must produce byte-identical artifacts."""
    first, first_dir = _run_request(_request_doc(), tmp_path, output="run-a")
    second, second_dir = _run_request(_request_doc(), tmp_path, output="run-b")

    assert first.status == "complete", first.reason
    assert second.status == "complete", second.reason
    first_bytes = {
        artifact["artifact_id"]: (first_dir / artifact["artifact_id"]).read_bytes()
        for artifact in first.artifacts
    }
    second_bytes = {
        artifact["artifact_id"]: (second_dir / artifact["artifact_id"]).read_bytes()
        for artifact in second.artifacts
    }
    assert first_bytes == second_bytes
    for artifact_id, content in first_bytes.items():
        assert hashlib.sha256(content).hexdigest() == next(
            item["sha256"] for item in first.artifacts if item["artifact_id"] == artifact_id
        )


def test_wrong_component_is_unavailable(tmp_path: Path) -> None:
    """A request for another component must not produce artifacts."""
    result, _ = _run_request(_request_doc(component_id="srev99-other"), tmp_path)

    assert result.status == "unavailable"
    assert "unsupported component" in result.reason
    assert result.artifacts == ()


def test_missing_capability_is_unavailable(tmp_path: Path) -> None:
    """Required capabilities outside the descriptor must stay unavailable."""
    result, _ = _run_request(_request_doc(required_capabilities=["video-overlay"]), tmp_path)

    assert result.status == "unavailable"
    assert "missing capabilities" in result.reason


def test_incompatible_required_version_is_unavailable(tmp_path: Path) -> None:
    """A newer required major version must not run against this component."""
    doc = _request_doc()
    doc["config"] = {**doc["config"], "required_component_version": "2.0.0"}
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "incompatible-required-version" in result.reason


def test_corrupt_scene_is_failed(tmp_path: Path) -> None:
    """Invalid scene selections must fail closed."""
    for scene in (
        {"frame_indices": [], "formats": ["svg"]},
        {"frame_indices": [0, 0], "formats": ["svg"]},
        {"frame_indices": [0], "formats": ["hologram"]},
        {"frame_indices": [0], "formats": ["svg"], "figure": {"dpi": 0}},
    ):
        doc = _request_doc()
        doc["config"] = {"scene": scene}
        result, _ = _run_request(doc, tmp_path)

        assert result.status == "failed", scene
        assert result.artifacts == ()


def test_out_of_range_frame_index_is_failed(tmp_path: Path) -> None:
    """Frame indices beyond the trace must fail, not render partial scenes."""
    doc = _request_doc()
    doc["config"] = {"scene": _scene(frame_indices=[0, 99])}
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "out of range" in result.reason


def test_empty_sources_are_failed(tmp_path: Path) -> None:
    """A request with no evidence cannot be rendered."""
    from robot_sf.analysis_workbench.review_contracts import ComponentRequest

    request = ComponentRequest(
        request_id="srev09-test",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory="out",
        config={"scene": _scene()},
    )
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "failed"
    assert "missing-evidence" in result.reason


def test_unusable_sources_are_unavailable(tmp_path: Path) -> None:
    """Sources in unsupported formats must not block, but yield no scenes."""
    doc = _request_doc(
        sources=[
            {
                "artifact_id": "video-0000",
                "uri": "trace-export.json",
                "format": "video/mp4",
            }
        ]
    )
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "unsupported-evidence" in result.reason
    assert result.artifacts == ()


def test_extra_skipped_sources_keep_partial_status(tmp_path: Path) -> None:
    """Usable traces still render; skipped extra sources are diagnosed."""
    doc = _request_doc(
        sources=[
            {
                "artifact_id": "trace-0002",
                "uri": "trace-export.json",
                "format": "simulation_trace_export.v1",
            },
            {
                "artifact_id": "video-0000",
                "uri": "trace-export.json",
                "format": "video/mp4",
            },
        ]
    )
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "partial", result.reason
    assert (out_dir / "frame-source-map.json").is_file()
    assert [item["artifact_id"] for item in result.diagnostics] == ["video-0000"]


def test_rendering_is_hermetic_against_polluted_rcparams(tmp_path: Path) -> None:
    """Foreign global style mutations must not change canvas geometry.

    Other suites set savefig.bbox=tight globally (e.g. the latex style
    helper); under xdist worker reuse those mutations leak into this
    component. Rendering pins its savefig contract explicitly.
    """
    import matplotlib

    with matplotlib.rc_context(
        {
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
        }
    ):
        result, out_dir = _run_request(_request_doc(), tmp_path)

    assert result.status == "complete", result.reason
    with Image.open(out_dir / "trace-0002-scene_000001.png") as image:
        assert image.size == (256, 192)


def test_output_collision_is_failed(tmp_path: Path) -> None:
    """Scenes must never silently overwrite an existing directory."""
    (tmp_path / "out").mkdir()
    result, _ = _run_request(_request_doc(), tmp_path)

    assert result.status == "failed"
    assert "output-collision" in result.reason


def test_absolute_source_uri_is_unavailable(tmp_path: Path) -> None:
    """The direct API must reject absolute source URIs even inside the base."""
    request = _typed_request(_request_doc(), uri=str(FIXTURES / "trace-export.json"))
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "unavailable"
    assert result.diagnostics[0]["reason"] == "unsafe-source-uri"


def test_absolute_output_directory_is_failed(tmp_path: Path) -> None:
    """The direct API must reject absolute output directories."""
    request = _typed_request(_request_doc(), output_directory=str(tmp_path / "absolute-output"))
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "failed"
    assert "unsafe-output-path" in result.reason


def test_source_symlink_escape_is_unavailable(tmp_path: Path) -> None:
    """A source symlink resolving outside source_base must not be consumed."""
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    outside_source = tmp_path / "outside-trace.json"
    outside_source.write_bytes((FIXTURES / "trace-export.json").read_bytes())
    (source_root / "trace.json").symlink_to(outside_source)
    request = _typed_request(_request_doc(), uri="trace.json")

    result = run(request, base=tmp_path / "output-root", source_base=source_root)

    assert result.status == "unavailable"
    assert result.diagnostics[0]["reason"] == "unsafe-source-uri"


def test_direct_artifact_id_escape_is_failed_before_staging(tmp_path: Path) -> None:
    """The typed API must not interpolate an unsafe source ID into a path."""
    doc = _request_doc()
    doc["sources"] = [{**doc["sources"][0], "artifact_id": "../escaped"}]
    result = run(_typed_request(doc), base=tmp_path, source_base=FIXTURES)

    assert result.status == "failed"
    assert "safe filename component" in result.reason
    assert not (tmp_path / "escaped-scene_000000.svg").exists()


def test_special_file_source_is_rejected_without_blocking(tmp_path: Path) -> None:
    """A FIFO source URI must return a diagnostic instead of blocking read_bytes."""
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    fifo = source_root / "trace.fifo"
    os.mkfifo(fifo)
    request = _typed_request(_request_doc(), uri="trace.fifo")
    results: list[Any] = []

    worker = threading.Thread(
        target=lambda: results.append(
            run(request, base=tmp_path / "output", source_base=source_root)
        ),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive(), "special-file source read blocked the API"
    assert results[0].status == "unavailable"
    assert "source must be a regular file" in results[0].diagnostics[0]["reason"]


def test_oversized_source_is_unavailable(tmp_path: Path) -> None:
    """A sparse source beyond the byte budget must not be read or parsed."""
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    source_path = source_root / "trace.json"
    source_path.write_bytes(b"{}")
    os.truncate(source_path, review_scene.MAX_SOURCE_BYTES + 1)
    request = _typed_request(_request_doc(), uri="trace.json")

    result = run(request, base=tmp_path / "output", source_base=source_root)

    assert result.status == "unavailable"
    assert "source exceeds maximum size" in result.diagnostics[0]["reason"]


def test_nested_frame_indices_return_schema_valid_failure(tmp_path: Path, capsys: Any) -> None:
    """Nested frame-index values must fail through both API and CLI envelopes."""
    doc = _request_doc()
    doc["config"] = {"scene": _scene(frame_indices=[[0]])}
    request = component_request_from_dict(doc)
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "failed"
    assert "frame indices must be non-negative integers" in result.reason
    component_result_from_dict(review_scene._result_payload(result))

    request_path = tmp_path / "nested-request.json"
    request_path.write_text(json.dumps(doc), encoding="utf-8")
    exit_code = review_scene.main(
        ["--input", str(request_path), "--output", "nested-output", "--base", str(tmp_path)]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    component_result_from_dict(payload)
    assert payload["status"] == "failed"
    assert "frame indices must be non-negative integers" in payload["reason"]


def test_resource_limits_fail_closed(tmp_path: Path) -> None:
    """Figure and frame budgets must reject oversized diagnostic requests."""
    oversized_figure = _request_doc()
    oversized_figure["config"] = {
        "scene": _scene(figure={"width_in": review_scene.MAX_FIGURE_WIDTH_IN + 1})
    }
    figure_result, _ = _run_request(oversized_figure, tmp_path, output="oversized-figure")
    assert figure_result.status == "failed"
    assert "resource-limit" in figure_result.reason

    oversized_frames = _request_doc()
    oversized_frames["config"] = {
        "scene": _scene(frame_indices=list(range(review_scene.MAX_RENDER_FRAMES + 1)))
    }
    frames_result, _ = _run_request(oversized_frames, tmp_path, output="oversized-frames")
    assert frames_result.status == "failed"
    assert "resource-limit" in frames_result.reason


def test_output_symlink_escape_is_failed(tmp_path: Path) -> None:
    """An output path through a symlink outside base must fail closed."""
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (base / "link").symlink_to(outside, target_is_directory=True)

    result = run(
        _typed_request(_request_doc(), output_directory="link/scenes"),
        base=base,
        source_base=FIXTURES,
    )

    assert result.status == "failed"
    assert "unsafe-output-path" in result.reason
    assert not (outside / "scenes").exists()


def test_cli_smoke_reports_complete(tmp_path: Path, capsys: Any) -> None:
    """The standalone CLI must wire input/config/output through run()."""
    out_dir = tmp_path / "cli-out"
    exit_code = review_scene.main(
        [
            "--input",
            str(FIXTURES / "request.json"),
            "--config",
            str(FIXTURES / "config.json"),
            "--output",
            "cli-out",
            "--base",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["schema_version"] == "component-result.v1"
    component_result_from_dict(payload)
    assert payload["status"] == "complete"
    assert (out_dir / "frame-source-map.json").is_file()


def test_cli_malformed_documents_return_schema_valid_failure(tmp_path: Path, capsys: Any) -> None:
    """Malformed request/config JSON must not escape as a traceback."""
    bad_request = tmp_path / "bad-request.json"
    bad_request.write_text("[]", encoding="utf-8")
    exit_code = review_scene.main(
        ["--input", str(bad_request), "--output", "bad-output", "--base", str(tmp_path)]
    )

    assert exit_code == 1
    request_result = json.loads(capsys.readouterr().out)
    component_result_from_dict(request_result)
    assert request_result["status"] == "failed"
    assert "JSON object" in request_result["reason"]

    config = tmp_path / "bad-config.json"
    config.write_text("[]", encoding="utf-8")
    exit_code = review_scene.main(
        [
            "--input",
            str(FIXTURES / "request.json"),
            "--config",
            str(config),
            "--output",
            "bad-config-output",
            "--base",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    config_result = json.loads(capsys.readouterr().out)
    component_result_from_dict(config_result)
    assert config_result["status"] == "failed"
    assert "config document must be a JSON object" in config_result["reason"]


def test_api_malformed_request_returns_schema_valid_failure(tmp_path: Path) -> None:
    """The API boundary must normalize non-object input to a failed result."""
    result = run(["not", "a", "request"], base=tmp_path)

    assert result.status == "failed"
    component_result_from_dict(review_scene._result_payload(result))
    assert "expected a component request object" in result.reason


def test_descriptor_matches_contract() -> None:
    """The shipped descriptor must declare this component's exact surface."""
    doc = descriptor()

    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert doc["required_capabilities"] == []
    assert doc["output_types"] == ["review-scene.v1"]
