"""Focused tests for the SREV-09 review-scene component (issue #9278)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from robot_sf.analysis_workbench.review_contracts import component_request_from_dict
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
    assert [row["step"] for row in source_map["frames"]] == [0, 1, 2]
    assert [row["time_s"] for row in source_map["frames"]] == [0.0, 0.5, 1.0]
    assert source_map["frames"][0]["robot"]["position_m"] == [0.0, 0.0]
    assert source_map["frames"][0]["robot"]["radius_defaulted"] is True
    assert source_map["sourcemap_sha256"] == result.provenance["sourcemap_sha256"]
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


def test_repeated_runs_compare_equal_source_map_bytes(tmp_path: Path) -> None:
    """Deterministic fixture runs must produce byte-identical source maps."""
    first, first_dir = _run_request(_request_doc(), tmp_path, output="run-a")
    second, second_dir = _run_request(_request_doc(), tmp_path, output="run-b")

    assert first.status == "complete", first.reason
    assert second.status == "complete", second.reason
    assert (first_dir / "frame-source-map.json").read_bytes() == (
        second_dir / "frame-source-map.json"
    ).read_bytes()


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
    assert '"status": "complete"' in captured.out
    assert (out_dir / "frame-source-map.json").is_file()


def test_descriptor_matches_contract() -> None:
    """The shipped descriptor must declare this component's exact surface."""
    doc = descriptor()

    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert doc["required_capabilities"] == []
    assert doc["output_types"] == ["review-scene.v1"]
