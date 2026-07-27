"""Contract tests for the OSM-SVG import adapter.

These tests lock the orchestration boundary in
:mod:`robot_sf.maps.import_svg_from_osm` without touching real SVG/OSM files.
Every ``pysocialforce.map_osm_converter`` entry point the adapter binds is
replaced with a mock, so the suite exercises only argument forwarding, the
extract -> scale-bar -> save call ordering, and the logging sequence.

The conversion fidelity of ``pysocialforce`` itself is explicitly out of scope
here; this is adapter contract coverage only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest
from loguru import logger

from robot_sf.maps.import_svg_from_osm import import_svg_from_osm

if TYPE_CHECKING:
    from pathlib import Path

_ADAPTER_MODULE = "robot_sf.maps.import_svg_from_osm"


@pytest.fixture
def converter_pipeline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the adapter's converter bindings with an ordered parent mock.

    The adapter imports the three ``pysocialforce.map_osm_converter`` functions
    into its own module namespace, so patching them there mocks the boundary the
    adapter actually calls. A shared parent mock preserves cross-mock call order.
    """
    pipeline = MagicMock(name="osm_converter_pipeline")
    monkeypatch.setattr(
        f"{_ADAPTER_MODULE}.extract_buildings_as_obstacle",
        pipeline.extract_buildings_as_obstacle,
    )
    monkeypatch.setattr(
        f"{_ADAPTER_MODULE}.add_scale_bar_to_root",
        pipeline.add_scale_bar_to_root,
    )
    monkeypatch.setattr(
        f"{_ADAPTER_MODULE}.save_root_as_svg",
        pipeline.save_root_as_svg,
    )
    return pipeline


def _capture_info_logs() -> tuple[list[str], int]:
    """Attach a throwaway loguru sink that records rendered INFO messages."""
    captured: list[str] = []
    handler_id = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="INFO",
    )
    return captured, handler_id


def test_pipeline_forwards_arguments_and_preserves_order(
    converter_pipeline: MagicMock,
    tmp_path: Path,
) -> None:
    """Extraction, scale-bar decoration, and save must run in order with exact args.

    The input path and map scale reach extraction; the extracted root flows into
    scale-bar decoration with the adapter's fixed bar length; the decorated root
    and output path flow into save.
    """
    input_path = str(tmp_path / "input.osm.svg")
    output_path = str(tmp_path / "output.svg")
    extracted_root = MagicMock(name="extracted_root")
    decorated_root = MagicMock(name="decorated_root")
    converter_pipeline.extract_buildings_as_obstacle.return_value = extracted_root
    converter_pipeline.add_scale_bar_to_root.return_value = decorated_root

    import_svg_from_osm(input_path, output_path, map_scale_factor=7.5)

    # One ordered assertion locks call sequence, argument forwarding, and the
    # data flow (extracted root -> decorated root) between the three boundaries.
    assert converter_pipeline.mock_calls == [
        call.extract_buildings_as_obstacle(input_path, map_scale_factor=7.5),
        call.add_scale_bar_to_root(extracted_root, line_length=50),
        call.save_root_as_svg(decorated_root, output_path),
    ]
    converter_pipeline.extract_buildings_as_obstacle.assert_called_once()
    converter_pipeline.add_scale_bar_to_root.assert_called_once()
    converter_pipeline.save_root_as_svg.assert_called_once()


def test_pipeline_logs_extraction_start_then_save_finish(
    converter_pipeline: MagicMock,
    tmp_path: Path,
) -> None:
    """The adapter must log an extraction-start line followed by a save-finish line."""
    input_path = str(tmp_path / "input.osm.svg")
    output_path = str(tmp_path / "output.svg")
    captured, handler_id = _capture_info_logs()
    try:
        import_svg_from_osm(input_path, output_path, map_scale_factor=2.0)
    finally:
        logger.remove(handler_id)

    start_lines = [m for m in captured if m.startswith("Extracting buildings from")]
    finish_lines = [m for m in captured if m.startswith("Saved extracted building obstacles to")]
    assert start_lines, f"missing extraction start log; captured={captured!r}"
    assert finish_lines, f"missing save finish log; captured={captured!r}"

    assert input_path in start_lines[0]
    assert "2.0" in start_lines[0]
    assert output_path in finish_lines[0]
    # The start log must strictly precede the finish log.
    assert captured.index(start_lines[0]) < captured.index(finish_lines[0])


@pytest.mark.parametrize(
    "failing_step",
    [
        "extract_buildings_as_obstacle",
        "add_scale_bar_to_root",
        "save_root_as_svg",
    ],
)
def test_converter_failure_propagates_without_false_success(
    converter_pipeline: MagicMock,
    tmp_path: Path,
    failing_step: str,
) -> None:
    """A converter error must propagate with no false success log or save write.

    When extraction or scale-bar decoration fails, the adapter must never reach
    the save boundary (no compensating file write). When save itself fails, it is
    reached exactly once. In every failure path the success log must stay absent.
    """
    getattr(converter_pipeline, failing_step).side_effect = RuntimeError("converter exploded")
    input_path = str(tmp_path / "input.osm.svg")
    output_path = str(tmp_path / "output.svg")
    captured, handler_id = _capture_info_logs()
    try:
        with pytest.raises(RuntimeError, match="converter exploded"):
            import_svg_from_osm(input_path, output_path, map_scale_factor=1.0)
    finally:
        logger.remove(handler_id)

    if failing_step == "save_root_as_svg":
        # Save is the failing boundary, so it is reached exactly once.
        converter_pipeline.save_root_as_svg.assert_called_once()
    else:
        # An upstream failure must short-circuit before any save (file write).
        converter_pipeline.save_root_as_svg.assert_not_called()

    # No false success log on any failure path.
    assert not any("Saved extracted building obstacles to" in message for message in captured)
