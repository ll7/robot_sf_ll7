"""Example telemetry renderer component (SREV-29 fixture, issue #9290).

Renders one tiny fixture episode trace to a small static PNG plus a caption
JSON using Matplotlib (Agg). Offline and deterministic for fixed inputs; the
test preset is 320x180. No video, no interactivity, no AI.

Evidence boundary: diagnostic fixture only. The figure illustrates the
checked-in fixture input; it establishes no benchmark, population, or causal
claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ComponentResult,
)

COMPONENT_ID = "srev29-example-renderer"
COMPONENT_VERSION = "1.0.0"
FIGURE_WIDTH = 320
FIGURE_HEIGHT = 180

DESCRIPTOR: dict[str, Any] = {
    "schema_version": "component-descriptor.v1",
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": ["component-request.v1"],
    "required_capabilities": ["bounded-execution"],
    "optional_capabilities": ["matplotlib-figure"],
    "output_types": ["renderer-caption.v1", "renderer-figure.v1"],
}


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        chunk_length = handle.read(4)
        chunk_type = handle.read(4)
        width = handle.read(4)
        height = handle.read(4)
    if signature != b"\x89PNG\r\n\x1a\n" or chunk_type != b"IHDR" or len(chunk_length) != 4:
        raise ValueError("not a PNG file")
    return int.from_bytes(width, "big"), int.from_bytes(height, "big")


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Render one fixture episode trace to PNG plus caption JSON.

    Args:
        request: Validated component request naming one JSON source.
        base: Base directory sources resolve under and output writes into.

    Returns:
        Complete result with figure/caption artifacts, failed for corrupt
        inputs or collisions, or unavailable when Matplotlib is missing or a
        capability/version is unsupported. Never raises for contract-level
        problems.
    """
    root = base if base is not None else Path.cwd()
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    missing = [
        name
        for name in request.required_capabilities
        if name not in ("bounded-execution", "matplotlib-figure")
    ]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="missing optional dependency: matplotlib is not installed",
        )
    if len(request.sources) != 1:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="renderer requires exactly one source artifact",
        )
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=f"output collision, already exists: {request.output_directory}",
        )
    try:
        document = json.loads((root / request.sources[0].uri).read_text(encoding="utf-8"))
        episodes = document["episodes"] if isinstance(document, dict) else None
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("source must carry a non-empty 'episodes' list")
        output_dir.mkdir(parents=True, exist_ok=False)
        figure = pyplot.figure(figsize=(FIGURE_WIDTH / 100.0, FIGURE_HEIGHT / 100.0), dpi=100)
        axes = figure.add_subplot(1, 1, 1)
        for episode in episodes:
            steps = episode.get("steps", [])
            xs = [float(point[0]) for point in steps]
            ys = [float(point[1]) for point in steps]
            axes.plot(xs, ys, marker="o", markersize=2, label=str(episode.get("episode_id", "")))
        axes.legend(fontsize=6)
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        figure_path = output_dir / "telemetry-figure.png"
        # Pin the savefig bounding box: a worker-global `savefig.bbox=tight`
        # (e.g. from a shared plotting style) would otherwise crop the canvas
        # and change the raster dimensions. `rc_context` restores globals after.
        with matplotlib.rc_context({"savefig.bbox": "standard"}):
            figure.savefig(str(figure_path), dpi=100)
        pyplot.close(figure)
        width, height = _png_dimensions(figure_path)
        figure_bytes = figure_path.read_bytes()
        figure_digest = hashlib.sha256(figure_bytes).hexdigest()
        caption = {
            "schema_version": "renderer-caption.v1",
            "request_id": request.request_id,
            "source_artifact_id": request.sources[0].artifact_id,
            "episode_count": len(episodes),
            "figure": {
                "artifact_id": "telemetry-figure.png",
                "width": width,
                "height": height,
                "sha256": figure_digest,
            },
        }
        caption_digest = _write_json(output_dir / "renderer-caption.json", caption)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason=f"corrupt source artifact: {error}",
        )
    prefix = Path(request.output_directory)
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="complete",
        artifacts=(
            {
                "artifact_id": "renderer-caption.json",
                "uri": str(prefix / "renderer-caption.json"),
                "sha256": caption_digest,
            },
            {
                "artifact_id": "telemetry-figure.png",
                "uri": str(prefix / "telemetry-figure.png"),
                "sha256": figure_digest,
            },
        ),
        provenance={"source_artifact_id": request.sources[0].artifact_id},
    )
