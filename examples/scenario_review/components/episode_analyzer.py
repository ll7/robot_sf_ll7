"""Example episode analyzer component (SREV-29 fixture, issue #9290).

Reads one tiny fixture episode-trace JSON artifact and writes a deterministic
summary. Offline, dependency-free, and strictly bounded: it reads a single
declared source, writes only the requested output directory, and reports
measured content (never requested config) in its artifacts.

Evidence boundary: diagnostic fixture only. Summaries describe the checked-in
fixture input; they establish no benchmark, population, or causal claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ComponentResult,
)

COMPONENT_ID = "srev29-example-analyzer"
COMPONENT_VERSION = "1.0.0"

DESCRIPTOR: dict[str, Any] = {
    "schema_version": "component-descriptor.v1",
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": ["component-request.v1"],
    "required_capabilities": ["bounded-execution"],
    "optional_capabilities": [],
    "output_types": ["analyzer-summary.v1"],
}


def _read_source(root: Path, uri: str) -> Any:
    candidate = root / uri
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source artifact {uri!r}: {error}") from error


def _episode_summary(episode: Any) -> dict[str, Any]:
    if not isinstance(episode, dict):
        raise ValueError("episode entries must be mappings")
    episode_id = episode.get("episode_id", "")
    steps = episode.get("steps", [])
    if not isinstance(steps, list) or not all(
        isinstance(point, list) and len(point) == 2 for point in steps
    ):
        raise ValueError(f"episode {episode_id!r} steps must be [x, y] pairs")
    points = [(float(point[0]), float(point[1])) for point in steps]
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"episode {episode_id!r} steps must be finite")
    displacement = math.dist(points[0], points[-1]) if len(points) >= 2 else 0.0
    segment_lengths = [
        math.dist(points[index], points[index + 1]) for index in range(len(points) - 1)
    ]
    mean_speed = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0
    return {
        "episode_id": str(episode_id),
        "step_count": len(points),
        "displacement_m": displacement,
        "mean_step_length_m": mean_speed,
    }


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Summarize one fixture episode-trace artifact.

    Args:
        request: Validated component request naming one JSON source.
        base: Base directory sources resolve under and output writes into.

    Returns:
        Complete result with the summary artifact, or failed/unavailable with
        a stable reason. Never raises for contract-level problems.
    """
    root = base if base is not None else Path.cwd()
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )
    missing = [name for name in request.required_capabilities if name != "bounded-execution"]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(missing))}",
        )
    if len(request.sources) != 1:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="analyzer requires exactly one source artifact",
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
        document = _read_source(root, request.sources[0].uri)
        episodes = document["episodes"] if isinstance(document, dict) else None
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("source must carry a non-empty 'episodes' list")
        summaries = [_episode_summary(episode) for episode in episodes]
        summary = {
            "schema_version": "analyzer-summary.v1",
            "request_id": request.request_id,
            "source_artifact_id": request.sources[0].artifact_id,
            "episode_count": len(summaries),
            "episodes": summaries,
        }
        digest = _write_json(output_dir / "analyzer-summary.json", summary)
    except (ValueError, KeyError, TypeError) as error:
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
                "artifact_id": "analyzer-summary.json",
                "uri": str(prefix / "analyzer-summary.json"),
                "sha256": digest,
            },
        ),
        provenance={"source_artifact_id": request.sources[0].artifact_id},
    )
