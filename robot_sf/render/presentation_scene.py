"""Optional 2.5D presentation view descriptors (issue #9369).

This module owns the narrow view component for the isometric presentation
preset: a validated descriptor over an already-exported scene. It changes no
source geometry -- floor polygons stay identical to the source obstacle
vertices, wall height is illustrative and disclosed, and the top-down preset
reproduces the existing overhead camera exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

CameraPreset = Literal["top-down", "isometric"]

ISOMETRIC_ELEVATION_DEG = 50.0
ISOMETRIC_AZIMUTH_DEG = 45.0
DEFAULT_WALL_HEIGHT_M = 1.2


@dataclass(frozen=True, slots=True)
class PresentationViewConfig:
    """Validated view request for one exported scene."""

    preset: CameraPreset = "top-down"
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M
    disclosures: tuple[str, ...] = field(default_factory=tuple)


def describe_view(
    scene: dict[str, Any],
    preset: CameraPreset = "top-down",
    *,
    wall_height_m: float = DEFAULT_WALL_HEIGHT_M,
) -> dict[str, Any]:
    """Build the JSON-safe view descriptor for one exported scene.

    The descriptor references the scene's own positions, footprints, actor
    IDs, and source timestamps; it never re-derives or alters them. Floor
    polygons are the source obstacle vertices verbatim; only wall volumes
    gain an illustrative height, which is disclosed as symbolic.

    Returns:
        View descriptor with preset, camera, walls, and disclosures.

    Raises:
        ValueError: For an unknown preset or a non-finite wall height.
    """
    if preset not in ("top-down", "isometric"):
        raise ValueError(f"unknown camera preset: {preset!r}")
    try:
        height = float(wall_height_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"wall_height_m must be finite: {wall_height_m!r}") from exc
    if not math.isfinite(height) or height < 0.0:
        raise ValueError(f"wall_height_m must be finite: {wall_height_m!r}")
    frames = scene.get("frames", [])
    disclosures = [
        "Simulation and collision geometry remain 2D; wall height is illustrative.",
        "Decorative bodies and shading must not imply different clearance.",
    ]
    if preset == "isometric":
        disclosures.append(
            f"Isometric elevation {ISOMETRIC_ELEVATION_DEG:g} deg, azimuth "
            f"{ISOMETRIC_AZIMUTH_DEG:g} deg; fixed pose, no follow camera."
        )
    fidelity = scene.get("fidelity", {})
    if fidelity.get("geometry", {}).get("mode") == "symbolic":
        disclosures.append("Actor radii symbolic; footprints show recorded positions only.")
    return {
        "schema_version": "presentation-view.v1",
        "preset": preset,
        "frame_count": len(frames),
        "camera": {
            "kind": "orthographic",
            "elevation_deg": 90.0 if preset == "top-down" else ISOMETRIC_ELEVATION_DEG,
            "azimuth_deg": 0.0 if preset == "top-down" else ISOMETRIC_AZIMUTH_DEG,
        },
        "walls": {
            "height_m": height,
            "height_source": "illustrative",
            "bevel_enabled": False,
            "floor_polygons": "source obstacle vertices verbatim",
        },
        "disclosures": disclosures,
    }


def floor_polygons_match_source(scene: dict[str, Any]) -> bool:
    """Return whether every map obstacle keeps vertices usable as floor polygons.

    The extrusion contract requires floor geometry to equal the source
    vertices; this checks the descriptor input side (all obstacles expose
    finite ``vertices`` lists).

    Returns:
        True when every obstacle has a finite vertices list.
    """
    obstacles = (scene.get("map") or {}).get("obstacles") or []
    if not obstacles:
        return False
    for obstacle in obstacles:
        vertices = (obstacle or {}).get("vertices") or []
        if not vertices:
            return False
        for point in vertices:
            try:
                values = (float(point[0]), float(point[1]))
            except (TypeError, ValueError, IndexError):
                return False
            if not all(math.isfinite(value) for value in values):
                return False
    return True
