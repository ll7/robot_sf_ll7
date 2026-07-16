"""Static scenario/planner gallery generation.

Aggregates existing scenario-thumbnail rendering and scenario-matrix metadata
into a single self-contained static HTML page. See issue #5796.

This package produces a discoverability/inspection artifact, not benchmark
evidence: the per-card "expected runtime" is a deterministic order-of-magnitude
estimate, and "supported planners" is a documented constant set of canonical
planner names, not a measured capability per scenario.
"""

from __future__ import annotations

from robot_sf.gallery.builder import (
    GALLERY_HTML_SCHEMA_VERSION,
    GalleryBuildResult,
    GalleryCard,
    build_gallery,
    estimate_expected_runtime_seconds,
    resolve_supported_planners,
)

__all__ = [
    "GALLERY_HTML_SCHEMA_VERSION",
    "GalleryBuildResult",
    "GalleryCard",
    "build_gallery",
    "estimate_expected_runtime_seconds",
    "resolve_supported_planners",
]
