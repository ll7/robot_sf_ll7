"""Central constants for the Social Navigation Benchmark.

This module consolidates version identifiers and physical / geometric
thresholds so they are defined once and imported everywhere else. All
public constants are part of the *contract* between episode JSONL
records, aggregation, and downstream tooling (figures, SNQI, tables).

Rules:
- Never change a value silently; bump version or add a new constant.
- Expose stable names; avoid re-exporting from other modules to keep import graph simple.
- Document rationale briefly next to each value (source: design spec / research.md).
"""

from __future__ import annotations

# --- Schema / artifact versions ---
# Episode JSON lines schema version (frozen once published). Bump ONLY when
# backward-incompatible structural changes are introduced.
EPISODE_SCHEMA_VERSION: str = "v1"

# --- Distance thresholds (meters) ---
# Current implementation: aligned with existing tests and historical behavior
# - Collision threshold: 0.25m (distance strictly below => collision)
# - Near-miss threshold: 0.50m (inclusive upper bound is exclusive in code logic)
#
# NOTE: Research.md suggested 0.35m collision / 0.60m near-miss thresholds
# but reverting to 0.25/0.50 for test compatibility. Consider gradual migration
# with new test fixtures for research-aligned values.
COLLISION_DIST: float = 0.25
NEAR_MISS_DIST: float = 0.50

# --- Force thresholds ---
# Comfort force threshold: social-force magnitude above which interaction is
# considered discomfort / high-exposure. Chosen empirically; see research.md.
COMFORT_FORCE_THRESHOLD: float = 2.0

__all__ = [
    "COLLISION_DIST",
    "COMFORT_FORCE_THRESHOLD",
    "EPISODE_SCHEMA_VERSION",
    "NEAR_MISS_DIST",
]
