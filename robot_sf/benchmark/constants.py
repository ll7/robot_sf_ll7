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
# Research decisions (see specs/120-social-navigation-benchmark-plan/research.md):
# - Collision threshold: 0.35m (distance strictly below => collision)
# - Near-miss threshold: 0.60m (inclusive upper bound is exclusive in code logic)
# Any change MUST be accompanied by design doc + version bump if it alters
# interpretation of historical data.
COLLISION_DIST: float = 0.35
NEAR_MISS_DIST: float = 0.60

# --- Force thresholds ---
# Comfort force threshold: social-force magnitude above which interaction is
# considered discomfort / high-exposure. Chosen empirically; see research.md.
COMFORT_FORCE_THRESHOLD: float = 2.0

__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "COLLISION_DIST",
    "NEAR_MISS_DIST",
    "COMFORT_FORCE_THRESHOLD",
]
