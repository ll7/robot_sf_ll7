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
EPISODE_SCHEMA_VERSION: str = "v1"

# --- Distance thresholds (meters) ---
# Rationale: thresholds aligned with historical tests (see tests/test_metrics.py)
# Collision: strict distance < 0.25m counts as collision.
# Near-miss: distance in [0.25m, 0.5m) counts as near-miss.
COLLISION_DIST: float = 0.25  # distance strictly below => collision event increment
NEAR_MISS_DIST: float = 0.50  # distance in [COLLISION_DIST, NEAR_MISS_DIST) => near-miss

# --- Force thresholds ---
# Comfort force threshold: social-force magnitude above which interaction is
# considered discomfort / high-exposure. Initial placeholder; to be refined.
COMFORT_FORCE_THRESHOLD: float = 2.0

__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "COLLISION_DIST",
    "NEAR_MISS_DIST",
    "COMFORT_FORCE_THRESHOLD",
]
