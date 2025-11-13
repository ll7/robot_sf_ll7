"""
Common utilities for robot_sf package.

This module provides shared type definitions, error handling,
seed management, and compatibility shims.
"""

# Re-export commonly used symbols
from robot_sf.common.artifact_paths import (
    ARTIFACT_CATEGORIES,
    ArtifactCategory,
    ensure_canonical_tree,
    find_legacy_artifact_paths,
    get_artifact_category,
    get_artifact_category_path,
    get_artifact_override_root,
    get_artifact_root,
    get_legacy_migration_plan,
    get_repository_root,
    iter_artifact_categories,
    resolve_artifact_path,
)
from robot_sf.common.compat import validate_compatibility
from robot_sf.common.errors import raise_fatal_with_remedy, warn_soft_degrade
from robot_sf.common.seed import SeedReport, set_global_seed
from robot_sf.common.types import (
    Circle2D,
    Line2D,
    MapBounds,
    Point2D,
    Range,
    Range2D,
    RobotPose,
    Vec2D,
    Zone,
)

__all__ = [  # noqa: RUF022 - Grouped by source module for clarity
    # Artifact helpers (from .artifact_paths)
    "ARTIFACT_CATEGORIES",
    "ArtifactCategory",
    "ensure_canonical_tree",
    "find_legacy_artifact_paths",
    "get_legacy_migration_plan",
    "get_artifact_category",
    "get_artifact_category_path",
    "get_artifact_override_root",
    "get_artifact_root",
    "get_repository_root",
    "iter_artifact_categories",
    "resolve_artifact_path",
    # Types (from .types)
    "Circle2D",
    "Line2D",
    "MapBounds",
    "Point2D",
    "Range",
    "Range2D",
    "RobotPose",
    "Vec2D",
    "Zone",
    # Seed management (from .seed)
    "SeedReport",
    "set_global_seed",
    # Errors (from .errors)
    "raise_fatal_with_remedy",
    "warn_soft_degrade",
    # Compatibility (from .compat)
    "validate_compatibility",
]
