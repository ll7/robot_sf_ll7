"""Map verification module for validating SVG maps.

This module provides tools to verify that all SVG maps in the repository
work correctly and meet quality standards. It includes:

- Map inventory management and filtering
- Geometric and metadata validation rules
- Environment instantiation testing
- Performance tracking and reporting
- Structured JSON manifest output

Key Components:
    - MapRecord: Representation of an SVG map with metadata
    - VerificationResult: Outcome of validating a single map
    - VerificationRunSummary: Aggregated results for a validation run

Example:
    >>> from robot_sf.maps.verification import run_verification
    >>> results = run_verification(scope='all', mode='local')
"""

__all__ = [
    "MapRecord",
    "VerificationResult",
    "VerificationRunSummary",
]

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MapRecord:
    """Logical representation of an SVG map with metadata.
    
    Attributes:
        map_id: Unique identifier (typically filename without extension)
        file_path: Path to the SVG file
        tags: Set of classification tags (e.g., 'classic', 'pedestrian_only')
        ci_enabled: Whether this map should be verified in CI
        metadata: Dict containing spawn zones, goals, and other map-specific data
        last_modified: Last modification timestamp
    """
    map_id: str
    file_path: Path
    tags: set[str]
    ci_enabled: bool
    metadata: dict
    last_modified: datetime


@dataclass
class VerificationResult:
    """Structured outcome for a single map validation.
    
    Attributes:
        map_id: Reference to the validated map
        status: One of 'pass', 'fail', or 'warn'
        rule_ids: List of rule identifiers that were checked or violated
        duration_ms: Time taken to verify this map in milliseconds
        factory_used: Which factory was used ('robot' or 'pedestrian')
        message: Human-readable summary of the result
        timestamp: When this verification completed
    """
    map_id: str
    status: str  # 'pass' | 'fail' | 'warn'
    rule_ids: list[str]
    duration_ms: float
    factory_used: str  # 'robot' | 'pedestrian'
    message: str
    timestamp: datetime


@dataclass
class VerificationRunSummary:
    """Aggregated results for a complete verification run.
    
    Attributes:
        run_id: Unique identifier for this run
        git_sha: Git commit SHA when run was executed
        total_maps: Total number of maps analyzed
        passed: Count of maps that passed all checks
        failed: Count of maps that failed checks
        warned: Count of maps with warnings
        slow_maps: List of map IDs that exceeded performance budgets
        artifact_path: Path where detailed results were written
        started_at: Run start timestamp
        finished_at: Run completion timestamp
    """
    run_id: str
    git_sha: str
    total_maps: int
    passed: int
    failed: int
    warned: int
    slow_maps: list[str]
    artifact_path: str
    started_at: datetime
    finished_at: datetime
