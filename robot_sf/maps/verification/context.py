"""Verification context and result data structures.

This module defines the core data classes for verification workflows:
- VerificationResult: Per-map validation outcome
- VerificationRunSummary: Aggregated run metadata
- VerificationContext: Runtime configuration and artifact routing

Alignment with data-model.md
-----------------------------
These classes implement the entities defined in specs/001-map-verification/data-model.md.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal


class VerificationStatus(str, Enum):
    """Verification outcome status."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class FactoryType(str, Enum):
    """Environment factory type used during instantiation."""

    ROBOT = "robot"
    PEDESTRIAN = "pedestrian"


@dataclass
class VerificationResult:
    """Structured outcome for a single map verification.

    Attributes
    ----------
    map_id : str
        Unique identifier for the map (typically filename without extension)
    status : VerificationStatus
        Overall verification status (pass/fail/warn)
    rule_ids : list[str]
        Identifiers of violated or triggered rules
    duration_ms : float
        Verification duration in milliseconds
    factory_used : FactoryType
        Which factory was used for environment instantiation
    message : str
        Human-readable diagnostic message or remediation hint
    timestamp : datetime
        When this verification completed
    """

    map_id: str
    status: VerificationStatus
    rule_ids: list[str]
    duration_ms: float
    factory_used: FactoryType
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate invariants from data-model.md."""
        # Ensure status aligns with rule_ids
        if self.status != VerificationStatus.PASS and not self.rule_ids:
            raise ValueError(f"Status {self.status} requires non-empty rule_ids")

        # Ensure duration is positive
        if self.duration_ms <= 0:
            raise ValueError(f"duration_ms must be > 0, got {self.duration_ms}")


@dataclass
class VerificationRunSummary:
    """Aggregated verification run metadata and results.

    Attributes
    ----------
    run_id : str
        Unique identifier for this verification run (UUID or timestamp-based)
    git_sha : str | None
        Git commit SHA if available
    total_maps : int
        Total number of maps analyzed
    passed : int
        Count of maps that passed
    failed : int
        Count of maps that failed
    warned : int
        Count of maps with warnings
    slow_maps : list[str]
        Map IDs that exceeded performance budget
    artifact_path : Path | None
        Path where manifest was written
    started_at : datetime
        Run start timestamp
    finished_at : datetime | None
        Run completion timestamp
    results : list[VerificationResult]
        Individual per-map results
    """

    run_id: str
    git_sha: str | None
    total_maps: int
    passed: int
    failed: int
    warned: int
    slow_maps: list[str]
    artifact_path: Path | None
    started_at: datetime
    finished_at: datetime | None = None
    results: list[VerificationResult] = field(default_factory=list)

    def __post_init__(self):
        """Validate invariants from data-model.md."""
        # Ensure counts add up
        if self.passed + self.failed + self.warned != self.total_maps:
            raise ValueError(
                f"passed ({self.passed}) + failed ({self.failed}) + warned ({self.warned}) "
                f"!= total_maps ({self.total_maps})"
            )

        # Ensure timestamps are monotonic
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")


@dataclass
class VerificationContext:
    """Runtime configuration for verification execution.

    Attributes
    ----------
    mode : Literal["local", "ci"]
        Execution mode (affects timeouts and exit behavior)
    artifact_root : Path
        Root directory for output artifacts
    seed : int | None
        Random seed for deterministic environment instantiation
    fix_enabled : bool
        Whether to attempt automatic remediation
    soft_timeout_s : float
        Soft performance budget per map (seconds)
    hard_timeout_s : float
        Hard timeout for entire run (seconds)
    """

    mode: Literal["local", "ci"]
    artifact_root: Path
    seed: int | None = None
    fix_enabled: bool = False
    soft_timeout_s: float = 20.0
    hard_timeout_s: float = 60.0

    @property
    def is_ci_mode(self) -> bool:
        """Check if running in CI mode."""
        return self.mode == "ci"

    def get_validation_output_dir(self) -> Path:
        """Get the validation artifact directory."""
        validation_dir = self.artifact_root / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        return validation_dir
