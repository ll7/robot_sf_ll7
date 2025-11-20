"""Verification context and artifact routing for map verification.

This module provides utilities for managing verification runs, including
output paths, run metadata, and artifact organization.
"""

import uuid
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger


ModeType = Literal["local", "ci"]


@dataclass
class VerificationContext:
    """Context for a map verification run.
    
    Attributes:
        run_id: Unique identifier for this verification run
        mode: Execution mode ('local' or 'ci')
        git_sha: Git commit SHA for this run
        started_at: When the verification run started
        output_path: Where to write the manifest file
        fix_mode: Whether to attempt auto-fixes
        seed: Random seed for deterministic testing
        perf_soft_budget_s: Soft performance budget per map (seconds)
        perf_hard_timeout_s: Hard timeout per map (seconds)
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode: ModeType = "local"
    git_sha: str = field(default_factory=lambda: _get_git_sha())
    started_at: datetime = field(default_factory=datetime.now)
    output_path: Path = field(default=Path("output/validation/map_verification.json"))
    fix_mode: bool = False
    seed: int | None = None
    perf_soft_budget_s: float = 20.0
    perf_hard_timeout_s: float = 60.0
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_ci_mode(self) -> bool:
        """Check if running in CI mode."""
        return self.mode == "ci"
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self.mode == "local"


def _get_git_sha() -> str:
    """Get current git commit SHA.
    
    Returns:
        Git SHA string, or 'unknown' if not in a git repo or git unavailable
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        logger.debug(f"Current git SHA: {sha[:8]}")
        return sha
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Could not determine git SHA: {e}")
        return "unknown"


def create_artifact_path(
    base_dir: Path,
    run_id: str,
    *,
    filename: str | None = None,
) -> Path:
    """Create an artifact output path with proper organization.
    
    Args:
        base_dir: Base directory for artifacts (e.g., output/validation)
        run_id: Unique run identifier
        filename: Optional specific filename
        
    Returns:
        Path for the artifact
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"verification_{run_id}.json"
    
    return base_dir / filename


def get_default_output_dir() -> Path:
    """Get the default output directory for verification artifacts.
    
    Returns:
        Path to output/validation directory
    """
    return Path("output/validation")
