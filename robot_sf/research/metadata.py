"""Reproducibility metadata collection for research reports.

This module collects comprehensive metadata to ensure research reproducibility,
including git provenance, package versions, hardware profiles, and experimental
configuration.

Key Components:
    - Git metadata (commit hash, branch, dirty state)
    - Python package versions
    - Hardware profiles (CPU, memory, optional GPU)
    - Experimental configuration tracking

Usage:
    >>> from robot_sf.research.metadata import collect_reproducibility_metadata
    >>> metadata = collect_reproducibility_metadata(
    ...     seeds=[42, 43, 44], config_paths={"expert": Path("configs/expert.yaml")}
    ... )
    >>> metadata["git_commit"]  # 'abc123...'
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from robot_sf.research.aggregation import compute_completeness_score
from robot_sf.research.exceptions import ValidationError
from robot_sf.research.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HardwareProfile:
    """Hardware configuration snapshot.

    Attributes:
        cpu_model: CPU model string
        cpu_cores: Number of CPU cores
        memory_gb: Total system memory in GB
        gpu_info: Optional GPU information (model, memory)
    """

    cpu_model: str
    cpu_cores: int
    memory_gb: float
    gpu_info: dict[str, Any] | None = None


@dataclass
class ReproducibilityMetadata:
    """Complete reproducibility metadata for an experiment.

    Attributes:
        timestamp: UTC timestamp of metadata collection
        git_commit: Git commit hash
        git_branch: Git branch name
        git_dirty: Whether working directory had uncommitted changes
        python_version: Python version string
        package_versions: Dict of package name → version
        hardware: Hardware profile
        seeds: Random seeds used in experiment
        config_paths: Dict of config type → file path
    """

    timestamp: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    python_version: str
    package_versions: dict[str, str]
    hardware: HardwareProfile
    seeds: list[int] = field(default_factory=list)
    config_paths: dict[str, str] = field(default_factory=dict)
    timing: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "timestamp": self.timestamp,
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "python_version": self.python_version,
            "packages": self.package_versions,
            "hardware": {
                "cpu_model": self.hardware.cpu_model,
                "cpu_cores": self.hardware.cpu_cores,
                "memory_gb": self.hardware.memory_gb,
                "gpu_info": self.hardware.gpu_info,
            },
            "experiment": {
                "seeds": self.seeds,
                "configs": self.config_paths,
            },
        }
        if self.timing:
            data["timing"] = self.timing
        return data


def get_git_metadata() -> tuple[str, str, bool]:
    """Extract git metadata from repository.

    Returns:
        Tuple of (commit_hash, branch_name, is_dirty)
    """
    try:
        # Get commit hash
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        # Get branch name
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Check if working directory is dirty
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        is_dirty = bool(status_output.strip())

        logger.debug("Collected git metadata", commit=commit[:8], branch=branch, dirty=is_dirty)
        return commit, branch, is_dirty

    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        msg = "Failed to retrieve git metadata. Ensure repository is initialized."
        logger.warning(msg, error=str(e))
        return "unknown", "unknown", False


def get_package_versions() -> dict[str, str]:
    """Get versions of key packages.

    Returns:
        Dictionary mapping package name to version string

    Note:
        Only includes packages relevant to research reporting:
        scipy, matplotlib, pandas, numpy, stable-baselines3, torch
    """
    import importlib.metadata

    packages = [
        "scipy",
        "matplotlib",
        "pandas",
        "numpy",
        "stable-baselines3",
        "torch",
        "robot_sf",  # Self-reference
    ]

    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Package not installed - record as unavailable for reproducibility tracking
            versions[pkg] = "not installed"

    logger.debug("Collected package versions", count=len(versions))
    return versions


def get_hardware_profile() -> HardwareProfile:
    """Collect hardware profile information.

    Returns:
        HardwareProfile with CPU, memory, and optional GPU info
    """
    cpu_model = platform.processor() or "Unknown CPU"
    cpu_cores = psutil.cpu_count(logical=False) or 1
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # Attempt to collect GPU info (optional)
    gpu_info = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "model": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            }
    except (ImportError, RuntimeError):
        pass  # GPU info optional

    logger.debug(
        "Collected hardware profile",
        cpu=cpu_model,
        cores=cpu_cores,
        memory_gb=f"{memory_gb:.1f}",
        gpu=gpu_info is not None,
    )

    return HardwareProfile(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        gpu_info=gpu_info,
    )


def collect_reproducibility_metadata(
    seeds: list[int] | None = None,
    config_paths: dict[str, Path] | None = None,
) -> ReproducibilityMetadata:
    """Collect complete reproducibility metadata.

    Args:
        seeds: Random seeds used in experiment (optional)
        config_paths: Dict mapping config type to file path (optional)

    Returns:
        ReproducibilityMetadata instance
    """
    timestamp = datetime.now(tz=UTC).isoformat()
    git_commit, git_branch, git_dirty = get_git_metadata()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    packages = get_package_versions()
    hardware = get_hardware_profile()

    # Convert config paths to strings
    config_paths_str = {}
    if config_paths:
        config_paths_str = {k: str(v) for k, v in config_paths.items()}

    metadata = ReproducibilityMetadata(
        timestamp=timestamp,
        git_commit=git_commit,
        git_branch=git_branch,
        git_dirty=git_dirty,
        python_version=python_version,
        package_versions=packages,
        hardware=hardware,
        seeds=seeds or [],
        config_paths=config_paths_str,
    )

    logger.info(
        "Collected reproducibility metadata",
        commit=git_commit[:8],
        branch=git_branch,
        seeds=len(seeds) if seeds else 0,
    )

    return metadata


def parse_tracker_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Parse a run-tracker manifest (JSON or JSONL) into a structured dict.

    The parser is intentionally tolerant: it accepts both JSON and JSONL files,
    returning the most recent record for JSONL inputs. Steps are summarized so
    completeness can be calculated for the execution flow.
    """

    path = Path(manifest_path)
    if not path.exists():
        msg = f"Tracker manifest does not exist: {path}"
        logger.warning(msg)
        raise ValidationError(msg)

    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            if not lines:
                raise ValidationError(f"Tracker manifest empty: {path}")
            payload = lines[-1]
        else:
            payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Failed to parse tracker manifest at {path}"
        logger.warning(msg, error=str(exc))
        raise ValidationError(msg) from exc

    steps = payload.get("steps") or []
    enabled_steps = payload.get("enabled_steps") or [
        s.get("step_id") for s in steps if s.get("step_id")
    ]
    completed_steps = [
        s.get("step_id")
        for s in steps
        if s.get("step_id") and s.get("status") in {"completed", "success", "succeeded"}
    ]
    failed_steps = [
        s.get("step_id")
        for s in steps
        if s.get("step_id") and s.get("status") in {"failed", "error", "timeout"}
    ]

    completeness = None
    if enabled_steps:
        completeness = compute_completeness_score(
            expected_seeds=enabled_steps,
            completed_seeds=completed_steps,
            failed_seeds=failed_steps,
        )

    return {
        "run_id": payload.get("run_id"),
        "status": payload.get("status"),
        "enabled_steps": enabled_steps,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "seeds": payload.get("seeds") or payload.get("summary", {}).get("seeds", []),
        "summary": payload.get("summary", {}),
        "completeness": completeness,
    }
