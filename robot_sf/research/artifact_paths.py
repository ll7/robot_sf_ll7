"""Artifact path management for research reports.

This module provides utilities for managing output paths within the canonical
artifact root (`output/research_reports/`), ensuring consistent directory
structure across all report generation workflows.

Path Structure:
    output/research_reports/<timestamp>_<experiment_name>/
    ├── report.md
    ├── report.tex (optional)
    ├── metadata.json
    ├── figures/
    │   ├── fig-learning-curve.pdf
    │   ├── fig-learning-curve.png
    │   └── ...
    ├── data/
    │   ├── metrics.json
    │   ├── metrics.csv
    │   └── hypothesis.json
    └── configs/
        ├── expert_ppo.yaml
        ├── bc_pretrain.yaml
        └── ppo_finetune.yaml

Usage:
    >>> from robot_sf.research.artifact_paths import ensure_report_tree
    >>> paths = ensure_report_tree("my_experiment")
    >>> paths["report"]  # Path("output/research_reports/.../report.md")
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from robot_sf.research.logging_config import get_logger

logger = get_logger(__name__)


def get_artifact_root() -> Path:
    """Get the canonical artifact root directory.

    Respects ROBOT_SF_ARTIFACT_ROOT environment variable for overrides.

    Returns:
        Path to artifact root (defaults to 'output/')
    """
    return Path(os.getenv("ROBOT_SF_ARTIFACT_ROOT", "output"))


def get_research_reports_root() -> Path:
    """Get the research reports subdirectory within artifact root.

    Returns:
        Path to research_reports/ directory
    """
    return get_artifact_root() / "research_reports"


def generate_report_id(experiment_name: str) -> str:
    """Generate unique report identifier with timestamp.

    Args:
        experiment_name: Human-readable experiment label

    Returns:
        Report ID in format: YYYYMMDD_HHMMSS_<experiment_name_sanitized>

    Example:
        >>> generate_report_id("BC Ablation Study")
        '20251121_143022_bc_ablation_study'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = experiment_name.lower().replace(" ", "_").replace("-", "_")
    # Remove any non-alphanumeric characters except underscores
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return f"{timestamp}_{sanitized}"


def ensure_report_tree(
    experiment_name: str, output_override: Path | None = None
) -> dict[str, Path]:
    """Create report directory structure and return path dictionary.

    Args:
        experiment_name: Human-readable experiment label
        output_override: Optional override for output directory (defaults to auto-generated)

    Returns:
        Dictionary mapping logical names to paths:
            - "root": Report root directory
            - "report": report.md path
            - "report_tex": report.tex path
            - "metadata": metadata.json path
            - "figures": figures/ subdirectory
            - "data": data/ subdirectory
            - "configs": configs/ subdirectory

    Raises:
        OSError: If directory creation fails
    """
    if output_override:
        report_dir = output_override
    else:
        report_id = generate_report_id(experiment_name)
        report_dir = get_research_reports_root() / report_id

    # Create directory structure
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "figures").mkdir(exist_ok=True)
    (report_dir / "data").mkdir(exist_ok=True)
    (report_dir / "configs").mkdir(exist_ok=True)

    logger.info(
        "Created report directory structure",
        report_dir=str(report_dir),
        experiment=experiment_name,
    )

    return {
        "root": report_dir,
        "report": report_dir / "report.md",
        "report_tex": report_dir / "report.tex",
        "metadata": report_dir / "metadata.json",
        "figures": report_dir / "figures",
        "data": report_dir / "data",
        "configs": report_dir / "configs",
    }


def get_output_paths(report_root: Path) -> dict[str, Path]:
    """Get standard output paths for an existing report directory.

    Args:
        report_root: Existing report root directory

    Returns:
        Dictionary mapping logical names to paths (same structure as ensure_report_tree)
    """
    return {
        "root": report_root,
        "report": report_root / "report.md",
        "report_tex": report_root / "report.tex",
        "metadata": report_root / "metadata.json",
        "figures": report_root / "figures",
        "data": report_root / "data",
        "configs": report_root / "configs",
    }
