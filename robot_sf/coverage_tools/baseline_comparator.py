"""
Baseline comparison utilities for CI/CD coverage monitoring.

Provides functionality to compare current coverage against a baseline,
detect decreases, and generate warnings for CI/CD pipelines.

This module implements the core comparison logic without I/O side effects,
following the library-first principle (Constitution XI).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class CoverageSnapshot:
    """
    A single coverage measurement snapshot.

    Attributes:
        total_coverage: Overall coverage percentage
        file_coverage: Dict mapping file paths to coverage percentages
        timestamp: ISO 8601 timestamp of measurement
        commit_sha: Git commit SHA (optional)
        branch: Git branch name (optional)
    """

    total_coverage: float
    file_coverage: dict[str, float]
    timestamp: str
    commit_sha: str | None = None
    branch: str | None = None

    @classmethod
    def from_coverage_json(cls, data: dict[str, Any], **kwargs) -> "CoverageSnapshot":
        """
        Create snapshot from coverage.json structure.

        Args:
            data: Coverage.json data dictionary
            **kwargs: Additional metadata (timestamp, commit_sha, branch)

        Returns:
            CoverageSnapshot instance
        """
        total_coverage = data.get("totals", {}).get("percent_covered", 0.0)

        file_coverage = {}
        for file_path, file_data in data.get("files", {}).items():
            file_coverage[file_path] = file_data.get("summary", {}).get("percent_covered", 0.0)

        return cls(
            total_coverage=total_coverage,
            file_coverage=file_coverage,
            timestamp=kwargs.get("timestamp", ""),
            commit_sha=kwargs.get("commit_sha"),
            branch=kwargs.get("branch"),
        )


@dataclass
class CoverageBaseline:
    """
    Reference baseline for coverage comparison.

    Attributes:
        snapshot: The baseline coverage snapshot
        source: Description of baseline source (e.g., "main branch", "PR base")
    """

    snapshot: CoverageSnapshot
    source: str = "unknown"


@dataclass
class CoverageDelta:
    """
    Comparison result between current and baseline coverage.

    Attributes:
        current_coverage: Current total coverage percentage
        baseline_coverage: Baseline total coverage percentage
        delta: Change in coverage (current - baseline)
        threshold: Warning threshold percentage
        changed_files: List of file-level changes
        warnings: List of warning messages
    """

    current_coverage: float
    baseline_coverage: float
    delta: float
    threshold: float
    changed_files: list[dict[str, Any]]
    warnings: list[str]

    @property
    def has_decrease(self) -> bool:
        """Check if coverage decreased beyond threshold."""
        return self.delta < -abs(self.threshold)

    @property
    def has_increase(self) -> bool:
        """Check if coverage increased."""
        return self.delta > abs(self.threshold)


def load_baseline(baseline_path: Path) -> CoverageBaseline | None:
    """
    Load baseline coverage from JSON file.

    Args:
        baseline_path: Path to baseline coverage JSON file

    Returns:
        CoverageBaseline if file exists and is valid, None otherwise

    Side Effects:
        Logs warnings for missing or invalid files
    """
    if not baseline_path.exists():
        logger.warning(f"Baseline file not found: {baseline_path}")
        return None

    try:
        data = json.loads(baseline_path.read_text())
        snapshot = CoverageSnapshot.from_coverage_json(
            data,
            timestamp=data.get("meta", {}).get("timestamp", ""),
        )
        return CoverageBaseline(snapshot=snapshot, source=str(baseline_path))
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Invalid baseline file {baseline_path}: {e}")
        return None


def compare(
    current_path: Path,
    baseline: CoverageBaseline | None,
    threshold: float = 1.0,
) -> CoverageDelta:
    """
    Compare current coverage against baseline.

    Args:
        current_path: Path to current coverage.json
        baseline: Baseline to compare against (None = no comparison)
        threshold: Warning threshold in percentage points (default: 1.0)

    Returns:
        CoverageDelta with comparison results

    Side Effects:
        Logs info messages about comparison results

    Raises:
        FileNotFoundError: If current_path doesn't exist
        ValueError: If current coverage data is invalid
    """
    if not current_path.exists():
        msg = f"Current coverage file not found: {current_path}"
        raise FileNotFoundError(msg)

    try:
        current_data = json.loads(current_path.read_text())
        current_snapshot = CoverageSnapshot.from_coverage_json(
            current_data,
            timestamp=current_data.get("meta", {}).get("timestamp", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        msg = f"Invalid current coverage file: {e}"
        raise ValueError(msg) from e

    # No baseline = no comparison
    if baseline is None:
        logger.info("No baseline available for comparison")
        return CoverageDelta(
            current_coverage=current_snapshot.total_coverage,
            baseline_coverage=current_snapshot.total_coverage,
            delta=0.0,
            threshold=threshold,
            changed_files=[],
            warnings=["No baseline available for comparison"],
        )

    # Compute delta
    delta = current_snapshot.total_coverage - baseline.snapshot.total_coverage

    # Find changed files
    changed_files = []
    for file_path, current_cov in current_snapshot.file_coverage.items():
        baseline_cov = baseline.snapshot.file_coverage.get(file_path, 0.0)
        file_delta = current_cov - baseline_cov

        if abs(file_delta) > 0.1:  # Changed by more than 0.1%
            changed_files.append(
                {
                    "file": file_path,
                    "current": current_cov,
                    "baseline": baseline_cov,
                    "delta": file_delta,
                }
            )

    # Sort by delta (worst first)
    changed_files.sort(key=lambda x: x["delta"])

    # Generate warnings
    warnings = []
    if delta < -abs(threshold):
        warnings.append(f"Coverage decreased by {abs(delta):.2f}% (threshold: {threshold:.2f}%)")
        if changed_files:
            worst = changed_files[0]
            warnings.append(f"Largest decrease: {worst['file']} ({worst['delta']:+.2f}%)")

    logger.info(
        f"Coverage comparison: current={current_snapshot.total_coverage:.2f}%, "
        f"baseline={baseline.snapshot.total_coverage:.2f}%, "
        f"delta={delta:+.2f}%"
    )

    return CoverageDelta(
        current_coverage=current_snapshot.total_coverage,
        baseline_coverage=baseline.snapshot.total_coverage,
        delta=delta,
        threshold=threshold,
        changed_files=changed_files,
        warnings=warnings,
    )


def generate_warning(delta: CoverageDelta, format_type: str = "github") -> str:
    """
    Generate warning message in specified format.

    Args:
        delta: Coverage delta from comparison
        format_type: Output format ('github', 'terminal', 'json')

    Returns:
        Formatted warning string

    Side Effects:
        None (pure function)
    """
    if format_type == "github":
        return _generate_github_annotation(delta)
    if format_type == "terminal":
        return _generate_terminal_warning(delta)
    if format_type == "json":
        return _generate_json_warning(delta)
    return f"Unknown format: {format_type}"


def _generate_github_annotation(delta: CoverageDelta) -> str:
    """Generate GitHub Actions annotation format."""
    if not delta.has_decrease:
        return ""

    lines = []
    lines.append(
        f"::warning title=Coverage Decreased::"
        f"Coverage dropped by {abs(delta.delta):.2f}% "
        f"({delta.baseline_coverage:.2f}% → {delta.current_coverage:.2f}%)"
    )

    for file_change in delta.changed_files[:3]:  # Top 3 worst files
        if file_change["delta"] < 0:
            lines.append(
                f"::warning file={file_change['file']}::"
                f"Coverage decreased by {abs(file_change['delta']):.2f}%"
            )

    return "\n".join(lines)


def _generate_terminal_warning(delta: CoverageDelta) -> str:
    """Generate terminal-friendly warning."""
    if not delta.has_decrease:
        return "✓ Coverage maintained or improved"

    lines = [
        "⚠️  WARNING: Coverage Decreased",
        f"   Baseline: {delta.baseline_coverage:.2f}%",
        f"   Current:  {delta.current_coverage:.2f}%",
        f"   Change:   {delta.delta:+.2f}%",
    ]

    if delta.changed_files:
        lines.append("\n   Top affected files:")
        for file_change in delta.changed_files[:5]:
            if file_change["delta"] < 0:
                lines.append(f"     - {file_change['file']}: {file_change['delta']:+.2f}%")

    return "\n".join(lines)


def _generate_json_warning(delta: CoverageDelta) -> str:
    """Generate JSON format warning."""
    return json.dumps(
        {
            "has_decrease": delta.has_decrease,
            "current_coverage": delta.current_coverage,
            "baseline_coverage": delta.baseline_coverage,
            "delta": delta.delta,
            "threshold": delta.threshold,
            "warnings": delta.warnings,
            "changed_files": delta.changed_files,
        },
        indent=2,
    )
