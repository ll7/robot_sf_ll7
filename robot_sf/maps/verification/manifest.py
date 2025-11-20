"""Manifest writer for structured verification results.

This module provides utilities to write verification results to JSON/JSONL
manifests for downstream consumption by dashboards, CI tooling, and documentation.
"""

import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from robot_sf.maps.verification import VerificationResult, VerificationRunSummary
from robot_sf.maps.verification.context import VerificationContext


def write_manifest(
    results: list[VerificationResult],
    context: VerificationContext,
    output_path: Path | None = None,
) -> Path:
    """Write verification results to a JSON manifest.
    
    Args:
        results: List of verification results
        context: Verification context
        output_path: Where to write the manifest (overrides context.output_path)
        
    Returns:
        Path where the manifest was written
    """
    if output_path is None:
        output_path = context.output_path
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary
    summary = create_verification_summary(results, context)
    
    # Build manifest structure
    manifest = {
        "summary": _summary_to_dict(summary),
        "results": [_result_to_dict(r) for r in results],
    }
    
    # Write to file
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"Wrote manifest to {output_path}")
    return output_path


def create_verification_summary(
    results: list[VerificationResult],
    context: VerificationContext,
) -> VerificationRunSummary:
    """Create a summary from verification results.
    
    Args:
        results: List of verification results
        context: Verification context
        
    Returns:
        Verification run summary
    """
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    
    # Identify slow maps (exceeded soft performance budget)
    slow_threshold_ms = context.perf_soft_budget_s * 1000
    slow_maps = [
        r.map_id
        for r in results
        if r.duration_ms > slow_threshold_ms
    ]
    
    return VerificationRunSummary(
        run_id=context.run_id,
        git_sha=context.git_sha,
        total_maps=len(results),
        passed=passed,
        failed=failed,
        warned=warned,
        slow_maps=slow_maps,
        artifact_path=str(context.output_path),
        started_at=context.started_at,
        finished_at=datetime.now(),
    )


def _summary_to_dict(summary: VerificationRunSummary) -> dict:
    """Convert VerificationRunSummary to dictionary.
    
    Args:
        summary: Summary object
        
    Returns:
        Dictionary representation
    """
    return {
        "run_id": summary.run_id,
        "git_sha": summary.git_sha,
        "total_maps": summary.total_maps,
        "passed": summary.passed,
        "failed": summary.failed,
        "warned": summary.warned,
        "slow_maps": summary.slow_maps,
        "artifact_path": summary.artifact_path,
        "started_at": summary.started_at.isoformat(),
        "finished_at": summary.finished_at.isoformat(),
        "duration_s": (summary.finished_at - summary.started_at).total_seconds(),
    }


def _result_to_dict(result: VerificationResult) -> dict:
    """Convert VerificationResult to dictionary.
    
    Args:
        result: Result object
        
    Returns:
        Dictionary representation
    """
    return {
        "map_id": result.map_id,
        "status": result.status,
        "rule_ids": result.rule_ids,
        "duration_ms": result.duration_ms,
        "factory_used": result.factory_used,
        "message": result.message,
        "timestamp": result.timestamp.isoformat(),
    }


def read_manifest(manifest_path: Path) -> dict:
    """Read and parse a verification manifest.
    
    Args:
        manifest_path: Path to the manifest file
        
    Returns:
        Parsed manifest dictionary
        
    Raises:
        FileNotFoundError: If manifest does not exist
        json.JSONDecodeError: If manifest is invalid JSON
    """
    with open(manifest_path) as f:
        return json.load(f)
