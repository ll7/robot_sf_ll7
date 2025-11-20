"""Manifest writer for structured verification output.

This module handles writing verification results to JSON/JSONL format
for downstream tooling, dashboards, and documentation.

Output Format
-------------
The manifest is a JSON file containing:
- Run metadata (ID, git SHA, timestamps)
- Aggregate counts (passed/failed/warned)
- Per-map results with rule IDs and messages
"""

import json
from pathlib import Path

from loguru import logger

from robot_sf.maps.verification.context import VerificationRunSummary


def write_manifest(summary: VerificationRunSummary, output_path: Path) -> None:
    """Write verification summary to JSON manifest.
    
    Parameters
    ----------
    summary : VerificationRunSummary
        Verification results to write
    output_path : Path
        Destination file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert summary to dict
    manifest_data = {
        "run_id": summary.run_id,
        "git_sha": summary.git_sha,
        "started_at": summary.started_at.isoformat(),
        "finished_at": summary.finished_at.isoformat() if summary.finished_at else None,
        "summary": {
            "total_maps": summary.total_maps,
            "passed": summary.passed,
            "failed": summary.failed,
            "warned": summary.warned,
            "slow_maps": summary.slow_maps,
        },
        "results": [
            {
                "map_id": result.map_id,
                "status": result.status.value,
                "rule_ids": result.rule_ids,
                "duration_ms": result.duration_ms,
                "factory_used": result.factory_used.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
            }
            for result in summary.results
        ],
    }
    
    # Write JSON
    with open(output_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    
    logger.debug(f"Wrote manifest to {output_path}")


def write_jsonl_manifest(summary: VerificationRunSummary, output_path: Path) -> None:
    """Write verification summary to JSONL format (one result per line).
    
    Parameters
    ----------
    summary : VerificationRunSummary
        Verification results to write
    output_path : Path
        Destination file path
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        # Write header with run metadata
        header = {
            "type": "run_metadata",
            "run_id": summary.run_id,
            "git_sha": summary.git_sha,
            "started_at": summary.started_at.isoformat(),
            "finished_at": summary.finished_at.isoformat() if summary.finished_at else None,
            "total_maps": summary.total_maps,
            "passed": summary.passed,
            "failed": summary.failed,
            "warned": summary.warned,
        }
        f.write(json.dumps(header) + "\n")
        
        # Write each result
        for result in summary.results:
            result_data = {
                "type": "map_result",
                "run_id": summary.run_id,
                "map_id": result.map_id,
                "status": result.status.value,
                "rule_ids": result.rule_ids,
                "duration_ms": result.duration_ms,
                "factory_used": result.factory_used.value,
                "message": result.message,
                "timestamp": result.timestamp.isoformat(),
            }
            f.write(json.dumps(result_data) + "\n")
    
    logger.debug(f"Wrote JSONL manifest to {output_path}")
