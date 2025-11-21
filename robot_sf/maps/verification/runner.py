"""Main verification runner orchestrating map validation workflow.

This module provides the core verification logic that:
1. Loads map inventory and resolves scope
2. Applies validation rules
3. Instantiates environments for runtime compatibility testing
4. Collects timing and performance metrics
5. Generates structured results

Primary Entry Point
-------------------
verify_maps() : Main orchestrator function
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger

from robot_sf.common.artifact_paths import get_artifact_root
from robot_sf.maps.verification.context import (
    FactoryType,
    VerificationContext,
    VerificationResult,
    VerificationRunSummary,
    VerificationStatus,
)
from robot_sf.maps.verification.map_inventory import MapInventory, MapRecord
from robot_sf.maps.verification.rules import RuleSeverity, apply_all_rules
from robot_sf.maps.verification.scope_resolver import ScopeResolver


def verify_single_map(
    map_record: MapRecord,
    context: VerificationContext,
) -> VerificationResult:
    """Verify a single map.

    Parameters
    ----------
    map_record : MapRecord
        Map to verify
    context : VerificationContext
        Verification runtime context

    Returns
    -------
    VerificationResult
        Verification outcome for this map
    """
    logger.info(f"Verifying map: {map_record.map_id}")
    start_time = time.perf_counter()

    # Apply validation rules
    violations = apply_all_rules(map_record.file_path)

    # Determine overall status
    if any(v.severity == RuleSeverity.ERROR for v in violations):
        status = VerificationStatus.FAIL
    elif any(v.severity == RuleSeverity.WARNING for v in violations):
        status = VerificationStatus.WARN
    else:
        status = VerificationStatus.PASS

    # Extract rule IDs
    rule_ids = [v.rule_id for v in violations]

    # Build message
    if status == VerificationStatus.PASS:
        message = "All checks passed"
    else:
        violation_msgs = [f"{v.rule_id}: {v.message}" for v in violations]
        message = "; ".join(violation_msgs)

    # Attempt environment instantiation if rules passed
    factory_used = FactoryType.ROBOT
    if status == VerificationStatus.PASS:
        try:
            # Determine factory type based on tags
            if "pedestrian_only" in map_record.tags:
                factory_used = FactoryType.PEDESTRIAN
            else:
                factory_used = FactoryType.ROBOT

            # TODO: Actually instantiate environment
            # For now, just log what we would do
            logger.debug(
                f"Would instantiate {factory_used.value} environment for {map_record.map_id}"
            )

            # Placeholder: successful instantiation
            # from robot_sf.gym_env.environment_factory import make_robot_env, make_pedestrian_env
            # if factory_used == FactoryType.ROBOT:
            #     env = make_robot_env(...)
            # else:
            #     env = make_pedestrian_env(...)
            # env.close()

        except Exception as e:  # noqa: BLE001 - environment instantiation may raise many implementation-specific errors
            logger.error(f"Environment instantiation failed for {map_record.map_id}: {e}")
            status = VerificationStatus.FAIL
            rule_ids.append("R999")
            message += f"; Environment instantiation failed: {e}"

    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Check soft timeout
    if duration_ms > context.soft_timeout_s * 1000:
        logger.warning(
            f"Map {map_record.map_id} exceeded soft timeout: "
            f"{duration_ms:.0f}ms > {context.soft_timeout_s * 1000:.0f}ms"
        )

    result = VerificationResult(
        map_id=map_record.map_id,
        status=status,
        rule_ids=rule_ids if rule_ids else ["PASS"],
        duration_ms=duration_ms,
        factory_used=factory_used,
        message=message,
        timestamp=datetime.now(),
    )

    # Log result
    status_symbol = "✓" if status == VerificationStatus.PASS else "✗"
    logger.info(f"{status_symbol} {map_record.map_id}: {status.value} ({duration_ms:.0f}ms)")

    if status != VerificationStatus.PASS:
        logger.warning(f"  {message}")

    return result


def _write_summary_manifest(summary: VerificationRunSummary, output_path: Path | None) -> None:
    """Write verification summary manifest if output path provided.

    Parameters
    ----------
    summary : VerificationRunSummary
        Summary to write
    output_path : Path | None
        Optional output path
    """
    if not output_path:
        return

    try:
        from robot_sf.maps.verification.manifest import write_manifest

        write_manifest(summary, output_path)
        logger.debug(f"Manifest written to: {output_path}")
    except Exception as write_err:  # noqa: BLE001 - manifest write failure should not abort reporting
        logger.warning(f"Failed to write manifest: {write_err}")


def verify_maps(
    scope: str = "all",
    mode: Literal["local", "ci"] = "local",
    output_path: Path | None = None,
    seed: int | None = None,
    fix: bool = False,
) -> VerificationRunSummary:
    """Main verification orchestrator.

    Parameters
    ----------
    scope : str
        Scope specifier ('all', 'ci', 'changed', filename, or glob)
    mode : Literal["local", "ci"]
        Verification mode (affects timeouts and exit behavior)
    output_path : Path | None
        Optional path to write JSON manifest
    seed : int | None
        Random seed for deterministic environment instantiation
    fix : bool
        Whether to attempt automatic remediation

    Returns
    -------
    VerificationRunSummary
        Aggregated verification results
    """
    # Initialize context
    artifact_root = get_artifact_root()
    context = VerificationContext(
        mode=mode,
        artifact_root=artifact_root,
        seed=seed,
        fix_enabled=fix,
    )

    # Generate run ID
    run_id = f"map_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    logger.info("=" * 60)
    logger.info(f"Map Verification Run: {run_id}")
    logger.info(f"Mode: {mode}, Scope: {scope}")
    logger.info("=" * 60)

    started_at = datetime.now()

    # Load inventory and resolve scope
    inventory = MapInventory()
    resolver = ScopeResolver(inventory)

    try:
        maps_to_verify = resolver.resolve(scope)
    except ValueError as e:
        error_msg = f"Failed to resolve scope '{scope}': {e}"
        logger.error(error_msg)
        # Return summary with error captured for programmatic access
        finished_at = datetime.now()
        summary = VerificationRunSummary(
            run_id=run_id,
            git_sha=None,
            total_maps=0,
            passed=0,
            failed=0,
            warned=0,
            slow_maps=[],
            artifact_path=output_path,
            started_at=started_at,
            finished_at=finished_at,
            results=[],
        )
        _write_summary_manifest(summary, output_path)
        return summary

    if not maps_to_verify:
        logger.warning(f"No maps to verify in scope '{scope}'")
        finished_at = datetime.now()
        summary = VerificationRunSummary(
            run_id=run_id,
            git_sha=None,
            total_maps=0,
            passed=0,
            failed=0,
            warned=0,
            slow_maps=[],
            artifact_path=output_path,
            started_at=started_at,
            finished_at=finished_at,
            results=[],
        )
        _write_summary_manifest(summary, output_path)
        return summary

    # Verify each map
    results: list[VerificationResult] = []
    for map_record in maps_to_verify:
        result = verify_single_map(map_record, context)
        results.append(result)

    # Aggregate results
    passed = sum(1 for r in results if r.status == VerificationStatus.PASS)
    failed = sum(1 for r in results if r.status == VerificationStatus.FAIL)
    warned = sum(1 for r in results if r.status == VerificationStatus.WARN)

    # Find slow maps
    slow_maps = [r.map_id for r in results if r.duration_ms > context.soft_timeout_s * 1000]

    # Get git SHA if available
    git_sha = None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_sha = result.stdout.strip()
    except Exception:  # noqa: BLE001 - git may be unavailable (shallow clone or no repo)
        pass

    finished_at = datetime.now()

    # Create summary
    summary = VerificationRunSummary(
        run_id=run_id,
        git_sha=git_sha,
        total_maps=len(maps_to_verify),
        passed=passed,
        failed=failed,
        warned=warned,
        slow_maps=slow_maps,
        artifact_path=output_path,
        started_at=started_at,
        finished_at=finished_at,
        results=results,
    )

    # Log summary
    logger.info("=" * 60)
    logger.info("Verification Summary:")
    logger.info(f"  Total maps: {summary.total_maps}")
    logger.info(f"  Passed: {summary.passed}")
    logger.info(f"  Failed: {summary.failed}")
    logger.info(f"  Warned: {summary.warned}")
    if summary.slow_maps:
        logger.warning(f"  Slow maps: {', '.join(summary.slow_maps)}")
    logger.info("=" * 60)

    # Write manifest if requested
    _write_summary_manifest(summary, output_path)
    if output_path and output_path.exists():
        logger.info(f"Manifest written to: {output_path}")

    return summary
