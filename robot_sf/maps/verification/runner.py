"""Verification runner for executing map validation workflows.

This module orchestrates the verification process:
- Loading maps based on scope
- Applying validation rules
- Optionally instantiating environments for runtime checks
- Tracking timing and performance
"""

import time
from datetime import datetime
from typing import Literal

from loguru import logger

from robot_sf.maps.verification import MapRecord, VerificationResult
from robot_sf.maps.verification.context import VerificationContext
from robot_sf.maps.verification.rules import apply_rules, get_default_rules


def verify_single_map(
    map_record: MapRecord,
    context: VerificationContext,
    *,
    skip_env_instantiation: bool = False,
) -> VerificationResult:
    """Verify a single map.
    
    Args:
        map_record: Map to verify
        context: Verification context with settings
        skip_env_instantiation: If True, skip environment instantiation checks
        
    Returns:
        Verification result for this map
    """
    start_time = time.time()
    logger.info(f"Verifying map: {map_record.map_id}")
    
    # Apply validation rules
    rule_results = apply_rules(map_record)
    
    # Determine overall status
    has_errors = any(r.severity == "error" and not r.passed for r in rule_results)
    has_warnings = any(r.severity == "warn" and not r.passed for r in rule_results)
    
    if has_errors:
        status = "fail"
    elif has_warnings:
        status = "warn"
    else:
        status = "pass"
    
    # Collect rule IDs
    rule_ids = [r.rule_id for r in rule_results]
    
    # Build message
    failed_rules = [r for r in rule_results if not r.passed]
    if failed_rules:
        messages = [f"{r.rule_id}: {r.message}" for r in failed_rules]
        message = "; ".join(messages)
    else:
        message = "All checks passed"
    
    # Determine factory (placeholder - will be enhanced when env instantiation is added)
    factory_used = "robot"  # Default
    if "pedestrian_focused" in map_record.tags or "pedestrian_only" in map_record.tags:
        factory_used = "pedestrian"
    
    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000
    
    # Performance warning
    if duration_ms > context.perf_soft_budget_s * 1000:
        logger.warning(
            f"Map {map_record.map_id} verification took {duration_ms:.0f}ms "
            f"(soft budget: {context.perf_soft_budget_s * 1000:.0f}ms)"
        )
    
    result = VerificationResult(
        map_id=map_record.map_id,
        status=status,
        rule_ids=rule_ids,
        duration_ms=duration_ms,
        factory_used=factory_used,
        message=message,
        timestamp=datetime.now(),
    )
    
    logger.info(
        f"Map {map_record.map_id}: {status.upper()} "
        f"({len(rule_ids)} rules, {duration_ms:.0f}ms)"
    )
    
    return result


def verify_maps(
    maps: list[MapRecord],
    context: VerificationContext,
) -> list[VerificationResult]:
    """Verify multiple maps.
    
    Args:
        maps: List of maps to verify
        context: Verification context
        
    Returns:
        List of verification results
    """
    logger.info(f"Starting verification of {len(maps)} maps")
    
    results = []
    for i, map_record in enumerate(maps, 1):
        logger.info(f"Progress: {i}/{len(maps)} maps")
        
        try:
            result = verify_single_map(map_record, context)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to verify {map_record.map_id}: {e}")
            # Create a failure result
            results.append(
                VerificationResult(
                    map_id=map_record.map_id,
                    status="fail",
                    rule_ids=["execution_error"],
                    duration_ms=0.0,
                    factory_used="unknown",
                    message=f"Verification failed with error: {e}",
                    timestamp=datetime.now(),
                )
            )
    
    # Summary statistics
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    
    logger.info(f"Verification complete: {passed} passed, {failed} failed, {warned} warned")
    
    return results
