"""Fail-closed admission for release checkpoint-staging receipts.

The staging CLI writes a receipt only after every checkpoint-bearing planner arm has
materialized a checksum-verified file.  Release execution consumes that receipt instead of
trusting a cheaper, network-free resolvability check.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.campaign.campaign_checkpoint_preflight import (
    iter_campaign_arm_checkpoint_references,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready_campaign import CampaignConfig


CHECKPOINT_STAGING_RECEIPT_SCHEMA = "campaign-checkpoint-staging-receipt.v1"
_ACCEPTED_ARM_STATUSES = frozenset({"present_local", "staged"})


class CheckpointStagingReceiptError(RobotSfError, RuntimeError):
    """Raised when a release checkpoint receipt is absent, stale, or mismatched."""


def _parse_generated_at(value: Any) -> datetime:
    """Parse a required UTC timestamp from a receipt.

    Returns:
        The normalized timezone-aware UTC timestamp.
    """
    if not isinstance(value, str) or not value.strip():
        raise CheckpointStagingReceiptError("receipt generated_at_utc is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CheckpointStagingReceiptError("receipt generated_at_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise CheckpointStagingReceiptError("receipt generated_at_utc must include a timezone")
    return parsed.astimezone(UTC)


def _reference_identity(reference: Any) -> tuple[str, str, str, str, bool]:
    """Return the stable identity tuple for a checkpoint reference."""
    return (
        reference.planner_key,
        reference.algo,
        reference.kind,
        reference.value,
        bool(reference.implicit),
    )


def _receipt_arm_identity(arm: dict[str, Any]) -> tuple[str, str, str, str, bool]:
    """Return the stable identity tuple for a receipt arm."""
    return (
        str(arm.get("planner_key", "")),
        str(arm.get("algo", "")),
        str(arm.get("kind", "")),
        str(arm.get("value", "")),
        bool(arm.get("implicit", False)),
    )


def _validate_receipt_header(
    payload: dict[str, Any],
    *,
    config_path: Path,
    max_age_hours: float,
    now: datetime | None,
) -> None:
    """Validate schema, staging mode, age, and campaign-config binding."""
    if payload.get("schema_version") != CHECKPOINT_STAGING_RECEIPT_SCHEMA:
        raise CheckpointStagingReceiptError(
            f"checkpoint staging receipt schema must be {CHECKPOINT_STAGING_RECEIPT_SCHEMA}"
        )
    if payload.get("status") != "ok":
        raise CheckpointStagingReceiptError("checkpoint staging receipt status is not ok")
    if payload.get("mode") != "enforced_staged" or payload.get("stage") is not True:
        raise CheckpointStagingReceiptError(
            "checkpoint receipt was not produced in enforced-staged mode"
        )
    if payload.get("submit_safe") is not True:
        raise CheckpointStagingReceiptError("checkpoint staging receipt has submit_safe=false")

    generated_at = _parse_generated_at(payload.get("generated_at_utc"))
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    if generated_at > current_time + timedelta(minutes=5):
        raise CheckpointStagingReceiptError("checkpoint staging receipt timestamp is in the future")
    if current_time - generated_at > timedelta(hours=max_age_hours):
        raise CheckpointStagingReceiptError(
            f"checkpoint staging receipt is stale (older than {max_age_hours:g} hours)"
        )
    if payload.get("campaign_config_sha256") != sha256_file(config_path):
        raise CheckpointStagingReceiptError(
            "checkpoint receipt campaign config hash does not match"
        )


def _validate_receipt_arms(cfg: CampaignConfig, payload: dict[str, Any]) -> None:
    """Validate arm coverage, statuses, materialized files, and checksums."""
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CheckpointStagingReceiptError("checkpoint staging receipt has no covered arms")
    if not all(isinstance(arm, dict) for arm in arms):
        raise CheckpointStagingReceiptError("checkpoint staging receipt arms must be JSON objects")
    expected = sorted(
        _reference_identity(ref) for ref in iter_campaign_arm_checkpoint_references(cfg)
    )
    observed = sorted(_receipt_arm_identity(arm) for arm in arms)
    if observed != expected:
        raise CheckpointStagingReceiptError(
            "checkpoint staging receipt arm/config identities do not match"
        )

    for arm in arms:
        planner_key = str(arm.get("planner_key", "<unknown>"))
        if arm.get("status") not in _ACCEPTED_ARM_STATUSES:
            raise CheckpointStagingReceiptError(
                f"checkpoint receipt arm {planner_key} has non-staged status {arm.get('status')!r}"
            )
        resolved_path = arm.get("resolved_path")
        expected_sha = arm.get("checkpoint_sha256")
        if not isinstance(resolved_path, str) or not Path(resolved_path).is_file():
            raise CheckpointStagingReceiptError(
                f"checkpoint receipt arm {planner_key} no longer resolves to a file"
            )
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise CheckpointStagingReceiptError(
                f"checkpoint receipt arm {planner_key} has no valid SHA-256"
            )
        if sha256_file(Path(resolved_path)) != expected_sha.lower():
            raise CheckpointStagingReceiptError(
                f"checkpoint receipt arm {planner_key} file checksum changed"
            )


def validate_checkpoint_staging_receipt(
    cfg: CampaignConfig,
    receipt_path: str | Path,
    *,
    campaign_config_path: str | Path,
    max_age_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a staged-checkpoint receipt against current config and files.

    Returns:
        The parsed receipt when every release-admission check passes.

    Raises:
        CheckpointStagingReceiptError: If the receipt cannot safely admit a release run.
    """
    path = Path(receipt_path).resolve()
    if not path.is_file():
        raise CheckpointStagingReceiptError(f"checkpoint staging receipt not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointStagingReceiptError(
            f"checkpoint staging receipt is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointStagingReceiptError("checkpoint staging receipt must be a JSON object")
    config_path = Path(campaign_config_path).resolve()
    _validate_receipt_header(
        payload,
        config_path=config_path,
        max_age_hours=max_age_hours,
        now=now,
    )
    _validate_receipt_arms(cfg, payload)
    return payload


__all__ = [
    "CHECKPOINT_STAGING_RECEIPT_SCHEMA",
    "CheckpointStagingReceiptError",
    "validate_checkpoint_staging_receipt",
]
