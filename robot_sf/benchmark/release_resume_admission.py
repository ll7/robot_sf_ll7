"""Fail-closed admission for resuming an immutable benchmark release campaign."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from pathlib import Path


RELEASE_RESUME_RECEIPT_SCHEMA = "benchmark-release-resume-receipt.v1"
_INFRASTRUCTURE_REASONS = frozenset(
    {
        "cluster_filesystem_interruption",
        "network_interruption",
        "node_failure",
        "scheduler_requeue",
        "walltime_kill",
    }
)
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SECRET_KEY = re.compile(r"(?:auth|credential|password|secret|token)", re.IGNORECASE)


class ReleaseResumeAdmissionError(RobotSfError, RuntimeError):
    """Raised when same-campaign release resume is not proven infrastructure-only."""


def campaign_has_prior_execution(campaign_root: Path) -> bool:
    """Return whether a fixed campaign directory contains prior planner execution state."""
    runs_dir = campaign_root / "runs"
    return runs_dir.is_dir() and any(runs_dir.iterdir())


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a required JSON object without including its contents in errors.

    Returns:
        Parsed JSON object.
    """
    if not path.is_file():
        raise ReleaseResumeAdmissionError(f"{label} is unavailable")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseResumeAdmissionError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReleaseResumeAdmissionError(f"{label} must be a JSON object")
    return payload


def _parse_created_at(value: Any) -> datetime:
    """Parse a timezone-aware receipt timestamp.

    Returns:
        Normalized timezone-aware UTC timestamp.
    """
    if not isinstance(value, str) or not value.strip():
        raise ReleaseResumeAdmissionError("resume receipt created_at_utc is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseResumeAdmissionError("resume receipt created_at_utc is invalid") from exc
    if parsed.tzinfo is None:
        raise ReleaseResumeAdmissionError("resume receipt created_at_utc must include a timezone")
    return parsed.astimezone(UTC)


def _contains_secret_key(value: Any) -> bool:
    """Return whether a nested payload contains a credential-shaped field name."""
    if isinstance(value, dict):
        return any(
            _SECRET_KEY.search(str(key)) is not None or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def _validate_receipt_header(
    payload: dict[str, Any],
    *,
    campaign_id: str,
    max_age_hours: float,
    now: datetime | None,
) -> None:
    """Validate the operator's explicit infrastructure-only resume ruling."""
    if payload.get("schema_version") != RELEASE_RESUME_RECEIPT_SCHEMA:
        raise ReleaseResumeAdmissionError(
            f"resume receipt schema must be {RELEASE_RESUME_RECEIPT_SCHEMA}"
        )
    if payload.get("status") != "approved" or payload.get("resume_same_campaign") is not True:
        raise ReleaseResumeAdmissionError("resume receipt does not approve same-campaign resume")
    if payload.get("interruption_class") != "infrastructure":
        raise ReleaseResumeAdmissionError("same-campaign resume is infrastructure-only")
    if payload.get("interruption_reason") not in _INFRASTRUCTURE_REASONS:
        raise ReleaseResumeAdmissionError("resume receipt interruption reason is not admitted")
    if payload.get("campaign_id") != campaign_id:
        raise ReleaseResumeAdmissionError("resume receipt campaign_id does not match")
    if _contains_secret_key(payload):
        raise ReleaseResumeAdmissionError("resume receipt contains a credential-shaped field")

    created_at = _parse_created_at(payload.get("created_at_utc"))
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    if created_at > current_time + timedelta(minutes=5):
        raise ReleaseResumeAdmissionError("resume receipt timestamp is in the future")
    if current_time - created_at > timedelta(hours=max_age_hours):
        raise ReleaseResumeAdmissionError(
            f"resume receipt is stale (older than {max_age_hours:g} hours)"
        )


def validate_release_resume_admission(  # noqa: C901, PLR0913
    *,
    campaign_root: Path,
    campaign_id: str,
    campaign_config_path: Path,
    checkpoint_receipt_path: Path,
    current_source_commit: str,
    resume_enabled: bool,
    resume_receipt_path: Path | None,
    max_age_hours: float = 24.0,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Validate a same-ID resume or prove that the campaign is a fresh execution.

    Returns:
        A sanitized, hash-bound resume receipt summary, or ``None`` for a fresh campaign.
    """
    prior_execution = campaign_has_prior_execution(campaign_root)
    if not prior_execution:
        if resume_receipt_path is not None:
            raise ReleaseResumeAdmissionError(
                "resume receipt is only valid when prior campaign execution exists"
            )
        return None

    if not resume_enabled:
        raise ReleaseResumeAdmissionError("campaign configuration disables resume")
    if resume_receipt_path is None:
        raise ReleaseResumeAdmissionError(
            "existing release campaign requires an infrastructure-only resume receipt"
        )
    if max_age_hours <= 0:
        raise ReleaseResumeAdmissionError("resume receipt max age must be positive")

    payload = _read_json_object(resume_receipt_path, label="resume receipt")
    _validate_receipt_header(
        payload,
        campaign_id=campaign_id,
        max_age_hours=max_age_hours,
        now=now,
    )

    source_commit = str(payload.get("source_commit", "")).strip().lower()
    current_commit = current_source_commit.strip().lower()
    if _SHA40.fullmatch(source_commit) is None or source_commit != current_commit:
        raise ReleaseResumeAdmissionError("resume receipt source commit does not match checkout")

    expected_config_sha = sha256_file(campaign_config_path)
    expected_checkpoint_sha = sha256_file(checkpoint_receipt_path)
    if payload.get("campaign_config_sha256") != expected_config_sha:
        raise ReleaseResumeAdmissionError("resume receipt campaign config hash does not match")
    if payload.get("checkpoint_receipt_sha256") != expected_checkpoint_sha:
        raise ReleaseResumeAdmissionError("resume receipt checkpoint hash does not match")

    prior_manifest_path = campaign_root / "campaign_manifest.json"
    prior_manifest = _read_json_object(prior_manifest_path, label="prior campaign manifest")
    if payload.get("prior_campaign_manifest_sha256") != sha256_file(prior_manifest_path):
        raise ReleaseResumeAdmissionError(
            "resume receipt prior campaign manifest hash does not match"
        )
    if prior_manifest.get("campaign_id") != campaign_id:
        raise ReleaseResumeAdmissionError("prior campaign manifest campaign_id does not match")
    prior_git = prior_manifest.get("git")
    prior_commit = prior_git.get("commit") if isinstance(prior_git, dict) else None
    if prior_commit != current_commit:
        raise ReleaseResumeAdmissionError(
            "same-campaign resume cannot cross a source commit; start a fresh campaign"
        )

    receipt_sha = sha256_file(resume_receipt_path)
    if _SHA256.fullmatch(receipt_sha) is None:  # defensive contract check
        raise ReleaseResumeAdmissionError("resume receipt checksum is invalid")
    return {
        "schema_version": RELEASE_RESUME_RECEIPT_SCHEMA,
        "path": resume_receipt_path,
        "sha256": receipt_sha,
        "created_at_utc": payload["created_at_utc"],
        "interruption_class": "infrastructure",
        "interruption_reason": payload["interruption_reason"],
        "campaign_id": campaign_id,
        "source_commit": current_commit,
        "campaign_config_sha256": expected_config_sha,
        "checkpoint_receipt_sha256": expected_checkpoint_sha,
        "prior_campaign_manifest_sha256": payload["prior_campaign_manifest_sha256"],
    }


__all__ = [
    "RELEASE_RESUME_RECEIPT_SCHEMA",
    "ReleaseResumeAdmissionError",
    "campaign_has_prior_execution",
    "validate_release_resume_admission",
]
