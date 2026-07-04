"""Readiness checks for issue #4018 density-curriculum comparison manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

READINESS_SCHEMA_VERSION = "issue_4018.density_curriculum_readiness.v1"
COMPARISON_SCHEMA_VERSION = "density_curriculum_comparison.v1"
ISSUE_ID = "ll7/robot_sf_ll7#4018"
CLAIM_BOUNDARY = "comparison readiness only; not benchmark evidence or a training-result claim"
NEXT_EMPIRICAL_ACTION = (
    "run the paired density-curriculum and fixed-density PPO smoke configs, then "
    "replace the dry-run manifest with completed training artifact paths before any "
    "performance or paper-facing claim"
)


@dataclass(frozen=True, slots=True)
class DensityCurriculumReadiness:
    """Fail-closed readiness verdict for a paired curriculum comparison manifest."""

    status: str
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    manifest_path: str | None = None
    schema_version: str = READINESS_SCHEMA_VERSION
    issue: str = ISSUE_ID
    claim_boundary: str = CLAIM_BOUNDARY
    next_empirical_action: str = NEXT_EMPIRICAL_ACTION
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready readiness packet."""
        return {
            "schema_version": self.schema_version,
            "issue": self.issue,
            "status": self.status,
            "claim_boundary": self.claim_boundary,
            "manifest_path": self.manifest_path,
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "comparison": self.comparison,
            "next_empirical_action": self.next_empirical_action,
        }


def evaluate_density_curriculum_readiness(
    manifest_path: Path,
) -> DensityCurriculumReadiness:
    """Evaluate whether a density-curriculum comparison manifest can support a smoke run.

    Returns:
        Fail-closed readiness packet for the input manifest.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    notes: list[str] = []

    if manifest.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        blockers.append("manifest schema_version must be density_curriculum_comparison.v1")
    if manifest.get("issue") != ISSUE_ID:
        blockers.append("manifest issue must be ll7/robot_sf_ll7#4018")
    if "no benchmark" not in str(manifest.get("claim_boundary", "")).lower():
        blockers.append("manifest claim_boundary must explicitly exclude benchmark claims")
    if bool(manifest.get("dry_run", True)):
        blockers.append("manifest is dry_run; paired training artifacts are not available")

    curriculum = _arm(manifest, "curriculum", blockers)
    baseline = _arm(manifest, "baseline", blockers)
    _check_pair(curriculum, baseline, blockers)
    _check_artifacts(manifest, blockers)

    if not blockers:
        notes.append("paired manifest is ready for diagnostic smoke comparison review")

    status = "ready_diagnostic_smoke" if not blockers else "blocked"
    return DensityCurriculumReadiness(
        status=status,
        blockers=tuple(blockers),
        notes=tuple(notes),
        manifest_path=str(manifest_path),
        comparison={
            "curriculum": _compact_arm(curriculum),
            "baseline": _compact_arm(baseline),
            "dry_run": bool(manifest.get("dry_run", True)),
        },
    )


def _arm(
    manifest: dict[str, Any],
    name: str,
    blockers: list[str],
) -> dict[str, Any]:
    raw = manifest.get(name)
    if not isinstance(raw, dict):
        blockers.append(f"{name} arm must be a mapping")
        return {}
    return raw


def _check_pair(
    curriculum: dict[str, Any],
    baseline: dict[str, Any],
    blockers: list[str],
) -> None:
    if curriculum.get("density_curriculum_enabled") is not True:
        blockers.append("curriculum arm must enable density_curriculum")
    if baseline.get("density_curriculum_enabled") is not False:
        blockers.append("baseline arm must disable density_curriculum")

    curriculum_steps = curriculum.get("total_timesteps")
    baseline_steps = baseline.get("total_timesteps")
    if curriculum_steps != baseline_steps:
        blockers.append("curriculum and baseline total_timesteps must match")
    if not isinstance(curriculum_steps, int) or curriculum_steps <= 0:
        blockers.append("paired total_timesteps must be a positive integer")

    curriculum_policy = curriculum.get("policy_id")
    baseline_policy = baseline.get("policy_id")
    if not curriculum_policy or not baseline_policy:
        blockers.append("both arms must declare policy_id")
    if curriculum_policy == baseline_policy:
        blockers.append("curriculum and baseline policy_id values must be distinct")


def _check_artifacts(manifest: dict[str, Any], blockers: list[str]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        blockers.append("manifest artifacts mapping is required for non-dry-run readiness")
        return

    for key in ("curriculum_checkpoint", "baseline_checkpoint"):
        value = artifacts.get(key)
        if not isinstance(value, str) or not value:
            blockers.append(f"artifacts.{key} must name a completed checkpoint path")


def _compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": arm.get("path"),
        "policy_id": arm.get("policy_id"),
        "total_timesteps": arm.get("total_timesteps"),
        "density_curriculum_enabled": arm.get("density_curriculum_enabled"),
    }
