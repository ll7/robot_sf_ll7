"""Post-readiness evidence-gap packet for adversarial Package B.

This module consumes the issue #3079 Package B readiness preflight and maps it
to the next safe decision. It does not run adversarial sampler comparisons,
submit Slurm jobs, or interpret benchmark evidence.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.adversarial_package_b_preflight import (
    PackageBPreflightResult,
    preflight_package_b_manifest,
)

SCHEMA_VERSION = "package-b-post-readiness-gap-packet.v1"
DEFAULT_MANIFEST = Path("configs/adversarial/issue_3079_package_b_budget_matched.yaml")


@dataclass(frozen=True)
class PackageBGapPacket:
    """Structured dry-run packet for the Package B readiness-to-evidence gap."""

    manifest_path: str
    status: str
    safe_next_decision: str
    readiness_ready: bool
    blockers: tuple[str, ...]
    expected_outputs: tuple[dict[str, str], ...]
    validation_commands: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    source_preflight: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """Return stable JSON payload for CLI snapshots and downstream review."""
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": self.manifest_path,
            "status": self.status,
            "safe_next_decision": self.safe_next_decision,
            "readiness_ready": self.readiness_ready,
            "blockers": list(self.blockers),
            "expected_outputs": list(self.expected_outputs),
            "validation_commands": list(self.validation_commands),
            "out_of_scope": list(self.out_of_scope),
            "source_preflight": self.source_preflight,
        }


def _expected_outputs(preflight: PackageBPreflightResult) -> tuple[dict[str, str], ...]:
    """Return expected Package B outputs without requiring them to exist yet."""
    artifacts = preflight.metadata.get("output_artifacts", {})
    output_dir = artifacts.get("output_dir")
    report_json = artifacts.get("report_json")
    outputs: list[dict[str, str]] = []

    if isinstance(output_dir, str) and output_dir:
        outputs.append(
            {
                "path": output_dir,
                "artifact_class": "worktree_local_campaign_output",
                "required_before": "local_or_slurm_campaign_execution",
                "durability": "not_durable_until_promoted_or_manifested",
            }
        )
    if isinstance(report_json, str) and report_json:
        outputs.append(
            {
                "path": report_json,
                "artifact_class": "package_b_sampler_comparison_report",
                "required_before": "evidence_review",
                "durability": "not_durable_until_promoted_or_manifested",
            }
        )

    outputs.append(
        {
            "path": "output/adversarial/issue_3079_package_b/post_readiness_gap_packet.json",
            "artifact_class": "dry_run_decision_packet",
            "required_before": "handoff_to_campaign_operator",
            "durability": "disposable_unless_copied_to_reviewable_evidence",
        }
    )
    return tuple(outputs)


def _validation_commands(manifest_path: str) -> tuple[str, ...]:
    """Return commands that validate readiness before any campaign launch.

    The manifest path is shell-quoted so the emitted copy-paste commands stay
    correct even if the path contains spaces or shell metacharacters.
    """
    quoted_path = shlex.quote(manifest_path)
    return (
        "uv run python scripts/tools/preflight_adversarial_package_b.py "
        f"--manifest {quoted_path}",
        "uv run python scripts/tools/prepare_package_b_post_readiness_gap_packet.py "
        f"--manifest {quoted_path} "
        "--output output/adversarial/issue_3079_package_b/post_readiness_gap_packet.json",
        "uv run pytest tests/benchmark/test_adversarial_package_b_gap_packet.py",
    )


def _blockers(preflight: PackageBPreflightResult) -> tuple[str, ...]:
    """Return readiness blockers plus the fixed campaign-submission guard."""
    blockers = list(preflight.blockers)
    if preflight.ready:
        blockers.append(
            "Slurm/GPU submission remains blocked until a human operator chooses to launch the "
            "campaign from a clean owning worktree after local dry-run validation."
        )
    return tuple(blockers)


def build_package_b_gap_packet(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repo_root: Path | None = None,
) -> PackageBGapPacket:
    """Build the Package B post-readiness gap packet without executing a campaign.

    Returns:
        Dry-run packet that maps readiness metadata to the next safe decision.
    """
    preflight = preflight_package_b_manifest(manifest_path, repo_root=repo_root)
    status = "ready_for_local_dry_run_handoff" if preflight.ready else "blocked_on_readiness"
    decision = (
        "Run only the listed local validation commands, then decide whether a separate "
        "Slurm-capable operator should launch Package B."
        if preflight.ready
        else "Repair the readiness blockers before any local or Slurm campaign decision."
    )

    return PackageBGapPacket(
        manifest_path=preflight.manifest_path,
        status=status,
        safe_next_decision=decision,
        readiness_ready=preflight.ready,
        blockers=_blockers(preflight),
        expected_outputs=_expected_outputs(preflight),
        validation_commands=_validation_commands(preflight.manifest_path),
        out_of_scope=(
            "no full benchmark campaign run",
            "no Slurm/GPU submission",
            "no paper/dissertation claim edits",
        ),
        source_preflight=preflight.to_payload(),
    )
