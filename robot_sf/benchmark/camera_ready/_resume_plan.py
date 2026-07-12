"""Resume-plan preflight for camera-ready benchmark campaigns (issue #5392).

Before a resumed campaign executes, this module inspects existing arm directories
to build a **resume plan** that states:

- per arm: episodes found on disk, episodes remaining, verdict
  (skip-complete | continue-from-N | fresh)
- the campaign-id/config-hash match check (mismatch = fail-closed)
- totals: episodes banked vs to-run

The plan is logged and written to ``resume_plan.json`` in the campaign root so
the operator can sanity-check projected walltime before GPU resources are consumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.camera_ready._config import _sanitize_name
from robot_sf.benchmark.camera_ready._util import _utc_now
from robot_sf.benchmark.utils import load_optional_json


@dataclass(frozen=True)
class ArmResumeVerdict:
    """Resume verdict for a single planner/kinematics arm."""

    planner_key: str
    kinematics: str
    arm_dir: Path
    episodes_found: int
    expected_total: int
    episodes_remaining: int
    verdict: str  # "skip-complete" | "continue-from-N" | "fresh"
    episodes_path: Path | None = None
    prior_summary: dict[str, Any] | None = None


class ResumeMismatchError(ValueError):
    """Raised when the campaign-id or config-hash on disk does not match."""


def _count_jsonl_episodes(episodes_path: Path) -> int:
    """Return the number of episode records in a JSONL file."""
    if not episodes_path.is_file():
        return 0
    count = 0
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_number} in {episodes_path}: {exc}"
                ) from exc
            count += 1
    return count


def _expected_jobs(scenarios: list[dict[str, Any]]) -> int:
    """Return the number of jobs ``run_batch`` would expand from *scenarios*.

    Mirrors the logic in ``runner._expand_jobs``: each scenario has a
    ``repeats`` field (default 1) since the campaign orchestrator never sets
    ``repeats_override``.
    """
    total = 0
    for sc in scenarios:
        repeats = sc.get("repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
            raise ValueError(f"scenario repeats must be a non-negative integer, got {repeats!r}")
        total += repeats
    return total


def _prior_config_hash(campaign_root: Path) -> str | None:
    """Extract config_hash from the on-disk campaign manifest.

    Returns:
        The config hash string if present on disk, otherwise None.
    """
    manifest = _load_resume_manifest(campaign_root)
    value = str(manifest.get("config_hash", "")).strip()
    if not value:
        raise ValueError("campaign manifest is missing a non-empty config_hash")
    return value


def _prior_campaign_id(campaign_root: Path) -> str | None:
    """Extract campaign_id from the on-disk campaign manifest.

    Returns:
        The campaign id string if present on disk, otherwise None.
    """
    manifest = _load_resume_manifest(campaign_root)
    value = str(manifest.get("campaign_id", "")).strip()
    if not value:
        raise ValueError("campaign manifest is missing a non-empty campaign_id")
    return value


def _load_resume_manifest(campaign_root: Path) -> dict[str, Any]:
    """Load the required campaign manifest or fail closed.

    Returns:
        Parsed campaign manifest mapping.
    """
    manifest_path = campaign_root / "campaign_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Campaign manifest missing at {manifest_path}")
    manifest = load_optional_json(str(manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f"Campaign manifest must be a JSON object: {manifest_path}")
    return manifest


def verify_resume_context(
    campaign_root: Path,
    *,
    campaign_id: str,
    config_hash: str,
) -> None:
    """Verify that the existing campaign on disk matches the current invocation.

    Args:
        campaign_root: Root directory of the campaign to resume.
        campaign_id: Campaign id for the current invocation.
        config_hash: Config hash for the current invocation.

    Raises:
        ResumeMismatchError: If the campaign-id or config-hash on disk does not
            match the current invocation.
    """
    errors: list[str] = []
    prior_id = _prior_campaign_id(campaign_root)
    if prior_id is not None and prior_id != campaign_id:
        errors.append(f"campaign-id mismatch: disk='{prior_id}' expected='{campaign_id}'")
    prior_hash = _prior_config_hash(campaign_root)
    if prior_hash is not None and prior_hash != config_hash:
        errors.append(f"config-hash mismatch: disk='{prior_hash}' expected='{config_hash}'")
    if errors:
        raise ResumeMismatchError(
            f"resume context mismatch for campaign root {campaign_root}: " + "; ".join(errors)
        )


def _build_verdict_str(
    episodes_found: int,
    expected_total: int,
) -> str:
    """Return a human-readable verdict label."""
    if expected_total <= 0:
        return "fresh"
    if episodes_found >= expected_total:
        return "skip-complete"
    return f"continue-from-{episodes_found}"


def build_resume_plan(
    runs_dir: Path,
    *,
    planners: list[dict[str, Any]],
    kinematics_matrix: list[str],
    scenarios: list[dict[str, Any]],
    expected_jobs: int | None = None,
) -> list[ArmResumeVerdict]:
    """Build a resume plan by inspecting existing arm directories.

    Args:
        runs_dir: The ``runs/`` directory under campaign root.
        planners: Enabled planner specs. Each planner must have ``key`` and
            optionally ``enabled`` (defaults True).
        kinematics_matrix: List of kinematics variants to run.
        scenarios: Scenario list that would be passed to ``run_batch``.
        expected_jobs: Optional precomputed expected job count. If not given,
            computed from scenario ``repeats`` fields.

    Returns:
        List of ``ArmResumeVerdict`` objects describing the plan for each arm.
    """
    if expected_jobs is None:
        expected_jobs = _expected_jobs(scenarios)
    elif isinstance(expected_jobs, bool) or not isinstance(expected_jobs, int) or expected_jobs < 0:
        raise ValueError(f"expected_jobs must be a non-negative integer, got {expected_jobs!r}")

    enabled_planners = [p for p in planners if isinstance(p, dict) and p.get("enabled", True)]

    verdicts: list[ArmResumeVerdict] = []
    for planner in enabled_planners:
        planner_key = str(planner.get("key", "unknown"))
        for kinematics in kinematics_matrix:
            arm_name = f"{_sanitize_name(planner_key)}__{_sanitize_name(kinematics)}"
            arm_dir = runs_dir / arm_name
            if arm_dir.exists() and not arm_dir.is_dir():
                raise NotADirectoryError(f"Resume arm path is not a directory: {arm_dir}")
            episodes_path = arm_dir / "episodes.jsonl" if arm_dir.exists() else None

            episodes_found = 0
            prior_summary = None
            if episodes_path is not None and episodes_path.exists():
                episodes_found = _count_jsonl_episodes(episodes_path)

            summary_path = arm_dir / "summary.json" if arm_dir.exists() else None
            if summary_path is not None and summary_path.exists():
                prior_summary = load_optional_json(str(summary_path))

            episodes_remaining = max(0, expected_jobs - episodes_found)
            verdict = _build_verdict_str(episodes_found, expected_jobs)

            verdicts.append(
                ArmResumeVerdict(
                    planner_key=planner_key,
                    kinematics=kinematics,
                    arm_dir=arm_dir,
                    episodes_found=episodes_found,
                    expected_total=expected_jobs,
                    episodes_remaining=episodes_remaining,
                    verdict=verdict,
                    episodes_path=episodes_path,
                    prior_summary=prior_summary,
                )
            )
    return verdicts


def resume_plan_summary(verdicts: list[ArmResumeVerdict]) -> dict[str, Any]:
    """Return plan totals and a compact arm summary."""
    episodes_banked = sum(v.episodes_found for v in verdicts)
    episodes_to_run = sum(v.episodes_remaining for v in verdicts)
    skips = [v for v in verdicts if v.verdict == "skip-complete"]
    continues = [v for v in verdicts if v.verdict.startswith("continue-from-")]
    freshes = [v for v in verdicts if v.verdict == "fresh"]

    return {
        "total_arms": len(verdicts),
        "arms_skip_complete": len(skips),
        "arms_continue": len(continues),
        "arms_fresh": len(freshes),
        "episodes_banked": episodes_banked,
        "episodes_to_run": episodes_to_run,
        "expected_total_episodes": sum(v.expected_total for v in verdicts),
        "arms": [
            {
                "planner_key": v.planner_key,
                "kinematics": v.kinematics,
                "episodes_found": v.episodes_found,
                "expected_total": v.expected_total,
                "episodes_remaining": v.episodes_remaining,
                "verdict": v.verdict,
            }
            for v in verdicts
        ],
    }


def emit_resume_plan_log(verdicts: list[ArmResumeVerdict]) -> None:
    """Emit a structured log summary of the resume plan."""
    summary = resume_plan_summary(verdicts)
    logger.info(
        "Resume plan: %d arms (%d skip, %d continue, %d fresh); %d episodes banked, %d to-run",
        summary["total_arms"],
        summary["arms_skip_complete"],
        summary["arms_continue"],
        summary["arms_fresh"],
        summary["episodes_banked"],
        summary["episodes_to_run"],
    )
    for v in verdicts:
        logger.info(
            "  arm '%s' (%s): %d/%d episodes -> %s",
            v.planner_key,
            v.kinematics,
            v.episodes_found,
            v.expected_total,
            v.verdict,
        )


def write_resume_plan(
    campaign_root: Path,
    *,
    config_hash: str,
    campaign_id: str,
    verdicts: list[ArmResumeVerdict],
) -> Path:
    """Write the resume plan JSON artifact to the campaign root.

    Returns:
        Path to the written ``resume_plan.json`` file.
    """
    plan_path = campaign_root / "resume_plan.json"
    payload = {
        "schema_version": "benchmark-resume-plan.v1",
        "campaign_id": campaign_id,
        "config_hash": config_hash,
        "generated_at_utc": _utc_now(),
        "context_check": {
            "config_hash_match": True,
            "campaign_id_match": True,
        },
        **resume_plan_summary(verdicts),
    }
    plan_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return plan_path
