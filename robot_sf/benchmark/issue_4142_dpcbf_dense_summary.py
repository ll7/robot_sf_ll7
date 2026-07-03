"""Plan-consuming result summarizer for the issue #4142 dense DPCBF comparison.

Issue #4142 asks for a bounded dense dynamic-obstacle comparison of three predeclared
Control Barrier Function (CBF) arms -- unfiltered (``cbf_off``), collision-cone CBF
(``cbf_collision_cone_on``), and the Dynamic Parabolic CBF variant
(``cbf_dynamic_parabolic_v1_on``). The upstream slices built the read-only *readiness*
surface (:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_readiness`, PR #4299) and the
packet-consuming *run planner*
(:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_runner`, PR #4318) that resolves the
packet schema ``robot_sf.issue_4142_dpcbf_dense_comparison.v1`` into an ordered three-arm
run plan with a per-arm output JSONL path each. The plan is executable-in-principle but
execution stays authorization-gated, so no arm output exists yet.

This module closes the next downstream gate: it consumes the resolved run plan and, for
each planned arm, reads that arm's per-episode JSONL output (if present) into a fail-closed
comparison **summary** under schema
``robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1``. It is the surface a future
authorized campaign's outputs flow through; until then it reports, honestly and
fail-closed, that results are incomplete.

Design boundaries (all fail-closed):

- **Single source of truth.** The set of arms, their output paths, the shared fail-closed
  row-status exclusion (``fallback``, ``degraded``, ``failed``, ``ineligible`` are caveats,
  never success evidence), and the plan/readiness gates all come from
  :func:`robot_sf.benchmark.issue_4142_dpcbf_dense_runner.build_run_plan`. Nothing is
  re-derived from the packet here.
- **Plan gate.** If the run plan is not ``plan_ready_campaign_gated`` the summary is
  ``plan_blocked``: it consumes no artifacts and surfaces the plan's blockers. There is no
  path that summarizes results from an invalid plan.
- **Artifact gate / caveat separation.** A row's status is compared against the plan's
  declared ``excluded_row_statuses``. Excluded rows are counted as **caveats**, broken out
  by status, and are *never* added to success-evidence counts. A missing, empty, or
  unparseable arm artifact is recorded in the artifact manifest with an explicit
  ``missing_status`` and keeps the comparison out of the ``complete`` state.
- **No execution.** Summarizing runs no episodes, launches no campaign, submits no
  Slurm/GPU job, and makes no safety-performance or collision-reduction claim.

Status semantics:

- ``plan_blocked`` -- the underlying run plan did not resolve; no artifacts were consumed.
- ``results_incomplete`` -- the plan is ready but at least one required arm artifact is
  missing/empty/unparseable, so the three-arm comparison cannot be called complete. This is
  the expected state while execution stays gated (no arm output exists yet).
- ``complete`` -- the plan is ready and all three required arms have a present, parseable
  artifact with at least one row. ``complete`` describes artifact *coverage*, not a
  benchmark conclusion; caveat rows are reported separately and never upgraded to success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import REQUIRED_ARMS
from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import (
    DEFAULT_OUTPUT_DIR,
    PACKET_PATH,
    DenseComparisonRunnerError,
    build_run_plan,
)

#: Output-contract schema for this comparison summary surface.
SUMMARY_SCHEMA_VERSION = "robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1"

#: Row-status labels that count as success evidence when *not* excluded by the plan. Mirrors
#: the accepted-success vocabulary in :mod:`robot_sf.benchmark.fallback_policy`
#: (``ok``/``benchmark_success``); anything unrecognized stays a caveat, never success.
_SUCCESS_ROW_STATUSES: frozenset[str] = frozenset(
    {"ok", "success", "successful_evidence", "benchmark_success", "complete"}
)

#: Claim boundary emitted with every summary so a reader never mistakes a resolved summary
#: for a benchmark result. Mirrors the packet's diagnostic-only, bounded evidence tier.
CLAIM_BOUNDARY = (
    "Bounded diagnostic comparison summary only. Summarizing runs no episodes, authorizes "
    "no campaign, submits no Slurm/GPU job, and makes no safety-performance or "
    "collision-reduction claim. Fallback, degraded, failed, or ineligible rows are caveats "
    "and are never success evidence; a 'complete' status reports artifact coverage, not a "
    "safety conclusion."
)


class DenseComparisonSummaryError(ValueError):
    """Raised when the run plan cannot be built into a summary at all."""


@dataclass(frozen=True, slots=True)
class ArmResultSummary:
    """Per-arm artifact/row summary for one predeclared comparison arm."""

    arm_key: str
    enabled: bool
    variant: str
    output_jsonl: str
    #: One of ``present``, ``missing``, ``empty``, or ``unparseable``.
    missing_status: str
    artifact_present: bool
    row_count: int
    success_evidence_rows: int
    caveat_rows: int
    #: Counts of caveat rows keyed by their (excluded) row status, sorted for stability.
    caveat_rows_by_status: dict[str, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    @property
    def has_success_evidence(self) -> bool:
        """True when this arm has at least one non-excluded success-evidence row."""
        return self.artifact_present and self.success_evidence_rows > 0


@dataclass(frozen=True, slots=True)
class DenseComparisonSummary:
    """Fail-closed, plan-gated comparison summary resolved from per-arm artifacts."""

    schema_version: str
    packet_path: str
    plan_schema_version: str
    plan_status: str
    readiness_status: str
    output_dir: str
    excluded_row_statuses: tuple[str, ...]
    fallback_excluded: bool
    arms: tuple[ArmResultSummary, ...]
    status: str
    blockers: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def all_arms_have_success_evidence(self) -> bool:
        """True when every required arm has at least one success-evidence row.

        This is *not* a benchmark conclusion; it only reports that each arm produced
        non-excluded rows. It is surfaced so a reader never conflates artifact coverage
        (``status == "complete"``) with the presence of comparable success rows per arm.
        """
        by_key = {arm.arm_key: arm for arm in self.arms}
        return all(key in by_key and by_key[key].has_success_evidence for key in REQUIRED_ARMS)


def _read_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], str, tuple[str, ...]]:
    """Read a per-arm episode JSONL file into row mappings.

    Each non-blank line must be a JSON object; the row's ``status`` field (default
    ``"unknown"``) drives success/caveat classification downstream.

    Returns:
        ``(rows, missing_status, errors)`` where ``missing_status`` is one of ``present``,
        ``missing``, ``empty``, or ``unparseable``.
    """
    if not path.is_file():
        return [], "missing", (f"artifact not found: {path.as_posix()}",)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: not valid JSON ({exc})")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"line {lineno}: JSON row is not an object")
            continue
        rows.append(parsed)

    if errors:
        # Any malformed content fails closed for the whole artifact: a partially parseable
        # file must never be summarized as if it were complete evidence.
        return rows, "unparseable", tuple(errors)
    if not rows:
        return [], "empty", (f"artifact has no JSON rows: {path.as_posix()}",)
    return rows, "present", ()


def _summarize_arm(
    *,
    arm_key: str,
    enabled: bool,
    variant: str,
    output_jsonl: str,
    repo_root: Path,
    excluded_statuses: frozenset[str],
) -> ArmResultSummary:
    """Read and classify one arm's artifact rows against the fail-closed exclusion set.

    Returns:
        A per-arm result summary with success/caveat counts and any structural errors.
    """
    artifact_abs = Path(output_jsonl)
    if not artifact_abs.is_absolute():
        artifact_abs = repo_root / artifact_abs

    rows, missing_status, errors = _read_jsonl_rows(artifact_abs)
    present = missing_status == "present"

    success_rows = 0
    caveat_rows = 0
    caveat_by_status: dict[str, int] = {}
    if present:
        for row in rows:
            status = str(row.get("status", "unknown")).strip().lower()
            if status in excluded_statuses:
                caveat_rows += 1
                caveat_by_status[status] = caveat_by_status.get(status, 0) + 1
            elif status in _SUCCESS_ROW_STATUSES:
                success_rows += 1
            else:
                # Unrecognized status: fail-closed as a caveat, never success evidence.
                caveat_rows += 1
                caveat_by_status[status] = caveat_by_status.get(status, 0) + 1

    return ArmResultSummary(
        arm_key=arm_key,
        enabled=enabled,
        variant=variant,
        output_jsonl=output_jsonl,
        missing_status=missing_status,
        artifact_present=present,
        row_count=len(rows) if present else 0,
        success_evidence_rows=success_rows,
        caveat_rows=caveat_rows,
        caveat_rows_by_status=dict(sorted(caveat_by_status.items())),
        errors=errors,
    )


def summarize_dense_comparison(
    repo_root: str | Path = ".",
    packet_path: str | Path = PACKET_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> DenseComparisonSummary:
    """Consume the resolved run plan into a fail-closed comparison summary.

    The run plan is the single source of truth for arms, output paths, and the fail-closed
    row-status exclusion. When the plan is not ``plan_ready_campaign_gated`` the summary is
    ``plan_blocked`` and consumes no artifacts. Otherwise each required arm's JSONL output
    is read (if present) and classified against the plan's ``excluded_row_statuses``.

    Args:
        repo_root: Directory that repo-relative packet/config/artifact paths resolve against.
        packet_path: Repo-relative (or absolute) path to the comparison packet.
        output_dir: Directory the per-arm output JSONL paths were planned under.

    Returns:
        The resolved comparison summary.

    Raises:
        DenseComparisonSummaryError: if the run plan cannot be built at all.
    """
    root = Path(repo_root)
    try:
        plan = build_run_plan(repo_root=root, packet_path=packet_path, output_dir=output_dir)
    except DenseComparisonRunnerError as exc:
        raise DenseComparisonSummaryError(str(exc)) from exc

    excluded_statuses = frozenset(str(s).strip().lower() for s in plan.excluded_row_statuses)

    if not plan.is_executable_in_principle:
        # Plan gate: never summarize results from an invalid/blocked plan.
        return DenseComparisonSummary(
            schema_version=SUMMARY_SCHEMA_VERSION,
            packet_path=str(packet_path),
            plan_schema_version=plan.schema_version,
            plan_status=plan.status,
            readiness_status=plan.readiness_status,
            output_dir=plan.output_dir,
            excluded_row_statuses=plan.excluded_row_statuses,
            fallback_excluded=plan.fallback_excluded,
            arms=(),
            status="plan_blocked",
            blockers=plan.blockers,
        )

    arms = tuple(
        _summarize_arm(
            arm_key=job.arm_key,
            enabled=job.enabled,
            variant=job.variant,
            output_jsonl=job.output_jsonl,
            repo_root=root,
            excluded_statuses=excluded_statuses,
        )
        for job in plan.arms
    )

    blockers: list[str] = []
    present_keys = {arm.arm_key for arm in arms if arm.artifact_present}
    missing_required = sorted(set(REQUIRED_ARMS) - present_keys)
    if missing_required:
        blockers.append(
            f"arm artifacts missing/empty/unparseable for required arms: {missing_required}"
        )
    for arm in arms:
        blockers.extend(arm.errors)

    # ``complete`` requires every required arm to have a present, parseable, non-empty
    # artifact. Caveat rows do not block completeness; they are reported separately.
    status = "complete" if not missing_required else "results_incomplete"

    return DenseComparisonSummary(
        schema_version=SUMMARY_SCHEMA_VERSION,
        packet_path=str(packet_path),
        plan_schema_version=plan.schema_version,
        plan_status=plan.status,
        readiness_status=plan.readiness_status,
        output_dir=plan.output_dir,
        excluded_row_statuses=plan.excluded_row_statuses,
        fallback_excluded=plan.fallback_excluded,
        arms=arms,
        status=status,
        blockers=tuple(blockers),
    )


def to_dict(summary: DenseComparisonSummary) -> dict[str, Any]:
    """Return a JSON-serializable view of the comparison summary."""
    return {
        "schema_version": summary.schema_version,
        "packet_path": summary.packet_path,
        "plan_schema_version": summary.plan_schema_version,
        "plan_status": summary.plan_status,
        "readiness_status": summary.readiness_status,
        "status": summary.status,
        "claim_boundary": summary.claim_boundary,
        "output_dir": summary.output_dir,
        "excluded_row_statuses": list(summary.excluded_row_statuses),
        "fallback_excluded": summary.fallback_excluded,
        "all_arms_have_success_evidence": summary.all_arms_have_success_evidence,
        "artifact_manifest": [
            {
                "arm_key": arm.arm_key,
                "enabled": arm.enabled,
                "variant": arm.variant,
                "output_jsonl": arm.output_jsonl,
                "missing_status": arm.missing_status,
                "artifact_present": arm.artifact_present,
                "row_count": arm.row_count,
                "success_evidence_rows": arm.success_evidence_rows,
                "caveat_rows": arm.caveat_rows,
                "caveat_rows_by_status": arm.caveat_rows_by_status,
                "has_success_evidence": arm.has_success_evidence,
                "errors": list(arm.errors),
            }
            for arm in summary.arms
        ],
        "blockers": list(summary.blockers),
    }


def render_markdown(summary: DenseComparisonSummary) -> str:
    """Render a compact Markdown report leading with the claim boundary and status.

    Returns:
        A Markdown string describing the resolved (or blocked) comparison summary.
    """
    lines: list[str] = []
    lines.append("# Issue #4142 dense DPCBF comparison summary")
    lines.append("")
    lines.append(f"Claim boundary: {summary.claim_boundary}")
    lines.append("")
    lines.append(
        f"- Status: `{summary.status}` (plan: `{summary.plan_status}`, "
        f"readiness: `{summary.readiness_status}`)"
    )
    lines.append(f"- Packet: `{summary.packet_path}`")
    lines.append(f"- Output dir (planned): `{summary.output_dir}`")
    lines.append(
        f"- Excluded row statuses (caveats, never success): "
        f"{', '.join(f'`{s}`' for s in summary.excluded_row_statuses) or '(none)'}"
    )
    lines.append(
        f"- Fallback/degraded excluded: {summary.fallback_excluded}; "
        f"all arms have success evidence: {summary.all_arms_have_success_evidence}"
    )
    lines.append("")
    lines.append("## Artifact manifest")
    lines.append("")
    if summary.arms:
        lines.append(
            "| arm_key | artifact | rows | success rows | caveat rows | caveat breakdown |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for arm in summary.arms:
            caveat_breakdown = (
                ", ".join(f"{k}={v}" for k, v in arm.caveat_rows_by_status.items()) or "-"
            )
            lines.append(
                f"| `{arm.arm_key}` | `{arm.missing_status}` | {arm.row_count} | "
                f"{arm.success_evidence_rows} | {arm.caveat_rows} | {caveat_breakdown} |"
            )
    else:
        lines.append("_No arm artifacts consumed (summary is plan-blocked; see below)._")
    lines.append("")
    if summary.blockers:
        lines.append("## Blockers (fail-closed)")
        lines.append("")
        for blocker in summary.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "CLAIM_BOUNDARY",
    "SUMMARY_SCHEMA_VERSION",
    "ArmResultSummary",
    "DenseComparisonSummary",
    "DenseComparisonSummaryError",
    "render_markdown",
    "summarize_dense_comparison",
    "to_dict",
]
