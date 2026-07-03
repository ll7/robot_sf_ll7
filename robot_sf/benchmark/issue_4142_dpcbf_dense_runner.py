"""Packet-consuming run planner for the issue #4142 dense DPCBF comparison.

Issue #4142 asks for a bounded dense dynamic-obstacle comparison of three predeclared
Control Barrier Function (CBF) arms -- unfiltered (``cbf_off``), collision-cone CBF
(``cbf_collision_cone_on``), and the Dynamic Parabolic CBF variant
(``cbf_dynamic_parabolic_v1_on``). PR #4299 added the read-only *readiness* surface
(:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_readiness`) that validates the predeclared
packet ``configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`` but explicitly left
one downstream gate open: *no packet-consuming runner is wired to schema*
``robot_sf.issue_4142_dpcbf_dense_comparison.v1``.

This module closes that first gate at the planning level. It consumes the packet schema and
resolves it into a concrete, ordered three-arm **run plan** -- one benchmark job per arm,
each pinned to the packet's shared algorithm, that arm's adapter config, the shared scenario
manifest, and a per-arm output path. The plan is what a future authorized executor would
run; building it makes the packet schema executable-in-principle and inspectable, so the
canonical command can enumerate the exact jobs instead of failing on an unrecognized schema.

Two hard boundaries are preserved, fail-closed:

- **Readiness gate.** The plan is built only when the canonical readiness validator reports
  ``inputs_ready_campaign_gated``. If any packet input is missing, invalid, or not
  fail-closed, :func:`build_run_plan` returns a plan with status
  ``prerequisites_incomplete``, no executable arm jobs, and the readiness blockers surfaced.
  There is deliberately no path that resolves executable jobs from an invalid packet.
- **Execution gate.** Running the dense comparison (episodes, Slurm/GPU) stays out of scope
  for this slice. :func:`execute_run_plan` never runs episodes; it fails closed with
  :class:`DenseComparisonExecutionGatedError`, because execution requires explicit
  human/Slurm authorization that this runner intentionally does not grant.

The fail-closed row-status exclusion (``fallback``, ``degraded``, ``failed``, ``ineligible``
are caveats, never success evidence) is carried verbatim from the packet into the resolved
plan, so any downstream summarizer inherits the exclusion from the plan rather than
re-deriving it.

Status semantics:

- ``prerequisites_incomplete`` -- readiness failed; no executable arm jobs were resolved.
- ``plan_ready_campaign_gated`` -- every input is valid; the three-arm plan is resolved and
  inspectable, but execution stays gated behind :data:`RUNNER_GATES`. This is the expected
  healthy state; it confirms the plan is reviewable, *not* that the comparison may run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    PACKET_SCHEMA_VERSION,
    REQUIRED_ARMS,
    REQUIRED_EXCLUDED_ROW_STATUSES,
    DenseComparisonReadiness,
    DpcbfDenseReadinessError,
    evaluate_readiness,
    load_packet,
)

#: Output-contract schema for the resolved run plan.
PLAN_SCHEMA_VERSION = "robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1"

#: Default output directory for the three per-arm episode JSONL files. Under the git-ignored
#: ``output/`` root per AGENTS.md; nothing is written by planning, only recorded in the plan.
DEFAULT_OUTPUT_DIR = "output/issue_4142_dpcbf_dense"

#: Claim boundary emitted with every plan so a reader never mistakes a resolved plan for a
#: result. Mirrors the packet's diagnostic-only, bounded evidence tier.
CLAIM_BOUNDARY = (
    "Resolved run plan for a bounded diagnostic comparison only. Building this plan runs no "
    "episodes, authorizes no campaign, submits no Slurm/GPU job, and makes no "
    "safety-performance or collision-reduction claim. Fallback, degraded, failed, or "
    "ineligible rows are caveats and are never success evidence."
)

#: Gates that keep a valid plan from executing here. Surfaced verbatim so
#: ``plan_ready_campaign_gated`` is never mistaken for a go-ahead to run.
RUNNER_GATES: tuple[str, ...] = (
    "executing the dense comparison requires explicit human/Slurm authorization and is out "
    "of scope for this runner slice; execute_run_plan() fails closed",
    "running episodes (CPU or GPU) is deferred to the authorized campaign, not performed by "
    "this planner",
)


class DenseComparisonRunnerError(ValueError):
    """Raised when the packet cannot be consumed into a run plan at all."""


class DenseComparisonExecutionGatedError(RuntimeError):
    """Raised when execution is attempted; execution stays authorization-gated here."""


@dataclass(frozen=True, slots=True)
class ArmJobPlan:
    """One resolved benchmark job for a single predeclared comparison arm."""

    arm_key: str
    enabled: bool
    variant: str
    algorithm: str
    algorithm_config: str
    scenario_manifest: str
    output_jsonl: str


@dataclass(frozen=True, slots=True)
class DenseComparisonRunPlan:
    """A fail-closed, campaign-gated run plan resolved from the comparison packet."""

    schema_version: str
    packet_path: str
    packet_schema_version: str
    algorithm: str | None
    scenario_manifest: str | None
    output_dir: str
    evidence_tier: str | None
    fallback_rows_are_success_evidence: bool
    excluded_row_statuses: tuple[str, ...]
    fallback_excluded: bool
    arms: tuple[ArmJobPlan, ...]
    status: str
    blockers: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY
    runner_gates: tuple[str, ...] = RUNNER_GATES
    campaign_gates: tuple[str, ...] = ()
    readiness_status: str = ""

    @property
    def is_executable_in_principle(self) -> bool:
        """True when the plan resolved fully and only the execution gate remains.

        Note that this is *not* a permission to run: :func:`execute_run_plan` still fails
        closed. It only means the packet resolved into a complete, reviewable three-arm plan.
        """
        return self.status == "plan_ready_campaign_gated"


def _output_jsonl_for(arm_key: str, output_dir: str) -> str:
    """Return the per-arm episode JSONL output path (as a POSIX-style string)."""
    return (Path(output_dir) / f"{arm_key}.jsonl").as_posix()


def _resolve_arm_jobs(
    readiness: DenseComparisonReadiness,
    *,
    algorithm: str,
    scenario_manifest: str,
    output_dir: str,
) -> tuple[ArmJobPlan, ...]:
    """Resolve one benchmark job per predeclared arm, preserving packet order.

    Only the required arms are turned into jobs; each is pinned to the shared algorithm, the
    arm's validated adapter config, the shared scenario manifest, and a per-arm output path.

    Returns:
        Ordered per-arm job plans (empty when no required arm resolved a config).
    """
    jobs: list[ArmJobPlan] = []
    for arm in readiness.arms:
        if arm.arm_key not in REQUIRED_ARMS:
            continue
        # Readiness already validated the adapter config exists and matches the arm; a
        # missing path here would mean readiness was not ready, so this stays defensive.
        if arm.algorithm_config_path is None:
            continue
        jobs.append(
            ArmJobPlan(
                arm_key=arm.arm_key,
                enabled=arm.enabled,
                variant=arm.variant,
                algorithm=algorithm,
                algorithm_config=arm.algorithm_config_path,
                scenario_manifest=scenario_manifest,
                output_jsonl=_output_jsonl_for(arm.arm_key, output_dir),
            )
        )
    return tuple(jobs)


def build_run_plan(
    repo_root: str | Path = ".",
    packet_path: str | Path = PACKET_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> DenseComparisonRunPlan:
    """Consume the comparison packet into a fail-closed, campaign-gated run plan.

    The plan is resolved into executable per-arm jobs only when the canonical readiness
    validator reports ``inputs_ready_campaign_gated``. Otherwise the returned plan carries
    status ``prerequisites_incomplete``, no arm jobs, and the readiness blockers -- there is
    no path that resolves executable jobs from an invalid packet.

    Args:
        repo_root: Directory that repo-relative packet/config paths resolve against.
        packet_path: Repo-relative (or absolute) path to the comparison packet.
        output_dir: Directory the per-arm output JSONL paths are planned under. Recorded in
            the plan only; nothing is written by planning.

    Returns:
        The resolved run plan.

    Raises:
        DenseComparisonRunnerError: if the packet cannot be loaded or parsed at all.
    """
    readiness = evaluate_readiness(repo_root=repo_root, packet_path=packet_path)
    output_dir_str = Path(output_dir).as_posix()

    try:
        packet = load_packet(_as_abs(repo_root, packet_path))
    except DpcbfDenseReadinessError as exc:  # pragma: no cover - readiness raised first
        raise DenseComparisonRunnerError(str(exc)) from exc

    algorithm = packet.get("algorithm")
    algorithm = str(algorithm) if algorithm is not None else None
    contract = packet.get("summary_contract")
    contract = contract if isinstance(contract, dict) else {}
    excluded = contract.get("excluded_row_statuses")
    excluded_tuple = tuple(str(s) for s in excluded) if isinstance(excluded, list) else ()
    fallback_flag = contract.get("fallback_rows_are_success_evidence", None)

    blockers = list(readiness.blockers)
    # Fail closed on the runner's own preconditions, independent of readiness wording, so the
    # run plan can never silently drop the shared algorithm or the fail-closed exclusion.
    if not algorithm:
        blockers.append("packet does not declare a shared 'algorithm' for the comparison")
    missing_excluded = sorted(set(REQUIRED_EXCLUDED_ROW_STATUSES) - set(excluded_tuple))
    if missing_excluded:
        blockers.append(
            f"packet summary_contract.excluded_row_statuses missing fail-closed statuses: "
            f"{missing_excluded}"
        )

    ready = readiness.inputs_ready and not blockers
    if ready and algorithm is not None and readiness.scenario_manifest_path is not None:
        arms = _resolve_arm_jobs(
            readiness,
            algorithm=algorithm,
            scenario_manifest=readiness.scenario_manifest_path,
            output_dir=output_dir_str,
        )
        # Defensive: never report a ready plan that is missing an arm the comparison needs.
        if {job.arm_key for job in arms} != set(REQUIRED_ARMS):
            blockers.append("resolved run plan is missing a required comparison arm")
            arms = ()
    else:
        arms = ()

    status = "plan_ready_campaign_gated" if arms and not blockers else "prerequisites_incomplete"

    return DenseComparisonRunPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        packet_path=str(packet_path),
        packet_schema_version=PACKET_SCHEMA_VERSION,
        algorithm=algorithm,
        scenario_manifest=readiness.scenario_manifest_path,
        output_dir=output_dir_str,
        evidence_tier=readiness.evidence_tier,
        fallback_rows_are_success_evidence=bool(fallback_flag),
        excluded_row_statuses=excluded_tuple,
        fallback_excluded=readiness.fallback_excluded,
        arms=arms,
        status=status,
        blockers=tuple(blockers),
        campaign_gates=readiness.campaign_gates,
        readiness_status=readiness.status,
    )


def _as_abs(repo_root: str | Path, packet_path: str | Path) -> Path:
    """Resolve a possibly repo-relative packet path against ``repo_root``.

    Returns:
        The absolute packet path.
    """
    packet_abs = Path(packet_path)
    if not packet_abs.is_absolute():
        packet_abs = Path(repo_root) / packet_abs
    return packet_abs


def execute_run_plan(plan: DenseComparisonRunPlan) -> None:
    """Fail closed: executing the dense comparison is out of scope for this runner slice.

    A valid plan is fully resolved and reviewable, but running it (episodes, Slurm/GPU) is
    deferred to an explicitly authorized campaign. This function never runs episodes; it
    always raises so no code path can accidentally launch the comparison from the planner.

    Raises:
        DenseComparisonExecutionGatedError: always.
    """
    raise DenseComparisonExecutionGatedError(
        "issue #4142 dense DPCBF comparison execution is authorization-gated: this runner "
        "slice resolves and validates the three-arm plan but does not run episodes. "
        "Executing requires explicit human/Slurm authorization and a benchmark-grade "
        f"campaign, which is out of scope here. Plan status: {plan.status}."
    )


def to_dict(plan: DenseComparisonRunPlan) -> dict[str, Any]:
    """Return a JSON-serializable view of the run plan."""
    return {
        "schema_version": plan.schema_version,
        "packet_path": plan.packet_path,
        "packet_schema_version": plan.packet_schema_version,
        "status": plan.status,
        "readiness_status": plan.readiness_status,
        "is_executable_in_principle": plan.is_executable_in_principle,
        "claim_boundary": plan.claim_boundary,
        "algorithm": plan.algorithm,
        "scenario_manifest": plan.scenario_manifest,
        "output_dir": plan.output_dir,
        "evidence_tier": plan.evidence_tier,
        "fallback_rows_are_success_evidence": plan.fallback_rows_are_success_evidence,
        "excluded_row_statuses": list(plan.excluded_row_statuses),
        "fallback_excluded": plan.fallback_excluded,
        "arms": [
            {
                "arm_key": job.arm_key,
                "enabled": job.enabled,
                "variant": job.variant,
                "algorithm": job.algorithm,
                "algorithm_config": job.algorithm_config,
                "scenario_manifest": job.scenario_manifest,
                "output_jsonl": job.output_jsonl,
            }
            for job in plan.arms
        ],
        "blockers": list(plan.blockers),
        "runner_gates": list(plan.runner_gates),
        "campaign_gates": list(plan.campaign_gates),
    }


def render_markdown(plan: DenseComparisonRunPlan) -> str:
    """Render a compact Markdown report leading with the claim boundary and status.

    Returns:
        A Markdown string describing the resolved (or blocked) run plan.
    """
    lines: list[str] = []
    lines.append("# Issue #4142 dense DPCBF comparison run plan")
    lines.append("")
    lines.append(f"Claim boundary: {plan.claim_boundary}")
    lines.append("")
    lines.append(f"- Status: `{plan.status}` (readiness: `{plan.readiness_status}`)")
    lines.append(f"- Packet: `{plan.packet_path}` (schema `{plan.packet_schema_version}`)")
    lines.append(f"- Algorithm: `{plan.algorithm}`")
    lines.append(f"- Scenario manifest: `{plan.scenario_manifest}`")
    lines.append(f"- Output dir (planned, not written): `{plan.output_dir}`")
    lines.append(
        f"- Evidence tier: `{plan.evidence_tier}`; fallback/degraded excluded: "
        f"{plan.fallback_excluded}"
    )
    lines.append(
        f"- Excluded row statuses (caveats, never success): "
        f"{', '.join(f'`{s}`' for s in plan.excluded_row_statuses) or '(none)'}"
    )
    lines.append("")
    lines.append("## Resolved arm jobs")
    lines.append("")
    if plan.arms:
        lines.append("| arm_key | enabled | variant | algorithm_config | output_jsonl |")
        lines.append("| --- | --- | --- | --- | --- |")
        for job in plan.arms:
            lines.append(
                f"| `{job.arm_key}` | {job.enabled} | `{job.variant}` | "
                f"`{job.algorithm_config}` | `{job.output_jsonl}` |"
            )
    else:
        lines.append("_No executable arm jobs resolved (plan is blocked; see below)._")
    lines.append("")
    if plan.blockers:
        lines.append("## Blockers (fail-closed)")
        lines.append("")
        for blocker in plan.blockers:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Execution gates (remain even when the plan is ready)")
    lines.append("")
    for gate in plan.runner_gates:
        lines.append(f"- {gate}")
    for gate in plan.campaign_gates:
        lines.append(f"- {gate}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "CLAIM_BOUNDARY",
    "DEFAULT_OUTPUT_DIR",
    "PLAN_SCHEMA_VERSION",
    "RUNNER_GATES",
    "ArmJobPlan",
    "DenseComparisonExecutionGatedError",
    "DenseComparisonRunPlan",
    "DenseComparisonRunnerError",
    "build_run_plan",
    "execute_run_plan",
    "render_markdown",
    "to_dict",
]
