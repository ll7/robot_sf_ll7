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
- **Execution gate.** Running the dense comparison locally is authorization-gated.
  :func:`execute_run_plan` fails closed with :class:`DenseComparisonExecutionGatedError`
  unless the caller supplies the exact public authorization ID (:data:`REQUIRED_AUTHORIZATION_ID`)
  through an explicit argument -- a boolean, environment variable, implicit TTY, or bare
  ``--execute`` flag is insufficient and no output files are created. With the correct ID the
  executor runs bounded local episodes via the canonical benchmark runner; it never submits
  Slurm/GPU jobs and knows nothing about ``sbatch``, SSH, tmux, or queue tooling.

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

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from collections.abc import Callable

#: Output-contract schema for the resolved run plan.
PLAN_SCHEMA_VERSION = "robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1"

#: Output-contract schema for the machine-readable execution manifest.
EXECUTION_MANIFEST_SCHEMA_VERSION = "robot_sf.issue_4142_dpcbf_dense_comparison_execution.v1"

#: Exact public authorization ID that must be supplied to actually run episodes. Anything
#: else -- a missing/empty value, a boolean, an environment variable, an implicit TTY, or a
#: bare ``--execute`` flag -- is insufficient and the executor fails closed before any write.
REQUIRED_AUTHORIZATION_ID = "RSF-DPCBF-DENSE-20260712"

#: Canonical episode-record schema the benchmark runner validates rows against.
EPISODE_SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

#: Filename of the execution manifest written under the plan's output directory. Also acts as
#: the provenance sidecar a repeated invocation reads to decide resume-vs-fail-closed.
EXECUTION_MANIFEST_FILENAME = "execution_manifest.json"

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
    "executing the dense comparison locally requires the exact public authorization ID passed "
    "through an explicit flag; without it execute_run_plan() fails closed and writes nothing",
    "this executor runs bounded local episodes only; it submits no Slurm/GPU job and knows "
    "nothing about sbatch, SSH, tmux, or queue tooling",
)


class DenseComparisonRunnerError(RobotSfError, ValueError):
    """Raised when the packet cannot be consumed into a run plan at all."""


class DenseComparisonExecutionGatedError(RobotSfError, RuntimeError):
    """Raised when execution is attempted without the exact public authorization ID."""


class DenseComparisonProvenanceMismatchError(RobotSfError, RuntimeError):
    """Raised when a repeated run would silently mix incompatible packet/config/git provenance."""


@dataclass(frozen=True, slots=True)
class DenseExecutionInputs:
    """Fixed, bounded execution inputs for the three-arm local episode run.

    These are intentionally small and diagnostic: a bounded runtime comparison, never a
    benchmark-grade campaign. Values come from the packet's optional ``execution`` block
    merged over :data:`DEFAULT_EXECUTION_INPUTS`; :func:`build_run_plan` fails closed if any
    override is out of bounds so a plan can never silently widen into a large run.
    """

    base_seed: int
    repeats: int
    horizon: int
    dt: float
    workers: int
    video_enabled: bool = False
    resume: bool = True


#: Fixed bounded defaults. Small horizon/repeats and a single worker keep this a bounded
#: diagnostic comparison; video stays off so no large artifacts are produced.
DEFAULT_EXECUTION_INPUTS = DenseExecutionInputs(
    base_seed=0,
    repeats=3,
    horizon=100,
    dt=0.1,
    workers=1,
    video_enabled=False,
    resume=True,
)

#: Upper bounds on packet-supplied overrides. A plan that requests more fails closed rather
#: than quietly turning the bounded diagnostic into a heavy run.
_EXECUTION_INPUT_BOUNDS = {
    "base_seed": (0, 1_000_000),
    "repeats": (1, 20),
    "horizon": (1, 2_000),
    "workers": (1, 8),
}


def _resolve_resume(block: dict[str, Any], blockers: list[str]) -> bool:
    """Validate the resume flag without coercing strings into execution semantics.

    Returns:
        Valid boolean resume value, or the safe default after recording a blocker.
    """
    resume = block.get("resume", DEFAULT_EXECUTION_INPUTS.resume)
    if not isinstance(resume, bool):
        blockers.append(f"packet execution.resume must be a boolean, got {resume!r}")
        return DEFAULT_EXECUTION_INPUTS.resume
    return resume


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
    execution_inputs: DenseExecutionInputs = field(default_factory=lambda: DEFAULT_EXECUTION_INPUTS)

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


def _resolve_execution_inputs(packet: dict[str, Any]) -> tuple[DenseExecutionInputs, list[str]]:
    """Merge the packet's optional ``execution`` block over the bounded defaults.

    Unknown/absent keys keep their default. Any override that is the wrong type or outside
    :data:`_EXECUTION_INPUT_BOUNDS` becomes a fail-closed blocker rather than a silent clamp,
    so a plan can never quietly grow into a heavy run.

    Returns:
        The resolved bounded execution inputs and any blockers found.
    """
    block = packet.get("execution")
    if block is None:
        return DEFAULT_EXECUTION_INPUTS, []
    if not isinstance(block, dict):
        return DEFAULT_EXECUTION_INPUTS, ["packet 'execution' block must be a mapping"]

    blockers: list[str] = []
    values: dict[str, Any] = {
        "base_seed": DEFAULT_EXECUTION_INPUTS.base_seed,
        "repeats": DEFAULT_EXECUTION_INPUTS.repeats,
        "horizon": DEFAULT_EXECUTION_INPUTS.horizon,
        "workers": DEFAULT_EXECUTION_INPUTS.workers,
    }
    for key, (low, high) in _EXECUTION_INPUT_BOUNDS.items():
        if key not in block:
            continue
        raw = block[key]
        if isinstance(raw, bool) or not isinstance(raw, int):
            blockers.append(f"packet execution.{key} must be an integer, got {raw!r}")
            continue
        if not (low <= raw <= high):
            blockers.append(f"packet execution.{key}={raw} is outside bounds [{low}, {high}]")
            continue
        values[key] = raw

    dt = DEFAULT_EXECUTION_INPUTS.dt
    if "dt" in block:
        raw_dt = block["dt"]
        if (
            isinstance(raw_dt, bool)
            or not isinstance(raw_dt, (int, float))
            or not (0 < raw_dt <= 1)
        ):
            blockers.append(f"packet execution.dt must be a float in (0, 1], got {raw_dt!r}")
        else:
            dt = float(raw_dt)

    # video is deliberately forced off: this is a bounded diagnostic run, not a render job.
    if bool(block.get("video_enabled", False)):
        blockers.append("packet execution.video_enabled must be false (bounded diagnostic run)")

    resume = _resolve_resume(block, blockers)

    inputs = DenseExecutionInputs(
        base_seed=int(values["base_seed"]),
        repeats=int(values["repeats"]),
        horizon=int(values["horizon"]),
        dt=dt,
        workers=int(values["workers"]),
        video_enabled=False,
        resume=resume,
    )
    return inputs, blockers


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

    execution_inputs, execution_blockers = _resolve_execution_inputs(packet)

    blockers = list(readiness.blockers)
    blockers.extend(execution_blockers)
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
        execution_inputs=execution_inputs,
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


@dataclass(frozen=True, slots=True)
class ArmExecutionResult:
    """Outcome of running one comparison arm through the canonical benchmark runner."""

    arm_key: str
    algorithm: str
    algorithm_config: str
    output_jsonl: str
    #: ``executed`` (all scheduled jobs written, no failures), ``failed`` (runner raised, wrote
    #: nothing, or reported failures). Failed/degraded rows are caveats, never success.
    status: str
    total_jobs: int
    written: int
    failed_jobs: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DenseExecutionManifest:
    """Machine-readable manifest describing one authorized local execution attempt."""

    schema_version: str
    packet_path: str
    packet_schema_version: str
    plan_schema_version: str
    authorization_id: str
    git_sha: str
    git_dirty: bool
    algorithm: str
    scenario_manifest: str
    output_dir: str
    provenance_key: str
    effective_arguments: dict[str, Any]
    arms: tuple[ArmExecutionResult, ...]
    started_at: str
    ended_at: str
    #: ``complete`` only when every required arm executed with success rows and no failures.
    #: Otherwise ``results_incomplete`` -- a caveat state, never a successful comparison.
    status: str
    excluded_row_statuses: tuple[str, ...]
    claim_boundary: str = CLAIM_BOUNDARY


def _git_provenance(repo_root: Path) -> tuple[str, bool]:
    """Return the current git SHA and whether the working tree is dirty.

    Returns:
        ``(sha, dirty)``; ``("unknown", True)`` if git is unavailable (dirty=True fails safe).
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode()
        return sha or "unknown", bool(porcelain.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return "unknown", True


def _provenance_key(
    plan: DenseComparisonRunPlan,
    git_sha: str,
    *,
    git_dirty: bool,
    schema_path: str | Path,
) -> str:
    """Build a stable provenance key that must match for a repeated run to safely resume.

    Combines the packet/plan schema lineage, the shared algorithm, every arm's config, the
    bounded execution inputs, and the git SHA. A change in any of these means the on-disk
    artifacts came from an incompatible run and resume must fail closed.

    Returns:
        A deterministic string key.
    """
    inputs = plan.execution_inputs
    parts = [
        plan.packet_schema_version,
        plan.schema_version,
        str(plan.packet_path),
        str(plan.algorithm),
        str(plan.scenario_manifest),
        f"output_dir={plan.output_dir}",
        f"seed={inputs.base_seed}",
        f"repeats={inputs.repeats}",
        f"horizon={inputs.horizon}",
        f"dt={inputs.dt}",
        f"workers={inputs.workers}",
        f"video={inputs.video_enabled}",
        f"resume={inputs.resume}",
        f"schema_path={Path(schema_path).as_posix()}",
        git_sha,
        f"git_dirty={git_dirty}",
    ]
    parts.extend(f"{job.arm_key}:{job.algorithm_config}" for job in plan.arms)
    return "|".join(parts)


def _abs_under_root(repo_root: Path, rel_or_abs: str) -> Path:
    """Resolve a possibly repo-relative path against ``repo_root``.

    Returns:
        The absolute path.
    """
    path = Path(rel_or_abs)
    return path if path.is_absolute() else repo_root / path


def _default_run_batch() -> Callable[..., dict[str, Any]]:
    """Import the canonical benchmark batch runner lazily.

    Returns:
        ``robot_sf.benchmark.runner.run_batch``.
    """
    from robot_sf.benchmark.runner import run_batch  # noqa: PLC0415

    return run_batch


def execute_run_plan(
    plan: DenseComparisonRunPlan,
    *,
    authorization: str | None = None,
    repo_root: str | Path = ".",
    schema_path: str | Path = EPISODE_SCHEMA_PATH,
    run_batch_fn: Callable[..., dict[str, Any]] | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> DenseExecutionManifest:
    """Run the three-arm dense comparison locally, gated on the exact authorization ID.

    The executor reuses the canonical benchmark runner (:func:`robot_sf.benchmark.runner.
    run_batch`) once per resolved arm, in packet order, each pinned to the shared scenario
    manifest and that arm's distinct algorithm config, and writes the planned per-arm JSONL.
    It is deliberately local-only: it knows nothing about ``sbatch``, SSH, tmux, queue tooling,
    or private ops.

    Fail-closed order (nothing is written until every gate passes):

    1. The plan must be fully resolved (``plan_ready_campaign_gated``); a blocked plan raises.
    2. ``authorization`` must equal :data:`REQUIRED_AUTHORIZATION_ID` exactly. A missing/empty
       value, a boolean, or any other string raises before a single output file is created.
    3. If an execution manifest already exists under the output directory, its provenance key
       must match the current plan+git provenance, otherwise the run fails closed rather than
       mixing incompatible artifacts. A matching key allows a provenance-safe resume.

    Args:
        plan: A resolved run plan from :func:`build_run_plan`.
        authorization: The exact public authorization ID; anything else fails closed.
        repo_root: Directory repo-relative plan paths resolve against.
        schema_path: Episode-record schema the runner validates rows against.
        run_batch_fn: Injection seam for tests; defaults to the canonical ``run_batch``.
        now_fn: Injection seam for timestamps; defaults to UTC ``datetime.now``.

    Returns:
        The execution manifest (also written to ``output_dir/execution_manifest.json``).

    Raises:
        DenseComparisonExecutionGatedError: if the plan is not ready or authorization is wrong.
        DenseComparisonProvenanceMismatchError: if a prior manifest has incompatible provenance.
    """
    # Gate 1: never execute a plan that did not fully resolve.
    if not plan.is_executable_in_principle:
        raise DenseComparisonExecutionGatedError(
            "issue #4142 dense DPCBF comparison cannot execute: the run plan is not fully "
            f"resolved (status {plan.status!r}). Resolve all packet inputs first."
        )

    # Gate 2: authorization. Checked before ANY filesystem write so a wrong/absent ID never
    # creates output files. A boolean, env var, TTY, or bare --execute flag is insufficient.
    if authorization != REQUIRED_AUTHORIZATION_ID:
        raise DenseComparisonExecutionGatedError(
            "issue #4142 dense DPCBF comparison execution is authorization-gated: pass the "
            f"exact public authorization ID {REQUIRED_AUTHORIZATION_ID!r} via the explicit "
            "--authorization flag. A missing/boolean/env/TTY value or a bare --execute flag "
            "is insufficient; no output files are created."
        )

    root = Path(repo_root)
    output_dir_abs = _abs_under_root(root, plan.output_dir)
    manifest_path = output_dir_abs / EXECUTION_MANIFEST_FILENAME
    schema_abs = _abs_under_root(root, str(schema_path)).resolve()

    git_sha, git_dirty = _git_provenance(root)
    provenance_key = _provenance_key(
        plan,
        git_sha,
        git_dirty=git_dirty,
        schema_path=schema_abs,
    )

    # Gate 3: provenance-safe resume. A pre-existing manifest with a different key means the
    # on-disk artifacts came from an incompatible run; fail closed instead of mixing them.
    if manifest_path.is_file():
        try:
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(prior, dict):
                raise ValueError("manifest JSON is not a dictionary")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise DenseComparisonProvenanceMismatchError(
                f"existing execution manifest {manifest_path} is unreadable: {exc}"
            ) from exc
        prior_key = prior.get("provenance_key")
        if prior_key != provenance_key:
            raise DenseComparisonProvenanceMismatchError(
                "refusing to resume: existing execution manifest provenance does not match "
                f"this run. existing={prior_key!r} current={provenance_key!r}. Remove "
                f"{output_dir_abs} or run with matching packet/config/git provenance."
            )

    inputs = plan.execution_inputs
    clock = now_fn or (lambda: datetime.now(UTC))
    run_batch = run_batch_fn or _default_run_batch()

    effective_arguments = {
        "authorization_id": REQUIRED_AUTHORIZATION_ID,
        "repo_root": str(root),
        "schema_path": str(schema_path),
        "base_seed": inputs.base_seed,
        "repeats": inputs.repeats,
        "horizon": inputs.horizon,
        "dt": inputs.dt,
        "workers": inputs.workers,
        "video_enabled": inputs.video_enabled,
        "resume": inputs.resume,
    }

    output_dir_abs.mkdir(parents=True, exist_ok=True)
    started_at = clock().isoformat()

    scenario_abs = _abs_under_root(root, str(plan.scenario_manifest))
    results: list[ArmExecutionResult] = []
    for job in plan.arms:
        out_abs = _abs_under_root(root, job.output_jsonl)
        algo_config_abs = _abs_under_root(root, job.algorithm_config)
        results.append(
            _execute_arm(
                run_batch,
                job=job,
                scenario_abs=scenario_abs,
                out_abs=out_abs,
                schema_abs=schema_abs,
                algo_config_abs=algo_config_abs,
                inputs=inputs,
            )
        )

    ended_at = clock().isoformat()
    all_ok = len(results) == len(REQUIRED_ARMS) and all(
        r.status == "executed" and r.written > 0 for r in results
    )
    status = "complete" if all_ok else "results_incomplete"

    manifest = DenseExecutionManifest(
        schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION,
        packet_path=plan.packet_path,
        packet_schema_version=plan.packet_schema_version,
        plan_schema_version=plan.schema_version,
        authorization_id=REQUIRED_AUTHORIZATION_ID,
        git_sha=git_sha,
        git_dirty=git_dirty,
        algorithm=str(plan.algorithm),
        scenario_manifest=str(plan.scenario_manifest),
        output_dir=plan.output_dir,
        provenance_key=provenance_key,
        effective_arguments=effective_arguments,
        arms=tuple(results),
        started_at=started_at,
        ended_at=ended_at,
        status=status,
        excluded_row_statuses=plan.excluded_row_statuses,
    )
    manifest_path.write_text(
        json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _execute_arm(
    run_batch: Callable[..., dict[str, Any]],
    *,
    job: ArmJobPlan,
    scenario_abs: Path,
    out_abs: Path,
    schema_abs: Path,
    algo_config_abs: Path,
    inputs: DenseExecutionInputs,
) -> ArmExecutionResult:
    """Run one arm through the canonical runner and classify its outcome.

    A runner exception or a run that writes nothing is recorded as ``failed`` and remains a
    visible caveat in the manifest; it never blocks the remaining arms and is never counted as
    success.

    Returns:
        The per-arm execution result.
    """
    try:
        summary = run_batch(
            scenarios_or_path=str(scenario_abs),
            out_path=str(out_abs),
            schema_path=str(schema_abs),
            base_seed=inputs.base_seed,
            repeats_override=inputs.repeats,
            horizon=inputs.horizon,
            dt=inputs.dt,
            algo=job.algorithm,
            algo_config_path=str(algo_config_abs),
            video_enabled=False,
            video_renderer="none",
            workers=inputs.workers,
            resume=inputs.resume,
            append=inputs.resume,
        )
    except Exception as exc:  # noqa: BLE001 - any runner failure is a visible caveat, not a crash
        return ArmExecutionResult(
            arm_key=job.arm_key,
            algorithm=job.algorithm,
            algorithm_config=job.algorithm_config,
            output_jsonl=job.output_jsonl,
            status="failed",
            total_jobs=0,
            written=0,
            failed_jobs=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    summary = summary if isinstance(summary, dict) else {}
    total_jobs = int(summary.get("total_jobs", 0) or 0)
    written = int(summary.get("written", 0) or 0)
    failures = summary.get("failures", [])
    failed_jobs = (
        len(failures) if isinstance(failures, list) else int(summary.get("failed_jobs", 0))
    )
    ok = total_jobs > 0 and written >= total_jobs and failed_jobs == 0
    return ArmExecutionResult(
        arm_key=job.arm_key,
        algorithm=job.algorithm,
        algorithm_config=job.algorithm_config,
        output_jsonl=job.output_jsonl,
        status="executed" if ok else "failed",
        total_jobs=total_jobs,
        written=written,
        failed_jobs=failed_jobs,
        error=None if ok else "run wrote fewer episodes than scheduled or reported failures",
    )


def manifest_to_dict(manifest: DenseExecutionManifest) -> dict[str, Any]:
    """Return a JSON-serializable view of the execution manifest."""
    return {
        "schema_version": manifest.schema_version,
        "packet_path": manifest.packet_path,
        "packet_schema_version": manifest.packet_schema_version,
        "plan_schema_version": manifest.plan_schema_version,
        "authorization_id": manifest.authorization_id,
        "git_sha": manifest.git_sha,
        "git_dirty": manifest.git_dirty,
        "algorithm": manifest.algorithm,
        "scenario_manifest": manifest.scenario_manifest,
        "output_dir": manifest.output_dir,
        "provenance_key": manifest.provenance_key,
        "effective_arguments": dict(manifest.effective_arguments),
        "status": manifest.status,
        "claim_boundary": manifest.claim_boundary,
        "excluded_row_statuses": list(manifest.excluded_row_statuses),
        "arms": [
            {
                "arm_key": arm.arm_key,
                "algorithm": arm.algorithm,
                "algorithm_config": arm.algorithm_config,
                "output_jsonl": arm.output_jsonl,
                "status": arm.status,
                "total_jobs": arm.total_jobs,
                "written": arm.written,
                "failed_jobs": arm.failed_jobs,
                "error": arm.error,
            }
            for arm in manifest.arms
        ],
        "started_at": manifest.started_at,
        "ended_at": manifest.ended_at,
    }


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
        "execution_inputs": {
            "base_seed": plan.execution_inputs.base_seed,
            "repeats": plan.execution_inputs.repeats,
            "horizon": plan.execution_inputs.horizon,
            "dt": plan.execution_inputs.dt,
            "workers": plan.execution_inputs.workers,
            "video_enabled": plan.execution_inputs.video_enabled,
            "resume": plan.execution_inputs.resume,
        },
        "authorization_required": REQUIRED_AUTHORIZATION_ID,
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
    "DEFAULT_EXECUTION_INPUTS",
    "DEFAULT_OUTPUT_DIR",
    "EPISODE_SCHEMA_PATH",
    "EXECUTION_MANIFEST_FILENAME",
    "EXECUTION_MANIFEST_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "REQUIRED_AUTHORIZATION_ID",
    "RUNNER_GATES",
    "ArmExecutionResult",
    "ArmJobPlan",
    "DenseComparisonExecutionGatedError",
    "DenseComparisonProvenanceMismatchError",
    "DenseComparisonRunPlan",
    "DenseComparisonRunnerError",
    "DenseExecutionInputs",
    "DenseExecutionManifest",
    "build_run_plan",
    "execute_run_plan",
    "manifest_to_dict",
    "render_markdown",
    "to_dict",
]
