"""Plan-level preflight checker for the paper-grade reactivity-vs-replay rank study (issue #3637).

Issue #3573 landed the *diagnostic*: the pure quantifier
``robot_sf.benchmark.reactivity_ablation.assess_reactivity_ablation`` (#3594) and the paired-run +
open-loop-replay pedestrian mode (#3612), on a small matrix (2 scenarios, goal + orca, 4 seeds).
Issue #3637 carries the **paper-grade** extension: the same ablation across >=3 planners at a seed
budget sufficient for rank stability, with the replay limitation stated explicitly.

This module is the *next evidence-control layer* for that extension. It is a **pure, side-effect
free** checker (mirroring the quantifier in
:mod:`robot_sf.benchmark.reactivity_ablation`): given a proposed run plan, it verifies the
**plan-level preconditions** required before the paper-grade campaign is launched, and emits a
manifest that bakes in the canonical replay limitation.

What it checks (preconditions, declared by the plan — not the run output):

* at least :data:`MIN_PLANNERS` planners (the paper-grade ablation needs >=3);
* exactly the two reactive/replay arms (:data:`~robot_sf.benchmark.reactivity_ablation.REACTIVITY_ARMS`);
* **paired seeds** — both arms use the identical seed set (common random numbers), the property that
  makes the per-planner contrast attributable to reactivity;
* a seed budget at or above the rank-stability floor and strictly above the #3573 diagnostic matrix;
* the **replay-limitation metadata** is present and correct: "replay" is robot->ped force-off in a
  live sim, **not** trajectory playback.

What it deliberately does **not** do (out of scope for #3637's implementation slice): run the
benchmark, compute or interpret rank stability, or make any paper-facing claim. Actual seed
*sufficiency* (CI half-width / rank-flip) is decided **post-run** by
``scripts/tools/seed_sufficiency_gate.py``; this preflight only gates that the *plan* is well-formed
and honestly labeled before any compute is spent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.reactivity_ablation import (
    REACTIVITY_ARMS,
    REPLAY_IS_TRAJECTORY_PLAYBACK,
    REPLAY_LIMITATION,
)

PREFLIGHT_SCHEMA = "reactivity_replay_rank_study_preflight.v1"
ISSUE = 3637

#: Paper-grade ablation requires at least this many planners (issue #3637 Definition of Done).
MIN_PLANNERS = 3

#: Seed-count floor for a rank-stability-capable plan. This is a *precondition* floor aligned with
#: the S20 schedule used elsewhere (``scripts/tools/seed_sufficiency_gate.py``), **not** a proof of
#: sufficiency: actual sufficiency is decided post-run from CI half-width / rank-flip evidence.
MIN_RANK_STABILITY_SEEDS = 20

#: The #3573 diagnostic small matrix used 4 seeds; a paper-grade plan must strictly exceed it.
DIAGNOSTIC_SEED_COUNT = 4

#: Metrics that must be preserved for the post-run rank-stability gate.
REQUIRED_RANK_STABILITY_METRICS = ("collision_rate", "near_miss_rate", "min_separation_m")

#: Canonical post-run gate command family. Packets may add arguments but must route here.
RANK_STABILITY_GATE_COMMAND = "scripts/tools/seed_sufficiency_gate.py"

#: Explicitly excluded actions for this preflight slice. These are machine-checked so a
#: launch packet cannot be mistaken for benchmark evidence or a compute submission request.
REQUIRED_OUT_OF_SCOPE = (
    "no_full_benchmark_campaign",
    "no_slurm_gpu_submission",
    "no_paper_dissertation_claim_edits",
)


@dataclass(frozen=True, slots=True)
class ReactivityReplayRunPlan:
    """A proposed paper-grade reactivity-vs-replay run plan to be preflight-checked.

    This is the declared *intent* of the run, not its result. ``arm_seeds`` maps each reactivity arm
    to its seed list; the checker verifies the two arms are paired (identical seeds).

    Attributes:
        planners: Planner / algo names to compare (canonical names from
            ``robot_sf.benchmark.algorithm_readiness``).
        arm_seeds: Mapping of arm tag -> seed list. Must contain exactly the reactive/replay arms.
        scenario_set: Path to the scenario set the campaign will run over.
        horizon: Episode horizon (steps); the contrast is near-null below ~150 on the diagnostic
            family, so the plan must declare a horizon long enough for the robot to reach pedestrians.
        replay_is_trajectory_playback: Whether the replay arm is pre-recorded trajectory playback.
            Must be ``False`` for this ablation (live force-off, not playback).
        replay_limitation: Human-readable limitation note that must accompany every artifact.
        rank_stability_analysis: Plan-level post-run analysis contract metadata. This is not a
            measured result and does not claim rank stability.
        min_planners: Override for the planner-count floor (defaults to :data:`MIN_PLANNERS`).
        min_seeds: Override for the seed-count floor (defaults to :data:`MIN_RANK_STABILITY_SEEDS`).
    """

    planners: tuple[str, ...]
    arm_seeds: dict[str, tuple[int, ...]]
    scenario_set: str
    horizon: int
    scenario_set_sha256: str | None = None
    replay_is_trajectory_playback: bool = REPLAY_IS_TRAJECTORY_PLAYBACK
    replay_limitation: str = REPLAY_LIMITATION
    rank_stability_analysis: dict[str, Any] = field(default_factory=dict)
    out_of_scope: tuple[str, ...] = REQUIRED_OUT_OF_SCOPE
    min_planners: int = MIN_PLANNERS
    min_seeds: int = MIN_RANK_STABILITY_SEEDS
    #: Minimum horizon for the contrast to register (diagnostic family observation, #3573).
    min_horizon: int = field(default=150)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of a single preflight check.

    Attributes:
        name: Stable check identifier.
        passed: Whether the precondition holds.
        detail: Human-readable explanation (the blocking reason when ``passed`` is False).
    """

    name: str
    passed: bool
    detail: str


def _check_planner_count(plan: ReactivityReplayRunPlan) -> CheckResult:
    """At least ``min_planners`` distinct planners are declared.

    Returns:
        CheckResult: The ``planner_count`` check outcome.
    """
    distinct = sorted(set(plan.planners))
    ok = len(distinct) >= plan.min_planners
    return CheckResult(
        name="planner_count",
        passed=ok,
        detail=(
            f"{len(distinct)} distinct planners ({', '.join(distinct) or 'none'}); "
            f"need >= {plan.min_planners}"
        ),
    )


def _check_arms(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Exactly the two canonical reactive/replay arms are present.

    Returns:
        CheckResult: The ``arms_present`` check outcome.
    """
    arms = set(plan.arm_seeds)
    expected = set(REACTIVITY_ARMS)
    ok = arms == expected
    return CheckResult(
        name="arms_present",
        passed=ok,
        detail=(
            f"arms {sorted(arms)}; need exactly {sorted(expected)}"
            if not ok
            else f"both arms present: {sorted(expected)}"
        ),
    )


def _check_paired_seeds(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Both arms use the identical seed set (common random numbers).

    Returns:
        CheckResult: The ``paired_seeds`` check outcome.
    """
    seed_sets = {arm: tuple(seeds) for arm, seeds in plan.arm_seeds.items()}
    if set(seed_sets) != set(REACTIVITY_ARMS):
        return CheckResult(
            name="paired_seeds",
            passed=False,
            detail="cannot verify pairing until both reactive/replay arms are present",
        )
    reactive_seeds = sorted(seed_sets[REACTIVITY_ARMS[0]])
    replay_seeds = sorted(seed_sets[REACTIVITY_ARMS[1]])
    ok = bool(reactive_seeds) and reactive_seeds == replay_seeds
    return CheckResult(
        name="paired_seeds",
        passed=ok,
        detail=(
            f"arms share identical {len(reactive_seeds)} seeds (common random numbers)"
            if ok
            else "reactive and replay arms must use the identical non-empty seed set"
        ),
    )


def _check_seed_budget(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Seed count meets the rank-stability floor and exceeds the diagnostic matrix.

    Returns:
        CheckResult: The ``seed_budget`` check outcome.
    """
    counts = {arm: len(set(seeds)) for arm, seeds in plan.arm_seeds.items()}
    min_count = min(counts.values()) if counts else 0
    ok = min_count >= plan.min_seeds and min_count > DIAGNOSTIC_SEED_COUNT
    return CheckResult(
        name="seed_budget",
        passed=ok,
        detail=(
            f"min distinct seeds across arms = {min_count}; need >= {plan.min_seeds} "
            f"and > {DIAGNOSTIC_SEED_COUNT} (the #3573 diagnostic matrix). "
            "Precondition floor only; sufficiency is decided post-run by the seed-sufficiency gate."
        ),
    )


def _check_horizon(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Horizon is long enough for the robot to reach pedestrians (contrast registers).

    Returns:
        CheckResult: The ``horizon`` check outcome.
    """
    ok = plan.horizon >= plan.min_horizon
    return CheckResult(
        name="horizon",
        passed=ok,
        detail=(
            f"horizon {plan.horizon} >= {plan.min_horizon} (contrast is near-null below this on the "
            "diagnostic family, #3573)"
            if ok
            else f"horizon {plan.horizon} below the {plan.min_horizon}-step floor; contrast may be "
            "near-null because the robot never reaches the crossing"
        ),
    )


def _check_scenario_set_digest(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Validate supplied scenario-set digest so launch packets fail closed on drift.

    Returns:
        CheckResult: ``scenario_set_sha256`` check outcome.
    """
    expected = (plan.scenario_set_sha256 or "").strip().lower()
    if not expected:
        return CheckResult(
            name="scenario_set_sha256",
            passed=True,
            detail="no scenario_set_sha256 supplied; drift guard not enforced",
        )
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        return CheckResult(
            name="scenario_set_sha256",
            passed=False,
            detail="scenario_set_sha256 must be a 64-character lowercase hex SHA-256 digest",
        )

    scenario_path = Path(plan.scenario_set)
    if not scenario_path.is_absolute():
        scenario_path = Path(__file__).resolve().parents[2] / scenario_path
    if not scenario_path.is_file():
        return CheckResult(
            name="scenario_set_sha256",
            passed=False,
            detail=f"scenario_set file not found for digest check: {plan.scenario_set}",
        )

    actual = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    ok = actual == expected
    return CheckResult(
        name="scenario_set_sha256",
        passed=ok,
        detail=(
            f"scenario_set digest matches {expected}"
            if ok
            else f"scenario_set digest mismatch: expected {expected}, got {actual}"
        ),
    )


def _check_rank_stability_analysis(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Validate the declared post-run rank-stability analysis contract.

    Returns:
        CheckResult: ``rank_stability_analysis`` check outcome.
    """
    analysis = plan.rank_stability_analysis
    if not analysis:
        return CheckResult(
            name="rank_stability_analysis",
            passed=False,
            detail="rank_stability_analysis metadata must be present in launch packet",
        )

    metrics = analysis.get("required_metrics")
    metric_set = set(metrics) if isinstance(metrics, list) else set()
    missing_metrics = sorted(set(REQUIRED_RANK_STABILITY_METRICS) - metric_set)
    rank_metric = analysis.get("rank_metric")
    gate_command = str(analysis.get("seed_sufficiency_gate_command") or "")
    claim_boundary = str(analysis.get("claim_boundary") or "").lower()
    paired_resampling = analysis.get("paired_seed_resampling") is True
    replay_caveat_required = analysis.get("replay_limitation_required") is True

    failures: list[str] = []
    if missing_metrics:
        failures.append(f"missing required_metrics: {', '.join(missing_metrics)}")
    if not isinstance(rank_metric, str) or rank_metric not in metric_set:
        failures.append("rank_metric must be one of required_metrics")
    if not paired_resampling:
        failures.append("paired_seed_resampling must be true")
    if RANK_STABILITY_GATE_COMMAND not in gate_command:
        failures.append(
            f"seed_sufficiency_gate_command must route through {RANK_STABILITY_GATE_COMMAND}"
        )
    if not replay_caveat_required:
        failures.append("replay_limitation_required must be true")
    if "no paper-facing" not in claim_boundary or "post-run" not in claim_boundary:
        failures.append("claim_boundary must state no paper-facing claim until post-run review")

    return CheckResult(
        name="rank_stability_analysis",
        passed=not failures,
        detail=(
            "post-run rank-stability contract declared: paired seed resampling, required metrics, "
            "seed-sufficiency gate, replay caveat, and no-paper-claim boundary"
            if not failures
            else "; ".join(failures)
        ),
    )


def _check_replay_limitation(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Replay is force-off (not trajectory playback) and the limitation note is present.

    Returns:
        CheckResult: The ``replay_limitation`` check outcome.
    """
    note = (plan.replay_limitation or "").strip()
    ok = (plan.replay_is_trajectory_playback is False) and bool(note)
    if plan.replay_is_trajectory_playback is not False:
        detail = (
            "replay_is_trajectory_playback must be False: this ablation disables the robot->ped force "
            "in a live sim, it is not pre-recorded trajectory playback"
        )
    elif not note:
        detail = "replay_limitation note must be present in the manifest and accompanying artifacts"
    else:
        detail = "replay limitation stated: live force-off, not trajectory playback"
    return CheckResult(name="replay_limitation", passed=ok, detail=detail)


def _check_out_of_scope(plan: ReactivityReplayRunPlan) -> CheckResult:
    """Validate the non-execution / non-claim boundary travels with the packet.

    Returns:
        CheckResult: ``out_of_scope`` check outcome.
    """
    declared = set(plan.out_of_scope)
    required = set(REQUIRED_OUT_OF_SCOPE)
    missing = sorted(required - declared)
    extras = sorted(declared - required)
    ok = not missing
    detail = (
        "explicitly excludes full benchmark campaign, Slurm/GPU submission, and "
        "paper/dissertation claim edits"
        if ok
        else f"missing required out_of_scope exclusions: {', '.join(missing)}"
    )
    if extras:
        detail = f"{detail}; extra exclusions declared: {', '.join(extras)}"
    return CheckResult(name="out_of_scope", passed=ok, detail=detail)


_CHECKS = (
    _check_planner_count,
    _check_arms,
    _check_paired_seeds,
    _check_seed_budget,
    _check_horizon,
    _check_scenario_set_digest,
    _check_rank_stability_analysis,
    _check_replay_limitation,
    _check_out_of_scope,
)


def check_run_plan(plan: ReactivityReplayRunPlan) -> list[CheckResult]:
    """Run every plan-level precondition check and return the results in declaration order.

    Returns:
        list[CheckResult]: One result per check, in declaration order.
    """
    return [check(plan) for check in _CHECKS]


def build_preflight_manifest(plan: ReactivityReplayRunPlan) -> dict[str, Any]:
    """Build the versioned preflight manifest for a proposed run plan.

    The manifest is deterministic and side-effect free (provenance such as git HEAD / timestamps is
    added by the CLI wrapper). It carries the replay limitation so the constraint travels with the
    artifact.

    Returns:
        dict[str, Any]: ``status`` is ``"ready"`` when every precondition passes, else ``"blocked"``
        with the blocking reasons. Always carries the replay-limitation metadata and a conservative
        claim boundary.
    """
    checks = check_run_plan(plan)
    blocking = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    status = "ready" if not blocking else "blocked"
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "issue": ISSUE,
        "status": status,
        "evidence_tier": "plan_preflight",
        "plan": {
            "planners": list(plan.planners),
            "arms": {arm: list(seeds) for arm, seeds in plan.arm_seeds.items()},
            "scenario_set": plan.scenario_set,
            "scenario_set_sha256": plan.scenario_set_sha256,
            "horizon": plan.horizon,
            "rank_stability_analysis": dict(plan.rank_stability_analysis),
            "out_of_scope": list(plan.out_of_scope),
            "min_planners": plan.min_planners,
            "min_seeds": plan.min_seeds,
        },
        "replay_limitation": {
            "note": plan.replay_limitation,
            "is_trajectory_playback": plan.replay_is_trajectory_playback,
            "mechanism": "peds_have_robot_repulsion=false (live social-force, robot->ped force off)",
        },
        "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks],
        "blocking_issues": blocking,
        "claim_boundary": (
            "Plan-level preflight only: verifies the proposed reactivity-vs-replay run is well-formed "
            "(>=3 planners, paired seeds, both arms, seed-budget floor, replay limitation stated) "
            "before any compute is spent. It does NOT run the benchmark, measure or interpret rank "
            "stability, or make any paper-facing claim. Actual seed sufficiency is decided post-run "
            "by scripts/tools/seed_sufficiency_gate.py; 'replay' is live force-off, not trajectory "
            "playback."
        ),
        "out_of_scope": list(plan.out_of_scope),
    }


def run_plan_from_packet(packet: dict[str, Any]) -> ReactivityReplayRunPlan:
    """Build a :class:`ReactivityReplayRunPlan` from a launch-packet mapping (e.g. parsed YAML).

    Expected shape (only the listed keys are read; other extras are ignored)::

        planners: [goal, orca, social_force]
        scenario_set: configs/scenarios/sets/classic_crossing_subset.yaml
        horizon: 300
        seeds: [101, 102, ...]            # paired seed set used by BOTH arms
        # optional per-arm override (rarely needed; defaults to the shared `seeds`):
        arm_seeds: {reactive: [...], replay: [...]}
        replay:
          is_trajectory_playback: false
          limitation: "..."              # defaults to the canonical REPLAY_LIMITATION
        rank_stability_analysis:         # required post-run analysis contract (fails closed)
          paired_seed_resampling: true
          required_metrics: [collision_rate, near_miss_rate, min_separation_m]
          rank_metric: collision_rate
          seed_sufficiency_gate_command: "uv run python scripts/tools/seed_sufficiency_gate.py ..."
          replay_limitation_required: true
          claim_boundary: "No paper-facing claim until post-run ..."
        min_planners: 3                  # optional override
        min_seeds: 20                    # optional override

    The packet parses with ``rank_stability_analysis`` defaulting to an empty mapping when absent,
    but the manifest then preflights as ``blocked`` because the contract check fails closed (see
    :func:`_check_rank_stability_analysis`).

    Returns:
        ReactivityReplayRunPlan: The parsed run plan.

    Raises:
        ValueError: If required keys are missing or malformed.
    """
    if not isinstance(packet, dict):
        raise ValueError("packet must be a mapping")

    planners = packet.get("planners")
    if not isinstance(planners, list) or not all(isinstance(p, str) for p in planners):
        raise ValueError("packet 'planners' must be a list of planner-name strings")

    scenario_set = packet.get("scenario_set")
    if not isinstance(scenario_set, str) or not scenario_set.strip():
        raise ValueError("packet 'scenario_set' must be a non-empty string")
    scenario_set_sha256 = _resolve_scenario_set_sha256(packet)

    horizon = packet.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("packet 'horizon' must be an integer")

    arm_seeds = _resolve_arm_seeds(packet)

    replay = packet.get("replay", {})
    if not isinstance(replay, dict):
        raise ValueError("packet 'replay' must be a mapping when present")
    is_playback = replay.get("is_trajectory_playback", REPLAY_IS_TRAJECTORY_PLAYBACK)
    if not isinstance(is_playback, bool):
        raise ValueError("packet 'replay.is_trajectory_playback' must be a boolean")
    limitation = replay.get("limitation", REPLAY_LIMITATION)
    if not isinstance(limitation, str):
        raise ValueError("packet 'replay.limitation' must be a string")
    analysis = _resolve_rank_stability_analysis(packet)
    out_of_scope = _resolve_out_of_scope(packet)

    overrides: dict[str, Any] = {}
    if "min_planners" in packet:
        overrides["min_planners"] = int(packet["min_planners"])
    if "min_seeds" in packet:
        overrides["min_seeds"] = int(packet["min_seeds"])

    return ReactivityReplayRunPlan(
        planners=tuple(planners),
        arm_seeds=arm_seeds,
        scenario_set=scenario_set,
        horizon=horizon,
        scenario_set_sha256=scenario_set_sha256,
        replay_is_trajectory_playback=is_playback,
        replay_limitation=limitation,
        rank_stability_analysis=dict(analysis),
        out_of_scope=out_of_scope,
        **overrides,
    )


def _resolve_arm_seeds(packet: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Resolve per-arm seed lists from an explicit ``arm_seeds`` or a shared ``seeds`` key.

    Returns:
        dict[str, tuple[int, ...]]: Arm tag -> seed tuple (both arms share the seed set when
        resolved from a shared ``seeds`` key).
    """

    def _as_seed_tuple(value: Any, where: str) -> tuple[int, ...]:
        if not isinstance(value, list) or not all(
            isinstance(s, int) and not isinstance(s, bool) for s in value
        ):
            raise ValueError(f"{where} must be a list of integer seeds")
        return tuple(value)

    explicit = packet.get("arm_seeds")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("packet 'arm_seeds' must be a mapping of arm -> seed list")
        return {
            arm: _as_seed_tuple(seeds, f"arm_seeds[{arm!r}]") for arm, seeds in explicit.items()
        }

    shared = packet.get("seeds")
    if shared is None:
        raise ValueError("packet must provide either 'seeds' (shared) or 'arm_seeds' (per-arm)")
    seeds = _as_seed_tuple(shared, "packet 'seeds'")
    # Paired by construction: both arms run the identical seed set (common random numbers).
    return dict.fromkeys(REACTIVITY_ARMS, seeds)


def _resolve_scenario_set_sha256(packet: dict[str, Any]) -> str | None:
    """Return optional packet scenario-set checksum.

    Returns:
        Optional SHA-256 hex digest supplied by the launch packet.
    """
    scenario_set_sha256 = packet.get("scenario_set_sha256")
    if scenario_set_sha256 is not None and not isinstance(scenario_set_sha256, str):
        raise ValueError("packet 'scenario_set_sha256' must be a string when present")
    return scenario_set_sha256


def _resolve_rank_stability_analysis(packet: dict[str, Any]) -> dict[str, Any]:
    """Return optional rank-stability analysis metadata from the launch packet."""
    analysis = packet.get("rank_stability_analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("packet 'rank_stability_analysis' must be a mapping when present")
    return dict(analysis)


def _resolve_out_of_scope(packet: dict[str, Any]) -> tuple[str, ...]:
    """Return packet-level exclusions that keep preflight from implying evidence."""
    raw = packet.get("out_of_scope", ())
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("packet 'out_of_scope' must be list strings")
    return tuple(raw)


__all__ = [
    "DIAGNOSTIC_SEED_COUNT",
    "ISSUE",
    "MIN_PLANNERS",
    "MIN_RANK_STABILITY_SEEDS",
    "PREFLIGHT_SCHEMA",
    "RANK_STABILITY_GATE_COMMAND",
    "REQUIRED_OUT_OF_SCOPE",
    "REQUIRED_RANK_STABILITY_METRICS",
    "CheckResult",
    "ReactivityReplayRunPlan",
    "build_preflight_manifest",
    "check_run_plan",
    "run_plan_from_packet",
]
