"""Frozen-state counterfactual replay: locate the last avoidable control action.

This module implements the offline analysis contract of issue #5442 (child of
#5440, depends on the report contract of #5441): given a controlled fixture that
can *deterministically* snapshot and restore its full state (including any random
number generator), branch over admissible robot actions at each decision point in
the danger window and decide whether — and how early — the collision was avoidable.

The engine is intentionally decoupled from any concrete simulator. A caller
supplies a :class:`CounterfactualModel` — the *smallest snapshot/restore seam* —
and this module drives it. The controlled kinematic fixture used to validate the
contract lives in :mod:`robot_sf.benchmark.last_avoidable_fixtures`; a real-
simulator adapter can implement the same protocol later (see the docs note for
issue #5442).

Determinations (fail-closed):

* ``avoidable`` — the baseline replay is deterministic, every decision point in
  the window offered at least one feasible action, and at least one admissible
  action prevented contact within the frozen horizon. ``t_uca`` (earliest
  avoidable unsafe control action) and ``t_inevitable`` (point of no return) are
  reported.
* ``already_unavoidable`` — deterministic baseline, full feasible-action coverage
  over the window, yet **no** admissible action at any decision point prevented
  contact. Contact was already unavoidable at ``t_danger``.
* ``unknown`` — the baseline replay is not deterministic, or feasible-action
  coverage over the window is incomplete, so avoidability cannot be tested. Per
  the issue contract this **never** collapses to ``unavoidable``.

The result is controlled-fixture diagnostic evidence only. It assigns no legal or
moral fault (``normative_fault`` is always ``not_assessed``) and is not a real-
episode root-cause claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

LAST_AVOIDABLE_REPLAY_SCHEMA = "last_avoidable_replay.v1"

VERDICT_AVOIDABLE = "avoidable"
VERDICT_ALREADY_UNAVOIDABLE = "already_unavoidable"
VERDICT_UNKNOWN = "unknown"

# Substitution modes: how a candidate avoidance action is injected.
SUBSTITUTION_SINGLE_STEP = "single_step"  # substitute at t, then resume baseline commands
SUBSTITUTION_HOLD = "hold"  # apply the substituted action for the whole frozen horizon
_SUBSTITUTION_MODES = (SUBSTITUTION_SINGLE_STEP, SUBSTITUTION_HOLD)


@runtime_checkable
class CounterfactualModel(Protocol):
    """The smallest deterministic snapshot/restore seam the engine drives.

    A model wraps one controlled fixture positioned at step 0 of a recorded
    baseline episode. Implementations must be *deterministic*: restoring a
    snapshot and applying the same actions must reproduce the same collision
    outcome. The snapshot must capture everything that affects future steps,
    including any RNG state (see the fixture in
    :mod:`robot_sf.benchmark.last_avoidable_fixtures`).
    """

    def snapshot(self) -> Any:
        """Return an opaque, restorable copy of the full simulation state.

        The snapshot must include actor state (poses, velocities) *and* any RNG
        state so that :meth:`restore` followed by identical actions is bit-for-bit
        deterministic.
        """
        ...

    def restore(self, snapshot: Any) -> None:
        """Restore the model to a previously captured :meth:`snapshot`."""
        ...

    def step(self, action: Any) -> None:
        """Advance the simulation one control tick applying ``action``."""
        ...

    def collision(self) -> bool:
        """Return whether the robot is in contact at the current state."""
        ...

    def feasible_actions(self) -> Sequence[Any]:
        """Return the admissible action lattice at the current state.

        An empty sequence means no admissible substitution exists here; the
        engine treats such a decision point as untestable (coverage gap), which
        drives an ``unknown`` determination rather than ``already_unavoidable``.
        """
        ...

    def action_label(self, action: Any) -> str:
        """Return a stable, human-readable label for ``action`` (for provenance)."""
        ...


@dataclass(frozen=True)
class ReplayConfig:
    """Versioned analysis configuration recorded in every report.

    Attributes:
        t_danger: First step of the danger window (inclusive) to search.
        t_contact: Baseline contact step; the search window is
            ``[t_danger, t_contact)`` and ``t_contact`` bounds the replay.
        horizon: Frozen horizon ``H`` (control ticks) simulated forward from each
            candidate step when testing whether an action prevents contact.
        substitution_mode: How a candidate action is injected — ``single_step``
            (substitute at ``t`` then resume baseline commands) or ``hold`` (apply
            the substituted action for the whole horizon).
        determinism_replays: Number of identical baseline replays used to verify
            deterministic reproduction of the contact outcome.
        action_set_id: Provenance label for the admissible action set.
        feasibility_filter: Provenance label for how feasible actions are filtered.
        collision_predicate: Provenance label for the collision predicate.
        pedestrian_response: Pedestrian response assumption for this run, e.g.
            ``replayed`` (pedestrian follows its recorded path) or ``closed_loop``
            (pedestrian reacts to the robot).
    """

    t_danger: int
    t_contact: int
    horizon: int
    substitution_mode: str = SUBSTITUTION_SINGLE_STEP
    determinism_replays: int = 5
    action_set_id: str = "unspecified"
    feasibility_filter: str = "unspecified"
    collision_predicate: str = "unspecified"
    pedestrian_response: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate window, horizon, replay count, and substitution mode."""
        if self.t_danger < 0:
            raise ValueError(f"t_danger must be >= 0 (got {self.t_danger})")
        if self.t_contact <= self.t_danger:
            raise ValueError(f"t_contact ({self.t_contact}) must be > t_danger ({self.t_danger})")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 (got {self.horizon})")
        if self.determinism_replays < 1:
            raise ValueError(f"determinism_replays must be >= 1 (got {self.determinism_replays})")
        if self.substitution_mode not in _SUBSTITUTION_MODES:
            raise ValueError(
                f"substitution_mode must be one of {_SUBSTITUTION_MODES} "
                f"(got {self.substitution_mode!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe provenance mapping for this config."""
        return {
            "t_danger": self.t_danger,
            "t_contact": self.t_contact,
            "horizon": self.horizon,
            "substitution_mode": self.substitution_mode,
            "determinism_replays": self.determinism_replays,
            "action_set_id": self.action_set_id,
            "feasibility_filter": self.feasibility_filter,
            "collision_predicate": self.collision_predicate,
            "pedestrian_response": self.pedestrian_response,
        }


@dataclass(frozen=True)
class DeterminismCheck:
    """Result of verifying the baseline replay reproduces the contact outcome."""

    replays: int
    collision_stable: bool
    contact_step_stable: bool
    observed_contact_steps: tuple[int | None, ...]

    @property
    def deterministic(self) -> bool:
        """Return whether both the collision flag and contact step were stable."""
        return self.collision_stable and self.contact_step_stable

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe determinism-check mapping."""
        return {
            "replays": self.replays,
            "collision_stable": self.collision_stable,
            "contact_step_stable": self.contact_step_stable,
            "deterministic": self.deterministic,
            "observed_contact_steps": [
                None if s is None else int(s) for s in self.observed_contact_steps
            ],
        }


@dataclass(frozen=True)
class TimeBranchResult:
    """Per-decision-point branching outcome over the admissible action set."""

    step: int
    feasible_count: int
    preventing_action_labels: tuple[str, ...]

    @property
    def any_prevented(self) -> bool:
        """Return whether at least one admissible action prevented contact."""
        return len(self.preventing_action_labels) > 0

    @property
    def has_feasible_actions(self) -> bool:
        """Return whether any admissible action was available to test."""
        return self.feasible_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe per-step branch mapping."""
        return {
            "step": self.step,
            "feasible_count": self.feasible_count,
            "any_prevented": self.any_prevented,
            "preventing_action_labels": list(self.preventing_action_labels),
        }


@dataclass(frozen=True)
class LastAvoidableReport:
    """Self-contained ``last_avoidable_replay.v1`` result.

    The field set is deliberately forward-compatible with the
    ``collision_causal_report.v1`` contract proposed in issue #5441: it exposes
    ``t_danger``/``t_uca``/``t_inevitable``/``t_contact`` as available/unavailable
    (``None``) fields, records competing-explanation-relevant provenance, and
    holds ``normative_fault`` at ``not_assessed``. When #5441 lands this report can
    be embedded as the counterfactual branch of that contract without re-running.
    """

    verdict: str
    config: ReplayConfig
    determinism: DeterminismCheck
    branches: tuple[TimeBranchResult, ...]
    t_uca: int | None
    t_inevitable: int | None
    feasible_coverage: float
    minimal_sufficient_interventions: tuple[dict[str, Any], ...]
    runtime_s: float | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    normative_fault: str = "not_assessed"
    claim_boundary: str = (
        "controlled-fixture diagnostic evidence; not a real-episode root-cause "
        "claim; assigns no legal or moral fault"
    )
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe ``last_avoidable_replay.v1`` payload."""
        return {
            "schema_version": LAST_AVOIDABLE_REPLAY_SCHEMA,
            "verdict": self.verdict,
            "normative_fault": self.normative_fault,
            "claim_boundary": self.claim_boundary,
            "t_danger": self.config.t_danger,
            "t_uca": self.t_uca,
            "t_inevitable": self.t_inevitable,
            "t_contact": self.config.t_contact,
            "feasible_coverage": self.feasible_coverage,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "config": self.config.to_dict(),
            "determinism": self.determinism.to_dict(),
            "branches": [b.to_dict() for b in self.branches],
            "minimal_sufficient_interventions": list(self.minimal_sufficient_interventions),
            "runtime_s": self.runtime_s,
            "notes": list(self.notes),
        }


def _replay_to_contact(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    max_step: int,
) -> int | None:
    """Restore the initial snapshot, replay baseline actions, return the contact step.

    Returns:
        The first step index (0-based, the step whose action produced contact) at
        which :meth:`CounterfactualModel.collision` becomes true, or ``None`` if no
        contact occurs within ``max_step`` applied actions.
    """
    model.restore(initial_snapshot)
    if model.collision():
        return 0
    limit = min(max_step, len(baseline_actions))
    for step in range(limit):
        model.step(baseline_actions[step])
        if model.collision():
            return step
    return None


def _verify_determinism(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> DeterminismCheck:
    """Replay the baseline ``config.determinism_replays`` times and compare outcomes.

    Returns:
        A :class:`DeterminismCheck` recording whether the collision flag and
        contact step were stable across all replays.
    """
    max_step = config.t_contact + config.horizon
    contact_steps: list[int | None] = []
    for _ in range(config.determinism_replays):
        contact_steps.append(
            _replay_to_contact(model, initial_snapshot, baseline_actions, max_step)
        )
    collided = [c is not None for c in contact_steps]
    collision_stable = len(set(collided)) == 1
    contact_step_stable = len(set(contact_steps)) == 1
    return DeterminismCheck(
        replays=config.determinism_replays,
        collision_stable=collision_stable,
        contact_step_stable=contact_step_stable,
        observed_contact_steps=tuple(contact_steps),
    )


def _capture_window_snapshots(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> dict[int, Any]:
    """Return snapshots at the start of each step in ``[t_danger, t_contact)``.

    ``snapshots[t]`` is the state from which baseline ``action[t]`` would be
    applied — i.e. the decision point at step ``t``.
    """
    model.restore(initial_snapshot)
    snapshots: dict[int, Any] = {}
    for step in range(config.t_contact):
        if config.t_danger <= step < config.t_contact:
            snapshots[step] = model.snapshot()
        if step >= len(baseline_actions):
            break
        model.step(baseline_actions[step])
    return snapshots


def _action_prevents_contact(
    model: CounterfactualModel,
    step_snapshot: Any,
    action: Any,
    step: int,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> bool:
    """Return whether substituting ``action`` at ``step`` prevents contact in the horizon."""
    model.restore(step_snapshot)
    for offset in range(config.horizon):
        if config.substitution_mode == SUBSTITUTION_HOLD:
            applied = action
        elif offset == 0:
            applied = action
        else:
            resume_index = step + offset
            if resume_index >= len(baseline_actions):
                break
            applied = baseline_actions[resume_index]
        model.step(applied)
        if model.collision():
            return False
    return not model.collision()


def _branch_over_window(
    model: CounterfactualModel,
    snapshots: dict[int, Any],
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> tuple[list[TimeBranchResult], list[dict[str, Any]]]:
    """Branch over admissible actions at each decision point in the window.

    Returns:
        A tuple of the per-step branch results and the minimal sufficient
        single-action interventions found (each prevents contact on its own).
    """
    branches: list[TimeBranchResult] = []
    interventions: list[dict[str, Any]] = []
    for step in range(config.t_danger, config.t_contact):
        step_snapshot = snapshots.get(step)
        if step_snapshot is None:
            branches.append(
                TimeBranchResult(step=step, feasible_count=0, preventing_action_labels=())
            )
            continue
        model.restore(step_snapshot)
        feasible = list(model.feasible_actions())
        preventing_labels: list[str] = []
        for action in feasible:
            if _action_prevents_contact(
                model, step_snapshot, action, step, baseline_actions, config
            ):
                label = model.action_label(action)
                preventing_labels.append(label)
                interventions.append(
                    {
                        "step": step,
                        "action_label": label,
                        "substitution_mode": config.substitution_mode,
                        "horizon": config.horizon,
                    }
                )
        branches.append(
            TimeBranchResult(
                step=step,
                feasible_count=len(feasible),
                preventing_action_labels=tuple(preventing_labels),
            )
        )
    return branches, interventions


def locate_last_avoidable(
    model: CounterfactualModel,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
    *,
    runtime_s: float | None = None,
) -> LastAvoidableReport:
    """Locate the last avoidable control action via frozen-state counterfactual replay.

    The engine (1) verifies the baseline replay deterministically reproduces the
    contact outcome, (2) captures a snapshot at every decision point in
    ``[t_danger, t_contact)``, and (3) branches over the admissible action set at
    each point, checking whether any single admissible action prevents contact
    within the frozen horizon.

    Fail-closed determination (see module docstring): a nondeterministic baseline
    or incomplete feasible-action coverage yields ``unknown`` — never
    ``unavoidable``. Only full coverage with no preventing action anywhere yields
    ``already_unavoidable``.

    Args:
        model: The deterministic snapshot/restore seam positioned at step 0.
        baseline_actions: The recorded applied commands, indexed by step.
        config: Versioned analysis configuration.
        runtime_s: Optional measured wall-clock runtime to record (offline; no
            online gate is required).

    Returns:
        A :class:`LastAvoidableReport` preserving every branch result.
    """
    initial_snapshot = model.snapshot()

    determinism = _verify_determinism(model, initial_snapshot, baseline_actions, config)
    if not determinism.deterministic:
        return LastAvoidableReport(
            verdict=VERDICT_UNKNOWN,
            config=config,
            determinism=determinism,
            branches=(),
            t_uca=None,
            t_inevitable=None,
            feasible_coverage=0.0,
            minimal_sufficient_interventions=(),
            runtime_s=runtime_s,
            abstained=True,
            abstain_reason="nondeterministic_baseline",
            notes=("baseline replay did not reproduce a stable contact outcome",),
        )

    snapshots = _capture_window_snapshots(model, initial_snapshot, baseline_actions, config)
    branches, interventions = _branch_over_window(model, snapshots, baseline_actions, config)

    window_size = config.t_contact - config.t_danger
    with_feasible = sum(1 for b in branches if b.has_feasible_actions)
    feasible_coverage = with_feasible / window_size if window_size else 0.0

    preventable_steps = sorted(b.step for b in branches if b.any_prevented)

    if preventable_steps:
        t_uca = preventable_steps[0]
        # Point of no return: after the latest step where avoidance still worked,
        # contact is inevitable (clamped to the contact step).
        t_inevitable = min(preventable_steps[-1] + 1, config.t_contact)
        return LastAvoidableReport(
            verdict=VERDICT_AVOIDABLE,
            config=config,
            determinism=determinism,
            branches=tuple(branches),
            t_uca=t_uca,
            t_inevitable=t_inevitable,
            feasible_coverage=feasible_coverage,
            minimal_sufficient_interventions=tuple(interventions),
            runtime_s=runtime_s,
        )

    # No admissible action prevented contact at any decision point.
    if with_feasible == 0 or feasible_coverage < 1.0:
        # Coverage gap: we could not test avoidability everywhere -> abstain.
        return LastAvoidableReport(
            verdict=VERDICT_UNKNOWN,
            config=config,
            determinism=determinism,
            branches=tuple(branches),
            t_uca=None,
            t_inevitable=None,
            feasible_coverage=feasible_coverage,
            minimal_sufficient_interventions=(),
            runtime_s=runtime_s,
            abstained=True,
            abstain_reason="incomplete_feasible_action_coverage",
            notes=(
                "no admissible action prevented contact, but at least one decision "
                "point lacked a feasible action set; avoidability is untested",
            ),
        )

    # Full coverage, deterministic baseline, nothing prevents -> already unavoidable.
    return LastAvoidableReport(
        verdict=VERDICT_ALREADY_UNAVOIDABLE,
        config=config,
        determinism=determinism,
        branches=tuple(branches),
        t_uca=None,
        t_inevitable=config.t_danger,
        feasible_coverage=feasible_coverage,
        minimal_sufficient_interventions=(),
        runtime_s=runtime_s,
        notes=(
            "contact was unavoidable from t_danger under the declared action set "
            "and pedestrian response assumption",
        ),
    )
