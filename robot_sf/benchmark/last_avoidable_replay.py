"""Frozen-state counterfactual replay: locate the last avoidable control action.

This module implements the offline analysis contract of issue #5442 (child of
#5440, depends on the report contract of #5441): given a replay model that can
*deterministically* snapshot and restore its full state (including any random
number generator), branch over admissible robot actions at each decision point in
the danger window and decide whether — and how early — the collision was avoidable.

The engine is intentionally decoupled from any concrete simulator. A caller
supplies a :class:`CounterfactualModel` — the *smallest snapshot/restore seam* —
and this module drives it. The controlled kinematic fixture used to validate the
contract lives in :mod:`robot_sf.benchmark.last_avoidable_fixtures`; the
diagnostic production-simulator implementation lives in
:mod:`robot_sf.benchmark.simulator_counterfactual_adapter`. Both paths preserve
the engine's fail-closed, offline-only boundary.

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

The result is source-tagged diagnostic replay evidence only. The source kind
identifies whether it came from a controlled fixture or a native simulator
adapter; it assigns no legal or moral fault (``normative_fault`` is always
``not_assessed``) and is not a real-episode root-cause claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from robot_sf.benchmark.typed_snapshot import NoOpStep, compare_continuation_traces

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


class _DefaultPedestrianResponse(str):
    """String-compatible marker for an omitted, model-bindable response mode."""


_DEFAULT_PEDESTRIAN_RESPONSE = _DefaultPedestrianResponse("unknown")
_REPLAY_PROVENANCE_FIELDS = (
    "action_set_id",
    "feasibility_filter",
    "collision_predicate",
    "pedestrian_response",
    "source_kind",
)


class _ReplayProvenanceError(ValueError):
    """Raised when a model-declared replay provenance field is malformed."""


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
        t_contact: Baseline contact state tick (the number of applied actions at
            which contact is first observed); the search window is
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
            (pedestrian reacts to the robot). An omitted value is serialized as the
            schema-safe ``unknown`` value when no model declaration is available,
            but binds to a model-declared response mode when one is available.
            Explicit ``unknown`` remains an intentional unknown declaration and is
            not silently rebound to a model mode.
        source_kind: Provenance classification for the replay source. Native live
            simulator adapters bind this to ``live_episode``; controlled fixtures
            must use ``synthetic_fixture`` before a causal join. The default
            ``unspecified`` is retained for diagnostic compatibility but is
            rejected by the causal join. This is a source label, not a causal or
            benchmark claim.
    """

    t_danger: int
    t_contact: int
    horizon: int
    substitution_mode: str = SUBSTITUTION_SINGLE_STEP
    determinism_replays: int = 5
    action_set_id: str = "unspecified"
    feasibility_filter: str = "unspecified"
    collision_predicate: str = "unspecified"
    pedestrian_response: str = _DEFAULT_PEDESTRIAN_RESPONSE
    source_kind: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate window, horizon, replay count, and substitution mode."""
        for field_name in _REPLAY_PROVENANCE_FIELDS:
            value = getattr(self, field_name)
            if field_name == "pedestrian_response" and value is _DEFAULT_PEDESTRIAN_RESPONSE:
                continue
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
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
            "source_kind": self.source_kind,
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
    trace_stable: bool = False
    first_divergence_step: int | None = None
    first_divergence_field: str | None = None

    @property
    def deterministic(self) -> bool:
        """Return whether terminal outcome, contact tick, and trace all agree."""
        return self.collision_stable and self.contact_step_stable and self.trace_stable

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe determinism-check mapping."""
        return {
            "replays": self.replays,
            "collision_stable": self.collision_stable,
            "contact_step_stable": self.contact_step_stable,
            "trace_stable": self.trace_stable,
            "first_divergence_step": self.first_divergence_step,
            "first_divergence_field": self.first_divergence_field,
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

    The report is source-tagged diagnostic evidence: controlled fixtures and
    native simulator adapters can use the same replay contract, but their source
    and claim boundaries remain distinct. ``config.t_contact`` is the declared
    contact state tick; ``determinism.observed_contact_steps`` records the replay
    observations, and a downstream causal join may expose ``t_contact`` as an
    observed timestamp only when every observation agrees with that declaration.
    The report records competing-explanation-relevant provenance, holds
    ``normative_fault`` at ``not_assessed``, and is not a real-episode root-cause,
    benchmark, or paper-grade claim.
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
        "source-tagged diagnostic replay evidence; interpret using source_kind; "
        "not a real-episode root-cause, benchmark, or paper-grade claim; assigns "
        "no legal or moral fault"
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
        The contact state tick (the number of applied actions, so the action at
        zero produces state tick one) at which
        :meth:`CounterfactualModel.collision` becomes true, or ``None`` if no
        contact occurs within ``max_step`` applied actions.
    """
    model.restore(initial_snapshot)
    if model.collision():
        return 0
    limit = min(max_step, len(baseline_actions))
    for step in range(limit):
        model.step(baseline_actions[step])
        if model.collision():
            return step + 1
    return None


def _trace_value(value: Any) -> Any:
    """Convert an opaque snapshot into deterministic comparator-friendly values.

    Returns:
        A nested value containing owned arrays and stable object fields.
    """
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: (
                    {"present": getattr(value, item.name) is not None}
                    if item.name == "residual_adversary"
                    else _trace_value(getattr(value, item.name))
                )
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: _trace_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_trace_value(item) for item in value)
    if isinstance(value, list):
        return [_trace_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_trace_value(item) for item in value]
        return sorted(normalized, key=repr)
    if hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {key: _trace_value(item) for key, item in vars(value).items()},
        }
    return value


def _replay_with_trace(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    max_step: int,
) -> tuple[int | None, list[NoOpStep]]:
    """Replay baseline actions and retain typed no-op steps for every state tick.

    Returns:
        The first contact state tick and its typed continuation trace.
    """
    model.restore(initial_snapshot)
    initial_terminal = bool(model.collision())
    trace = [
        NoOpStep(
            step=0,
            state=_trace_value(model.snapshot()),
            terminal=initial_terminal,
        )
    ]
    if initial_terminal:
        return 0, trace
    limit = min(max_step, len(baseline_actions))
    for step in range(limit):
        action = baseline_actions[step]
        model.step(action)
        terminal = bool(model.collision())
        trace.append(
            NoOpStep(
                step=step + 1,
                state=_trace_value(model.snapshot()),
                applied_action=_trace_value(action),
                terminal=terminal,
            )
        )
        if terminal:
            return step + 1, trace
    return None, trace


def _verify_determinism(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> DeterminismCheck:
    """Replay the baseline and compare terminal outcomes plus typed traces.

    Returns:
        A :class:`DeterminismCheck` recording whether the collision flag, contact
        step, and every declared no-op trace field were stable across all replays.
    """
    max_step = config.t_contact + config.horizon
    contact_steps: list[int | None] = []
    traces: list[list[NoOpStep]] = []
    for _ in range(config.determinism_replays):
        contact_step, trace = _replay_with_trace(
            model, initial_snapshot, baseline_actions, max_step
        )
        contact_steps.append(contact_step)
        traces.append(trace)
    collided = [c is not None for c in contact_steps]
    collision_stable = len(set(collided)) == 1
    contact_step_stable = len(set(contact_steps)) == 1
    trace_stable = True
    first_divergence_step = None
    first_divergence_field = None
    if traces:
        expected_trace = traces[0]
        for actual_trace in traces[1:]:
            comparison = compare_continuation_traces(expected_trace, actual_trace)
            if comparison.equivalent:
                continue
            trace_stable = False
            first_divergence_step = comparison.first_divergence_step
            first_divergence_field = comparison.first_divergence_field
            break
    return DeterminismCheck(
        replays=config.determinism_replays,
        collision_stable=collision_stable,
        contact_step_stable=contact_step_stable,
        observed_contact_steps=tuple(contact_steps),
        trace_stable=trace_stable,
        first_divergence_step=first_divergence_step,
        first_divergence_field=first_divergence_field,
    )


def _capture_window_snapshots(
    model: CounterfactualModel,
    initial_snapshot: Any,
    baseline_actions: Sequence[Any],
    config: ReplayConfig,
) -> dict[int, Any]:
    """Return pre-contact snapshots at the start of each step in the window.

    ``snapshots[t]`` is the state from which baseline ``action[t]`` would be
    applied — i.e. the decision point at step ``t``. If contact is observed
    before the declared window end, later snapshots are omitted so post-contact
    states cannot be evaluated as counterfactual starts.
    """
    model.restore(initial_snapshot)
    if len(baseline_actions) < config.t_contact:
        raise ValueError(
            "baseline_actions must contain at least t_contact actions to capture "
            f"the full decision window (got {len(baseline_actions)}, "
            f"need {config.t_contact})"
        )
    snapshots: dict[int, Any] = {}
    for step in range(config.t_contact):
        if model.collision():
            break
        if config.t_danger <= step < config.t_contact:
            snapshots[step] = model.snapshot()
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
    # Keep the literal here so this small contract probe can execute the
    # function body without importing module-level constants.
    if config.substitution_mode == "single_step":
        # The candidate consumes action ``step``.  Every remaining horizon tick
        # must have a recorded baseline command to resume, including the last
        # tick at ``step + horizon - 1``.
        required = step + config.horizon
        if len(baseline_actions) < required:
            raise ValueError(
                "single_step substitution requires a recorded baseline suffix "
                f"through action {required - 1} (got {len(baseline_actions)} actions)"
            )
    for offset in range(config.horizon):
        if config.substitution_mode == SUBSTITUTION_HOLD:
            applied = action
        elif offset == 0:
            applied = action
        else:
            resume_index = step + offset
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
        if model.collision():
            # A post-contact snapshot is not a decision point from which a
            # counterfactual intervention can be attributed. Preserve it as a
            # coverage gap so callers fail closed instead of testing actions
            # after the baseline has already contacted.
            branches.append(
                TimeBranchResult(step=step, feasible_count=0, preventing_action_labels=())
            )
            continue
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


def _model_metadata(model: CounterfactualModel, field_name: str) -> str | None:
    """Read an optional model-declared provenance field.

    The protocol intentionally remains small for existing fixture models. Native
    adapters may expose these fields as properties (or zero-argument methods),
    allowing the engine to bind report provenance to the contract actually used.

    Returns:
        The normalized metadata value, or ``None`` when the model does not
        declare the requested field.
    """
    try:
        value = getattr(model, field_name, None)
        if callable(value):
            value = value()
    except Exception as exc:
        raise _ReplayProvenanceError(
            f"model replay provenance field {field_name!r} could not be read"
        ) from exc
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise _ReplayProvenanceError(
            f"model replay provenance field {field_name!r} must be a non-empty string"
        )
    return value


def _snapshot_state_is_complete(model: CounterfactualModel) -> bool | None:
    """Read an optional model declaration that its replay snapshot is complete.

    Returns:
        ``True`` or ``False`` when the model declares completeness; ``None`` when
        the legacy protocol has no completeness declaration.
    """
    try:
        declared = getattr(model, "replay_state_complete", None)
        if callable(declared):
            declared = declared()
    except Exception:  # noqa: BLE001 - malformed completeness declarations fail closed
        return False
    if declared is None:
        return None
    if type(declared) is bool:
        return declared
    if type(declared) is not str:
        return False
    normalized = declared.strip().lower()
    if normalized in {"true", "1", "yes", "complete"}:
        return True
    if normalized in {"false", "0", "no", "incomplete"}:
        return False
    return False


def _bind_model_metadata(
    model: CounterfactualModel, config: ReplayConfig
) -> tuple[ReplayConfig, tuple[str, ...]]:
    """Bind optional native provenance fields and report declaration conflicts.

    Returns:
        A possibly enriched config and a tuple of declaration mismatch messages.
    """
    bound_config = config
    mismatches: list[str] = []
    # These fields are model-bound rather than caller assertions. In particular,
    # the native adapter's action lattice and source kind must be reflected in the
    # report before any replay result can be considered attributable.
    model_fields = {
        "source_kind": "replay_source_kind",
        "action_set_id": "action_set_id",
        "feasibility_filter": "feasibility_filter",
        "collision_predicate": "collision_predicate",
        "pedestrian_response": "pedestrian_response",
    }
    for field_name, model_field in model_fields.items():
        declared = getattr(bound_config, field_name)
        actual = _model_metadata(model, model_field)
        if actual is None:
            continue
        # The omitted pedestrian-response default is a string-compatible marker,
        # so existing callers still observe/serialize ``unknown`` while native
        # adapters can bind it to their actual response mode. A caller that
        # explicitly supplies ``unknown`` remains distinct and fails closed
        # against a declared native mode. ``unspecified`` retains its legacy
        # model-binding semantics for all provenance fields.
        if declared is _DEFAULT_PEDESTRIAN_RESPONSE or declared == "unspecified":
            bound_config = replace(bound_config, **{field_name: actual})
        elif declared != actual:
            mismatches.append(f"{field_name}: declared={declared!r}, actual={actual!r}")
    return bound_config, tuple(mismatches)


def locate_last_avoidable(  # noqa: C901 - explicit fail-closed verdict state machine
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
    try:
        bound_config, metadata_mismatches = _bind_model_metadata(model, config)
    except _ReplayProvenanceError as exc:
        determinism = DeterminismCheck(
            replays=config.determinism_replays,
            collision_stable=False,
            contact_step_stable=False,
            observed_contact_steps=(None,) * config.determinism_replays,
        )
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
            abstain_reason="invalid_replay_provenance",
            notes=(str(exc),),
        )
    if metadata_mismatches:
        determinism = DeterminismCheck(
            replays=bound_config.determinism_replays,
            collision_stable=False,
            contact_step_stable=False,
            observed_contact_steps=(None,) * bound_config.determinism_replays,
        )
        return LastAvoidableReport(
            verdict=VERDICT_UNKNOWN,
            config=bound_config,
            determinism=determinism,
            branches=(),
            t_uca=None,
            t_inevitable=None,
            feasible_coverage=0.0,
            minimal_sufficient_interventions=(),
            runtime_s=runtime_s,
            abstained=True,
            abstain_reason="metadata_mismatch",
            notes=(
                "declared replay provenance does not match the model contract: "
                + "; ".join(metadata_mismatches),
            ),
        )
    if _snapshot_state_is_complete(model) is False:
        determinism = DeterminismCheck(
            replays=config.determinism_replays,
            collision_stable=False,
            contact_step_stable=False,
            observed_contact_steps=(None,) * config.determinism_replays,
        )
        return LastAvoidableReport(
            verdict=VERDICT_UNKNOWN,
            config=bound_config,
            determinism=determinism,
            branches=(),
            t_uca=None,
            t_inevitable=None,
            feasible_coverage=0.0,
            minimal_sufficient_interventions=(),
            runtime_s=runtime_s,
            abstained=True,
            abstain_reason="incomplete_snapshot_state",
            notes=(
                "the model declared that its replay snapshot omits mutable state "
                "required for deterministic continuation",
            ),
        )
    config = bound_config

    # A baseline shorter than the decision window cannot support branch capture.
    # Single-step substitutions additionally need the recorded suffix they
    # resume; hold substitutions do not consume baseline actions after the
    # decision window.  Report missing support as a fail-closed non-evaluation
    # rather than treating an empty suffix as a successful avoidance witness.
    # ``t_contact`` is a state tick / applied-action count. The action at index
    # ``t_contact - 1`` produces the contact state, so exactly ``t_contact``
    # recorded actions are required to replay the inclusive contact prefix.
    contact_prefix = config.t_contact
    required_baseline = (
        max(contact_prefix, config.t_contact + config.horizon - 1)
        if config.substitution_mode == SUBSTITUTION_SINGLE_STEP
        else contact_prefix
    )
    if len(baseline_actions) < required_baseline:
        determinism = DeterminismCheck(
            replays=config.determinism_replays,
            collision_stable=False,
            contact_step_stable=False,
            observed_contact_steps=(None,) * config.determinism_replays,
        )
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
            abstain_reason="insufficient_baseline_actions",
            notes=(
                "baseline action data did not cover the declared contact plus "
                f"horizon ({len(baseline_actions)} < {required_baseline})",
            ),
        )

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

    if all(contact_step is None for contact_step in determinism.observed_contact_steps):
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
            abstain_reason="baseline_no_contact",
            notes=(
                "baseline replay did not contact within the configured horizon; "
                "avoidability is untested",
            ),
        )

    observed_contact = determinism.observed_contact_steps[0]
    if observed_contact == 0:
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
            abstain_reason="baseline_initial_contact",
            notes=(
                "baseline was already in contact at the initial snapshot; "
                "no pre-contact unsafe control action can be identified",
            ),
        )
    if observed_contact is None or not (config.t_danger <= observed_contact <= config.t_contact):
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
            abstain_reason="baseline_contact_outside_declared_window",
            notes=(
                "baseline contact was observed outside the declared "
                f"[{config.t_danger}, {config.t_contact}] contact bounds",
            ),
        )
    if observed_contact != config.t_contact:
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
            abstain_reason="baseline_contact_tick_mismatch",
            notes=(
                "baseline contact tick did not match the declared contact tick: "
                f"observed={observed_contact}, declared={config.t_contact}; "
                "exact agreement is required before branch evaluation",
            ),
        )

    snapshots = _capture_window_snapshots(model, initial_snapshot, baseline_actions, config)
    try:
        branches, interventions = _branch_over_window(model, snapshots, baseline_actions, config)
    except ValueError:
        # This should be caught by the preflight above for a complete window,
        # but retaining the guard makes the fail-closed contract robust to a
        # model-specific horizon requirement.
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
            abstain_reason="insufficient_replay_horizon_support",
            notes=("at least one single-step branch lacked a recorded continuation",),
        )

    window_size = config.t_contact - config.t_danger
    with_feasible = sum(1 for b in branches if b.has_feasible_actions)
    feasible_coverage = with_feasible / window_size if window_size else 0.0

    # A preventing witness is not enough to certify avoidability when another
    # decision point could not be evaluated.  Keep the entire branch table for
    # diagnosis, but fail closed before emitting any non-unknown verdict.
    if feasible_coverage < 1.0:
        witness_note = (
            "a finite avoidance witness was observed, but incomplete feasible-action "
            "coverage prevents an avoidability determination"
            if any(branch.any_prevented for branch in branches)
            else "at least one decision point lacked a feasible action set; avoidability is untested"
        )
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
            notes=(witness_note,),
        )

    preventable_steps = sorted(b.step for b in branches if b.any_prevented)

    if preventable_steps:
        t_uca = preventable_steps[0]
        return LastAvoidableReport(
            verdict=VERDICT_AVOIDABLE,
            config=config,
            determinism=determinism,
            branches=tuple(branches),
            t_uca=t_uca,
            t_inevitable=min(preventable_steps[-1] + 1, config.t_contact),
            feasible_coverage=feasible_coverage,
            minimal_sufficient_interventions=tuple(interventions),
            runtime_s=runtime_s,
            notes=(),
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
