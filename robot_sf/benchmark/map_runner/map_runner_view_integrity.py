"""Runtime fail-closed guard for degenerate planner observation views (#3634).

Motivation: ``stream_gap`` was *silently blind* in the real benchmark runner (#3556/#3567) — its
standalone extractor read the wrong observation keys, so it saw ``robot=[0,0], n_peds=0`` and drove
blind every episode while still emitting collision "results". #3567 fixed the extractor and #3568
audited the seven headline planners with a *static* contract test, but explicitly deferred the
*runtime* guard. This module is that guard: the comprehensive, planner-agnostic version.

The invariant this enforces, at the ``map_runner`` planner-adapter boundary:

    the observation handed to the planner carries a nonzero pedestrian count
    AND the planner's own extractor sees zero pedestrians
    AND the planner is NOT declared pedestrian-blind by design
    --> the row is degenerate; ``map_runner`` must FAIL CLOSED
        (per docs/context/issue_691_benchmark_fallback_policy.md) instead of
        silently recording collision metrics produced by a blind planner.

Design choices that keep the guard conservative (no false trips):

* Ground truth is the *observation's own* pedestrian count — what was actually presented to the
  planner — not the raw simulator count. The SocNav observation is visibility-masked and truncated
  to ``max_pedestrians`` (see ``robot_sf.sensor.socnav_observation``), so a scenario where every
  pedestrian is occluded/out of range legitimately presents zero pedestrians; that is not
  degenerate and must not trip the guard.
* "What the planner sees" is probed through the planner adapter's *own* extractor
  (``_socnav_fields`` for the shared classical/reference planners, ``_extract_state`` for
  ``stream_gap``). When no extractor is discoverable, the view is treated as *not probeable* and the
  guard stays silent — it never fabricates a failure.
* A planner that ignores pedestrians by design (e.g. the ``goal`` reference, whose declared
  ``observation_spec.inputs`` omit ``pedestrians``) declares itself via that contract and is
  exempt. #3568 rejected a behavioural "reacts to pedestrians" guard for exactly this reason: the
  check inspects *what the planner sees*, not how it reacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
from robot_sf.errors import RobotSfError

#: Canonical degraded-reason code recorded when the guard trips (named in issue #3634/#3568).
DEGENERATE_PLANNER_VIEW_REASON = "degenerate_planner_view"


class DegeneratePlannerViewError(RobotSfError, RuntimeError):
    """Raised to fail a benchmark episode closed when a planner's effective view is degenerate.

    Carrying the structured diagnostic lets callers record an explicit non-success row
    (``degraded_reason: degenerate_planner_view``) rather than a misleading benchmark result.
    """

    def __init__(self, diagnostic: EffectiveViewIntegrity) -> None:
        """Store the integrity diagnostic and build a human-readable fail-closed message."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message())


@dataclass(frozen=True)
class EffectiveViewIntegrity:
    """Outcome of the planner effective-view integrity diagnostic for one observation.

    Attributes:
        degraded: ``True`` when the planner's effective view is degenerate (fail-closed trigger).
        degraded_reason: Reason code when degraded, else ``None``.
        observation_ped_count: Pedestrian count carried by the observation handed to the planner.
        extracted_ped_count: Pedestrian count the planner's own extractor saw, or ``None`` when the
            view could not be probed.
        pedestrian_blind_by_design: ``True`` when the planner declares it ignores pedestrians.
        probed: ``True`` when the planner's effective extraction was actually inspected.
    """

    degraded: bool
    degraded_reason: str | None
    observation_ped_count: int
    extracted_ped_count: int | None
    pedestrian_blind_by_design: bool
    probed: bool

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for episode/row metadata.

        Returns:
            dict[str, Any]: Stable diagnostic payload with the canonical reason key.
        """
        return {
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "observation_ped_count": self.observation_ped_count,
            "extracted_ped_count": self.extracted_ped_count,
            "pedestrian_blind_by_design": self.pedestrian_blind_by_design,
            "probed": self.probed,
        }

    def message(self) -> str:
        """Return a human-readable fail-closed explanation.

        Returns:
            str: Diagnostic message describing the degenerate-view trigger.
        """
        return (
            "degenerate planner observation view: the observation carried "
            f"{self.observation_ped_count} pedestrian(s) but the planner extracted "
            f"{self.extracted_ped_count}; failing closed with "
            f"degraded_reason={DEGENERATE_PLANNER_VIEW_REASON} "
            "(planner is not declared pedestrian-blind by design)."
        )


def _ped_count_from_value(value: Any) -> int:
    """Resolve a non-negative integer pedestrian count from a scalar/array payload.

    Returns:
        int: Parsed count, or ``0`` when the payload is missing/malformed.
    """
    if value is None:
        return 0
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return 0
    if arr.size == 0:
        return 0
    first = arr[0]
    if not np.isfinite(first):
        return 0
    return max(0, int(first))


def _rows_count(value: Any) -> int:
    """Count finite ``(N, 2)`` position rows in a positions payload.

    Returns:
        int: Number of position rows, or ``0`` when the payload is empty/malformed.
    """
    if value is None:
        return 0
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return 0
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        return 1 if arr.size >= 2 else 0
    return int(arr.shape[0])


def observation_ped_count(observation: Any) -> int:
    """Return the pedestrian count carried by a benchmark observation (nested or flat).

    This is the ground truth for the guard: it is exactly what was presented to the planner. The
    declared ``count`` field is preferred (it already reflects visibility masking/truncation); the
    number of position rows is used as a fallback.

    Returns:
        int: Non-negative pedestrian count, or ``0`` when no pedestrian payload is present.
    """
    if not isinstance(observation, dict):
        return 0
    peds = observation.get("pedestrians")
    if isinstance(peds, dict):
        if peds.get("count") is not None:
            return _ped_count_from_value(peds.get("count"))
        return _rows_count(peds.get("positions"))
    # Flat map-runner observation keys.
    if observation.get("pedestrians_count") is not None:
        return _ped_count_from_value(observation.get("pedestrians_count"))
    return _rows_count(observation.get("pedestrians_positions"))


def is_pedestrian_blind_by_design(algo_meta: Any) -> bool:
    """Return whether a planner declares (via its observation contract) that it ignores pedestrians.

    A planner is pedestrian-blind by design when its ``observation_spec.inputs`` contract omits the
    ``pedestrians`` channel (e.g. the ``goal`` reference). Such planners must not be false-flagged,
    so they are exempt from the degenerate-view guard.

    Returns:
        bool: ``True`` when the declared inputs omit the pedestrian channel.
    """
    if not isinstance(algo_meta, dict):
        return False
    spec = algo_meta.get("observation_spec")
    if not isinstance(spec, dict):
        return False
    inputs = spec.get("inputs")
    if inputs is None:
        return False
    try:
        declared = {str(value).strip().lower() for value in inputs}
    except TypeError:
        return False
    return "pedestrians" not in declared


def _resolve_adapter(policy_fn: Any) -> Any:
    """Return the planner adapter attached to a built policy callable, if any.

    Returns:
        Any: The adapter object, or ``None`` when the policy exposes no adapter.
    """
    return getattr(policy_fn, "_planner_adapter", None)


def probe_extracted_ped_count(policy_fn: Any, observation: Any) -> int | None:
    """Probe how many pedestrians the planner's *own* extractor sees in ``observation``.

    The probe invokes the adapter's own extraction so the result reflects the planner's effective
    view, mirroring the silent-blind failure class: a planner whose extractor reads the wrong keys
    returns zero here even when the observation carries pedestrians.

    Supported extractors (checked in order):

    * ``adapter._socnav_fields(obs) -> (robot, goal, ped_state)`` — shared classical/reference
      planners; the pedestrian count is read from ``ped_state``.
    * ``adapter._extract_state(obs) -> (robot_pos, heading, goal_pos, ped_pos, ped_vel)`` —
      ``stream_gap``; the pedestrian count is ``ped_pos`` row count.

    Returns:
        int | None: Extracted pedestrian count, or ``None`` when the view is not probeable
        (no recognised extractor, or the extractor raised).
    """
    adapter = _resolve_adapter(policy_fn)
    if adapter is None:
        return None

    socnav_fields = getattr(adapter, "_socnav_fields", None)
    if callable(socnav_fields):
        try:
            _robot, _goal, ped_state = socnav_fields(observation)
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        if isinstance(ped_state, dict):
            if ped_state.get("count") is not None:
                return _ped_count_from_value(ped_state.get("count"))
            return _rows_count(ped_state.get("positions"))
        return None

    extract_state = getattr(adapter, "_extract_state", None)
    if callable(extract_state):
        try:
            extracted = extract_state(observation)
        except (TypeError, ValueError, KeyError, IndexError):
            return None
        # (robot_pos, heading, goal_pos, ped_pos, ped_vel)
        if isinstance(extracted, tuple) and len(extracted) >= 4:
            return _rows_count(extracted[3])
        return None

    return None


def evaluate_effective_view_integrity(
    *,
    policy_fn: Callable[..., Any],
    observation: Any,
    algo_meta: Any,
) -> EffectiveViewIntegrity:
    """Evaluate whether a planner's effective observation view is degenerate (fail-closed guard).

    The guard trips only when *all* of the following hold:

    1. the observation carries a nonzero pedestrian count (something was presented to the planner);
    2. the planner is not declared pedestrian-blind by design;
    3. the planner's own extractor was probeable and saw zero pedestrians.

    Returns:
        EffectiveViewIntegrity: Structured diagnostic; ``degraded`` is ``True`` only on a real trip.
    """
    obs_count = observation_ped_count(observation)
    blind_by_design = is_pedestrian_blind_by_design(algo_meta)

    if obs_count <= 0 or blind_by_design:
        return EffectiveViewIntegrity(
            degraded=False,
            degraded_reason=None,
            observation_ped_count=obs_count,
            extracted_ped_count=None,
            pedestrian_blind_by_design=blind_by_design,
            probed=False,
        )

    extracted_count = probe_extracted_ped_count(policy_fn, observation)
    probed = extracted_count is not None
    degraded = probed and extracted_count == 0
    return EffectiveViewIntegrity(
        degraded=degraded,
        degraded_reason=DEGENERATE_PLANNER_VIEW_REASON if degraded else None,
        observation_ped_count=obs_count,
        extracted_ped_count=extracted_count,
        pedestrian_blind_by_design=blind_by_design,
        probed=probed,
    )
