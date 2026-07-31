"""Canonical local-planner protocol and the baseline->local family adapter.

This module establishes the single ``LocalPlannerProtocol`` that the native
``plan() -> tuple`` family (``robot_sf/planner/``) and the baseline
``step() -> dict`` family (``robot_sf/baselines/``) converge toward, plus the
explicit canonical adapter that bridges the baseline family onto the protocol
surface.

It is introduced by issue #6492 as the proof-of-contract for the broader
unification tracked by parent #6487. It deliberately migrates no other planner:
it only (1) defines the protocol, (2) defines the minimum diagnostics schema and
its fail-closed normalization, and (3) provides one canonical adapter wrapping a
baseline ``step()`` planner. Implementing this protocol does not commit a
planner to any benchmark, fallback, metric, or performance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from inspect import Signature, signature
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DIAGNOSTICS_UNAVAILABLE_KEY",
    "DIAGNOSTICS_UNAVAILABLE_REASON_KEY",
    "PLANNER_TYPE_KEY",
    "BaselineStepToLocalAdapter",
    "LocalPlannerProtocol",
    "normalize_planner_diagnostics",
]

#: Minimum required diagnostics key. Every normalized diagnostics payload must
#: carry a string ``planner_type`` identifying the producing planner family.
PLANNER_TYPE_KEY = "planner_type"

#: Key recording the diagnostics fields that could not be supplied by the
#: underlying planner, so missing data stays explicit instead of silent.
DIAGNOSTICS_UNAVAILABLE_KEY = "diagnostics_unavailable"

#: Human-readable reason explaining why a diagnostics field was synthesized.
DIAGNOSTICS_UNAVAILABLE_REASON_KEY = "diagnostics_unavailable_reason"


@runtime_checkable
class LocalPlannerProtocol(Protocol):
    """Canonical local-planner protocol for the ``plan() -> tuple`` family.

    A local planner consumes a structured observation each step and returns a
    ``(linear_speed, angular_rate)`` command tuple. ``reset``, ``diagnostics``,
    and ``close`` carry lifecycle, observability, and resource-cleanup
    semantics matching the canonical local-planner runner call sites.

    Notes:
        - ``reset`` accepts a keyword-only ``seed``. Planners that do not use a
          seed must still accept the parameter and discard it; canonical
          adapters forward the seed when supported and ignore it otherwise.
        - ``diagnostics`` returns a free-form dict that, after
          :func:`normalize_planner_diagnostics`, must carry a string
          ``planner_type``. Raw payloads that cannot supply it are normalized
          fail-closed (with an explicit reason) rather than dropped.
        - ``close`` must be idempotent.

    Example::

        class MyLocalPlanner:
            def plan(self, observation: dict[str, Any]) -> tuple[float, float]: ...
            def reset(self, *, seed: int | None = None) -> None: ...
            def diagnostics(self) -> dict[str, Any]: ...
            def close(self) -> None: ...
    """

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the ``(linear_speed, angular_rate)`` command for an observation.

        Args:
            observation: Structured planner observation payload.

        Returns:
            The ``(linear, angular)`` command tuple.
        """

    def reset(self, *, seed: int | None = None) -> None:
        """Reset planner state, forwarding ``seed`` when the planner uses one.

        Args:
            seed: Optional RNG seed.
        """

    def diagnostics(self) -> dict[str, Any]:
        """Return planner execution diagnostics.

        Returns:
            A diagnostics payload. After normalization it must include
            ``planner_type`` as a string.
        """

    def close(self) -> None:
        """Release held resources. Must be idempotent."""


def normalize_planner_diagnostics(
    raw: Mapping[str, Any] | Any,
    *,
    fallback_planner_type: str,
) -> dict[str, Any]:
    """Normalize a raw diagnostics payload to the minimum protocol schema.

    Guarantees the returned dict carries a string ``planner_type``. When the raw
    payload is missing, not a mapping, or lacks a valid ``planner_type``, the
    ``fallback_planner_type`` is synthesized and the loss is recorded explicitly
    via ``diagnostics_unavailable`` / ``diagnostics_unavailable_reason`` instead
    of being silently dropped.

    Args:
        raw: Raw diagnostics returned by a planner, or any non-mapping value.
        fallback_planner_type: ``planner_type`` to synthesize when the raw
            payload cannot supply a valid one.

    Returns:
        Normalized diagnostics dict with at least a string ``planner_type``,
        preserving any other keys the raw payload carried.
    """
    if isinstance(raw, Mapping):
        payload: dict[str, Any] = dict(raw)
    else:
        payload = {}

    unavailable: list[str] = []
    reasons: list[str] = []

    candidate = payload.get(PLANNER_TYPE_KEY)
    if isinstance(candidate, str) and candidate.strip():
        planner_type = candidate
    else:
        planner_type = fallback_planner_type
        unavailable.append(PLANNER_TYPE_KEY)
        if not isinstance(raw, Mapping):
            reasons.append(f"diagnostics() did not return a mapping (got {type(raw).__name__})")
        elif candidate is None:
            reasons.append("diagnostics() omitted planner_type")
        else:
            reasons.append(
                f"diagnostics() returned a non-string/empty planner_type ({candidate!r})"
            )

    payload[PLANNER_TYPE_KEY] = planner_type
    if unavailable:
        payload[DIAGNOSTICS_UNAVAILABLE_KEY] = unavailable
        payload[DIAGNOSTICS_UNAVAILABLE_REASON_KEY] = "; ".join(reasons)
    return payload


def _action_dict_to_command(action: Any, planner_type: str) -> tuple[float, float]:
    """Convert a baseline action dict to a ``(linear, angular)`` command tuple.

    Args:
        action: Action dict carrying ``{"v", "omega"}`` (unicycle) or
            ``{"vx", "vy"}`` (holonomic) keys.
        planner_type: Planner name used in failure messages.

    Returns:
        The ``(linear_speed, angular_rate)`` command tuple.

    Raises:
        TypeError: When ``action`` is not a mapping.
        ValueError: When the action dict carries neither recognized key pair.
    """
    if not isinstance(action, Mapping):
        raise TypeError(f"{planner_type}.step() must return a dict, got {type(action).__name__}")
    if "v" in action and "omega" in action:
        return float(action["v"]), float(action["omega"])
    if "vx" in action and "vy" in action:
        # Holonomic velocity cannot yield an angular rate from a single sample
        # without held heading state; the angular component is explicitly zeroed
        # (documented incompatibility deferred to parent #6487).
        return math.hypot(float(action["vx"]), float(action["vy"])), 0.0
    raise ValueError(
        f"{planner_type}.step() action must contain v/omega or vx/vy keys; got {dict(action)!r}"
    )


def _safe_diagnostics(planner: Any) -> Any:
    """Return the wrapped planner's diagnostics payload, failing closed on error.

    Returns:
        The planner's ``diagnostics()`` result, ``None`` when the method is
        absent, or a dict recording the failure when calling it raised.
    """
    diagnostics = getattr(planner, "diagnostics", None)
    if not callable(diagnostics):
        return None
    try:
        return diagnostics()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash the adapter
        return {"_diagnostics_call_error": f"{type(exc).__name__}: {exc}"}


def _signature_accepts(call_signature: Signature, *args: Any, **kwargs: Any) -> bool:
    """Return whether a callable signature accepts the supplied arguments."""
    try:
        call_signature.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def _reset_with_optional_seed(reset: Any, *, seed: int | None) -> None:
    """Invoke a reset hook without swallowing ``TypeError`` from its implementation.

    Signature inspection chooses the compatible legacy call shape before the
    hook runs. This avoids retrying a seed-aware reset when its body raises an
    unrelated ``TypeError``, which could otherwise duplicate state changes.
    """
    try:
        reset_signature = signature(reset)
    except (TypeError, ValueError):
        # Some extension callables do not expose a signature. Preserve the
        # canonical keyword call when a seed exists and the seedless call when
        # it does not; any TypeError from the hook must propagate unchanged.
        if seed is None:
            reset()
        else:
            reset(seed=seed)
        return

    if _signature_accepts(reset_signature, seed=seed):
        reset(seed=seed)
        return
    if seed is None and _signature_accepts(reset_signature):
        reset()
        return
    if _signature_accepts(reset_signature, seed):
        reset(seed)
        return
    if _signature_accepts(reset_signature):
        reset()
        return
    raise TypeError("reset hook accepts neither an optional seed nor a seedless call")


class BaselineStepToLocalAdapter:
    """Wrap a baseline ``step() -> dict`` planner as a ``LocalPlannerProtocol``.

    Bridges the baseline family (``robot_sf/baselines/``, ``step()`` returning
    ``{"vx", "vy"}`` holonomic or ``{"v", "omega"}`` unicycle action dicts) onto
    the canonical local-planner ``plan() -> tuple[float, float]`` surface.

    Unicycle actions map directly to ``(linear, angular)``. Holonomic
    ``{"vx", "vy"}`` actions are converted to ``(speed, 0.0)`` because a single
    velocity sample cannot derive an angular rate without held heading state;
    this lossy-but-explicit conversion is a documented incompatibility deferred
    to parent #6487 for full holonomic support.
    """

    def __init__(self, planner: Any, *, planner_type: str | None = None) -> None:
        """Initialize the adapter with a concrete baseline planner.

        Args:
            planner: Baseline planner exposing ``step(obs) -> dict``. Should also
                expose ``reset`` and ``close`` when supported.
            planner_type: Optional explicit ``planner_type`` for diagnostics.
                Defaults to the wrapped planner's class name.
        """
        self.planner = planner
        self._planner_type = planner_type or type(planner).__name__
        self._closed = False

    def reset(self, *, seed: int | None = None) -> None:
        """Forward reset to the wrapped planner, tolerating seedless signatures.

        Selects the protocol keyword form, a positional legacy form, or a
        seedless call from the hook signature before invoking it.
        """
        reset = getattr(self.planner, "reset", None)
        if not callable(reset):
            return
        _reset_with_optional_seed(reset, seed=seed)

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Delegate to the wrapped baseline ``step`` and convert to a command tuple.

        Args:
            observation: Structured planner observation payload forwarded to the
                wrapped baseline planner's ``step``.

        Returns:
            The ``(linear_speed, angular_rate)`` command tuple.
        """
        action = self.planner.step(observation)
        return _action_dict_to_command(action, self._planner_type)

    def diagnostics(self) -> dict[str, Any]:
        """Return protocol-normalized diagnostics, fail-closed when unavailable."""
        raw = _safe_diagnostics(self.planner)
        return normalize_planner_diagnostics(raw, fallback_planner_type=self._planner_type)

    def close(self) -> None:
        """Release the wrapped planner's resources once; idempotent thereafter."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self.planner, "close", None)
        if callable(close):
            close()
