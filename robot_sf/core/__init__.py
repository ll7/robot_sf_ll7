"""Small, additive simulation-state contracts for Robot SF.

The package intentionally keeps its public surface lazy.  Importing
``robot_sf.core`` therefore does not pull in the simulator, rendering stack, or
learning backends; consumers opt into the typed contract symbols they need.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

CORE_CONTRACT_VERSION = "core_contract.v1"

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .contract import (
        ActorId,
        ActorState,
        EpisodeRecord,
        ForceBreakdown,
        ForceComponent,
        ObservationSnapshot,
        Pose2D,
        TrackId,
        TransitionRecord,
        WorldFrame,
    )
    from .time import SimTime, Twist2D

__all__ = [
    "CORE_CONTRACT_VERSION",
    "ActorId",
    "ActorState",
    "EpisodeRecord",
    "ForceBreakdown",
    "ForceComponent",
    "ObservationSnapshot",
    "Pose2D",
    "SimTime",
    "TrackId",
    "TransitionRecord",
    "Twist2D",
    "WorldFrame",
]

_TIME_EXPORTS = frozenset({"SimTime", "Twist2D"})
_CONTRACT_EXPORTS = frozenset(set(__all__) - _TIME_EXPORTS - {"CORE_CONTRACT_VERSION"})


def __getattr__(name: str) -> Any:
    """Resolve typed contracts only when a caller requests one.

    Returns:
        Any: The requested lazily imported contract symbol.
    """

    if name in _TIME_EXPORTS:
        value = getattr(import_module(".time", __name__), name)
    elif name in _CONTRACT_EXPORTS:
        value = getattr(import_module(".contract", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy contract symbols in interactive discovery.

    Returns:
        list[str]: Available package attributes and lazy exports.
    """

    return sorted(set(globals()) | set(__all__))
