"""Fail-closed release-parameter freeze guard and 0.0.7 -> 0.0.8 arm-slot mapping.

Issue #9751 (author amendment on #9668, 2026-09-28) replaces the four hybrid
arm slots of the 0.0.8 roster with hybrid v4 arms under new, v4-named keys.
The v4 parameters may be frozen only after the #9748 development-split tuning
protocol finishes. Until then, the 0.0.8 campaign template binds each replaced
slot to a *placeholder* algorithm config that declares a
``release_parameter_freeze`` block whose status is not ``frozen``.

Every execution path refuses such a config:

* the map runner's algorithm-config parser and the shared candidate-manifest
  resolver raise :class:`UnfrozenReleaseParametersError`;
* campaign preflight raises before any output directory is created;
* release-manifest planner-roster admission reports a blocker.

Loading or inspecting the template stays possible, so contract tests and
tooling can still read the roster.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RELEASE_PARAMETER_FREEZE_KEY = "release_parameter_freeze"
FROZEN_STATUS = "frozen"

COMPARISON_PAIRED = "paired"
COMPARISON_IMPLEMENTATION_REPLACED = "implementation replaced"


class UnfrozenReleaseParametersError(ValueError):
    """Raised when an algorithm config declares release parameters that are not frozen."""


def release_parameter_freeze_blocker(config: Mapping[str, Any], *, label: str) -> str | None:
    """Return a blocker message when ``config`` declares unfrozen release parameters.

    A config without a ``release_parameter_freeze`` block is not governed by this
    guard and yields ``None``. A declared block must carry ``status: frozen``;
    any other status, a missing status, or a malformed block is a blocker.

    Returns:
        Human-readable blocker message, or ``None`` when the config may run.
    """
    if RELEASE_PARAMETER_FREEZE_KEY not in config:
        return None
    block = config.get(RELEASE_PARAMETER_FREEZE_KEY)
    if not isinstance(block, Mapping):
        return f"{label}: {RELEASE_PARAMETER_FREEZE_KEY} must be a mapping"
    status = str(block.get("status") or "").strip()
    if status == FROZEN_STATUS:
        return None
    gate = str(block.get("required_gate") or "unspecified gate").strip()
    return (
        f"{label}: release parameters are not frozen (status={status or 'missing'!r}); "
        f"this is a placeholder that cannot run before {gate} freezes its parameters"
    )


def assert_release_parameters_frozen(config: Mapping[str, Any], *, label: str) -> None:
    """Raise when ``config`` declares unfrozen release parameters.

    Raises:
        UnfrozenReleaseParametersError: If the config is an unfrozen placeholder.
    """
    blocker = release_parameter_freeze_blocker(config, label=label)
    if blocker is not None:
        raise UnfrozenReleaseParametersError(blocker)


def unfrozen_planner_config_blockers(planners: Iterable[Any]) -> list[str]:
    """Return freeze blockers for every enabled planner whose algo config is unfrozen.

    ``planners`` are campaign ``PlannerSpec``-like objects exposing ``key``,
    ``enabled`` and ``algo_config_path``. A missing or unreadable config is left
    to the existing loaders; this guard only reports declared unfrozen blocks.

    Returns:
        Sorted blocker messages, one per unfrozen enabled planner.
    """
    blockers: list[str] = []
    for planner in planners:
        if not getattr(planner, "enabled", True):
            continue
        raw_path = getattr(planner, "algo_config_path", None)
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        blocker = release_parameter_freeze_blocker(
            payload, label=f"planner {getattr(planner, 'key', '?')!s}"
        )
        if blocker is not None:
            blockers.append(blocker)
    return sorted(blockers)


@dataclass(frozen=True)
class ArmSlot:
    """One arm slot of the fixed 14-slot S30/H600 roster across releases."""

    key_0_0_7: str
    key_0_0_8: str
    comparison: str


# Order follows the 0.0.7 release manifest roster. Only the four hybrid slots
# change keys; every other slot keeps its key (corrected, versioned configs bound
# to version-free keys are disclosed in the release manifest, not here).
ARM_SLOTS_0_0_7_TO_0_0_8: tuple[ArmSlot, ...] = (
    ArmSlot("prediction_planner", "prediction_planner", COMPARISON_PAIRED),
    ArmSlot("goal", "goal", COMPARISON_PAIRED),
    ArmSlot("social_force", "social_force", COMPARISON_PAIRED),
    ArmSlot("orca", "orca", COMPARISON_PAIRED),
    ArmSlot("ppo", "ppo", COMPARISON_PAIRED),
    ArmSlot("socnav_sampling", "socnav_sampling", COMPARISON_PAIRED),
    ArmSlot("sacadrl", "sacadrl", COMPARISON_PAIRED),
    ArmSlot(
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
        COMPARISON_IMPLEMENTATION_REPLACED,
    ),
    ArmSlot(
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
        "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
        COMPARISON_IMPLEMENTATION_REPLACED,
    ),
    ArmSlot(
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v4_fast_progress_static_escape",
        COMPARISON_IMPLEMENTATION_REPLACED,
    ),
    ArmSlot(
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "hybrid_rule_v4_fast_progress_static_escape_continuous",
        COMPARISON_IMPLEMENTATION_REPLACED,
    ),
    ArmSlot("guarded_ppo", "guarded_ppo", COMPARISON_PAIRED),
    ArmSlot("predictive_mppi", "predictive_mppi", COMPARISON_PAIRED),
    ArmSlot("risk_dwa", "risk_dwa", COMPARISON_PAIRED),
)


def arm_slot_for_0_0_8_key(key: str) -> ArmSlot:
    """Return the arm slot that a 0.0.8 key fills.

    Raises:
        KeyError: If ``key`` is not a 0.0.8 roster key.

    Returns:
        The matching slot record.
    """
    for slot in ARM_SLOTS_0_0_7_TO_0_0_8:
        if slot.key_0_0_8 == key:
            return slot
    raise KeyError(f"not a 0.0.8 roster key: {key!r}")


__all__ = [
    "ARM_SLOTS_0_0_7_TO_0_0_8",
    "COMPARISON_IMPLEMENTATION_REPLACED",
    "COMPARISON_PAIRED",
    "FROZEN_STATUS",
    "RELEASE_PARAMETER_FREEZE_KEY",
    "ArmSlot",
    "UnfrozenReleaseParametersError",
    "arm_slot_for_0_0_8_key",
    "assert_release_parameters_frozen",
    "release_parameter_freeze_blocker",
    "unfrozen_planner_config_blockers",
]
