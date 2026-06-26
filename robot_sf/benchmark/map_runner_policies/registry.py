"""Registry dispatch helpers for map-runner policy builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

# Policies return a unicycle ``(linear, angular)`` tuple, or a structured payload
# (e.g. holonomic world-velocity commands from ORCA/HRVO) once those families are
# migrated into the registry; keep both shapes in the contract.
type PolicyCallable = Callable[[dict[str, Any]], tuple[float, float] | dict[str, Any]]
type PolicyBuildResult = tuple[PolicyCallable, dict[str, Any]]


class PolicyBuilder(Protocol):
    """Callable contract for per-family map-runner policy builders."""

    def __call__(
        self,
        algo_key: str,
        algo_config: dict[str, Any],
        *,
        robot_kinematics: str | None = None,
        robot_command_mode: str | None = None,
    ) -> PolicyBuildResult:
        """Build a policy callable and metadata for ``algo_key``."""


def build_registered_policy(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    builders: Mapping[str, PolicyBuilder],
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
) -> PolicyBuildResult | None:
    """Build a policy from the migrated registry, or return ``None``.

    The legacy ``map_runner._build_policy`` dispatcher still owns families not
    migrated to ``map_runner_policies``. This helper keeps the registry lookup
    testable while preserving the fall-through behavior for unmigrated keys.

    Returns:
        Built policy and metadata when ``algo_key`` is registered, otherwise
        ``None`` so the caller can continue legacy dispatch.
    """

    builder = builders.get(algo_key)
    if builder is None:
        return None
    return builder(
        algo_key,
        algo_config,
        robot_kinematics=robot_kinematics,
        robot_command_mode=robot_command_mode,
    )
