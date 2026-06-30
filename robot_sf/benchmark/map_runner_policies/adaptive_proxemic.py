"""Builder for adaptive-proxemic-selector map-runner policy family.

Continuation of ``_build_policy`` decomposition (#3384). This module is a
behavior-preserving move of the adaptive proxemic selector branch from
``robot_sf.benchmark.map_runner`` into the policy-builder registry package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.adaptive_proxemic_selector import (
    AdaptiveProxemicSelectorAdapter,
    build_adaptive_proxemic_selector_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable


#: Algorithm keys handled by the adaptive proxemic selector builder.
ADAPTIVE_PROXEMIC_SELECTOR_KEYS = frozenset(
    {"adaptive_proxemic_selector_v0", "adaptive_proxemic_selector_v1"}
)


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build adaptive proxemic selector adapter policy metadata.

    Args:
        algo_key: Normalized algorithm key (one :data:`ADAPTIVE_PROXEMIC_SELECTOR_KEYS`).
        algo_config: Algorithm configuration payload.
        robot_kinematics: Runtime robot kinematics label for metadata enrichment.
        robot_command_mode: Runtime robot command mode (for holonomic metadata labels).

    Returns:
        Policy callable and enriched metadata dictionary.
    """
    if algo_key not in ADAPTIVE_PROXEMIC_SELECTOR_KEYS:
        supported = ", ".join(sorted(ADAPTIVE_PROXEMIC_SELECTOR_KEYS))
        raise ValueError(
            f"Unsupported adaptive proxemic selector algo_key {algo_key!r}; "
            f"expected one of: {supported}"
        )

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    selector_algo_config = dict(algo_config)
    selector_algo_config.setdefault(
        "selector_version",
        "v1" if algo_key == "adaptive_proxemic_selector_v1" else "v0",
    )
    selector_config = build_adaptive_proxemic_selector_config(selector_algo_config)
    adapter = AdaptiveProxemicSelectorAdapter(config=selector_config)
    meta: dict[str, Any] = {
        "adaptive_proxemic_selector": {
            "status": "enabled",
            "selector_version": selector_config.selector_version,
            "diagnostic_only": bool(selector_config.diagnostic_only),
            "claim_boundary": selector_config.claim_boundary,
            "profile_sources": [
                selector_config.profiles[name].source_candidate
                for name in ("conservative", "neutral", "open")
            ],
        }
    }
    return build_adapter_policy(
        algo_key=algo_key,
        algo_config=selector_algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="AdaptiveProxemicSelectorAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations=(
            "diagnostic-only selector over fixed proxemic profiles; "
            "not benchmark or comfort evidence"
        ),
    )
