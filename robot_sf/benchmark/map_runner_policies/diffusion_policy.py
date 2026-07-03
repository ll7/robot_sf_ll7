"""Map-runner builder for the issue #4010 diffusion-policy planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_observations import obs_to_ppo_format
from robot_sf.benchmark.map_runner_policy_common import build_adapter_policy
from robot_sf.planner.diffusion_policy import (
    CLAIM_BOUNDARY,
    EVIDENCE_TIER,
    DiffusionPolicyAdapter,
    build_diffusion_policy_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable


DIFFUSION_POLICY_KEYS = frozenset(
    {"diffusion_policy", "diffusion_rl", "diffusion_local_policy", "colson_style_diffusion"}
)


class _MapRunnerDiffusionPolicyAdapter:
    """Convert map-runner observations before calling the diffusion adapter."""

    def __init__(self, adapter: DiffusionPolicyAdapter) -> None:
        self._adapter = adapter

    def reset(self, *, seed: int | None = None) -> None:
        """Reset underlying diffusion sampler state."""
        self._adapter.reset(seed=seed)

    def close(self) -> None:
        """Close underlying adapter."""
        self._adapter.close()

    def plan(self, obs: dict[str, Any]) -> tuple[float, float]:
        """Plan from normalized Robot SF state observations.

        Returns:
            tuple[float, float]: Linear and angular velocity command.
        """
        return self._adapter.plan(obs_to_ppo_format(obs))

    def diagnostics(self) -> dict[str, Any]:
        """Expose underlying diffusion diagnostics.

        Returns:
            dict[str, Any]: Compact diagnostic metadata.
        """
        return self._adapter.diagnostics()


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the diffusion-policy map-runner adapter.

    Returns:
        tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
        Policy callable and enriched metadata.
    """
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    config = build_diffusion_policy_config(algo_config)
    adapter = _MapRunnerDiffusionPolicyAdapter(DiffusionPolicyAdapter(config))
    meta: dict[str, Any] = {
        "algorithm": algo_key,
        "diffusion_policy": {
            "status": "ok",
            "evidence_tier": EVIDENCE_TIER,
            "allow_untrained_smoke": config.allow_untrained_smoke,
            "checkpoint_status": "untrained_smoke"
            if config.allow_untrained_smoke
            else "checkpoint_loaded",
            "normalizer_status": "not_required" if config.allow_untrained_smoke else "loaded",
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "planner_kinematics": {
            "planner_command_space": "unicycle_vw",
            "adapter_impact_eval": bool(adapter_impact_eval),
            "supports_native_commands": False,
            "supports_adapter_commands": True,
            "default_execution_mode": "adapter",
            "default_adapter_name": "DiffusionPolicyAdapter",
            "benchmark_command_space": "unicycle_vw",
            "projection_policy": "bounded_diffusion_action_to_unicycle_vw",
            "projection_documented": True,
        },
    }
    return build_adapter_policy(
        algo_key="diffusion_policy",
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="DiffusionPolicyAdapter",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations=(
            "diagnostic_only_untrained_smoke_not_benchmark_evidence"
            if config.allow_untrained_smoke
            else "diagnostic_only_smoke_checkpoint_not_benchmark_evidence"
        ),
    )
