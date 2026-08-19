"""Map-runner bridge for the bounded BRNE exploration planner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.baselines.brne import BRNE_PINNED_SHA, BRNEPlanner, build_brne_config
from robot_sf.benchmark.map_runner.map_runner_observations import obs_to_brne_format
from robot_sf.benchmark.map_runner_policies.map_runner_policy_common import build_adapter_policy

if TYPE_CHECKING:
    from collections.abc import Callable


BRNE_KEYS = frozenset({"brne"})


class _MapRunnerBRNEAdapter:
    """Adapt BRNE's baseline ``step`` contract to map-runner ``plan``."""

    def __init__(self, planner: BRNEPlanner) -> None:
        """Store the already-validated BRNE planner."""
        self._planner = planner

    def reset(self, *, seed: int | None = None) -> None:
        """Reset BRNE state between episodes."""
        self._planner.reset(seed=seed)

    def close(self) -> None:
        """Release the staged BRNE module reference."""
        self._planner.close()

    def diagnostics(self) -> dict[str, Any]:
        """Expose the planner's effective sample-count provenance.

        Returns:
            dict[str, Any]: Runtime planner metadata.
        """
        return {"planner_metadata": self._planner.get_metadata()}

    def plan(self, obs: dict[str, Any]) -> tuple[float, float]:
        """Run BRNE and return the native unicycle command.

        Returns:
            tuple[float, float]: Native ``(linear_velocity, angular_velocity)``.
        """
        action = self._planner.step(obs_to_brne_format(obs))
        return float(action["v"]), float(action["omega"])


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the native-only BRNE map-runner policy.

    Returns:
        Policy callable and explicit diagnostic metadata.

    Raises:
        RuntimeError: If the staged upstream BRNE core is unavailable.
        ValueError: If a fallback-enabled configuration is supplied.
    """
    del adapter_impact_eval
    if bool(algo_config.get("fallback_on_error", False)):
        raise ValueError("BRNE diagnostic preflight requires fallback_on_error: false")
    if bool(algo_config.get("include_in_paper", False)):
        raise ValueError("BRNE diagnostic preflight cannot set include_in_paper: true")

    planner = BRNEPlanner(build_brne_config(algo_config), seed=None)
    planner_metadata = planner.get_metadata()
    if (
        planner_metadata.get("status") != "ok"
        or planner_metadata.get("source_commit") != BRNE_PINNED_SHA
        or planner_metadata.get("source_pin") != BRNE_PINNED_SHA
        or planner_metadata.get("source_integrity") != "clean_pinned_worktree"
    ):
        reason = planner_metadata.get("status", "missing_dependency")
        planner.close()
        raise RuntimeError(
            "BRNE staged core is unavailable for the native diagnostic preflight "
            f"({reason}; source provenance is not the pinned clean checkout). "
            "Stage the pinned local-only repository first."
        )

    adapter = _MapRunnerBRNEAdapter(planner)
    meta: dict[str, Any] = {
        "algorithm": algo_key,
        "brne_diagnostic": {
            "status": "native_core_via_adapter",
            "execution_semantics": "native_upstream_core_through_robot_sf_adapter",
            "evidence_tier": "smoke_diagnostic",
            "claim_boundary": (
                "corridor-only native execution and non-degenerate behavior; "
                "not benchmark ranking, safety, realism, or paper evidence"
            ),
            "fallback_policy": "disabled; fallback/degraded rows are unavailable",
            "scenario_scope": "single-passage corridor-class maps only",
        },
        "upstream_reference": {
            "repo_url": "https://github.com/MurpheyLab/brne",
            "commit": "633a5cd",
            "pinned_sha": BRNE_PINNED_SHA,
            "staged_path": planner.config.stage_path,
            "license": "GPL-3.0 (local-only staging; not vendored/redistributed)",
        },
        "planner_metadata": planner_metadata,
    }
    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    return build_adapter_policy(
        algo_key="brne",
        algo_config=algo_config,
        meta=meta,
        adapter=adapter,
        adapter_name="BRNEPlanner",
        robot_kinematics=robot_kinematics,
        normalized_robot_command_mode=normalized_robot_command_mode,
        limitations="corridor_only_native_diagnostic_not_benchmark_evidence",
    )


__all__ = ["BRNE_KEYS", "build"]
