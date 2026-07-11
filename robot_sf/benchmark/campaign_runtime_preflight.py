"""CPU-only runtime dependency and map checks for camera-ready campaigns.

The camera-ready campaign preflight must reject failures that would otherwise surface only after a
GPU allocation starts.  These checks deliberately import only required policy modules and resolve
only prepared scenario maps; they do not load checkpoints or execute episodes.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.errors import RobotSfError
from robot_sf.training.scenario_loader import resolve_map_definition

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec


# These are modules imported by the selected policy construction path, rather than optional
# convenience imports. ORCA has its dedicated rvo2 preflight, so it is intentionally not repeated
# here. Hybrid global RL always constructs either a PPO or SAC local policy (SAC by default).
_POLICY_IMPORTS: dict[str, tuple[str, ...]] = {
    "ppo": ("stable_baselines3",),
    "sac": ("stable_baselines3",),
    "guarded_ppo": ("stable_baselines3",),
    "hybrid_global_rl": ("stable_baselines3",),
    "global_rl_local": ("stable_baselines3",),
    "route_conditioned_rl": ("stable_baselines3",),
    "hybrid_route_rl": ("stable_baselines3",),
    "drl_vo": ("torch",),
    "distributional_rl": ("torch",),
    "qr_dqn": ("torch",),
    "diffusion_policy": ("torch",),
    "diffusion_rl": ("torch",),
    "diffusion_local_policy": ("torch",),
    "colson_style_diffusion": ("torch",),
}


class CampaignPolicyDependencyPreflightError(RobotSfError, RuntimeError):
    """Raised when an enabled campaign arm cannot import a required policy dependency."""

    def __init__(self, message: str, *, arms: tuple[str, ...]) -> None:
        """Store the actionable error plus the affected campaign arms."""
        super().__init__(message)
        self.arms = arms


class CampaignScenarioMapPreflightError(RobotSfError, RuntimeError):
    """Raised when a prepared camera-ready scenario has an unresolvable map file."""

    def __init__(self, message: str, *, scenarios: tuple[str, ...]) -> None:
        """Store the actionable error plus the affected prepared scenarios."""
        super().__init__(message)
        self.scenarios = scenarios


@dataclass(frozen=True)
class _ImportFailure:
    """One arm/module import failure retained for an actionable aggregate error."""

    planner_key: str
    algo: str
    module: str
    error: Exception


def _required_policy_modules(planner: PlannerSpec) -> tuple[str, ...]:
    """Return runtime modules required by an enabled planner's policy construction path."""
    if not planner.enabled:
        return ()
    return _POLICY_IMPORTS.get(planner.algo.strip().lower(), ())


def check_campaign_arm_policy_dependencies_preflight(
    cfg: CampaignConfig,
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, Any]:
    """Import every enabled arm's required policy modules in the active interpreter.

    Returns:
        Summary naming each checked arm/module pair.

    Raises:
        CampaignPolicyDependencyPreflightError: If one or more required imports fail.
    """
    checks = [
        (planner, module)
        for planner in cfg.planners
        for module in _required_policy_modules(planner)
    ]
    failures: list[_ImportFailure] = []
    for planner, module in checks:
        try:
            import_module(module)
        except (ImportError, OSError) as exc:
            failures.append(
                _ImportFailure(
                    planner_key=planner.key,
                    algo=planner.algo,
                    module=module,
                    error=exc,
                )
            )

    if failures:
        details = "\n".join(
            f"  - arm '{failure.planner_key}' (algo={failure.algo!r}) requires "
            f"'{failure.module}': {type(failure.error).__name__}: {failure.error}"
            for failure in failures
        )
        message = (
            f"Campaign policy-dependency preflight failed for {len(failures)} import(s):\n"
            f"{details}\n"
            "Install the camera-ready policy dependencies in this same interpreter with:\n"
            "  uv sync --all-extras\n"
            "Aborting before starting the benchmark campaign."
        )
        logger.error(message)
        raise CampaignPolicyDependencyPreflightError(
            message,
            arms=tuple(sorted({failure.planner_key for failure in failures})),
        )

    logger.info("Policy-dependency preflight passed for {} arm-module import(s).", len(checks))
    return {
        "checked": len(checks),
        "arms": [
            {"planner_key": planner.key, "algo": planner.algo, "module": module}
            for planner, module in checks
        ],
    }


def check_campaign_scenario_maps_preflight(
    scenarios: list[dict[str, Any]],
    *,
    scenario_path: Path = Path("."),
    resolve_map: Callable[[str | None], Any] | None = None,
) -> dict[str, Any]:
    """Resolve every prepared scenario map with the same repository-root semantics as a run.

    ``run_batch`` receives the prepared scenario list and defaults its ``scenario_path`` to ``.``.
    Keeping that behavior here verifies the parent-normalized ``map_file`` values that both the
    in-process and subprocess campaign paths execute.

    Returns:
        Summary containing the number of scenarios and map references swept.

    Raises:
        CampaignScenarioMapPreflightError: If any named ``map_file`` cannot be loaded.
    """
    resolver = resolve_map
    if resolver is None:

        def resolver(map_file: str | None) -> Any:
            return resolve_map_definition(map_file, scenario_path=scenario_path)

    failures: list[tuple[str, str, Exception | None]] = []
    checked = 0
    for scenario in scenarios:
        map_file = scenario.get("map_file")
        if not isinstance(map_file, str) or not map_file.strip():
            continue
        checked += 1
        scenario_name = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        try:
            if resolver(map_file) is None:
                failures.append((scenario_name, map_file, None))
        except (OSError, ValueError) as exc:
            failures.append((scenario_name, map_file, exc))

    if failures:
        details = "\n".join(
            f"  - scenario '{name}', map_file '{map_file}'"
            + (f": {type(error).__name__}: {error}" if error is not None else "")
            for name, map_file, error in failures
        )
        message = (
            f"Campaign map-resolvability preflight failed for {len(failures)} scenario(s):\n"
            f"{details}\n"
            "Fix each scenario's map_file so it resolves from the repository root after campaign "
            "normalization. Aborting before starting the benchmark campaign."
        )
        logger.error(message)
        raise CampaignScenarioMapPreflightError(
            message,
            scenarios=tuple(name for name, _, _ in failures),
        )

    logger.info("Map-resolvability preflight passed for {} scenario map reference(s).", checked)
    return {"checked": checked, "scenario_count": len(scenarios)}


__all__ = [
    "CampaignPolicyDependencyPreflightError",
    "CampaignScenarioMapPreflightError",
    "check_campaign_arm_policy_dependencies_preflight",
    "check_campaign_scenario_maps_preflight",
]
