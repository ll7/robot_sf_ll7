"""Builder for the deterministic stand-still map-runner reference policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark import planner_command_contract as planner_commands
from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.utils import _config_hash

if TYPE_CHECKING:
    from collections.abc import Callable

STAND_STILL_ALGO_KEYS = frozenset({"stand_still"})


def metadata_seed() -> dict[str, Any]:
    """Declare the reference contract without changing frozen shared metadata.

    Returns:
        Fields required by the shared metadata enricher before policy construction.
    """
    return {
        "baseline_category": "classical",
        "policy_semantics": "constant_zero_velocity_reference",
        "planner_kinematics": {
            "planner_command_space": "unicycle_vw",
            "supports_native_commands": True,
            "supports_adapter_commands": False,
            "execution_detail": "Emits an exact zero linear and angular command on every step.",
        },
    }


def apply_observation_contract(meta: dict[str, Any]) -> dict[str, Any]:
    """Record that the stand-still policy consumes no observation fields.

    Returns:
        The supplied metadata with its effective observation contract corrected.
    """
    observation_spec = meta["observation_spec"]
    observation_spec["inputs"] = []
    observation_spec["notes"] = "Stationary reference ignores all observation content."
    contract = meta["planner_contract"]["observation_contract"]
    contract["required_inputs"] = []
    contract["notes"] = observation_spec["notes"]
    return meta


def build(
    algo_key: str,
    algo_config: dict[str, Any],
    *,
    robot_kinematics: str | None = None,
    robot_command_mode: str | None = None,
    adapter_impact_eval: bool = False,
) -> tuple[Callable[[dict[str, Any]], tuple[float, float]], dict[str, Any]]:
    """Build the zero-command policy and its native execution metadata.

    The callable deliberately does not inspect its observation or configuration.
    The runner records the active observation contract for comparability, while
    this reference always emits zero linear and angular velocity.

    Returns:
        Policy callable and metadata with native execution identity.
    """
    del adapter_impact_eval
    if algo_key not in STAND_STILL_ALGO_KEYS:
        supported = ", ".join(sorted(STAND_STILL_ALGO_KEYS))
        raise ValueError(f"Unsupported stand-still policy key '{algo_key}'. Expected: {supported}")

    normalized_robot_command_mode = (
        str(robot_command_mode).strip().lower() if robot_command_mode is not None else None
    )
    meta: dict[str, Any] = {
        "algorithm": algo_key,
        "status": "ok",
        "config": algo_config,
        "config_hash": _config_hash(algo_config),
        **metadata_seed(),
    }
    meta = enrich_algorithm_metadata(
        algo=algo_key,
        metadata=meta,
        execution_mode="native",
        robot_kinematics=robot_kinematics,
    )
    # This reference is intentionally local to the oracle runner. Keep the
    # shared algorithm-metadata source byte-for-byte frozen because historical
    # adversarial campaign contracts pin its SHA-256.
    apply_observation_contract(meta)
    planner_meta = meta.get("planner_kinematics")
    if isinstance(planner_meta, dict):
        planner_meta["planner_command_space"] = planner_commands.default_robot_command_space(
            robot_kinematics,
            algo_config,
            robot_command_mode=normalized_robot_command_mode,
        )

    def _policy(obs: dict[str, Any]) -> tuple[float, float]:
        """Emit the stationary reference command without reading observations.

        Returns:
            tuple[float, float]: Exact zero linear and angular velocity.
        """
        del obs
        return 0.0, 0.0

    return _policy, meta
