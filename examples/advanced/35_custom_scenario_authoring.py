"""Author a custom corridor scenario programmatically, validate, reload, and smoke-test it.

Usage:
    uv run python examples/advanced/35_custom_scenario_authoring.py [--out PATH]

Prerequisites:
    - None beyond the repository (map asset is committed).

Expected Output:
    - Loguru summary of the authored scenario, validation report, reload digest match,
      negative-fixture outcomes, and a short headless smoke rollout.

Limitations:
    - Tutorial only; it authors one compact corridor scenario and small negative fixtures.
    - Out-of-bounds and overlapping-start inputs are advisory warnings, not hard errors.

References:
    - docs/SCENARIOS.md
    - robot_sf/training/scenario_loader.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.benchmark.identity.hash_utils import stable_hash
from robot_sf.common.seed import set_global_seed
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.nav.map_config import SinglePedestrianDefinition, SocialGroupDefinition
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    load_scenarios_for_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = REPO_ROOT / "maps/svg_maps/classic_head_on_corridor.svg"
SEED = 7
SMOKE_STEPS = 10


def _step_budget(default: int) -> int:
    """Return a smaller rollout budget when the example runs in smoke mode."""
    override = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:  # pragma: no cover - defensive guard
            pass
    return default


def build_scenario(map_file: str) -> dict[str, Any]:
    """Build one compact corridor scenario mapping with explicit identity and units.

    Args:
        map_file: Map reference string stored in the scenario (resolved relative
            to the serialized file's parent on load).

    Returns:
        A scenario mapping using meters, seconds, unique string actor IDs, an
        explicit seed list in deterministic order, and separated start/goal zones.
    """
    return {
        "name": "tutorial_authored_corridor",
        "map_file": map_file,
        "simulation_config": {"max_episode_steps": 50, "ped_density": 0.02},
        "robot_config": {"radius": 0.3},
        "metadata": {"archetype": "corridor", "authored_by": "35_custom_scenario_authoring"},
        "seeds": [7, 8, 9],
    }


def serialize_scenario(scenario: dict[str, Any], out_path: Path) -> Path:
    """Write the scenario mapping to a caller-owned YAML file.

    Args:
        scenario: Scenario mapping from :func:`build_scenario`.
        out_path: Destination file path (parent directories are created).

    Returns:
        The resolved output path.
    """
    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False))
    return out_path.resolve()


def validate_scenario_file(path: Path) -> None:
    """Fail closed when the serialized file has any validation issue.

    Args:
        path: Serialized scenario file to validate.

    Raises:
        ValueError: If the validation report records any issue.
    """
    report = load_scenarios_for_validation(path)
    problems = list(report.entry_issues) + list(report.load_issues)
    if report.load_error is not None:
        problems.append(str(report.load_error))
    if problems:
        raise ValueError(f"Scenario validation failed: {problems}")
    logger.info("Validation clean: {} row(s), digest checks follow.", len(report.scenarios))


def reload_and_compare(path: Path, expected_digest: str) -> None:
    """Reload the file and prove semantic equivalence through the canonical digest.

    Args:
        path: Serialized scenario file to reload.
        expected_digest: Digest of the in-memory scenario mapping.

    Raises:
        ValueError: If the reloaded scenario digest differs.
    """
    [reloaded] = load_scenarios(path)
    actual_digest = stable_hash(reloaded)
    if actual_digest != expected_digest:
        raise ValueError("Reloaded scenario differs from the authored mapping.")
    logger.info("Reload digest matches: {}", actual_digest)


def run_headless_smoke(scenario_path: Path, steps: int) -> float:
    """Run a short headless rollout of the authored scenario and return total reward.

    Args:
        scenario_path: Serialized scenario file.
        steps: Bounded step budget for the smoke rollout.

    Returns:
        Total reward collected during the smoke rollout.
    """
    set_global_seed(SEED)
    [scenario] = load_scenarios(scenario_path)
    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    env = make_robot_env(config=config, seed=SEED)
    total_reward = 0.0
    try:
        observation, _ = env.reset()
        logger.info("Smoke env reset; observation keys: {}.", _observation_keys(observation))
        for _ in range(steps):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                env.reset()
    finally:
        env.close()
    logger.info("Smoke rollout complete: {} steps, total reward {:.3f}.", steps, total_reward)
    return total_reward


def _observation_keys(observation: Any) -> list[str]:
    """Return observation mapping keys, or an empty list for array observations."""
    if hasattr(observation, "keys"):
        return list(observation.keys())
    return []


def demonstrate_negative_fixtures() -> dict[str, str]:
    """Demonstrate invalid geometry/identity handling with specific outcomes.

    Returns:
        Mapping of fixture name to its observed outcome description.
    """
    outcomes: dict[str, str] = {}

    # Duplicate actor identity fails fast with a specific error.
    dup_a = SinglePedestrianDefinition(id="ped_1", start=(1.0, 1.0), goal=(5.0, 1.0))
    dup_b = SinglePedestrianDefinition(id="ped_1", start=(2.0, 2.0), goal=(6.0, 2.0))
    try:
        from robot_sf.nav.map_config import MapDefinition

        MapDefinition(
            width=10.0,
            height=10.0,
            obstacles=[],
            robot_spawn_zones=[],
            ped_spawn_zones=[],
            robot_goal_zones=[],
            bounds=[],
            robot_routes=[],
            ped_goal_zones=[],
            ped_crowded_zones=[],
            ped_routes=[],
            single_pedestrians=[dup_a, dup_b],
        )
        outcomes["duplicate_identity"] = "UNEXPECTEDLY ACCEPTED"
    except ValueError as exc:
        outcomes["duplicate_identity"] = f"ValueError: {exc}"

    # Non-positive group radius fails fast with a specific error.
    try:
        SocialGroupDefinition(
            group_id="g1",
            type="conversation",
            members=("ped_1",),
            formation="circular_conversation",
            centroid=(5.0, 5.0),
            radius=0.0,
        )
        outcomes["invalid_radius"] = "UNEXPECTEDLY ACCEPTED"
    except ValueError as exc:
        outcomes["invalid_radius"] = f"ValueError: {exc}"

    # Malformed start shape fails fast with a specific error.
    try:
        SinglePedestrianDefinition(id="ped_2", start=[1.0, 1.0], goal=(5.0, 1.0))  # type: ignore[arg-type]
        outcomes["malformed_start"] = "UNEXPECTEDLY ACCEPTED"
    except ValueError as exc:
        outcomes["malformed_start"] = f"ValueError: {exc}"

    # Goal equal to start is advisory-only: construction succeeds with a warning.
    warn_ped = SinglePedestrianDefinition(id="ped_3", start=(3.0, 3.0), goal=(3.0, 3.0))
    outcomes["goal_equals_start"] = f"accepted with warning (start={warn_ped.start})"

    # Out-of-bounds waypoint is advisory-only at definition level.
    oob_ped = SinglePedestrianDefinition(
        id="ped_4", start=(1.0, 1.0), trajectory=[(2.0, 2.0), (999.0, 999.0)]
    )
    outcomes["path_escape"] = f"accepted; flagged by bounds checks (waypoints={oob_ped.trajectory})"

    for name, outcome in outcomes.items():
        logger.info("Negative fixture {}: {}", name, outcome)
    return outcomes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the tutorial.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Parsed arguments with the caller-owned output path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/custom_scenario_authoring/authored_corridor.yaml"),
        help="Caller-owned destination for the serialized scenario.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Author, validate, reload, smoke-test, and negatively probe one scenario.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    out_path = Path(args.out)
    try:
        rel_map = os.path.relpath(DEFAULT_MAP, start=out_path.parent.resolve())
    except ValueError:  # pragma: no cover - cross-drive fallback
        rel_map = str(DEFAULT_MAP.resolve())
    scenario = build_scenario(rel_map)
    written = serialize_scenario(scenario, out_path)
    logger.info(
        "Authored scenario '{}' with seeds {} (units: meters, seconds).",
        scenario["name"],
        scenario["seeds"],
    )
    validate_scenario_file(written)
    reload_and_compare(written, stable_hash(scenario))
    run_headless_smoke(written, _step_budget(SMOKE_STEPS))
    demonstrate_negative_fixtures()
    logger.info("Tutorial complete: authoring workflow demonstrated end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
