"""Short integration probes in a disjoint namespace, never eligible as receipt episodes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from robot_sf.benchmark.receipt_contract import episode_inputs
from robot_sf.benchmark.receipt_stages import (
    atomic_json,
    compare_episode_observation,
    encode_row,
    stage_key,
)


def run_canary(derived: list[tuple], output: Path, source_sha: str, steps: int) -> int:  # noqa: C901
    """Reset/step selected real arm/slots and drive production writer/scorer/comparator.

    The default selects every arm/slot. A scheduler canary can select a handful with the
    existing --arms/--scenarios flags; its report records coverage explicitly.

    Returns:
        The validated stage result."""
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.benchmark.schema_validator import load_schema
    from scripts.validation import run_empty_world_sweep as sweep

    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=True)
    schema = load_schema(root / "robot_sf/benchmark/schemas/episode.schema.v1.json")
    rows, coverage = [], []

    def execute(cfg, arm, scenario, kin, label, reset_hook=None):
        scoped, kwargs = episode_inputs(cfg, arm, copy.deepcopy(scenario), kin, root)
        scoped["name"] = f"canary/{label}/{scoped['name']}"
        scoped.pop("id", None)
        scoped.pop("scenario_id", None)
        scoped["seeds"] = [1001]
        scoped["metadata"] = {"diagnostic_variant": "receipt-canary.v1"}
        scoped.setdefault("simulation_config", {})["max_episode_steps"] = steps
        kwargs["horizon"] = steps
        row = runner._run_map_episode(scoped, 1001, **kwargs, pair_reset_hook=reset_hook)
        path = output / "runs" / arm.key / "episodes.jsonl"
        atomic_json(
            path.parent / "receipt_execution.json",
            {
                "schema_version": "receipt-execution.v1",
                "source_sha": source_sha,
                "execution_key": stage_key(
                    "canary-execution", {"source_sha": source_sha, "namespace": "canary"}
                ),
                "run_id": str(output.resolve()),
            },
            immutable=True,
        )
        runner._write_validated(path, schema, row)
        # Consume persisted bytes through the same reader and outcome comparator.
        persisted = json.loads(path.read_text().splitlines()[-1])
        compare_episode_observation(persisted, persisted)
        rows.append(persisted)
        return row

    for suite, _, _, cfg in derived:
        for arm in cfg.planners:
            for kin in cfg.kinematics_matrix or ("differential_drive",):
                for scenario in _load_campaign_scenarios(cfg):
                    execute(cfg, arm, scenario, kin, f"{suite}/{arm.key}")
                    coverage.append([suite, arm.key, kin, scenario["name"]])

    # Exercise actual terminal states with deterministic post-reset physical fixtures.
    _, _, _, cfg = derived[0]
    arm = next((arm for arm in cfg.planners if arm.algo == "goal"), cfg.planners[0])
    scenario = _load_campaign_scenarios(cfg)[0]
    kin = (cfg.kinematics_matrix or ("differential_drive",))[0]

    def at_goal(env, _obs):
        sim = env.unwrapped.simulator
        nav = sim.robot_navs[0]
        goal = tuple(nav.waypoints[-1])
        nav.waypoint_id = len(nav.waypoints) - 1
        nav.pos = goal
        sim.robots[0].reset_state((goal, 0.0))
        return {"diagnostic_fixture": "goal"}

    def at_boundary(env, _obs):
        env.unwrapped.simulator.robots[0].reset_state(((0.0, 0.0), 0.0))
        return {"diagnostic_fixture": "collision"}

    finalization = {}
    for name, hook in (("goal", at_goal), ("collision", at_boundary), ("timeout", None)):
        row = execute(cfg, arm, scenario, kin, f"finalization/{name}", hook)
        outcome = sweep.classify_outcome(row)
        finalization[name] = outcome
        expected = "success" if name == "goal" else name
        if outcome != expected:
            raise ValueError(f"canary finalization {name} produced {outcome}")
    invalid = copy.deepcopy(scenario)
    invalid["map_file"] = str(output / "deliberately-absent-map.svg")
    try:
        execute(cfg, arm, invalid, kin, "invalid-config")
    except (ValueError, FileNotFoundError, RuntimeError):
        finalization["invalid_config"] = "rejected"
    else:
        raise ValueError("canary invalid configuration unexpectedly executed")
    report = {
        "schema_version": "receipt-canary.v1",
        "source_sha": source_sha,
        "namespace": "canary",
        "receipt_eligible": False,
        "coverage": coverage,
        "episodes": len(rows),
        "finalization": finalization,
        "comparison_key": stage_key("canary-comparison", [encode_row(row) for row in rows]),
    }
    atomic_json(output / "canary.json", report)
    print(
        json.dumps(
            {"canary_episodes": len(rows), "receipt_eligible": False, "finalization": finalization}
        )
    )
    return 0
