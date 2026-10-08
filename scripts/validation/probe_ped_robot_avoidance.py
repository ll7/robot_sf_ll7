"""Diagnostic onset sensitivity against a stationary robot on dev seeds 1001-1030.

This two-force fixture is not a benchmark or behaviour-gate receipt.
"""

import json

import numpy as np
from pysocialforce.config import DesiredForceConfig, SceneConfig
from pysocialforce.forces import DesiredForce
from pysocialforce.scene import PedState

from robot_sf.ped_npc.ped_robot_force import (
    PedRobotForce,
    PedRobotForceConfig,
    PedRobotForceV2Config,
)


def run_probe() -> list[dict]:
    """Return passing/clearance diagnostics for legacy and four new onsets.

    Returns:
        One summary per cutoff for thirty fixed development seeds.
    """
    results = []
    for edge in [None, 1.3, 1.65, 2.0, 2.15]:
        clearances = []
        onsets = []
        passed = 0
        overlaps = 0
        for seed in range(1001, 1031):
            rng = np.random.default_rng(seed)
            y = rng.uniform(-0.3, 0.3)
            peds = PedState(np.array([[-5.0, y, 1.3, 0.0, 5.0, y, 0.5]]), [[0]], SceneConfig())
            peds.assign_desired_speeds(np.array([1.3]))
            config = (
                PedRobotForceConfig() if edge is None else PedRobotForceV2Config(edge_onset=edge)
            )
            robot = PedRobotForce(config, peds, lambda: (0.0, 0.0))
            desired = DesiredForce(DesiredForceConfig(), peds)
            min_distance = 100.0
            onset_x = None
            for _ in range(200):
                force = robot()
                if onset_x is None and np.linalg.norm(force) > 0:
                    onset_x = float(peds.pos()[0, 0])
                peds.step(desired() + force)
                min_distance = min(min_distance, float(np.linalg.norm(peds.pos()[0])))
                if peds.pos()[0, 0] > 3:
                    passed += 1
                    break
            clearances.append(min_distance - 1.35)
            onsets.append(onset_x)
            overlaps += min_distance < 1.35
        results.append(
            {
                "edge_onset": edge,
                "cutoff": 3.35 if edge is None else 1.35 + edge,
                "passed": passed,
                "contacts": overlaps,
                "min_clearance": min(clearances),
                "median_clearance": float(np.median(clearances)),
                "mean_onset_x": float(np.mean(onsets)),
            }
        )
    return results


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
