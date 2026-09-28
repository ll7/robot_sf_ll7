#!/usr/bin/env python3
"""Diagnostic-only VV-5 force mutants; never edits historical assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_sf.benchmark.metrics import robot_force_reductions
from robot_sf.ped_npc.adversial_ped_force import AdversarialPedForceConfig
from robot_sf.ped_npc.ped_robot_force import PedRobotForceConfig
from robot_sf.planner.socnav import (
    SOCIAL_FORCE_PLANNER_LEGACY_V1,
    SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)
from robot_sf.sim import simulator as simulator_module

ROOT = Path(__file__).resolve().parents[2]
GRID_EXTENT = 16.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_force(version: str, resolution: float) -> list[float]:
    cells = round(GRID_EXTENT / resolution)
    origin = -GRID_EXTENT / 2
    coords = origin + (np.arange(cells) + 0.5) * resolution
    _xs, ys = np.meshgrid(coords, coords)
    grid = np.zeros((3, cells, cells), dtype=np.float32)
    grid[0] = (ys >= 3.0).astype(np.float32)
    observation = {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([1.0], dtype=np.float32),
        },
        "goal": {
            "current": np.array([20.0, 0.0], dtype=np.float32),
            "next": np.array([21.0, 1.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=np.float32),
            "velocities": np.zeros((0, 2), dtype=np.float32),
            "count": np.array([0.0], dtype=np.float32),
        },
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.array([origin, origin], dtype=np.float32),
        "occupancy_grid_meta_resolution": np.array([resolution], dtype=np.float32),
        "occupancy_grid_meta_size": np.array([GRID_EXTENT, GRID_EXTENT], dtype=np.float32),
        "occupancy_grid_meta_use_ego_frame": np.array([1.0], dtype=np.float32),
        "occupancy_grid_meta_channel_indices": np.array([0, 1, 2], dtype=np.float32),
    }
    adapter = SocialForcePlannerAdapter(SocNavPlannerConfig(social_force_planner_version=version))
    force = adapter._compute_obstacle_force(
        observation, np.zeros(2), 0.0, np.zeros(2), observation["robot"]
    )
    return [float(value) for value in force]


def _wall_case() -> dict:
    def measure(version: str) -> dict:
        coarse = np.asarray(_wall_force(version, 0.2))
        fine = np.asarray(_wall_force(version, 0.1))
        ratio = float(np.linalg.norm(fine - coarse) / np.linalg.norm(fine))
        return {
            "version": version,
            "coarse_force_xy": coarse.tolist(),
            "fine_force_xy": fine.tolist(),
            "relative_change": ratio,
            "invariance_pass": bool(np.linalg.norm(fine) > 0.05 and ratio <= 0.25),
        }

    clean = measure(SOCIAL_FORCE_PLANNER_RESOLUTION_INDEPENDENT_V2)
    mutant = measure(SOCIAL_FORCE_PLANNER_LEGACY_V1)
    return {
        "fault": "per_cell_wall_force_sum",
        "fixture": "flat wall lower face y=3 m; robot radius 1 m; 0.2 and 0.1 m ego grids",
        "check": "production #9724 obstacle-force path plus <=25% resolution-invariance assertion",
        "clean": clean,
        "mutant": mutant,
        "detected": clean["invariance_pass"] and not mutant["invariance_pass"],
        "boundary": "Component-level force check, not a nominal episode or release admission.",
    }


class _Peds:
    def __init__(self) -> None:
        self.agent_radius = 0.4
        self._position = np.array([[1.8, 0.0]], dtype=float)

    def pos(self) -> np.ndarray:
        return self._position

    def size(self) -> int:
        return 1


def _pedestrian_case() -> dict:
    peds = _Peds()
    robot = SimpleNamespace(pos=(0.0, 0.0), config=SimpleNamespace(radius=1.0))
    original = simulator_module.pysf_make_forces
    simulator_module.pysf_make_forces = lambda _sim, _config: []
    try:
        variants = {}
        for label, active in (("clean", True), ("mutant", False)):
            forces = simulator_module._make_ped_forces(
                sim=SimpleNamespace(peds=peds),
                config=SimpleNamespace(),
                robots=[robot],
                peds_have_obstacle_forces=True,
                prf_config=PedRobotForceConfig(
                    is_active=active,
                    robot_radius=1.0,
                    activation_threshold=2.0,
                    force_multiplier=10.0,
                ),
                apf_config=AdversarialPedForceConfig(is_active=False),
            )
            components = [
                force
                for force in forces
                if getattr(force, "component_type", None) == "pedestrian_robot"
            ]
            force_xy = sum(
                (np.asarray(component(), dtype=float)[0] for component in components), np.zeros(2)
            )
            reduced = robot_force_reductions(force_xy.reshape(1, 1, 2), dt=0.1, reference=1.0)
            variants[label] = {
                "prf_active": active,
                "component_ids": [component.component_id for component in components],
                "force_xy": force_xy.tolist(),
                "impulse_total": reduced["robot_force_impulse_total"],
                "metrics_validator_pass": True,
            }
    finally:
        simulator_module.pysf_make_forces = original
    return {
        "fault": "pedestrian_robot_force_disabled",
        "fixture": "one pedestrian 1.8 m from stationary radius-1.0-m robot; one pre-integration force sample",
        "check": "production _make_ped_forces and robot_force_reductions; no explicit expected component roster gate",
        "clean": variants["clean"],
        "mutant": variants["mutant"],
        "physical_difference_observed": variants["clean"]["impulse_total"] > 0
        and variants["mutant"]["impulse_total"] == 0,
        "detected": variants["clean"]["metrics_validator_pass"]
        and not variants["mutant"]["metrics_validator_pass"],
        "boundary": "Factory wiring and one force evaluation, not a pedestrian trajectory/contact trial. Zero force reduces to valid zero metrics, so this production metric path misses the disabled component.",
        "proposed_check": "Bind expected active pedestrian-robot component roster and declared PRF configuration to pre-integration force capture; reject missing component under active scenario contract. Then confirm with a stopped-robot multi-seed trace.",
    }


def main() -> None:
    """Run the selected bounded diagnostic and write a checksummed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [_wall_case(), _pedestrian_case()]
    paths = [
        Path(__file__),
        ROOT / "robot_sf/planner/socnav_social_force.py",
        ROOT / "robot_sf/planner/socnav_base.py",
        ROOT / "robot_sf/sim/simulator.py",
        ROOT / "robot_sf/ped_npc/ped_robot_force.py",
        ROOT / "robot_sf/benchmark/metrics.py",
    ]
    report = {
        "schema_version": "issue_9735_force_faults_diagnostic.v1",
        "evidence_tier": "diagnostic_fixture_only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha(path) for path in paths if path != Path(__file__)
        },
        "fixture_script_sha256": _sha(Path(__file__)),
        "cases": cases,
        "sensitivity": {
            "detected": sum(bool(case["detected"]) for case in cases),
            "injected": len(cases),
        },
        "release_admitted": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "sha256": _sha(report_path),
                "sensitivity": report["sensitivity"],
                "cases": cases,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
