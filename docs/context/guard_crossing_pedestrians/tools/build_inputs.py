"""Build the 4-cell x 2-route-condition scenario matrix for the guard crossing study.

traverse = the release 0.0.8 cell exactly as authored (its pedestrian routes already
lead through the restriction). outside = the same map, robot start/goal, pedestrian
count, speed model and start-state distribution, with routes that never enter the
restriction. Run from the repository root.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

PACKET = Path("docs/context/guard_crossing_pedestrians")
RELEASE_MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
CELLS = [
    "classic_doorway_medium",
    "classic_doorway_high",
    "classic_bottleneck_medium",
    "classic_bottleneck_high",
]
# Doorway: the authored crowd route (6,22)->(22,13.9)->(38,6) passes the 3.3 m door in
# the x=21..23 wall. The outside route keeps the same spawn zone and total length
# (35.73 m, so the same density yields the same pedestrian count), approaches to 2.5 m
# of the wall face and turns back on the robot-goal side without entering the door.
DOORWAY_TRAVERSE_LENGTH = math.dist((6, 22), (22, 13.9)) + math.dist((22, 13.9), (38, 6))
_v = [(6.0, 22.0), (18.5, 15.8), (4.0, 12.0)]
_used = math.dist(_v[0], _v[1]) + math.dist(_v[1], _v[2])
DOORWAY_OUTSIDE_ROUTE = [*_v, (4.0, round(12.0 - (DOORWAY_TRAVERSE_LENGTH - _used), 3))]
# Bottleneck: the authored single pedestrians walk from (x,6) to (x,34) through the
# 4.3 m x 16 m corridor. Outside keeps the same start and default speed and walks a
# lap in the lower room, at least 3 m below the corridor mouth (y=12.08).
BOTTLENECK_LAP = [(15.5, 3.0), (24.5, 3.0), (15.5, 3.0)]


def load_release_cells() -> dict[str, dict]:
    from robot_sf.training.scenario_loader import load_scenarios  # noqa: PLC0415

    out = {}
    for s in load_scenarios(RELEASE_MATRIX):
        name = s.get("name")
        if name in CELLS:
            s = copy.deepcopy(dict(s))
            s["map_file"] = str(Path(s["map_file"]).as_posix()).replace("../../", "")
            s.pop("map_id", None)
            s.pop("seeds", None)
            out[name] = s
    assert sorted(out) == sorted(CELLS), out.keys()
    return out


def single_peds(map_file: str):
    from robot_sf.nav.svg_map_parser import convert_map  # noqa: PLC0415

    return convert_map(map_file).single_pedestrians


def main() -> None:
    cells = load_release_cells()
    scenarios = []
    for name in CELLS:
        base = cells[name]
        for cond in ("traverse", "outside"):
            s = copy.deepcopy(base)
            s["name"] = f"{name}__{cond}"
            md = s.setdefault("metadata", {})
            md["xped_base_cell"] = name
            md["xped_route_condition"] = cond
            if cond == "outside":
                if "doorway" in name:
                    s["route_overrides_file"] = str(PACKET / "inputs/doorway_outside_routes.yaml")
                else:
                    s["single_pedestrians"] = [
                        {
                            "id": p.id,
                            "goal": None,
                            "trajectory": [[float(p.start[0]), 3.0], *map(list, BOTTLENECK_LAP)],
                        }
                        for p in single_peds(s["map_file"])
                    ]
            scenarios.append(s)
    (PACKET / "inputs/doorway_outside_routes.yaml").write_text(
        yaml.safe_dump(
            {
                "ped_routes": [
                    {"spawn_id": 0, "goal_id": 99, "waypoints": [list(p) for p in DOORWAY_OUTSIDE_ROUTE]}
                ]
            },
            sort_keys=False,
        )
    )
    (PACKET / "inputs/scenarios.yaml").write_text(
        yaml.safe_dump({"scenarios": scenarios}, sort_keys=False)
    )
    print("doorway traverse length", round(DOORWAY_TRAVERSE_LENGTH, 3), "outside route", DOORWAY_OUTSIDE_ROUTE)


if __name__ == "__main__":
    main()
