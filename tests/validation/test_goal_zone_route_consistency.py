"""Goal-zone/route consistency for the goal_zone_entry_v1 matrix (issue #9762).

Every robot route bound to a goal zone must end inside that zone: under
``goal_zone_entry_v1`` success is polygon entry (``covers`` on the robot
centre), so a final waypoint outside the zone scores a timeout even when the
robot parks exactly on its route end.  The check collects all offenders
before failing so one repair pass can fix the whole matrix.
"""

from pathlib import Path

import yaml
from shapely.geometry import Point

from robot_sf.nav.navigation import _goal_zone_polygon
from robot_sf.nav.svg_map_parser import _load_single_svg

REPO_ROOT = Path(__file__).parents[2]
MATRIX = REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"


def _matrix_scenario_files() -> list[Path]:
    """Expand the matrix ``includes`` chain to per-scenario manifest files."""
    with open(MATRIX, encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    matrix_dir = MATRIX.parent
    manifests: list[Path] = []
    queue = [(matrix_dir / include).resolve() for include in matrix.get("includes", [])]
    seen: set[Path] = set()
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        seen.add(path)
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        nested = data.get("includes", [])
        if nested:
            queue.extend((path.parent / item).resolve() for item in nested)
        elif data.get("scenarios"):
            manifests.append(path)
    return manifests


def _scenario_map_files() -> list[Path]:
    """Resolve every ``scenarios[].map_file`` entry to an SVG path."""
    svg_files: list[Path] = []
    for manifest in _matrix_scenario_files():
        with open(manifest, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for scenario in data.get("scenarios", []):
            map_file = scenario.get("map_file")
            if map_file:
                svg_files.append((manifest.parent / map_file).resolve())
    return sorted(set(svg_files))


def test_goal_zone_entry_routes_end_inside_goal_zone() -> None:
    """Fail closed listing every route whose final waypoint misses its zone."""
    offenders: list[str] = []
    checked = 0
    for svg_file in _scenario_map_files():
        definitions = _load_single_svg(svg_file, strict=False)
        for map_def in definitions.values():
            for route in map_def.robot_routes or []:
                if not route.waypoints or not route.goal_zone:
                    continue
                checked += 1
                final = route.waypoints[-1]
                try:
                    inside = bool(_goal_zone_polygon(route.goal_zone).covers(Point(final)))
                except Exception as exc:  # report as offender, not error
                    offenders.append(f"{svg_file.name}: zone unusable ({exc})")
                    continue
                if not inside:
                    offenders.append(
                        f"{svg_file.name}: final waypoint {tuple(final)} "
                        f"outside goal zone {tuple(route.goal_zone)}"
                    )
    assert checked > 0, "matrix resolved to zero goal-zone routes"
    assert not offenders, (
        f"{len(offenders)} route(s) end outside their goal zone "
        f"(robot parks on-route yet scores timeout under goal_zone_entry_v1):\n"
        + "\n".join(offenders)
    )
