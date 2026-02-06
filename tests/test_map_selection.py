"""Tests for deterministic map selection via map_id."""

from __future__ import annotations

from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool


def _make_map(label: str, x0: float) -> MapDefinition:
    width = 4.0
    height = 4.0
    spawn_zone = ((x0, 0.5), (x0 + 0.5, 0.5), (x0 + 0.5, 1.0))
    goal_zone = ((x0 + 2.0, 2.5), (x0 + 2.5, 2.5), (x0 + 2.5, 3.0))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(x0 + 0.75, 0.75), (x0 + 2.25, 2.75)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def test_map_id_selects_map() -> None:
    """Ensure map_id pins a map so scenario runs are deterministic and reproducible."""
    map_a = _make_map("a", 0.0)
    map_b = _make_map("b", 1.0)
    pool = MapDefinitionPool(map_defs={"a": map_a, "b": map_b})
    cfg = EnvSettings(map_pool=pool)
    cfg.map_id = "b"
    cfg.backend = "dummy"

    env = BaseEnv(env_config=cfg)
    try:
        assert env.map_def is map_b
    finally:
        env.exit()
