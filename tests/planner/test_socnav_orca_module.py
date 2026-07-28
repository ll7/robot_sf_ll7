"""Focused coverage for the extracted ORCA + HRVO planner-family module."""

import subprocess
import sys

import numpy as np
import pytest

from robot_sf.planner import socnav
from robot_sf.planner import socnav_orca as orca


def _observation(*, goal=(5.0, 0.0), heading=0.0, pedestrians=None) -> dict:
    """Build a compact nested SocNav observation for adapter unit tests.

    Mirrors the helper shape used by tests/test_socnav_planner_adapter.py so the
    ORCA heuristic path receives a well-formed robot/goal/pedestrian payload.
    """
    if pedestrians is None:
        positions = np.zeros((1, 2), dtype=np.float32)
        count = 0.0
    else:
        positions = np.asarray(pedestrians, dtype=np.float32).reshape(-1, 2)
        count = float(positions.shape[0])
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "speed": np.array([0.0, 0.0], dtype=np.float32),
            "radius": np.array([0.5], dtype=np.float32),
        },
        "goal": {
            "current": np.asarray(goal, dtype=np.float32),
            "next": np.array([0.0, 0.0], dtype=np.float32),
        },
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "radius": np.array([0.4], dtype=np.float32),
            "count": np.array([count], dtype=np.float32),
        },
        "map": {"size": np.array([10.0, 10.0], dtype=np.float32)},
        "sim": {"timestep": np.array([0.1], dtype=np.float32)},
    }


def _with_occupancy_grid(
    observation: dict,
    *,
    obstacle_cells: tuple[tuple[int, int], ...] = (),
) -> dict:
    """Attach the minimal ego-frame occupancy-grid contract used by SocNav planners."""
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, column in obstacle_cells:
        grid[0, row, column] = 1.0
        grid[3, row, column] = 1.0
    observation["occupancy_grid"] = grid
    observation["occupancy_grid_meta_origin"] = np.array([-2.0, -2.0], dtype=np.float32)
    observation["occupancy_grid_meta_resolution"] = np.array([1.0], dtype=np.float32)
    observation["occupancy_grid_meta_size"] = np.array([4.0, 4.0], dtype=np.float32)
    observation["occupancy_grid_meta_use_ego_frame"] = np.array([1.0], dtype=np.float32)
    observation["occupancy_grid_meta_channel_indices"] = np.array(
        [0.0, 1.0, 2.0, 3.0],
        dtype=np.float32,
    )
    return observation


def test_facade_wildcard_import_resolves_lazy_public_exports() -> None:
    """The facade stays lazy until wildcard import resolves the ORCA/HRVO exports."""
    expected = {"ORCAPlannerAdapter", "HRVOPlannerAdapter", "make_orca_policy", "make_hrvo_policy"}
    assert expected <= set(dir(socnav))
    assert expected <= set(socnav.__all__)
    assert expected <= socnav._ORCA_LAZY_EXPORTS
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from robot_sf.planner import socnav\n"
                "assert 'robot_sf.planner.socnav_orca' not in sys.modules\n"
                "from robot_sf.planner.socnav import *\n"
                "assert 'robot_sf.planner.socnav_orca' in sys.modules\n"
                "from robot_sf.planner import socnav_orca\n"
                "assert ORCAPlannerAdapter is socnav_orca.ORCAPlannerAdapter\n"
                "assert HRVOPlannerAdapter is socnav_orca.HRVOPlannerAdapter\n"
                "assert make_orca_policy is socnav_orca.make_orca_policy\n"
                "assert make_hrvo_policy is socnav_orca.make_hrvo_policy\n"
            ),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_resolution_caches_into_facade_globals(monkeypatch) -> None:
    """Accessing a lazy name binds it into the facade module globals after first resolution."""
    expected = {
        "ORCAPlannerAdapter": orca.ORCAPlannerAdapter,
        "HRVOPlannerAdapter": orca.HRVOPlannerAdapter,
        "make_orca_policy": orca.make_orca_policy,
        "make_hrvo_policy": orca.make_hrvo_policy,
    }
    for name, value in expected.items():
        monkeypatch.delitem(socnav.__dict__, name, raising=False)
        assert getattr(socnav, name) is value
        assert name in socnav.__dict__


def test_orca_adapter_importable_and_instantiable() -> None:
    """The ORCA adapter can be imported and instantiated from the extracted module."""
    adapter = orca.ORCAPlannerAdapter()
    assert isinstance(adapter, orca.SamplingPlannerAdapter)
    assert adapter.config is not None
    assert adapter.config.max_linear_speed == 3.0


def test_hrvo_adapter_is_orca_subclass() -> None:
    """HRVO is a pure-Python velocity-obstacle adapter subclassing ORCAPlannerAdapter."""
    adapter = orca.HRVOPlannerAdapter()
    assert isinstance(adapter, orca.ORCAPlannerAdapter)
    assert isinstance(adapter, orca.SamplingPlannerAdapter)


def test_factories_wrap_correct_adapter_types() -> None:
    """Factory functions wrap the correct adapter inside the policy."""
    assert isinstance(orca.make_orca_policy().adapter, orca.ORCAPlannerAdapter)
    assert isinstance(orca.make_hrvo_policy().adapter, orca.HRVOPlannerAdapter)


def test_orca_heuristic_fallback_produces_finite_action(monkeypatch) -> None:
    """Facade re-exported ORCA adapter falls back to the heuristic plan when rvo2 is unavailable.

    Patching the facade's ``rvo2`` handle propagates to the extracted module because ORCA reads
    the live ``_socnav.rvo2`` attribute at call time (mirroring the SA-CADRL ``_socnav.tf`` path).
    """
    monkeypatch.setattr(socnav, "rvo2", None)
    from robot_sf.planner.socnav import ORCAPlannerAdapter

    adapter = ORCAPlannerAdapter(allow_fallback=True)
    linear, angular = adapter.plan(_observation(goal=(2.0, 0.0)))
    assert np.isfinite(linear)
    assert np.isfinite(angular)
    assert linear >= 0.0


def test_orca_heuristic_fallback_slows_for_head_on_pedestrian(monkeypatch) -> None:
    """The ORCA heuristic path reduces speed for a blocking head-on pedestrian."""
    monkeypatch.setattr(socnav, "rvo2", None)
    adapter = orca.ORCAPlannerAdapter(allow_fallback=True)

    linear_free, _angular_free = adapter.plan(_observation(goal=(5.0, 0.0)))
    linear_blocked, angular_blocked = adapter.plan(
        _observation(goal=(5.0, 0.0), pedestrians=[(2.0, 0.0)])
    )
    assert linear_blocked < linear_free
    assert np.isfinite(angular_blocked)


def test_orca_missing_rvo2_fails_closed_without_explicit_fallback(monkeypatch) -> None:
    """The extracted adapter preserves the benchmark-ready optional-dependency contract."""
    monkeypatch.setattr(socnav, "rvo2", None)

    with pytest.raises(RuntimeError, match="rvo2 is required"):
        orca.ORCAPlannerAdapter().plan(_observation())


@pytest.mark.parametrize(
    "points, message",
    [
        ([1.0, 2.0, 3.0], "even number of coordinates"),
        (np.zeros((1, 2, 2)), r"convertible to an \(N, 2\) array"),
    ],
)
def test_bound_static_obstacle_points_reject_malformed_shapes(points, message) -> None:
    """Malformed bound geometry continues to fail before planner state is mutated."""
    adapter = orca.ORCAPlannerAdapter()

    with pytest.raises(ValueError, match=message):
        adapter.bind_static_obstacle_points(points, spacing=0.25)


def test_bound_static_obstacles_replace_grid_and_reset_cached_rvo2_state() -> None:
    """Exact bound geometry takes precedence and invalidates cached simulator state."""
    adapter = orca.ORCAPlannerAdapter()
    adapter._rvo2_sim = object()
    adapter._rvo2_signature = ("cached",)
    adapter._rvo2_robot_id = 7
    adapter._rvo2_ped_ids = [8]

    adapter.bind_static_obstacle_points([1.0, 0.0, 1.5, 0.0], spacing=0.5)

    centers, radii = adapter._extract_obstacles_from_grid(
        _with_occupancy_grid(_observation(), obstacle_cells=((0, 0),)),
        np.zeros(2, dtype=float),
        0.0,
    )
    assert centers.shape == (1, 2)
    assert radii.shape == (1,)
    assert centers[0, 0] > 0.0
    assert adapter._rvo2_sim is None
    assert adapter._rvo2_signature is None
    assert adapter._rvo2_robot_id is None
    assert adapter._rvo2_ped_ids == []

    adapter.bind_static_obstacle_points([], spacing=0.5)
    assert adapter._bound_static_obstacle_points.shape == (0, 2)
    assert adapter._bound_static_obstacle_spacing == 0.0


def test_ego_grid_obstacle_extraction_and_forward_probe_contract() -> None:
    """The extracted module preserves ego-grid conversion and blocked-corridor probing."""
    adapter = orca.ORCAPlannerAdapter()
    observation = _with_occupancy_grid(_observation(), obstacle_cells=((2, 3),))
    robot_position = np.zeros(2, dtype=float)
    goal_direction = np.array([1.0, 0.0], dtype=float)

    centers, radii = adapter._extract_obstacles_from_grid(
        observation,
        robot_position,
        0.0,
    )

    assert centers.shape == (1, 2)
    assert radii.shape == (1,)
    assert np.all(np.isfinite(centers))
    assert np.all(radii > 0.0)
    assert adapter._direct_path_blocked(
        robot_pos=robot_position,
        robot_heading=0.0,
        goal_direction_world=goal_direction,
        observation=observation,
    )
