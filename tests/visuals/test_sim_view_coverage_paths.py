"""Coverage-focused tests for SimulationView helper branches."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from robot_sf.common.types import Rect
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy_grid import GridChannel
from robot_sf.render import sim_view as sim_view_mod
from robot_sf.render.sim_view import SimulationView, VisualizableAction, VisualizableSimState


def _map_def_with_routes_and_obstacles() -> MapDefinition:
    """Build a compact map that exercises routes, zones, and obstacle drawing."""
    width = 5.0
    height = 5.0
    spawn_zone: Rect = ((0.2, 0.2), (1.2, 0.2), (1.2, 1.2))
    goal_zone: Rect = ((3.6, 3.6), (4.6, 3.6), (4.6, 4.6))
    bounds = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.6, 0.6), (2.5, 2.5), (4.0, 4.0)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    obstacle = Obstacle([(1.6, 1.6), (2.4, 1.6), (2.4, 2.4), (1.6, 2.4)])
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[obstacle],
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


def _make_state(*, timestep: int = 3) -> VisualizableSimState:
    """Create a state that exercises robot, ped, and ego-ped render paths."""
    robot_action = VisualizableAction(
        pose=((1.0, 1.0), 0.2),
        action=(0.5, 0.1),
        goal=(4.0, 4.0),
    )
    ego_ped_action = VisualizableAction(
        pose=((1.2, 1.1), -0.3),
        action=(0.3, -0.1),
        goal=(1.8, 1.4),
    )
    return VisualizableSimState(
        timestep=timestep,
        robot_action=robot_action,
        robot_pose=((1.0, 1.0), 0.2),
        pedestrian_positions=np.array([[1.2, 1.3], [2.0, 1.8]], dtype=np.float64),
        ray_vecs=np.array([[[1.0, 1.0], [1.5, 1.5]]], dtype=np.float64),
        ped_actions=np.array(
            [
                [[1.2, 1.3], [1.35, 1.45]],
                [[2.0, 1.8], [2.2, 1.9]],
            ],
            dtype=np.float64,
        ),
        ego_ped_pose=((1.4, 1.0), -0.3),
        ego_ped_ray_vecs=np.array([[[1.4, 1.0], [1.8, 1.0]]], dtype=np.float64),
        ego_ped_action=ego_ped_action,
        planned_path=[(1.0, 1.0), (2.0, 2.0), (3.0, 2.5)],
        time_per_step_in_secs=0.1,
    )


class _DummyGrid:
    """Minimal occupancy-grid shim used by simulation view tests."""

    def __init__(self, *, use_ego_frame: bool, data: np.ndarray) -> None:
        self.config = SimpleNamespace(
            resolution=0.5,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
            use_ego_frame=use_ego_frame,
        )
        self._grid_origin = (0.1, 0.2)
        self._data = data

    def to_observation(self) -> np.ndarray:
        """Return a synthetic occupancy observation tensor."""
        return self._data


def _valid_grid_data() -> np.ndarray:
    """Build a tiny 2-channel occupancy grid with occupied cells."""
    return np.array(
        [
            [[0.0, 0.8], [0.0, 0.0]],
            [[0.9, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )


def _write_sprite(path) -> None:
    """Create a tiny sprite image for sprite rendering branches."""
    pygame.init()
    surface = pygame.Surface((6, 6), pygame.SRCALPHA)
    surface.fill((200, 80, 50, 255))
    pygame.image.save(surface, str(path))


def test_sim_view_render_pipeline_and_helpers(monkeypatch) -> None:
    """Render a full frame and call helper methods to cover UI/drawing branches."""
    view = SimulationView(
        width=120,
        height=90,
        scaling=20,
        map_def=_map_def_with_routes_and_obstacles(),
        record_video=False,
        display_text=True,
        display_help=True,
        ped_velocity_scale=1.5,
        show_occupancy_grid=True,
        show_telemetry_panel=True,
    )
    view.occupancy_grid = _DummyGrid(use_ego_frame=False, data=_valid_grid_data())
    view.grid_channel_visibility = {0: False, 1: True}
    view.telemetry_session = SimpleNamespace(render_surface=lambda: pygame.Surface((16, 10)))
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    state = _make_state()
    view.render(state, target_fps=30)

    # Cover explicit helper calls that are not always triggered via render()
    view._draw_spawn_zones()
    view._draw_goal_zones()
    view._draw_pedestrian_routes()
    view._draw_robot_routes()
    view._draw_coordinates(4, 5)
    view._add_minimal_hint()
    view.toggle_grid_channel_visibility(1)
    assert view.grid_channel_visibility[1] is False
    assert view.redraw_needed is True


def test_sim_view_occupancy_grid_error_and_rotation_paths(monkeypatch) -> None:
    """Exercise occupancy-grid render branches: runtime error, empty, and ego rotation."""
    view = SimulationView(
        width=120,
        height=90,
        scaling=25,
        map_def=_map_def_with_routes_and_obstacles(),
        show_occupancy_grid=True,
        record_video=False,
    )
    pose = ((1.0, 1.0), 0.5)

    # RuntimeError branch
    view.occupancy_grid = SimpleNamespace(
        config=SimpleNamespace(
            resolution=0.5,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS],
            use_ego_frame=False,
        ),
        to_observation=lambda: (_ for _ in ()).throw(RuntimeError("grid not ready")),
    )
    view._render_occupancy_grid(pose)

    # Empty data branch
    view.occupancy_grid = _DummyGrid(
        use_ego_frame=False,
        data=np.array([], dtype=np.float32).reshape((0, 0, 0)),
    )
    view._render_occupancy_grid(pose)

    # Ego-frame rotation branch + slow-path logging branch (>10ms)
    view.occupancy_grid = _DummyGrid(use_ego_frame=True, data=_valid_grid_data())
    ticks = iter([0, 20])
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: next(ticks))
    view._render_occupancy_grid(pose)


def test_sim_view_key_controls_and_event_loops(monkeypatch) -> None:
    """Cover keyboard control mappings and event queue processing paths."""
    view = SimulationView(record_video=False, width=80, height=60, scaling=10)

    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_CTRL)
    view._handle_keydown(SimpleNamespace(key=pygame.K_PLUS))
    view._handle_keydown(SimpleNamespace(key=pygame.K_MINUS))

    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_ALT)
    view._handle_keydown(SimpleNamespace(key=pygame.K_LEFT))
    view._handle_keydown(SimpleNamespace(key=pygame.K_RIGHT))
    view._handle_keydown(SimpleNamespace(key=pygame.K_UP))
    view._handle_keydown(SimpleNamespace(key=pygame.K_DOWN))

    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)
    for key in (
        pygame.K_r,
        pygame.K_f,
        pygame.K_p,
        pygame.K_h,
        pygame.K_q,
        pygame.K_t,
        pygame.K_o,
    ):
        view._handle_keydown(SimpleNamespace(key=key))

    assert view.focus_on_robot is False
    assert view.focus_on_ego_ped is True
    assert view.display_help is True
    assert view.display_text is True
    assert view.show_observation_space is True
    assert view.display_robot_info in (0, 1, 2)
    assert view.redraw_needed is True

    # _process_events covers QUIT/VIDEORESIZE/KEYDOWN dispatch.
    resize_event = pygame.event.Event(pygame.VIDEORESIZE, {"w": 140, "h": 100})
    key_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r})
    quit_event = pygame.event.Event(pygame.QUIT, {})
    monkeypatch.setattr(pygame.event, "get", lambda: [resize_event, key_event, quit_event])
    monkeypatch.setattr(sim_view_mod, "MOVIEPY_AVAILABLE", False)
    view.record_video = True
    view.frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    view._process_events()
    assert view.size_changed is True
    assert view.width == 140
    assert view.height == 100
    assert view.is_exit_requested is True

    # _process_event_queue while loop and tick branch.
    view.is_exit_requested = False
    queue_events = [pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_t})]
    monkeypatch.setattr(pygame.event, "get", lambda: queue_events)
    monkeypatch.setattr(
        view, "_handle_keydown", lambda _e: setattr(view, "is_exit_requested", True)
    )
    view._process_event_queue()


def test_sim_view_sprite_helpers_and_camera(monkeypatch, tmp_path) -> None:
    """Cover sprite lookup/load failure and camera/resize helpers."""
    sprite_path = tmp_path / "ego_sprite.png"
    _write_sprite(sprite_path)
    view = SimulationView(
        record_video=True,
        map_def=_map_def_with_routes_and_obstacles(),
        ego_ped_render_mode="sprite",
        ego_ped_sprite_path=str(sprite_path),
    )

    # Successful sprite draw path for ego ped.
    draw_calls: list[float | None] = []

    def _spy_draw_sprite(_sprite, _center, _radius, theta=None) -> None:
        draw_calls.append(theta)

    monkeypatch.setattr(view, "_draw_sprite", _spy_draw_sprite)
    view._draw_ego_ped(((1.0, 1.0), 0.25))
    assert len(draw_calls) == 1
    assert draw_calls[0] is not None
    assert abs(float(draw_calls[0]) - (0.25 + (np.pi / 2.0))) < 1e-9

    # Invalid entity path
    with pytest.raises(ValueError):
        view._sprite_path_for_entity("unknown")

    # Sprite load failure branch
    view.robot_sprite_path = "/tmp/load_failure_robot.png"
    monkeypatch.setattr(sim_view_mod.os.path, "exists", lambda _p: True)

    load_calls: list[str] = []

    def _raise_load(path: str):
        load_calls.append(path)
        raise pygame.error("load failed")

    monkeypatch.setattr(pygame.image, "load", _raise_load)
    assert view._get_entity_sprite("robot") is None
    assert load_calls  # second call should hit cache without reloading
    assert view._get_entity_sprite("robot") is None

    # Camera movement + resize helper branch
    view.focus_on_robot = True
    view.focus_on_ego_ped = True
    view._move_camera(_make_state())
    view.width = 140
    view.height = 100
    view._resize_window()
    view.clear()


def test_robot_action_uses_speed_times_horizon_without_extra_scaling(monkeypatch) -> None:
    """Robot speed vector should use distance = speed * horizon (meters), then one screen scaling."""
    view = SimulationView(
        record_video=True, scaling=10, map_def=_map_def_with_routes_and_obstacles()
    )
    view.action_horizon_s = 1.0
    view.direction_line_length_m = 0.5
    view.speed_line_scale = 1.0

    line_calls: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    original_line = pygame.draw.line

    def _spy_line(_screen, _color, start, end, width=1):
        line_calls.append((start, end, width))
        return original_line(_screen, _color, start, end, width)

    monkeypatch.setattr(pygame.draw, "line", _spy_line)
    action = VisualizableAction(
        pose=((1.0, 1.0), 0.0), action=np.array([2.0, 0.0]), goal=(0.0, 0.0)
    )
    view._augment_action(action, (255, 0, 0))

    # Last line is the speed line. Start at (10,10), end at (30,10) for 2 m/s * 1 s horizon.
    assert line_calls
    speed_start, speed_end, speed_width = line_calls[-1]
    assert speed_width == view.speed_line_width_px
    assert abs(speed_start[0] - 10.0) < 1e-6
    assert abs(speed_start[1] - 10.0) < 1e-6
    assert abs(speed_end[0] - 30.0) < 1e-6
    assert abs(speed_end[1] - 10.0) < 1e-6


def test_robot_measured_kinematics_draws_speed_and_acceleration(monkeypatch) -> None:
    """Measured robot overlays should draw blue speed and yellow acceleration vectors."""
    view = SimulationView(
        record_video=True, scaling=10, map_def=_map_def_with_routes_and_obstacles()
    )
    view.action_horizon_s = 1.0
    view.acceleration_horizon_s = 1.0
    view.speed_line_scale = 1.0
    view.acceleration_line_scale = 1.0
    view._prev_robot_pose_xy = (0.0, 0.0)
    view._prev_robot_speed_vec = np.array([0.0, 0.0], dtype=float)

    line_calls: list[
        tuple[tuple[int, int, int], tuple[float, float], tuple[float, float], int]
    ] = []

    def _spy_line(_screen, color, start, end, width=1):
        line_calls.append((color, start, end, width))

    monkeypatch.setattr(pygame.draw, "line", _spy_line)
    state = VisualizableSimState(
        timestep=1,
        robot_action=None,
        robot_pose=((1.0, 0.0), 0.0),
        pedestrian_positions=np.empty((0, 2)),
        ray_vecs=np.empty((0, 2, 2)),
        ped_actions=np.empty((0, 2, 2)),
        time_per_step_in_secs=1.0,
    )
    view._augment_robot_measured_kinematics(state)

    # Expect: direction line (blue), speed line (blue), acceleration line (yellow).
    assert len(line_calls) == 3
    assert line_calls[0][0] == sim_view_mod.ROBOT_ACTION_COLOR
    assert line_calls[1][0] == sim_view_mod.ROBOT_ACTION_COLOR
    assert line_calls[2][0] == sim_view_mod.ROBOT_ACCEL_COLOR
    # Speed line should reach +1m at scaling 10 => +10 px from x=10 to x=20.
    _, s_start, s_end, s_width = line_calls[1]
    assert s_width == view.speed_line_width_px
    assert abs(s_start[0] - 10.0) < 1e-6
    assert abs(s_end[0] - 20.0) < 1e-6


def test_robot_measured_kinematics_resets_on_episode_start(monkeypatch) -> None:
    """Episode start (timestep=0) should clear prior kinematic history spikes."""
    view = SimulationView(
        record_video=True, scaling=10, map_def=_map_def_with_routes_and_obstacles()
    )
    view._prev_robot_pose_xy = (50.0, 50.0)
    view._prev_robot_speed_vec = np.array([100.0, 0.0], dtype=float)

    line_calls: list[tuple[tuple[float, float], tuple[float, float], int]] = []

    def _spy_line(_screen, _color, start, end, width=1):
        line_calls.append((start, end, width))

    monkeypatch.setattr(pygame.draw, "line", _spy_line)
    state = VisualizableSimState(
        timestep=0,
        robot_action=None,
        robot_pose=((1.0, 0.0), 0.0),
        pedestrian_positions=np.empty((0, 2)),
        ray_vecs=np.empty((0, 2, 2)),
        ped_actions=np.empty((0, 2, 2)),
        time_per_step_in_secs=1.0,
    )
    view._augment_robot_measured_kinematics(state)

    # 2nd line is speed vector; it should collapse to zero length at episode start.
    assert len(line_calls) == 3
    speed_start, speed_end, _ = line_calls[1]
    assert speed_start == speed_end
    assert view._prev_robot_pose_xy == (1.0, 0.0)
    assert np.array_equal(view._prev_robot_speed_vec, np.array([0.0, 0.0], dtype=float))


def test_robot_rotation_action_draws_clamped_directional_arc(monkeypatch) -> None:
    """Rotation action should render an orange arc with turn direction and pi clamp."""
    view = SimulationView(
        record_video=True, scaling=10, map_def=_map_def_with_routes_and_obstacles()
    )
    arc_calls: list[tuple[list[tuple[float, float]], int]] = []

    def _spy_lines(_screen, _color, _closed, points, width=1):
        arc_calls.append((list(points), int(width)))

    monkeypatch.setattr(pygame.draw, "lines", _spy_lines)
    pose = ((1.0, 1.0), 0.2)
    # Command exceeds clamp; expected span is pi.
    action = VisualizableAction(pose=pose, action=np.array([0.0, 10.0]), goal=(0.0, 0.0))
    view._augment_robot_rotation_action(pose, action)
    assert arc_calls
    points, width = arc_calls[-1]
    assert width == view.rotation_arc_width_px
    assert len(points) >= 2
    center = np.array(view._scale_tuple(pose[0]), dtype=float)
    first = np.array(points[0], dtype=float) - center
    last = np.array(points[-1], dtype=float) - center
    start_angle = float(np.arctan2(first[1], first[0]))
    end_angle = float(np.arctan2(last[1], last[0]))
    # First point anchored at robot heading.
    assert abs(start_angle - pose[1]) < 1e-6
    # Span is pi (mod 2pi) due to clamp.
    span = (end_angle - start_angle + 2.0 * np.pi) % (2.0 * np.pi)
    assert abs(span - np.pi) < 1e-6


def test_ped_action_direction_line_persists_when_speed_zero(monkeypatch) -> None:
    """Pedestrians with zero current speed should still show a thin direction line from last movement."""
    view = SimulationView(
        record_video=True, scaling=10, map_def=_map_def_with_routes_and_obstacles()
    )
    view.action_horizon_s = 1.0
    view.direction_line_length_m = 0.5
    view.speed_line_scale = 1.0

    draw_calls: list[tuple[tuple[float, float], tuple[float, float], int]] = []

    def _spy_line(_screen, _color, start, end, width=1):
        draw_calls.append((start, end, width))

    monkeypatch.setattr(pygame.draw, "line", _spy_line)
    # First frame: moving upward (stores last direction).
    view._augment_ped_actions(np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=float))
    # Second frame: zero speed -> should reuse stored direction for thin line.
    view._augment_ped_actions(np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=float))

    # Two lines per call; for the second call, first line is the direction line.
    assert len(draw_calls) >= 4
    second_call_direction = draw_calls[2]
    start, end, width = second_call_direction
    assert width == view.direction_line_width_px
    assert abs(start[0] - end[0]) < 1e-6  # keeps upward direction (x unchanged)
    assert end[1] > start[1]  # y increases in world-up direction after scaling


def test_sim_view_exit_and_telemetry_paths(monkeypatch) -> None:
    """Exercise exit handling and telemetry panel layout/fallback branches."""
    view = SimulationView(record_video=False, width=120, height=90)

    # _handle_exit without and with abortion.
    view.is_abortion_requested = False
    view._handle_exit()

    exit_called = {"value": False}
    monkeypatch.setattr(sim_view_mod.sys, "exit", lambda: exit_called.__setitem__("value", True))
    view.is_abortion_requested = True
    view._handle_exit()
    assert exit_called["value"] is True

    # Telemetry panel: missing method, None surface, and both layouts.
    view.show_telemetry_panel = True
    view.telemetry_session = object()
    view._render_telemetry_panel()

    view.telemetry_session = SimpleNamespace(render_surface=lambda: None)
    view._render_telemetry_panel()

    pane_surface = pygame.Surface((20, 12))
    view.telemetry_session = SimpleNamespace(render_surface=lambda: pane_surface)
    view.telemetry_layout = "horizontal_split"
    view._render_telemetry_panel()
    view.telemetry_layout = "vertical_split"
    view._render_telemetry_panel()

    # Lidar augmentation edge branches (None, empty, non-sized object)
    view._augment_lidar(None)
    view._augment_lidar(np.empty((0, 2, 2), dtype=np.float32))
    view._augment_lidar(5)  # type: ignore[arg-type]
