"""Characterization baseline for the SimulationView render pipeline.

This is a pre-split lock (see issue #4965 / epic #4770): it golden-pins the
observable behavior of the *pure, surface-independent* helpers in
``SimulationView`` (``robot_sf/render/sim_view.py``) so the upcoming god-class
split refactor can prove behavior preservation by diffing against these tests.

Scope of the lock (no production code is modified here):

1. **Text-overlay content** -- ``_build_text_lines``, ``_get_display_info_lines``,
   ``_get_robot_info_lines``, ``_get_pedestrian_info_lines`` and
   ``_has_pedestrian_data`` for a fixed ``VisualizableSimState``.
2. **Frame-cadence logic** -- ``_effective_video_fps`` and
   ``_should_capture_recording_frame`` across a fixed set of ``target_fps``
   values (these helpers are pure given the internal counters, so the exact
   capture sequence is deterministic).
3. **Coordinate scaling** -- ``_scale_tuple`` and ``_timestep_text_pos`` for
   fixed inputs.

All assertions record the exact value produced by the current implementation on
``main``; the values are intentionally literal so a behavior change during the
split is surfaced as a concrete diff rather than a soft mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest

pygame = pytest.importorskip("pygame")  # skip the whole module if pygame is missing

from robot_sf.render.sim_view import (  # noqa: E402  (import after pygame guard)
    SimulationView,
    VisualizableAction,
    VisualizableSimState,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def view() -> SimulationView:
    """A deterministic, headless SimulationView with small dimensions.

    Constructed with the defaults that the helpers read (scaling/offset/width)
    so the recorded values are stable. Headless rendering is forced by the
    session-level ``SDL_VIDEODRIVER=dummy`` marker / environment.
    """
    return SimulationView(width=64, height=64, scaling=10.0)


def _state_with_robot(*, timestep: int = 7) -> VisualizableSimState:
    """Fixed state carrying a robot action/pose/goal but no ego pedestrian."""
    robot_action = VisualizableAction(
        pose=((1.0, 1.0), 0.2),
        action=(0.5, 0.1),
        goal=(3.0, 3.0),
    )
    return VisualizableSimState(
        timestep=timestep,
        robot_action=robot_action,
        robot_pose=((1.0, 1.0), 0.2),
        pedestrian_positions=np.zeros((0, 2)),
        ray_vecs=np.zeros((0, 2, 2)),
        ped_actions=np.zeros((0, 2, 2)),
        time_per_step_in_secs=0.1,
    )


def _state_with_pedestrian() -> VisualizableSimState:
    """Fixed state carrying both a robot and an ego pedestrian payload."""
    robot_action = VisualizableAction(
        pose=((1.0, 1.0), 0.2),
        action=(0.5, 0.1),
        goal=(3.0, 3.0),
    )
    ego_action = VisualizableAction(
        pose=((2.0, 2.0), 1.0),
        action=(1.0, 0.0),
        goal=(0.0, 0.0),
    )
    return VisualizableSimState(
        timestep=7,
        robot_action=robot_action,
        robot_pose=((1.0, 1.0), 0.2),
        pedestrian_positions=np.zeros((0, 2)),
        ray_vecs=np.zeros((0, 2, 2)),
        ped_actions=np.zeros((0, 2, 2)),
        ego_ped_pose=((2.0, 2.0), 1.0),
        ego_ped_action=ego_action,
        time_per_step_in_secs=0.1,
    )


# --------------------------------------------------------------------------- #
# 1. Text-overlay content
# --------------------------------------------------------------------------- #


class TestTextOverlayContent:
    """Pin exact text-overlay line lists produced by the current helpers."""

    def test_has_pedestrian_data_true(self, view: SimulationView) -> None:
        state = _state_with_pedestrian()
        assert view._has_pedestrian_data(state) is True

    def test_has_pedestrian_data_false_when_missing(self, view: SimulationView) -> None:
        state = _state_with_robot()
        assert view._has_pedestrian_data(state) is False

    def test_get_robot_info_lines_exact(self, view: SimulationView) -> None:
        state = _state_with_robot()
        assert view._get_robot_info_lines(state) == [
            "RobotPose: ((1.0, 1.0), 0.2)",
            "RobotAction: (0.5, 0.1)",
            "RobotGoal: (3.0, 3.0)",
        ]

    def test_get_robot_info_lines_empty_without_action(self, view: SimulationView) -> None:
        state = VisualizableSimState(
            timestep=1,
            robot_action=None,
            robot_pose=((0.0, 0.0), 0.0),
            pedestrian_positions=np.zeros((0, 2)),
            ray_vecs=np.zeros((0, 2, 2)),
            ped_actions=np.zeros((0, 2, 2)),
            time_per_step_in_secs=0.1,
        )
        assert view._get_robot_info_lines(state) == []

    def test_get_pedestrian_info_lines_exact(self, view: SimulationView) -> None:
        # DistanceRobot: euclid_dist((2.0,2.0),(1.0,1.0)) = sqrt(2) = 1.4142.. -> "1.41"
        state = _state_with_pedestrian()
        assert view._get_pedestrian_info_lines(state) == [
            "PedestrianPose: ((2.0, 2.0), 1.0)",
            "PedestrianAction: (1.0, 0.0)",
            "PedestrianGoal: (0.0, 0.0)",
            "DistanceRobot: 1.41",
        ]

    def test_get_pedestrian_info_lines_resets_mode_when_missing(self, view: SimulationView) -> None:
        """When pedestrian data is absent the helper disables the panel.

        ``_get_pedestrian_info_lines`` returns ``[]`` and mutates
        ``display_robot_info`` back to ``0`` so the overlay stops polling it.
        """
        state = _state_with_robot()
        view.display_robot_info = 2
        assert view._get_pedestrian_info_lines(state) == []
        assert view.display_robot_info == 0

    def test_get_display_info_lines_dispatch_mode0(self, view: SimulationView) -> None:
        state = _state_with_robot()
        view.display_robot_info = 0
        assert view._get_display_info_lines(state) == []

    def test_get_display_info_lines_dispatch_mode1(self, view: SimulationView) -> None:
        state = _state_with_robot()
        view.display_robot_info = 1
        assert view._get_display_info_lines(state) == [
            "RobotPose: ((1.0, 1.0), 0.2)",
            "RobotAction: (0.5, 0.1)",
            "RobotGoal: (3.0, 3.0)",
        ]

    def test_get_display_info_lines_dispatch_mode2(self, view: SimulationView) -> None:
        state = _state_with_pedestrian()
        view.display_robot_info = 2
        assert view._get_display_info_lines(state) == [
            "PedestrianPose: ((2.0, 2.0), 1.0)",
            "PedestrianAction: (1.0, 0.0)",
            "PedestrianGoal: (0.0, 0.0)",
            "DistanceRobot: 1.41",
        ]

    def test_build_text_lines_default_display(self, view: SimulationView) -> None:
        """Full overlay when not recording (fps/speedup lines present).

        ``clock.get_fps()`` is ``0.0`` before any frame is ticked, and the view
        defaults to ``current_target_fps=60.0``; both feed the recorded lines.
        """
        state = _state_with_robot(timestep=7)
        view.current_target_fps = 60.0
        info = view._get_robot_info_lines(state)
        assert view._build_text_lines(7, state, info) == [
            "step: 7",
            "scaling: 10.0",
            "target fps: 0.0/60.0",
            "speedup: 0.0x",
            "x-offset: 0.00",
            "y-offset: 0.00",
            "RobotPose: ((1.0, 1.0), 0.2)",
            "RobotAction: (0.5, 0.1)",
            "RobotGoal: (3.0, 3.0)",
            "(Press h for help)",
        ]

    def test_build_text_lines_omits_fps_when_recording(self) -> None:
        """While recording the fps/speedup lines are dropped from the overlay."""
        recording = SimulationView(width=64, height=64, scaling=10.0, record_video=True)
        state = _state_with_robot(timestep=7)
        recording.current_target_fps = 30.0
        assert recording._build_text_lines(7, state, []) == [
            "step: 7",
            "scaling: 10.0",
            "x-offset: 0.00",
            "y-offset: 0.00",
            "(Press h for help)",
        ]

    def test_build_text_lines_reflects_offset_and_scaling(self, view: SimulationView) -> None:
        """Offset and scaling are echoed into their respective lines.

        Offset is reported in world units (divided by scaling).
        """
        state = _state_with_robot(timestep=3)
        view.scaling = 4.0
        view.offset = np.array([8.0, 12.0])
        view.current_target_fps = 60.0
        lines = view._build_text_lines(3, state, [])
        assert "scaling: 4.0" in lines
        assert "x-offset: 2.00" in lines  # 8.0 / 4.0
        assert "y-offset: 3.00" in lines  # 12.0 / 4.0


# --------------------------------------------------------------------------- #
# 2. Frame-cadence logic
# --------------------------------------------------------------------------- #


class TestFrameCadence:
    """Pin ``_effective_video_fps`` and ``_should_capture_recording_frame``."""

    def test_effective_fps_uses_current_target_when_unset(self, view: SimulationView) -> None:
        view.current_target_fps = 60.0
        assert view.video_fps is None
        assert view._effective_video_fps() == 60.0

    def test_effective_fps_prefers_explicit_video_fps(self, view: SimulationView) -> None:
        view.current_target_fps = 60.0
        view.video_fps = 25.0
        assert view._effective_video_fps() == 25.0

    def test_effective_fps_float_return_type(self, view: SimulationView) -> None:
        view.video_fps = 30
        assert isinstance(view._effective_video_fps(), float)
        assert view._effective_video_fps() == 30.0

    @pytest.mark.parametrize(
        "capture_fps, render_fps",
        [
            (60.0, 60.0),  # equal -> capture every frame
            (120.0, 60.0),  # capture faster than render -> capture every frame
        ],
    )
    def test_capture_always_when_capture_ge_render(
        self, capture_fps: float, render_fps: float
    ) -> None:
        view = SimulationView(width=64, height=64, video_fps=capture_fps)
        view.current_target_fps = render_fps
        results = [view._should_capture_recording_frame(render_fps) for _ in range(8)]
        assert results == [True] * 8

    def test_capture_downsamples_60_to_10_sequence(self) -> None:
        """Rendering at 60 fps with a 10 fps target captures a fixed 10 of 60.

        The first frame is always captured; thereafter a credit accumulator at
        ``capture_fps / render_fps = 1/6`` gates each frame. The exact boolean
        sequence for the first 12 renders is pinned.
        """
        view = SimulationView(width=64, height=64, video_fps=10.0)
        view.current_target_fps = 60.0
        results = [view._should_capture_recording_frame(60.0) for _ in range(60)]
        assert sum(results) == 10
        assert results[:12] == [
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
        ]

    def test_capture_downsamples_60_to_30_sequence(self) -> None:
        """Rendering at 60 fps with a 30 fps target alternates capture/skip."""
        view = SimulationView(width=64, height=64, video_fps=30.0)
        view.current_target_fps = 60.0
        results = [view._should_capture_recording_frame(60.0) for _ in range(12)]
        assert results == [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        assert sum(results) == 6

    def test_capture_true_when_capture_fps_nonpositive(self) -> None:
        view = SimulationView(width=64, height=64, video_fps=0.0)
        view.current_target_fps = 60.0
        assert view._should_capture_recording_frame(60.0) is True

    def test_capture_true_when_render_fps_nonpositive(self) -> None:
        view = SimulationView(width=64, height=64, video_fps=10.0)
        view.current_target_fps = 60.0
        # Degenerate render fps falls back to "capture every frame".
        assert view._should_capture_recording_frame(0.0) is True

    def test_first_frame_always_captured(self) -> None:
        view = SimulationView(width=64, height=64, video_fps=10.0)
        view.current_target_fps = 60.0
        assert view._should_capture_recording_frame(60.0) is True


# --------------------------------------------------------------------------- #
# 3. Coordinate scaling
# --------------------------------------------------------------------------- #


class TestCoordinateScaling:
    """Pin ``_scale_tuple`` (offset and camera branches) and ``_timestep_text_pos``."""

    def test_scale_tuple_offset_branch_default(self, view: SimulationView) -> None:
        # default scaling=10, offset=[0,0], no camera center
        assert view._scale_tuple((1.0, 2.0)) == (10.0, 20.0)

    def test_scale_tuple_origin(self, view: SimulationView) -> None:
        assert view._scale_tuple((0.0, 0.0)) == (0.0, 0.0)

    def test_scale_tuple_with_offset(self, view: SimulationView) -> None:
        view.offset = np.array([5.0, 7.0])
        assert view._scale_tuple((1.0, 2.0)) == (15.0, 27.0)

    def test_scale_tuple_respects_scaling_factor(self) -> None:
        view = SimulationView(width=100, height=100, scaling=2.5)
        assert view._scale_tuple((3.0, 4.0)) == (7.5, 10.0)

    def test_scale_tuple_camera_branch_identity_rotation(self, view: SimulationView) -> None:
        """Camera-centered branch with zero rotation centers the world point.

        With width/height=64, center world (5,5) and identity rotation, the
        world point (5,5) maps to the screen center (32, 32).
        """
        view._camera_center_world = (5.0, 5.0)
        view._camera_rotation_rad = 0.0
        view._camera_rotation_cos = 1.0
        view._camera_rotation_sin = 0.0
        assert view._scale_tuple((5.0, 5.0)) == (32.0, 32.0)
        assert view._scale_tuple((6.0, 5.0)) == (42.0, 32.0)

    def test_timestep_text_pos_constant(self, view: SimulationView) -> None:
        assert view._timestep_text_pos == (16, 16)
