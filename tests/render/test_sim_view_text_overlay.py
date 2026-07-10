"""Focused tests for the ``SimViewTextOverlay`` collaborator (god-class split #4989).

These complement the behavior-preservation golden lines in
``test_sim_view_characterization.py`` (#4965), which pin the *delegating* surface
on ``SimulationView``. Here we cover the split-specific contract directly:

* the new collaborator module exists and is wired into ``SimulationView``;
* delegation forwards every extracted method with identical results;
* the host remains authoritative for shared mutable render state
  (``display_robot_info`` is written back through the host back-reference);
* the ``InteractivePlayback`` subclass ``super()._add_text(...)`` chain still
  works after the delegation refactor.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame = pytest.importorskip("pygame")  # skip if pygame is missing

from robot_sf.render.sim_view import (  # noqa: E402
    SimulationView,
    VisualizableAction,
    VisualizableSimState,
)
from robot_sf.render.sim_view_text_overlay import (  # noqa: E402
    TEXT_BACKGROUND,
    TEXT_COLOR,
    TEXT_OUTLINE_COLOR,
    SimViewTextOverlay,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cleanup_pygame() -> Iterator[None]:
    yield
    pygame.quit()


@pytest.fixture
def view() -> SimulationView:
    """A deterministic, headless SimulationView with small dimensions."""
    return SimulationView(width=64, height=64, scaling=10.0)


def _state_with_robot(*, timestep: int = 7) -> VisualizableSimState:
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


class TestCollaboratorWiring:
    """The collaborator exists, is the right type, and holds a host back-reference."""

    def test_host_holds_text_overlay_instance(self, view: SimulationView) -> None:
        assert isinstance(view._text_overlay, SimViewTextOverlay)

    def test_collaborator_back_references_host(self, view: SimulationView) -> None:
        assert view._text_overlay._host is view

    def test_collaborator_module_constants_match_host(self) -> None:
        """Color constants must be identical to the pre-split ``sim_view`` values."""
        from robot_sf.render import sim_view

        assert TEXT_COLOR == sim_view.TEXT_COLOR
        assert TEXT_BACKGROUND == sim_view.TEXT_BACKGROUND
        assert TEXT_OUTLINE_COLOR == sim_view.TEXT_OUTLINE_COLOR


class TestDelegation:
    """Every extracted method delegates with identical results to the collaborator."""

    def test_get_robot_info_lines_delegates(self, view: SimulationView) -> None:
        state = _state_with_robot()
        assert view._get_robot_info_lines(state) == view._text_overlay._get_robot_info_lines(state)

    def test_has_pedestrian_data_delegates_both_branches(self, view: SimulationView) -> None:
        assert view._has_pedestrian_data(_state_with_pedestrian()) is True
        assert view._has_pedestrian_data(_state_with_robot()) is False
        # matches the collaborator directly
        assert view._has_pedestrian_data(
            _state_with_pedestrian()
        ) == view._text_overlay._has_pedestrian_data(_state_with_pedestrian())

    def test_build_text_lines_delegates(self, view: SimulationView) -> None:
        view.current_target_fps = 60.0
        state = _state_with_robot(timestep=7)
        assert view._build_text_lines(7, state, []) == view._text_overlay._build_text_lines(
            7, state, []
        )

    def test_get_display_info_lines_delegates_all_modes(self, view: SimulationView) -> None:
        state = _state_with_robot()
        for mode in (0, 1):
            view.display_robot_info = mode
            assert view._get_display_info_lines(
                state
            ) == view._text_overlay._get_display_info_lines(state)

    def test_pedestrian_info_lines_writes_back_through_host(self, view: SimulationView) -> None:
        """The ``display_robot_info`` reset side-effect lands on the host, not a copy."""
        state = _state_with_robot()  # no pedestrian data
        view.display_robot_info = 2
        assert view._get_pedestrian_info_lines(state) == []
        # The collaborator wrote back to the host's attribute (single source of truth).
        assert view.display_robot_info == 0


class TestRenderingDelegation:
    """``_render_text_*`` and ``_add_text`` run end-to-end through the collaborator."""

    def test_add_text_renders_without_error(self, view: SimulationView) -> None:
        # Exercises _add_text -> _get_display_info_lines -> _build_text_lines
        # -> _render_text_display -> _render_text_line, all through the host delegator.
        view.display_robot_info = 1
        view._add_text(7, _state_with_robot())  # must not raise

    def test_render_text_display_blits_to_host_screen(self, view: SimulationView) -> None:
        before = view.screen.copy()
        view._render_text_display(["step: 1", "scaling: 10.0"])
        after = view.screen.copy()
        # The overlay is drawn onto the host's screen surface (pixels changed).
        assert not np.array_equal(pygame.surfarray.array3d(before), pygame.surfarray.array3d(after))


class TestSubclassCompatibility:
    """The ``InteractivePlayback`` ``super()._add_text(...)`` chain still works."""

    def test_subclass_super_add_text_still_works(self, view: SimulationView) -> None:
        """A subclass overriding ``_add_text`` and calling ``super()`` must not regress.

        ``InteractivePlayback._add_text`` calls ``super()._add_text(...)`` which now
        delegates to the collaborator; this synthesises that pattern to lock it in.
        """

        class _Sub(SimulationView):
            def _add_text(self, timestep, state):  # type: ignore[override]
                super()._add_text(timestep, state)

        sub = _Sub(width=64, height=64, scaling=10.0)
        sub.display_robot_info = 1
        # Should not raise: super() delegation reaches the collaborator.
        sub._add_text(3, _state_with_robot())
        assert isinstance(sub._text_overlay, SimViewTextOverlay)
