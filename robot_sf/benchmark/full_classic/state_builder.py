"""State builder for SimulationView rendering (T037).

Transforms a `ReplayEpisode` into a sequence of minimal objects exposing the
attributes expected by `SimulationView.render` for the current placeholder
integration. This keeps the adapter logic isolated so future enrichment
(pedestrians, actions, sensor rays) only touches this module.

Current scope: robot pose only (as a tuple[(x,y), heading]) and timestep index.
Pedestrian, action, and sensor information are left absent (SimulationView
guards with hasattr checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from .replay import ReplayEpisode


@dataclass(slots=True)
class _MinimalState:
    timestep: int
    robot_pose: tuple[tuple[float, float], float]


def build_minimal_states(ep: ReplayEpisode) -> Iterable[_MinimalState]:
    """Yield minimal state objects for each replay step.

    Parameters
    ----------
    ep : ReplayEpisode
        Source replay episode with ordered steps.
    """
    for idx, step in enumerate(ep.steps):
        yield _MinimalState(timestep=idx, robot_pose=((step.x, step.y), step.heading))


def iter_states(ep: ReplayEpisode) -> Generator[_MinimalState, None, None]:  # thin alias
    yield from build_minimal_states(ep)


__all__ = ["build_minimal_states", "iter_states"]
