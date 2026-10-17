"""Explicit boundary between endpoint geometry and PySocialForce obstacle inputs."""

from collections.abc import Iterable


def endpoint_segments_to_pysf(
    segments: Iterable[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Convert ``(x1, y1, x2, y2)`` walls to ``(x1, x2, y1, y2)``.

    ``pysocialforce.scene.EnvState`` consumes coordinates grouped by axis,
    but its ``obstacles_raw[:, :4]`` readback uses endpoint order. Apply this
    conversion exactly once before passing endpoint geometry to the simulator.

    Returns:
        Obstacle input segments in PySocialForce's axis-grouped order.
    """
    return [(x1, x2, y1, y2) for x1, y1, x2, y2 in segments]
