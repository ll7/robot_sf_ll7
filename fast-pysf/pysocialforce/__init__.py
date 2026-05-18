"""Numpy implementation of the Social Force model."""

__version__ = "2.0.0"

import importlib

from .config import (
    DesiredForceConfig,
    GroupCoherenceForceConfig,
    GroupGazeForceConfig,
    GroupReplusiveForceConfig,
    ObstacleForceConfig,
    PedSpawnConfig,
    SceneConfig,
    SimulatorConfig,
    SocialForceConfig,
)
from .forces import (
    DebuggableForce,
    DesiredForce,
    Force,
    GroupCoherenceForceAlt,
    GroupGazeForceAlt,
    GroupRepulsiveForce,
    ObstacleForce,
    SocialForce,
)
from .logging import logger
from .map_config import Circle, GlobalRoute, Line2D, MapDefinition, Obstacle, Rect, Vec2D, Zone
from .map_loader import load_map
from .simulator import Simulator, Simulator_v2


def __getattr__(name):
    """Lazy-load pygame-backed visualization exports on demand.

    Returns:
        Any: Requested visualization object from ``pysocialforce.sim_view``.
    """
    if name in {"SimulationView", "VisualizableSimState", "to_visualizable_state"}:
        sim_view = importlib.import_module("pysocialforce.sim_view")
        return getattr(sim_view, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
