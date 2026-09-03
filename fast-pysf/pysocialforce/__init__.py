"""Numpy implementation of the Social Force model."""

__version__ = "2.0.0"

import importlib

from .config import (
    DEFAULT_OBSTACLE_FORCE_LAW,
    LEGACY_SHIFTED_GRADIENT_V1,
    OBSTACLE_FORCE_DISTANCE_FLOOR,
    OBSTACLE_FORCE_LAW_METADATA_SCHEMA,
    OBSTACLE_FORCE_LAW_RESOLUTION_MODES,
    OBSTACLE_FORCE_LAW_SELECTOR_KEYS,
    OBSTACLE_FORCE_LAW_VERSIONS,
    SURFACE_DISTANCE_UNIT_NORMAL_V2,
    DesiredForceConfig,
    GroupCoherenceForceConfig,
    GroupGazeForceConfig,
    GroupReplusiveForceConfig,
    ObstacleForceConfig,
    PedSpawnConfig,
    SceneConfig,
    SimulatorConfig,
    SocialForceConfig,
    obstacle_force_law_metadata,
    resolve_obstacle_force_law,
    resolve_obstacle_force_law_with_mode,
)
from .force_trace import (
    ForceComponentOperation,
    ForceComponentResult,
    ForceComputationResult,
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
    all_obstacle_forces_for_law,
    obstacle_force_for_law,
    obstacle_force_surface_distance_unit_normal,
    surface_distance_unit_normal_force,
    surface_distance_unit_normal_force_vectors,
)
from .logging import logger
from .map_config import (
    Circle,
    GlobalRoute,
    Line2D,
    MapDefinition,
    Obstacle,
    Rect,
    Vec2D,
    Zone,
)
from .map_loader import load_map
from .scene import PedestrianStepDiagnostics
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
