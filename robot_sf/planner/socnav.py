"""
Compatibility facade for SocNav-family planner adapters.

The original SocNav planner module remains import-compatible while planner-family
implementations are split into focused modules. Shared occupancy-grid helpers now
live in `robot_sf.planner.socnav_occupancy`; this module re-exports
`OccupancyAwarePlannerMixin` for existing imports. Shared base/config classes
(`SocNavPlannerConfig`, the reference/sampling adapters, and the policy wrappers)
live in `robot_sf.planner.socnav_base` and are re-exported here unchanged.
"""

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import atan2, pi
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.common.forecast_variants import FORECAST_VARIANT_CHOICES
from robot_sf.common.math_utils import wrap_angle_pi, wrap_angle_pi_closed

# Convention: optional-import guards catch ImportError only (ModuleNotFoundError is a
# subclass); bind the exception as `exc` for consistency across the codebase.
try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import tensorflow.compat.v1 as tf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tf = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import rvo2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rvo2 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pysocialforce import forces as sf_forces  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sf_forces = None  # type: ignore[assignment]

from robot_sf.models import resolve_model_path
from robot_sf.nav.occupancy_grid_utils import world_to_ego
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    LocalObstacleFeatureExtractor,
    infer_predictive_feature_schema,
    normalize_obstacle_lines,
    obstacle_lines_from_map,
    obstacle_lines_from_observation,
    validate_predictive_runtime_feature_schema,
)

try:  # pragma: no cover - exercised in minimal environments without torch
    from robot_sf.planner.predictive_model import (
        PredictiveTrajectoryModel,
        load_predictive_checkpoint,
    )
except ImportError:  # pragma: no cover - optional dependency
    PredictiveTrajectoryModel = Any  # type: ignore[misc,assignment]
    load_predictive_checkpoint = None  # type: ignore[assignment]

from robot_sf.planner.socnav_base import (
    SamplingPlannerAdapter,
    SocNavBenchComplexPolicy,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    TrivialReferencePlannerAdapter,
)
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

if TYPE_CHECKING:
    from robot_sf.planner.socnav_orca import (
        HRVOPlannerAdapter,
        ORCAPlannerAdapter,
        make_hrvo_policy,
        make_orca_policy,
    )
    from robot_sf.planner.socnav_prediction import (
        PredictionPlannerAdapter,
        SocNavBenchSamplingAdapter,
        make_prediction_policy,
    )
    from robot_sf.planner.socnav_sacadrl import SACADRLPlannerAdapter, make_sacadrl_policy
    from robot_sf.planner.socnav_social_force import (
        SocialForcePlannerAdapter,
        make_social_force_policy,
    )

_SACADRL_LAZY_EXPORTS = {
    "_SACADRLModel",
    "_SACADRL_STATE_ORDER",
    "_sacadrl_actions",
    "_sacadrl_session_config",
    "SACADRLPlannerAdapter",
    "make_sacadrl_policy",
}

_SOCIAL_FORCE_LAZY_EXPORTS = {
    "SocialForcePlannerAdapter",
    "make_social_force_policy",
}

_ORCA_LAZY_EXPORTS = {
    "HRVOPlannerAdapter",
    "ORCAPlannerAdapter",
    "make_hrvo_policy",
    "make_orca_policy",
}

_PREDICTION_LAZY_EXPORTS = {
    "PredictionPlannerAdapter",
    "SocNavBenchSamplingAdapter",
    "make_prediction_policy",
}


def __getattr__(name: str) -> Any:
    """Resolve extracted planner-family symbols without importing them eagerly.

    Returns:
        Any: Requested symbol from the extracted family module.
    """
    if name in _SACADRL_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_sacadrl")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SOCIAL_FORCE_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_social_force")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _ORCA_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_orca")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _PREDICTION_LAZY_EXPORTS:
        module = import_module("robot_sf.planner.socnav_prediction")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public names in module introspection and wildcard imports.

    Returns:
        list[str]: Sorted names exposed by the facade and its lazy exports.
    """
    return sorted(
        set(globals())
        | {name for name in _ORCA_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _SACADRL_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _SOCIAL_FORCE_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _PREDICTION_LAZY_EXPORTS if not name.startswith("_")}
    )


__all__ = [
    "FORECAST_VARIANT_CHOICES",
    "PREDICTIVE_OBSTACLE_FEATURE_SCHEMA",
    "Any",
    "Callable",
    "HRVOPlannerAdapter",
    "LocalObstacleFeatureExtractor",
    "ORCAPlannerAdapter",
    "OccupancyAwarePlannerMixin",
    "Path",
    "PredictionPlannerAdapter",
    "PredictiveTrajectoryModel",
    "SACADRLPlannerAdapter",
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavBenchSamplingAdapter",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "SocialForcePlannerAdapter",
    "TrivialReferencePlannerAdapter",
    "atan2",
    "dataclass",
    "import_module",
    "infer_predictive_feature_schema",
    "load_predictive_checkpoint",
    "logger",
    "make_hrvo_policy",
    "make_orca_policy",
    "make_prediction_policy",
    "make_sacadrl_policy",
    "make_social_force_policy",
    "normalize_obstacle_lines",
    "np",
    "obstacle_lines_from_map",
    "obstacle_lines_from_observation",
    "os",
    "pi",
    "resolve_model_path",
    "rvo2",
    "sf_forces",
    "sys",
    "tf",
    "threading",
    "torch",
    "validate_predictive_runtime_feature_schema",
    "world_to_ego",
    "wrap_angle_pi",
    "wrap_angle_pi_closed",
]
