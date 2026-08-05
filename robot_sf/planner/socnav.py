"""Thin compatibility facade for the SocNav planner-family modules."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from robot_sf.planner.socnav_base import (
    SamplingPlannerAdapter,
    SocNavBenchComplexPolicy,
    SocNavPlannerConfig,
    SocNavPlannerPolicy,
    TrivialReferencePlannerAdapter,
)
from robot_sf.planner.socnav_occupancy import OccupancyAwarePlannerMixin

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


def __getattr__(name: str) -> Any:
    """Resolve family exports and deferred compatibility dependencies.

    Returns:
        Any: The requested family object or deferred compatibility value.
    """
    for exports, module_name in (
        (_SACADRL_LAZY_EXPORTS, "robot_sf.planner.socnav_sacadrl"),
        (_SOCIAL_FORCE_LAZY_EXPORTS, "robot_sf.planner.socnav_social_force"),
        (_ORCA_LAZY_EXPORTS, "robot_sf.planner.socnav_orca"),
        (_PREDICTION_LAZY_EXPORTS, "robot_sf.planner.socnav_prediction"),
    ):
        if name in exports:
            value = getattr(import_module(module_name), name)
            globals()[name] = value
            return value

    # The extracted family modules still read these optional backends through
    # the compatibility module. Resolve them only when a family or an existing
    # caller actually needs one; they are intentionally not facade exports.
    if name in {"torch", "tf", "rvo2", "sf_forces"}:
        module_name = {
            "torch": "torch",
            "tf": "tensorflow.compat.v1",
            "rvo2": "rvo2",
            "sf_forces": "pysocialforce.forces",
        }[name]
        try:
            value = import_module(module_name)
        except ImportError:  # pragma: no cover - optional dependency
            value = None
        globals()[name] = value
        return value

    if name in {"PredictiveTrajectoryModel", "load_predictive_checkpoint"}:
        try:
            module = import_module("robot_sf.planner.predictive_model")
        except ImportError:  # pragma: no cover - optional dependency
            value = Any if name == "PredictiveTrajectoryModel" else None
        else:
            try:
                value = getattr(module, name)
            except ImportError:  # pragma: no cover - optional torch dependency
                value = Any if name == "PredictiveTrajectoryModel" else None
        globals()[name] = value
        return value

    if name == "resolve_model_path":
        value = getattr(import_module("robot_sf.models"), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy public names in facade introspection and wildcard imports.

    Returns:
        list[str]: Sorted names exposed by the facade.
    """
    return sorted(
        set(globals())
        | set(__all__)
        | {name for name in _ORCA_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _SACADRL_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _SOCIAL_FORCE_LAZY_EXPORTS if not name.startswith("_")}
        | {name for name in _PREDICTION_LAZY_EXPORTS if not name.startswith("_")}
    )


__all__ = [
    "HRVOPlannerAdapter",
    "ORCAPlannerAdapter",
    "OccupancyAwarePlannerMixin",
    "PredictionPlannerAdapter",
    "SACADRLPlannerAdapter",
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavBenchSamplingAdapter",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "SocialForcePlannerAdapter",
    "TrivialReferencePlannerAdapter",
    "make_hrvo_policy",
    "make_orca_policy",
    "make_prediction_policy",
    "make_sacadrl_policy",
    "make_social_force_policy",
]
