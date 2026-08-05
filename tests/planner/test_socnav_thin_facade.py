"""Contract tests for the thin SocNav compatibility facade."""

import ast
from pathlib import Path

from robot_sf.planner import (
    socnav,
    socnav_base,
    socnav_occupancy,
    socnav_orca,
    socnav_prediction,
    socnav_sacadrl,
    socnav_social_force,
)

_FAMILY_EXPORTS = {
    socnav_base: (
        "SamplingPlannerAdapter",
        "SocNavBenchComplexPolicy",
        "SocNavPlannerConfig",
        "SocNavPlannerPolicy",
        "TrivialReferencePlannerAdapter",
    ),
    socnav_orca: (
        "HRVOPlannerAdapter",
        "ORCAPlannerAdapter",
        "make_hrvo_policy",
        "make_orca_policy",
    ),
    socnav_prediction: (
        "PredictionPlannerAdapter",
        "SocNavBenchSamplingAdapter",
        "make_prediction_policy",
    ),
    socnav_sacadrl: (
        "SACADRLPlannerAdapter",
        "make_sacadrl_policy",
    ),
    socnav_social_force: (
        "SocialForcePlannerAdapter",
        "make_social_force_policy",
    ),
    socnav_occupancy: ("OccupancyAwarePlannerMixin",),
}


def test_facade_contains_no_inline_planner_family_class_bodies() -> None:
    """Planner-family classes must live in their focused modules."""
    source = Path(socnav.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)] == []


def test_all_facade_names_resolve_to_identical_family_objects() -> None:
    """Every public facade name must be an identity-preserving family re-export."""
    expected_names = {name for names in _FAMILY_EXPORTS.values() for name in names}
    assert set(socnav.__all__) == expected_names

    for family_module, names in _FAMILY_EXPORTS.items():
        for name in names:
            assert getattr(socnav, name) is getattr(family_module, name)
