"""Conformance tests for the #6505 ``diagnostics()`` migration.

These tests are scoped to issue #6505: every ``robot_sf/planner/`` local planner
that participates in the local-planner ``plan() -> tuple`` family must expose a
``diagnostics()`` method returning at least ``{"planner_type": <ClassName>}``.

The protocol module itself (``LocalPlannerProtocol``, the baseline adapter, and
the fail-closed normalizer) is owned by issue #6492 / PR #6519 and is intentionally
not exercised here. These tests only validate the per-planner ``diagnostics()``
payloads that #6505 owns.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Regression guard: diagnostics() insertion must not erase a preceding return
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path,cls_name,method_name",
    [
        ("robot_sf/planner/risk_dwa.py", "RiskDWAPlannerAdapter", "plan"),
        ("robot_sf/planner/mppi_social.py", "MPPISocialPlannerAdapter", "plan"),
        ("robot_sf/planner/predictive_mppi.py", "PredictiveMPPIAdapter", "plan"),
        ("robot_sf/planner/safety_barrier.py", "SafetyBarrierPlannerAdapter", "plan"),
        ("robot_sf/planner/stream_gap.py", "StreamGapPlannerAdapter", "plan"),
        ("robot_sf/planner/guarded_ppo.py", "GuardedPPOAdapter", "_violated_constraints"),
    ],
)
def test_diagnostics_insertion_did_not_drop_preceding_return(
    rel_path: str, cls_name: str, method_name: str
) -> None:
    """Regression guard (#6505): ``diagnostics()`` insertion must not erase a return.

    The mechanical insertion of ``diagnostics()`` replaced the final ``return`` of
    several ``plan()``/helper methods, silently making them fall through to
    ``None``. Existing behavioral coverage did not reach those final returns, so
    the regression passed CI. Assert each affected concrete method still ends in
    a ``return`` statement so the regression fails fast if it recurs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tree = ast.parse((repo_root / rel_path).read_text())
    cls = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == cls_name
    )
    method = next(
        member
        for member in cls.body
        if isinstance(member, ast.FunctionDef) and member.name == method_name
    )
    assert isinstance(method.body[-1], ast.Return), (
        f"{cls_name}.{method_name} in {rel_path} lost its terminal return statement"
    )


# ---------------------------------------------------------------------------
# Conformance: every protocol-member planner exposes diagnostics()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name,cls_name",
    [
        ("robot_sf.planner.crowdnav_height", "CrowdNavHeightAdapter"),
        ("robot_sf.planner.crowdnav_pred_attng", "CrowdNavPredAttnGraphAdapter"),
        ("robot_sf.planner.fast_pysf_planner", "FastPysfPlannerPolicy"),
        ("robot_sf.planner.gap_prediction", "GapAwarePredictionAdapter"),
        ("robot_sf.planner.grid_route", "GridRoutePlannerAdapter"),
        ("robot_sf.planner.guarded_ppo", "GuardedPPOAdapter"),
        ("robot_sf.planner.learned_policy_adapter", "DummyLearnedLocalPolicyAdapter"),
        ("robot_sf.planner.mppi_social", "MPPISocialPlannerAdapter"),
        ("robot_sf.planner.predictive_mppi", "PredictiveMPPIAdapter"),
        ("robot_sf.planner.risk_dwa", "RiskDWAPlannerAdapter"),
        ("robot_sf.planner.safety_barrier", "SafetyBarrierPlannerAdapter"),
        ("robot_sf.planner.socnav", "PredictionPlannerAdapter"),
        ("robot_sf.planner.socnav", "SocNavBenchSamplingAdapter"),
        ("robot_sf.planner.socnav_base", "SamplingPlannerAdapter"),
        ("robot_sf.planner.socnav_orca", "ORCAPlannerAdapter"),
        ("robot_sf.planner.socnav_social_force", "SocialForcePlannerAdapter"),
        ("robot_sf.planner.sonic_crowdnav", "SonicCrowdNavAdapter"),
        (
            "robot_sf.planner.social_navigation_pyenvs_force_model",
            "SocialNavigationPyEnvsForceModelAdapter",
        ),
        (
            "robot_sf.planner.social_navigation_pyenvs_hsfm",
            "SocialNavigationPyEnvsHSFMAdapter",
        ),
        (
            "robot_sf.planner.social_navigation_pyenvs_orca",
            "SocialNavigationPyEnvsORCAAdapter",
        ),
        ("robot_sf.planner.stream_gap", "StreamGapPlannerAdapter"),
        ("robot_sf.planner.teb_commitment", "TEBCommitmentPlannerAdapter"),
    ],
)
def test_protocol_member_diagnostics_payload(module_name: str, cls_name: str) -> None:
    """Every planner in the #6505 migration returns the minimum diagnostics payload."""
    cls = getattr(import_module(module_name), cls_name)
    planner = cls.__new__(cls)

    assert planner.diagnostics() == {"planner_type": cls_name}
