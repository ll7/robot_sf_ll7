"""Regression coverage for production guards that must survive Python optimization."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_OPTIMIZED_GUARD_SCRIPT = textwrap.dedent(
    """
    from dataclasses import replace
    from types import SimpleNamespace
    from pathlib import Path

    import numpy as np

    import robot_sf.benchmark.camera_ready._config as camera_ready_config
    from robot_sf.benchmark.camera_ready._config import (
        RadiusSweepBindingPreflightError,
        _apply_radius_sweep_binding,
    )
    from robot_sf.benchmark.camera_ready._config_types import RadiusSweepConfig
    from robot_sf.baselines.ppo import PPOPlanner
    from robot_sf.baselines.social_force import SFPlannerConfig, SocialForcePlanner
    from robot_sf.benchmark.issue_5303_search_promotion_preregistration import (
        preflight_issue_5303_contract,
    )
    from robot_sf.benchmark.issue_5303_search_promotion_preregistration_v2 import (
        downstream_activation_errors,
        preflight_issue_5303_powered_contract,
    )
    import robot_sf.benchmark.collision.collision_pressure_report as collision_pressure_report
    from robot_sf.benchmark.collision.collision_pressure_report import (
        CollisionPressureReportError,
    )
    from robot_sf.benchmark.tie_aware_ranking import (
        TieAwareRankingError,
        _intervals_overlap_or_contact,
        _normalise_item,
    )
    from robot_sf.feature_extractors.attention_extractor import MultiHeadAttention
    from robot_sf.nav.occupancy_grid import GridConfig, OccupancyGrid
    import robot_sf.nav.svg_map_parser as svg
    from robot_sf.planner.dwa import DWAPlannerAdapter
    from robot_sf.planner.guarded_ppo import GuardedPPOAdapter
    from robot_sf.planner.predictive_mppi import (
        PredictiveMPPIAdapter,
        build_predictive_mppi_config,
    )
    from robot_sf.planner.nmpc_social import NMPCSocialConfig, NMPCSocialPlannerAdapter
    from robot_sf.planner.mppi_social import MPPISocialConfig, MPPISocialPlannerAdapter
    from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter
    import robot_sf.planner.socnav as socnav
    import robot_sf.scenario_certification.v1 as cert
    from robot_sf.sim.simulator import init_simulators


    def expect(label, exc_type, message_part, call):
        try:
            call()
        except Exception as exc:
            if type(exc) is not exc_type:
                raise RuntimeError(
                    f"{label}: expected {exc_type.__name__}, "
                    f"got {type(exc).__name__}: {exc}"
                )
            if message_part not in str(exc):
                raise RuntimeError(
                    f"{label}: missing {message_part!r} in {str(exc)!r}"
                )
            print(f"PASS {label}: {type(exc).__name__}: {exc}")
            return
        raise RuntimeError(f"{label}: expected {exc_type.__name__}, no exception raised")


    for label, preflight in (
        ("issue_5303_v1_preflight", preflight_issue_5303_contract),
        ("issue_5303_v2_preflight", preflight_issue_5303_powered_contract),
    ):
        result = preflight(repo_root=Path.cwd())
        if not result.ready:
            raise RuntimeError(f"{label}: preflight blocked: {result.blockers}")
        print(f"PASS {label}: ready")

    terminal_mapping_errors = downstream_activation_errors(object())
    if terminal_mapping_errors != ["terminal result must be a mapping"]:
        raise RuntimeError(
            "issue_5303_terminal_mapping: unexpected errors " f"{terminal_mapping_errors!r}"
        )
    print(f"PASS issue_5303_terminal_mapping: fail-closed: {terminal_mapping_errors[0]}")

    tie_item = _normalise_item(
        {
            "key": "item",
            "score": 1.0,
            "uncertainty": {"low": 0.5, "high": 1.5, "source": "fixture"},
        },
        0,
    )
    expect(
        "tie_aware_interval_bounds",
        TieAwareRankingError,
        "interval comparison requires both uncertainty bounds",
        lambda: _intervals_overlap_or_contact(
            replace(tie_item, uncertainty_low=None), tie_item
        ),
    )

    original_ledger_from_row = collision_pressure_report._ledger_from_row
    original_ledger_exclusion_reason = collision_pressure_report._ledger_exclusion_reason
    collision_pressure_report._ledger_from_row = lambda _row: None
    collision_pressure_report._ledger_exclusion_reason = lambda _ledger: None
    try:
        expect(
            "collision_pressure_ledger_guard",
            CollisionPressureReportError,
            "selected row is missing an auditable event ledger",
            lambda: collision_pressure_report._select_rows(
                [{"episode_id": "episode-1", "scenario_family": "family_a"}],
                ["family_a"],
            ),
        )
    finally:
        collision_pressure_report._ledger_from_row = original_ledger_from_row
        collision_pressure_report._ledger_exclusion_reason = original_ledger_exclusion_reason


    expect(
        "attention_positive_heads",
        ValueError,
        "num_heads must be positive, got 0",
        lambda: MultiHeadAttention(8, 0),
    )
    expect(
        "attention_divisibility",
        ValueError,
        "embed_dim=8 must be divisible by num_heads=3",
        lambda: MultiHeadAttention(8, 3),
    )
    expect(
        "simulator_map_definition",
        TypeError,
        "map_def should be of type MapDefinition",
        lambda: init_simulators(None, object()),
    )

    converter = svg.SvgMapConverter.__new__(svg.SvgMapConverter)
    converter.map_definition = object()
    expect(
        "svg_getter_result",
        TypeError,
        "Map definition is not of type MapDefinition",
        converter.get_map_definition,
    )

    original_converter = svg.SvgMapConverter
    svg.SvgMapConverter = lambda _path, *, geometry_contract: SimpleNamespace(
        map_definition=object()
    )
    try:
        converted = svg.convert_map("unused.svg")
        if converted is not None:
            raise RuntimeError(
                f"svg_converter_result: expected None, got {converted!r}"
            )
        print("PASS svg_converter_result: TypeError caught by convert_map policy")
    finally:
        svg.SvgMapConverter = original_converter

    social_force = SocialForcePlanner(SFPlannerConfig())
    expect(
        "social_force_observation",
        TypeError,
        "SocialForcePlanner requires Observation, got object",
        lambda: social_force.step(object()),
    )

    ppo = PPOPlanner.__new__(PPOPlanner)
    ppo._ensure_model_loaded = lambda: None
    ppo._uses_dict_observation = lambda: False
    expect(
        "ppo_observation",
        TypeError,
        "PPOPolicy requires Observation, got object",
        lambda: ppo.step(object()),
    )

    grid_config = GridConfig()
    grid_config.width = 0.0
    grid = OccupancyGrid(grid_config)
    expect(
        "occupancy_shape",
        ValueError,
        "Invalid grid shape: (2, 200, 0)",
        lambda: grid.generate([], [], ((0.0, 0.0), 0.0)),
    )

    dwa = DWAPlannerAdapter()
    expect(
        "dwa_obstacle_clearance_observation",
        ValueError,
        "DWA obstacle clearance requires observation when grid_payload is absent",
        lambda: dwa._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )

    predictive_mppi = PredictiveMPPIAdapter(
        build_predictive_mppi_config({"sample_count": 1, "iterations": 1}),
        allow_fallback=True,
    )
    expect(
        "predictive_mppi_obstacle_clearance_observation",
        ValueError,
        "Predictive MPPI obstacle clearance requires observation when grid_payload is absent",
        lambda: predictive_mppi._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )

    nmpc_social = NMPCSocialPlannerAdapter(NMPCSocialConfig(horizon_steps=1))
    expect(
        "nmpc_social_obstacle_clearance_observation",
        ValueError,
        "NMPC Social obstacle clearance requires observation when grid_payload is absent",
        lambda: nmpc_social._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )
    expect(
        "nmpc_social_occupancy_observation",
        ValueError,
        "NMPC Social occupancy cost requires observation when grid_payload is absent",
        lambda: nmpc_social._occupancy_cost(np.zeros(2, dtype=float)),
    )

    mppi_social = MPPISocialPlannerAdapter(
        MPPISocialConfig(sample_count=1, iterations=1, horizon_steps=1)
    )
    expect(
        "mppi_social_obstacle_clearance_observation",
        ValueError,
        "MPPI Social obstacle clearance requires observation when grid_payload is absent",
        lambda: mppi_social._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )
    risk_dwa = RiskDWAPlannerAdapter()
    expect(
        "risk_dwa_obstacle_clearance_observation",
        ValueError,
        "Risk-DWA obstacle clearance requires observation when grid_payload is absent",
        lambda: risk_dwa._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )
    guarded_ppo = GuardedPPOAdapter()
    expect(
        "guarded_ppo_obstacle_clearance_observation",
        ValueError,
        "Guarded PPO obstacle clearance requires observation when grid_payload is absent",
        lambda: guarded_ppo._min_obstacle_clearance(np.zeros(2, dtype=float)),
    )

    radius_config = RadiusSweepConfig(
        issue=6642,
        parent_issue=6600,
        arm_key="r0p5",
        radius_m=0.5,
        baseline_arm=False,
        runtime_binding_status="bound_runtime",
        binding_contract_version="radius_binding_canary.v1",
        gate1_canary_issue=6641,
        gate1_receipt_sha256="a" * 64,
        gate1_source_commit="b" * 40,
    )
    original_radius_metadata = camera_ready_config._radius_binding_metadata
    camera_ready_config._radius_binding_metadata = lambda _config: None
    try:
        expect(
            "camera_ready_radius_binding_metadata",
            RadiusSweepBindingPreflightError,
            "radius-sweep binding metadata could not be constructed",
            lambda: _apply_radius_sweep_binding([{"name": "s1"}], radius_config),
        )
    finally:
        camera_ready_config._radius_binding_metadata = original_radius_metadata

    original_torch = socnav.torch
    planner = socnav.PredictionPlannerAdapter.__new__(socnav.PredictionPlannerAdapter)
    planner._baseline_predictor = None
    planner._ensure_model = lambda: object()
    socnav.torch = None
    try:
        expect(
            "predictive_pytorch_capability",
            RuntimeError,
            "PyTorch is required for predictive model inference but is not available",
            lambda: planner._predict_trajectories(
                np.zeros((1, 4), dtype=np.float32),
                np.ones(1, dtype=np.float32),
            ),
        )
    finally:
        socnav.torch = original_torch

    route = SimpleNamespace(
        source_label="route",
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.0, 0.0)],
    )
    map_def = SimpleNamespace(width=10.0, height=10.0)
    config = SimpleNamespace(
        robot_config=SimpleNamespace(radius=1.0),
        sim_config=SimpleNamespace(ped_radius=0.4),
    )
    original_start_goal = cert._route_start_goal
    original_line = cert._line_from_points
    original_union = cert._obstacle_union
    original_validate = cert._validate_route_shape
    cert._obstacle_union = lambda _map: None
    cert._validate_route_shape = lambda *_args, **_kwargs: []
    try:
        cases = (
            ("route_start", (None, (1.0, 1.0)), object(), "None start"),
            ("route_goal", ((0.0, 0.0), None), object(), "None goal"),
            (
                "route_line",
                ((0.0, 0.0), (1.0, 1.0)),
                None,
                "None route_line",
            ),
        )
        for label, endpoints, route_line, message in cases:
            cert._route_start_goal = (
                lambda _route, endpoints=endpoints: endpoints
            )
            cert._line_from_points = (
                lambda _points, route_line=route_line: route_line
            )
            expect(
                label,
                RuntimeError,
                message,
                lambda: cert._certify_route(
                    route,
                    map_def=map_def,
                    map_name="map",
                    config=config,
                    settings=object(),
                ),
            )
    finally:
        cert._route_start_goal = original_start_goal
        cert._line_from_points = original_line
        cert._obstacle_union = original_union
        cert._validate_route_shape = original_validate
    """
)

_EXPECTED_MARKERS = (
    "PASS issue_5303_v1_preflight: ready",
    "PASS issue_5303_v2_preflight: ready",
    "PASS issue_5303_terminal_mapping: fail-closed",
    "PASS tie_aware_interval_bounds: TieAwareRankingError",
    "PASS collision_pressure_ledger_guard: CollisionPressureReportError",
    "PASS attention_positive_heads: ValueError",
    "PASS attention_divisibility: ValueError",
    "PASS simulator_map_definition: TypeError",
    "PASS svg_getter_result: TypeError",
    "PASS svg_converter_result: TypeError",
    "PASS social_force_observation: TypeError",
    "PASS ppo_observation: TypeError",
    "PASS occupancy_shape: ValueError",
    "PASS dwa_obstacle_clearance_observation: ValueError",
    "PASS predictive_mppi_obstacle_clearance_observation: ValueError",
    "PASS nmpc_social_obstacle_clearance_observation: ValueError",
    "PASS nmpc_social_occupancy_observation: ValueError",
    "PASS mppi_social_obstacle_clearance_observation: ValueError",
    "PASS risk_dwa_obstacle_clearance_observation: ValueError",
    "PASS guarded_ppo_obstacle_clearance_observation: ValueError",
    "PASS camera_ready_radius_binding_metadata: RadiusSweepBindingPreflightError",
    "PASS predictive_pytorch_capability: RuntimeError",
    "PASS route_start: RuntimeError",
    "PASS route_goal: RuntimeError",
    "PASS route_line: RuntimeError",
)

_EXPECTED_MESSAGES = (
    "terminal result must be a mapping",
    "interval comparison requires both uncertainty bounds",
    "selected row is missing an auditable event ledger",
    "num_heads must be positive, got 0",
    "embed_dim=8 must be divisible by num_heads=3",
    "map_def should be of type MapDefinition",
    "Map definition is not of type MapDefinition",
    "SVG map converter produced unexpected type",
    "SocialForcePlanner requires Observation, got object",
    "PPOPolicy requires Observation, got object",
    "Invalid grid shape: (2, 200, 0)",
    "DWA obstacle clearance requires observation when grid_payload is absent",
    "Predictive MPPI obstacle clearance requires observation when grid_payload is absent",
    "NMPC Social obstacle clearance requires observation when grid_payload is absent",
    "NMPC Social occupancy cost requires observation when grid_payload is absent",
    "MPPI Social obstacle clearance requires observation when grid_payload is absent",
    "Risk-DWA obstacle clearance requires observation when grid_payload is absent",
    "Guarded PPO obstacle clearance requires observation when grid_payload is absent",
    "radius-sweep binding metadata could not be constructed",
    "PyTorch is required for predictive model inference but is not available",
    "None start",
    "None goal",
    "None route_line",
)


def test_converted_guards_survive_python_optimized_mode() -> None:
    """Converted guards retain explicit exception contracts when asserts are removed."""
    assert not any(
        isinstance(node, ast.Assert) for node in ast.walk(ast.parse(_OPTIMIZED_GUARD_SCRIPT))
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", _OPTIMIZED_GUARD_SCRIPT],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined_output = f"{result.stdout}\n{result.stderr}"

    assert result.returncode == 0, combined_output
    for marker in _EXPECTED_MARKERS:
        assert marker in combined_output
    for message in _EXPECTED_MESSAGES:
        assert message in combined_output


@pytest.mark.parametrize(
    "relative_path",
    (
        "robot_sf/benchmark/issue_5303_search_promotion_preregistration.py",
        "robot_sf/benchmark/issue_5303_search_promotion_preregistration_v2.py",
        "robot_sf/benchmark/tie_aware_ranking.py",
        "robot_sf/benchmark/collision/collision_pressure_report.py",
    ),
)
def test_scoped_production_modules_have_no_production_asserts(
    relative_path: str,
) -> None:
    """Scoped production guards must not disappear when Python runs with ``-O``."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))
