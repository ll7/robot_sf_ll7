"""Validation for the h600 extended-roster benchmark matrix."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec

ROOT = Path(__file__).resolve().parents[2]
PROBE_CONFIG = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon600_probe.yaml"
)
EXTENDED_CONFIG = ROOT / "configs/benchmarks/paper_experiment_matrix_v1_h600_extended_roster.yaml"
PREDICTION_MPC_CBF_CONFIG = ROOT / "configs/algos/prediction_mpc_cv_cbf_collision_cone.yaml"


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_h600_extended_roster_preserves_probe_contract_and_appends_new_arms() -> None:
    """Extended h600 matrix keeps the bounded probe surface and adds two resolvable arms."""
    probe_payload = _load_yaml(PROBE_CONFIG)
    extended_payload = _load_yaml(EXTENDED_CONFIG)

    probe_planners = probe_payload["planners"]
    extended_planners = extended_payload["planners"]

    assert extended_payload["horizon"] == probe_payload["horizon"] == 600
    assert extended_payload["dt"] == probe_payload["dt"]
    assert extended_payload["seed_policy"] == probe_payload["seed_policy"]
    assert extended_payload["kinematics_matrix"] == probe_payload["kinematics_matrix"]
    assert extended_payload["scenario_matrix"] == probe_payload["scenario_matrix"]
    assert extended_planners[: len(probe_planners)] == probe_planners
    assert len(extended_planners) == len(probe_planners) + 2

    appended_keys = [planner["key"] for planner in extended_planners[len(probe_planners) :]]
    assert appended_keys == ["prediction_mpc", "prediction_mpc_cbf"]


def test_h600_extended_roster_loads_and_planner_algorithms_resolve() -> None:
    """Camera-ready loader accepts the matrix and every algorithm key is registered."""
    cfg = load_campaign_config(EXTENDED_CONFIG)

    assert cfg.name == "paper_experiment_matrix_v1_h600_extended_roster"
    assert cfg.horizon == 600
    assert cfg.seed_policy.mode == "seed-set"
    assert cfg.seed_policy.seed_set == "eval"
    assert len(cfg.planners) == 9

    for planner in cfg.planners:
        assert get_algorithm_readiness(planner.algo) is not None, planner.algo
        if planner.algo_config_path is not None:
            if not planner.algo_config_path.is_file():
                raise FileNotFoundError(
                    f"Algorithm config path not found or is not file: {planner.algo_config_path}"
                )

    prediction_mpc = next(planner for planner in cfg.planners if planner.key == "prediction_mpc")
    prediction_mpc_cbf = next(
        planner for planner in cfg.planners if planner.key == "prediction_mpc_cbf"
    )

    assert prediction_mpc.algo == "prediction_mpc"
    assert prediction_mpc.algo_config_path == (ROOT / "configs/algos/prediction_mpc_cv.yaml")
    assert prediction_mpc_cbf.algo == "prediction_mpc"
    assert prediction_mpc_cbf.algo_config_path == PREDICTION_MPC_CBF_CONFIG


def test_prediction_mpc_cbf_arm_uses_supported_adapter_config_surface() -> None:
    """The CBF arm resolves through the migrated adapter-policy builder."""
    algo_config = _load_yaml(PREDICTION_MPC_CBF_CONFIG)
    cbf_config = algo_config["cbf_safety_filter"]

    assert cbf_config == {"enabled": True}
    spec = build_registered_adapter_policy_spec("prediction_mpc", algo_config)

    assert spec is not None
    assert spec.algo_key == "prediction_mpc"
    assert spec.adapter_name == "PredictionMPCPlannerAdapter"
