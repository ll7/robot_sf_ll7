"""Identity checks for the opt-in 0.0.8 social-force kernel candidate."""

from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.training.scenario_loader import load_scenarios


def test_versioned_candidate_resolves_wrapped_kernel_on_both_sides() -> None:
    """The candidate binds wrapped simulation and planner kernels after loading."""
    root = Path(__file__).resolve().parents[2]
    cfg = load_campaign_config(
        root
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml",
        repository_root=root,
    )

    assert cfg.name.endswith("v0_0_8_candidate")
    scenarios = load_scenarios(cfg.scenario_matrix_path)
    assert scenarios
    assert all(
        scenario["simulation_config"]["social_force_kernel_version"] == "wrapped_v2"
        for scenario in scenarios
    )

    planner = next(planner for planner in cfg.planners if planner.key == "social_force")
    assert planner.algo_config_path is not None
    algo_config = yaml.safe_load(planner.algo_config_path.read_text(encoding="utf-8"))
    assert algo_config["social_force_kernel_version"] == "wrapped_v2"


def test_007_inputs_remain_byte_identical() -> None:
    """The candidate must not alter frozen scenario or planner inputs."""
    import hashlib

    root = Path(__file__).resolve().parents[2]
    expected = {
        "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml": "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c",
        "configs/algos/social_force_resolution_independent_v2.yaml": "4d32a90dd58c3e0273c175ee03d4de018e0d370e8a3892230942d21ca4a5a81d",
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml": "7dc9a2dd9df8585593c9bc8ecc001bed0d2ddff4ebb3803dfb92e8dad8762881",
    }
    for path, digest in expected.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest


def test_paired_diagnostic_uses_only_development_seeds() -> None:
    """Neither diagnostic arm may use the reserved evaluation range."""
    root = Path(__file__).resolve().parents[2]
    for arm in ("legacy_v1", "wrapped_v2"):
        path = root / f"configs/scenarios/issue_9764_kernel_paired_{arm}.yaml"
        matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert matrix["scenario_overrides"]["seeds"] == [103, 104, 105]
