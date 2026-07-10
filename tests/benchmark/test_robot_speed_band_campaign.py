"""Tests for the robot speed-band campaign axis (issue #5144).

These tests prove the ``robot_speed_band`` campaign axis is a real runtime-bound
capability, not a metadata-only launch packet: the swept drive-model speed cap
flows to the actual drive settings, the bicycle model is selected, #4976 braking
realism scales with the band, the axis is independent of the pedestrian-speed
axis (#4972), and the cap reaches the real ``BicycleDriveRobot`` action space.

This is a capability/binding test, not benchmark evidence: no campaign is run
and no planner-ranking claim is made.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.fidelity_sensitivity import validate_fidelity_sensitivity_config
from robot_sf.robot.actuation_envelope import actuation_envelope_from_drive_config
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "research" / "robot_speed_band_v1.yaml"
SCENARIO_SET = REPO_ROOT / "configs" / "scenarios" / "sets" / "paper_cross_kinematics_v1.yaml"


def _load_campaign_runner() -> ModuleType:
    module_path = REPO_ROOT / "scripts" / "benchmark" / "run_fidelity_sensitivity_campaign.py"
    spec = importlib.util.spec_from_file_location("robot_speed_band_campaign_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["robot_speed_band_campaign_runner"] = module
    spec.loader.exec_module(module)
    return module


campaign_runner = _load_campaign_runner()


def _variant_map() -> dict[str, campaign_runner.VariantSpec]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        variant.key: variant
        for variant in campaign_runner.load_variant_specs(config, include_all_variants=True)
    }


def _real_config() -> object:
    scenarios = list(load_scenarios(SCENARIO_SET))
    assert scenarios, f"scenario set produced no scenarios: {SCENARIO_SET}"
    return build_robot_config_from_scenario(scenarios[0], scenario_path=SCENARIO_SET)


def test_speed_band_config_validates_against_fidelity_schema() -> None:
    """The robot speed-band launch packet must satisfy the campaign schema."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validated = validate_fidelity_sensitivity_config(config)
    assert validated["schema_version"] == "fidelity-sensitivity.v1"
    assert validated["issue"] == 5144
    axis_keys = {axis["key"] for axis in validated["axes"]}
    assert "robot_speed_band" in axis_keys


def test_robot_speed_band_axis_binds_to_drive_speed_cap_not_unsupported() -> None:
    """The axis must resolve to a real runtime binding, not 'unsupported'."""
    variants = _variant_map()
    band_variants = [v for v in variants.values() if v.axis == "robot_speed_band"]
    assert band_variants, "config must define robot_speed_band variants"
    for variant in band_variants:
        assert variant.runtime_binding == "robot_config.drive_speed_cap"
        assert variant.runtime_binding != "unsupported"


def test_speed_band_variant_switches_to_bicycle_and_sweeps_max_velocity() -> None:
    """The variant must select the bicycle drive model and raise the speed cap."""
    variants = _variant_map()
    config = _real_config()
    # Baseline bicycle band at 2.0 m/s.
    campaign_runner.apply_variant(config, variants["baseline"], seed=111)
    assert isinstance(config.robot_config, BicycleDriveSettings)
    assert config.robot_config.max_velocity == pytest.approx(2.0)
    # Nominal bicycle band at 3.0 m/s (the bicycle model's existing ceiling).
    campaign_runner.apply_variant(
        config, variants["robot_speed_band__bicycle_3_0_mps_nominal"], seed=111
    )
    assert isinstance(config.robot_config, BicycleDriveSettings)
    assert config.robot_config.max_velocity == pytest.approx(3.0)
    # Micromobility-direction band at 4.0 m/s (above the differential-drive cap).
    campaign_runner.apply_variant(
        config, variants["robot_speed_band__bicycle_4_0_mps_micromobility"], seed=111
    )
    assert isinstance(config.robot_config, BicycleDriveSettings)
    assert config.robot_config.max_velocity == pytest.approx(4.0)


def test_braking_authority_and_stopping_distance_scale_with_speed_band() -> None:
    """#4976 actuation envelope must grow with the swept speed band."""
    variants = _variant_map()
    config = _real_config()
    stopping_distances: list[float] = []
    for key in (
        "baseline",
        "robot_speed_band__bicycle_3_0_mps_nominal",
        "robot_speed_band__bicycle_4_0_mps_micromobility",
    ):
        campaign_runner.apply_variant(config, variants[key], seed=111)
        envelope = actuation_envelope_from_drive_config(config.robot_config)
        assert envelope is not None, f"actuation envelope missing for {key}"
        # Braking must be distinct from forward acceleration so it is a real
        # #4976 deceleration cap, not the legacy symmetric default.
        assert envelope["braking_distinct_from_accel"] is True
        assert envelope["peak_forward_speed_m_s"] == pytest.approx(config.robot_config.max_velocity)
        stopping_distances.append(float(envelope["stopping_distance_envelope_m"]))
    # v^2 / (2*decel) with decel == speed => 2/2, 3/2, 4/2 -> strictly increasing.
    assert stopping_distances == sorted(stopping_distances)
    assert stopping_distances[0] < stopping_distances[-1]


def test_robot_speed_band_axis_is_independent_of_pedestrian_speed_axis() -> None:
    """The robot speed-band axis must not mutate pedestrian-speed state (#4972)."""
    variants = _variant_map()
    config = _real_config()
    # Capture pedestrian-speed state before applying the robot speed band.
    config.sim_config = SimulationSettings(ped_speed_tier="typical", ped_radius=0.4)
    before_tier = config.sim_config.ped_speed_tier
    before_desired_mean = config.sim_config.desired_speed_mean
    before_ped_radius = config.sim_config.ped_radius

    campaign_runner.apply_variant(
        config, variants["robot_speed_band__bicycle_4_0_mps_micromobility"], seed=111
    )
    # The robot speed band mutated robot_config only.
    assert isinstance(config.robot_config, BicycleDriveSettings)
    assert config.robot_config.max_velocity == pytest.approx(4.0)
    # Pedestrian-speed state is untouched by the robot speed-band axis.
    assert config.sim_config.ped_speed_tier == before_tier
    assert config.sim_config.desired_speed_mean == before_desired_mean
    assert config.sim_config.ped_radius == before_ped_radius

    # Conversely, the pedestrian-speed archetype axis must not change the
    # robot's resolved speed cap.
    arch_variant = variants["social_force_speed_archetypes__rush_hour"]
    cap_before = campaign_runner._robot_speed_cap(config.robot_config)
    campaign_runner.apply_variant(config, arch_variant, seed=111)
    assert campaign_runner._robot_speed_cap(config.robot_config) == pytest.approx(cap_before)
    assert isinstance(config.robot_config, BicycleDriveSettings)


def test_robot_speed_cap_reader_is_drive_model_agnostic() -> None:
    """The cap reader must read max_velocity (bicycle) and max_linear_speed (diff)."""
    assert campaign_runner._robot_speed_cap(
        BicycleDriveSettings(max_velocity=5.0)
    ) == pytest.approx(5.0)
    assert campaign_runner._robot_speed_cap(
        DifferentialDriveSettings(max_linear_speed=2.5)
    ) == pytest.approx(2.5)


def test_speed_band_variant_flows_to_real_bicycle_robot_action_space() -> None:
    """The swept cap must reach the actual BicycleDriveRobot via make_robot_env."""
    pytest.importorskip("gymnasium")
    from robot_sf.gym_env.environment_factory import make_robot_env

    variants = _variant_map()
    config = _real_config()
    campaign_runner.apply_variant(
        config, variants["robot_speed_band__bicycle_4_0_mps_micromobility"], seed=111
    )
    env = make_robot_env(config=config, seed=111, debug=False)
    try:
        robot = env.simulator.robots[0]
        assert isinstance(robot, BicycleDriveRobot)
        assert robot.config.max_velocity == pytest.approx(4.0)
        # The bicycle action space is [acceleration, steering]; the accel bound
        # carries the band-scaled braking-authority prerequisite (#4976).
        assert env.action_space.high[0] == pytest.approx(robot.config.max_accel)
        env.reset(seed=111)
        action = campaign_runner._env_action(env, {"v": 2.0, "omega": 1.5})
        # The runner consumes unicycle-style velocity commands, whereas the
        # bicycle environment consumes [acceleration, steering_angle]. A direct
        # omega-as-steering pass-through would be a different (and incorrect)
        # value for this non-zero target speed.
        assert action[0] == pytest.approx(robot.config.max_accel)
        assert action[1] == pytest.approx(math.atan(1.5 * robot.config.wheelbase / 2.0))
        assert action[1] != pytest.approx(1.5)
        obs, _reward, terminated, _truncated, _info = env.step(env.action_space.sample())
        assert obs is not None
        assert terminated is False
    finally:
        env.close()


def test_differential_drive_speed_band_sweeps_max_linear_speed() -> None:
    """The axis must also bind a raised cap to a differential-drive fallback."""
    config = _real_config()
    assert isinstance(config.robot_config, DifferentialDriveSettings)
    variant = campaign_runner.VariantSpec(
        axis="robot_speed_band",
        key="robot_speed_band__diff_3_0_mps",
        source_key="diff_3_0_mps",
        baseline=False,
        patch={
            "robot_config": {
                "type": "differential_drive",
                "max_linear_speed": 3.0,
                "max_linear_decel": 3.0,
            }
        },
        observation_noise={},
        runtime_binding="robot_config.drive_speed_cap",
    )
    campaign_runner.apply_variant(config, variant, seed=111)
    assert isinstance(config.robot_config, DifferentialDriveSettings)
    assert config.robot_config.max_linear_speed == pytest.approx(3.0)
    assert config.robot_config.max_linear_decel == pytest.approx(3.0)
    envelope = actuation_envelope_from_drive_config(config.robot_config)
    assert envelope is not None
    assert envelope["drive_model"] == "differential_drive"
    assert envelope["peak_forward_speed_m_s"] == pytest.approx(3.0)


def test_speed_band_axis_rejects_unsupported_drive_model() -> None:
    """A drive model outside bicycle/differential must fail closed."""
    config = SimpleNamespace(
        robot_config=SimpleNamespace(max_velocity=1.0, radius=0.3),  # not a drive settings type
        sim_config=SimulationSettings(),
    )
    variant = campaign_runner.VariantSpec(
        axis="robot_speed_band",
        key="robot_speed_band__holonomic",
        source_key="holonomic",
        baseline=False,
        patch={"robot_config": {"type": "holonomic", "max_velocity": 5.0}},
        observation_noise={},
        runtime_binding="robot_config.drive_speed_cap",
    )
    with pytest.raises(ValueError, match="only supports"):
        campaign_runner.apply_variant(config, variant, seed=111)


def test_run_episode_records_speed_band_metadata_with_actuation_envelope() -> None:
    """Episode rows must carry the resolved drive model, speed, and #4976 envelope."""
    variants = _variant_map()
    scenario = next(iter(load_scenarios(SCENARIO_SET)))
    row = campaign_runner.run_episode(
        scenario,
        scenario_path=SCENARIO_SET,
        variant=variants["robot_speed_band__bicycle_4_0_mps_micromobility"],
        planner_name="goal_seek",
        seed=111,
        horizon=8,
    )
    assert row["axis"] == "robot_speed_band"
    assert row["runtime_binding"] == "robot_config.drive_speed_cap"
    speed_band = row["speed_band"]
    assert speed_band["drive_model"] == "bicycle_drive"
    assert speed_band["peak_forward_speed_mps"] == pytest.approx(4.0)
    envelope = speed_band["actuation_envelope"]
    assert envelope["drive_model"] == "bicycle_drive"
    assert envelope["peak_forward_speed_m_s"] == pytest.approx(4.0)
    assert envelope["stopping_distance_envelope_m"] == pytest.approx(2.0)
    assert envelope["braking_distinct_from_accel"] is True
