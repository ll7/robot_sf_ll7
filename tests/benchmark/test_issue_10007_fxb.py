"""Controller regressions on real release map-runner observation bytes (seed 1001)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.planner import socnav_sampling_v2
from robot_sf.planner.socnav import (
    ORCAPlannerAdapter,
    SACADRLPlannerAdapter,
    SamplingPlannerAdapter,
    SocialForcePlannerAdapter,
    SocNavPlannerConfig,
)

FIXTURE = Path(__file__).parent / "fixtures/issue_10007_fxb/head_on_1001_reset.npz"


@pytest.fixture
def observation():
    """Load the first real policy observation from head-on corridor seed 1001."""
    with np.load(FIXTURE, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _sf():
    config = yaml.safe_load(Path("configs/algos/social_force_release_v0_0_8.yaml").read_text())
    return SocialForcePlannerAdapter(SocNavPlannerConfig(**config))


def test_social_force_uses_real_flat_sim_dt(observation):
    """The release runner's flat dt must drive integration, rather than tau=0.5."""
    assert _sf()._resolve_dt(observation) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "clocks, expected, source",
    [
        ({"dt": 0.2}, 0.2, "observation.dt"),
        ({"sim_timestep": 0.3, "dt": 0.2}, 0.3, "observation.sim_timestep"),
        (
            {"sim": {"timestep": 0.4}, "sim_timestep": 0.3, "dt": 0.2},
            0.4,
            "observation.sim.timestep",
        ),
    ],
)
def test_social_force_clock_source_and_precedence(observation, clocks, expected, source):
    """Accept the normalizer's dt alias and record the highest-priority clock."""
    observation.pop("sim_timestep")
    observation.update(clocks)
    adapter = _sf()
    assert adapter._resolve_dt(observation) == pytest.approx(expected)
    assert adapter._last_simulation_timestep == {"seconds": expected, "source": source}


def test_invalid_primary_clock_does_not_fall_through_to_dt(observation):
    """A valid alias must not conceal a corrupt higher-priority observed clock."""
    observation["sim_timestep"] = 0.0
    observation["dt"] = 0.2
    with pytest.raises(ValueError, match="timestep"):
        _sf()._resolve_dt(observation)


def test_social_force_flat_and_nested_timestep_actions_agree(observation):
    """Identical physical state and dt must yield identical first commands."""
    nested_dt = {**observation, "sim": {"timestep": observation["sim_timestep"]}}
    assert _sf().plan(observation) == pytest.approx(_sf().plan(nested_dt))


def _sacadrl(monkeypatch):
    adapter = SACADRLPlannerAdapter(SocNavPlannerConfig(max_linear_speed=2.0))
    # Isolate action decoding from checkpoint choice; 0.05 rad / 0.2 s = 0.25 rad/s.
    model = SimpleNamespace(actions=np.array([[1.0, 0.05]]), predict=lambda _: [[1.0]])
    monkeypatch.setattr(adapter, "_ensure_model", lambda: model)
    monkeypatch.setattr(adapter, "_build_network_input", lambda _: (np.zeros(138), 1.0, 10.0))
    return adapter


def test_sacadrl_uses_flat_nondefault_dt(observation, monkeypatch):
    """A real flat observation at dt=0.2 decodes a 0.05-rad action to 0.25 rad/s."""
    observation["sim_timestep"] = np.array([0.2], dtype=np.float32)
    assert _sacadrl(monkeypatch).plan(observation)[1] == pytest.approx(0.25)


def test_native_orca_solver_uses_flat_nondefault_dt(observation):
    """Check the real RVO2 solver clock, independent of the adapter's own diagnostics."""
    observation["sim_timestep"] = np.array([0.2], dtype=np.float32)
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(max_linear_speed=2.0))
    adapter.plan(observation)
    assert adapter._rvo2_sim.getTimeStep() == pytest.approx(0.2)


def test_sampling_rollouts_use_flat_nondefault_dt(observation, monkeypatch):
    """Every real rollout must receive the observed clock, even when commands tie."""
    observation["sim_timestep"] = np.array([0.2], dtype=np.float32)
    dts = []
    original = socnav_sampling_v2._rollout

    def observed_rollout(*args, **kwargs):
        dts.append(args[6] if len(args) > 6 else kwargs["dt"])
        return original(*args, **kwargs)

    monkeypatch.setattr(socnav_sampling_v2, "_rollout", observed_rollout)
    adapter = SamplingPlannerAdapter(SocNavPlannerConfig(socnav_sampling_version="bounded_v2"))
    adapter.plan(observation)
    assert dts
    np.testing.assert_allclose(dts, 0.2)


@pytest.mark.parametrize(
    "manifest",
    [
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml",
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml",
    ],
)
def test_orca_release_binding_sets_native_solver_speed_cap(observation, manifest):
    """Resolve real release manifest/config bytes through the actual policy builder."""
    matrix = yaml.safe_load((Path("configs/benchmarks") / manifest).read_text())
    arm = next(p for p in matrix["planners"] if p["key"] == "orca")
    config_path = arm.get("algo_config")
    config = yaml.safe_load(Path(config_path).read_text()) if config_path else {}
    policy, _ = _build_policy("orca", config)
    policy(observation)
    adapter = policy._planner_adapter
    assert adapter._rvo2_sim.getAgentMaxSpeed(adapter._rvo2_robot_id) == pytest.approx(2.0)


@pytest.mark.parametrize("planner", ["sacadrl", "sampling", "social_force", "orca"])
@pytest.mark.parametrize("bad_dt", [None, 0.0, -0.1, np.nan, np.inf])
def test_timestep_consumers_reject_missing_or_invalid_dt(observation, monkeypatch, planner, bad_dt):
    """Malformed observed time must fail at the contract, before generating commands."""
    observation.pop("sim_timestep")
    if bad_dt is not None:
        observation["sim"] = {"timestep": np.array([bad_dt])}
    adapters = {
        "sacadrl": lambda: _sacadrl(monkeypatch),
        "sampling": lambda: SamplingPlannerAdapter(
            SocNavPlannerConfig(socnav_sampling_version="bounded_v2")
        ),
        "social_force": _sf,
        "orca": lambda: ORCAPlannerAdapter(SocNavPlannerConfig(max_linear_speed=2.0)),
    }
    adapter = adapters[planner]()
    with pytest.raises(ValueError, match="timestep"):
        adapter.plan(observation)


def test_orca_occupancy_penalty_scales_requested_speed(observation, monkeypatch):
    """Below the cap, a 25% penalty reduces a 1 m/s aligned request to 0.75 m/s."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(max_linear_speed=2.0))
    heading = float(observation["robot_heading"][0])
    velocity = np.array([np.cos(heading), np.sin(heading)])
    # Keep direction aligned to isolate penalty application from heading slowdown.
    monkeypatch.setattr(
        adapter, "_get_safe_heading", lambda _pos, direction, _obs: (direction, 0.25)
    )
    linear, angular = adapter._velocity_world_to_command(
        velocity_world=velocity,
        robot_pos=observation["robot_position"],
        robot_heading=heading,
        observation=observation,
    )
    assert linear == pytest.approx(0.75)
    assert angular == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize(
    "angle, expected",
    [(0.0, 1.5), (np.pi / 4, 1.060660172), (np.pi / 2, 0.0), (3 * np.pi / 4, 0.0), (np.pi, 0.0)],
)
def test_orca_default_adapter_projects_forward_component(observation, angle, expected):
    """Release-default heading slowdown must bind even below the physical speed cap."""
    adapter = ORCAPlannerAdapter(SocNavPlannerConfig(max_linear_speed=2.0))
    # The actual observation bytes determine robot heading; remove the grid to
    # isolate differential-drive conversion from occupancy steering.
    observation = {
        key: value for key, value in observation.items() if not key.startswith("occupancy")
    }
    heading = float(observation["robot_heading"][0])
    velocity = 1.5 * np.array([np.cos(heading + angle), np.sin(heading + angle)])
    linear, angular = adapter._velocity_world_to_command(
        velocity_world=velocity,
        robot_pos=observation["robot_position"],
        robot_heading=heading,
        observation=observation,
    )
    assert linear == pytest.approx(expected, abs=1e-8)
    if angle >= np.pi / 2:
        assert abs(angular) == pytest.approx(1.0)
