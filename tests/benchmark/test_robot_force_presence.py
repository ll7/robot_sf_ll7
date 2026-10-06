"""Force-input validity through the production writer and SNQI consumers."""

from dataclasses import asdict, replace

import numpy as np
import pytest
from pysocialforce.config import SocialForceConfig

from robot_sf.benchmark.metrics import robot_force_metrics
from tests.benchmark.test_robot_attributable_force import CFG, _data


def post_loop(forces, *, force_counts=(1, 1)):
    from robot_sf.benchmark.map_runner.map_runner_episode import _compute_post_loop_metrics
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    return _compute_post_loop_metrics(
        robot_positions=[np.zeros(2), np.zeros(2)],
        robot_headings=[0.0, 0.0],
        ped_positions=[np.array([[2.0, 0.0]])] * 2,
        ped_forces=[np.zeros((1, 2))] * 2,
        robot_force_samples=[
            {
                "peds_pos": [[2.0 + index, 0.0] for index in range(count)],
                "forces": [forces] * count,
                "components": [{**CFG, "robot_pos": [0.0, 0.0]}],
                "social_force_config": asdict(SocialForceConfig()),
                "ped_radius_m": 0.35,
            }
            for count in force_counts
        ],
        visibility_trace=[],
        track_confidence_trace=[],
        visibility_evidence_statuses=[],
        visibility_evidence_reasons=[],
        reached_goal_step=None,
        collision_seen=False,
        ped_collision_seen=False,
        obstacle_collision_seen=False,
        robot_collision_seen=False,
        map_def=None,
        goal_vec=np.array([10.0, 0.0]),
        scenario={},
        config=RobotSimulationConfig(),
        horizon_val=2,
        record_forces=True,
        experimental_ped_impact=False,
        ped_impact_radius_m=2.0,
        ped_impact_window_steps=5,
    )


@pytest.mark.parametrize("forces", [[np.nan, np.nan], [np.nan, 1.0]])
def test_writer_refuses_nonfinite_force_for_present_pedestrian(forces):
    with pytest.raises(ValueError, match="present pedestrian"):
        post_loop(forces)
    valid = post_loop([3.0, 4.0]).metrics_raw
    assert valid["robot_force_impulse_total"] == pytest.approx(1.0)
    assert valid["robot_force_invalid_present_samples"] == 0


def test_absent_slot_requires_nan_padding():
    data = _data([[[2.0, 0.0]]])
    data.robot_force_config = CFG
    data.social_force_config = asdict(SocialForceConfig())
    data.robot_force_presence = np.array([[False]])
    data.robot_ped_forces = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="absent pedestrian"):
        robot_force_metrics(data)
    data.robot_ped_forces[:] = np.nan
    assert robot_force_metrics(data)["robot_force_impulse_total"] == 0.0


@pytest.mark.parametrize("invalid_count", [1, True, None], ids=["nonzero", "boolean", "missing"])
@pytest.mark.parametrize(
    "source", ["robot_force_impulse_total", "robot_force_pp_equiv_impulse_total"]
)
def test_snqi_refuses_invalid_present_sample_count(source, invalid_count):
    from robot_sf.benchmark.snqi.compute import normalize_snqi_v2_terms
    from tests.unit.benchmark.test_snqi_v2 import fixture_spec, metrics

    spec = replace(
        fixture_spec(),
        force_source=source,
        calibration_rho=0.95 if source.startswith("robot_force_pp_equiv") else 0.8,
    )
    row = metrics(
        robot_force_invalid_present_samples=0,
        robot_force_pp_equiv_invalid_present_samples=0,
        robot_force_metadata={
            "pp_equiv_status": "experimental_counterfactual",
            "pp_equiv_velocity_rule": "backward_difference_first_forward",
        },
    )
    row[source] = 0.0
    prefix = source.removesuffix("_impulse_total")
    if invalid_count is None:
        del row[prefix + "_invalid_present_samples"]
    else:
        row[prefix + "_invalid_present_samples"] = invalid_count
    with pytest.raises(ValueError, match="invalid present"):
        normalize_snqi_v2_terms(row, spec)

    row[prefix + "_invalid_present_samples"] = 0
    assert normalize_snqi_v2_terms(row, spec)["F"] == 0.0


def test_writer_refuses_counterfactual_nan_for_present_pedestrian(monkeypatch):
    from robot_sf.benchmark import metrics

    with monkeypatch.context() as patch:
        patch.setattr(
            metrics, "_pedestrian_pair_force", lambda delta, *args: np.full_like(delta, np.nan)
        )
        with pytest.raises(ValueError, match="present pedestrian"):
            post_loop([0.0, 0.0])
    assert post_loop([0.0, 0.0]).metrics_raw["robot_force_pp_equiv_invalid_present_samples"] == 0


def test_writer_presence_mask_reaches_both_force_reductions(monkeypatch):
    from robot_sf.benchmark import metrics
    from robot_sf.benchmark.map_runner import map_runner_episode

    expected = np.array([[True, True], [True, False]])
    masks = []
    original = metrics.robot_force_reductions

    def reduce(forces, **kwargs):
        masks.append(kwargs.get("presence"))
        return original(forces, **kwargs)

    def compute(data, **kwargs):
        np.testing.assert_array_equal(getattr(data, "robot_force_presence", None), expected)
        # Isolate stacking and consumer admission from counterfactual velocity estimation.
        monkeypatch.setattr(
            metrics, "robot_force_pp_equivalent", lambda episode: episode.robot_ped_forces
        )
        return metrics.robot_force_metrics(data)

    monkeypatch.setattr(map_runner_episode, "compute_all_metrics", compute)
    monkeypatch.setattr(metrics, "robot_force_reductions", reduce)
    result = post_loop([3.0, 4.0], force_counts=(2, 1)).metrics_raw
    assert len(masks) == 2
    for mask in masks:
        np.testing.assert_array_equal(mask, expected)
    assert result["robot_force_impulse_total"] == pytest.approx(1.5)
    assert result["robot_force_pp_equiv_impulse_total"] == pytest.approx(1.5)
