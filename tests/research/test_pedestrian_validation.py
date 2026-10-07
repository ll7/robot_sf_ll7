"""Independent synthetic trajectory answers for source measurement definitions."""

import numpy as np
import pytest

from robot_sf.research import pedestrian_validation as m


def test_acceleration_fit_uses_early_phase_not_late_plateau():
    t = 0.35 + np.arange(201) * 0.1
    v = 1.3 * (1 - np.exp(-(t - 0.35) / 0.54))
    v[t > 6] = 2.1  # A late plateau must not replace the acceleration-fit speed.
    out = m.acceleration_fit(v, t)
    assert out["fitted_desired_speed_m_s"] == pytest.approx(1.3, abs=1e-8)
    assert out["fitted_tau_s"] == pytest.approx(0.54, abs=1e-8)


def test_aperture_temporal_speed_drop_known_linear_deceleration():
    t = np.arange(51) * 0.1
    p = np.column_stack([4 + t - 0.04 * t * t, np.zeros(len(t))])
    # v=1-.08t, approach [0,2] mean .92, passage t=5 speed .60.
    out = m.aperture_drop(p, t, plane_m=8)
    assert out["speed_drop_m_s"] == pytest.approx(0.32, abs=1e-10)
    assert out["capture_time_s"] == 0
    assert out["passage_time_s"] == pytest.approx(5)
    assert out["reduction_event"] is True
    stopped = p.copy()
    stopped[:, 0] = np.minimum(stopped[:, 0], 7)
    assert m.aperture_drop(stopped, t, plane_m=8)["speed_drop_m_s"] is None


def test_bottleneck_interpolation_and_stability_known_pipeline():
    t = np.arange(141) * 0.1
    x = t[:, None] - 0.5 - np.arange(12)[None, :]
    p = np.stack([x, np.zeros_like(x)], axis=-1)
    out = m.bottleneck_flow(p, t, width_m=2, plane_m=1, expected_n=12)
    assert out["source_crossing_times_s"] == pytest.approx(np.arange(12) + 1.5)
    assert out["all_data_specific_flow_persons_m_s"] == pytest.approx(6 / 11)
    assert out["steady_window_s"] == (1.0, 12.0)
    assert out["steady_specific_flow_persons_m_s"] == pytest.approx(0.5)
    short = m.bottleneck_flow(p[:50], t[:50], width_m=2, plane_m=1, expected_n=12)
    assert short["all_data_flow_persons_s"] is None
    assert short["steady_flow_persons_s"] is None


def test_circumvention_cm_to_cylinder_edge_not_body_edge():
    p = np.array([[-1, 0.75], [0, 0.75], [1, 0.75]])
    out = m.circumvention_clearance(p, [0, 1, 2], centre_xy=(0, 0), obstacle_radius_m=0.25)
    assert out["lateral_cm_to_edge_m"] == 0.5
    assert out["minimum_cm_to_edge_m"] == 0.5
    assert out["abeam_time_s"] == 1


def test_huber_filtered_straight_line_and_known_onset_coordinate():
    t = np.arange(101) * 0.1
    p = np.column_stack([t, np.clip(t - 3, 0, 1) * 0.4])
    q = np.column_stack([10 - t, np.zeros(len(t))])
    straight = np.column_stack([t, np.zeros(len(t))])
    np.testing.assert_allclose(m.huber_filtered(straight, t), straight, atol=1e-12)
    out = m.turning_onset(p, q, t, [straight] * 5)
    # Physical onset rejects Gaussian tails instead of saturating at the window edge.
    assert out["pomd_own_x_m"] == pytest.approx(5)
    assert out["onset_m"] == pytest.approx(2.6, abs=1e-8)
    assert out["onset_threshold_rad_s"] == 0.05
    assert len(out["baseline_maxima_rad_s"]) == 5
    assert m.turning_onset(straight, q, t, [straight] * 5)["onset_m"] is None


def test_pair_steps_strict_thresholds_group_split_and_empty_bank():
    p = np.zeros((3, 3, 2))
    p[0, :, 0] = [0, 2, 4]
    p[1, :, 0] = [0, 0.4, 1]
    p[2, :, 0] = [0, 0.56, 0.45]
    out = m.pair_overlap(p, 0.28, [[0, 1]])
    assert out["group"]["pair_steps"] == 2
    assert out["group"]["share_below_2r"] == 0.5
    assert out["group"]["share_below_0_45"] == 0.5
    assert out["non_group"]["pair_steps"] == 4
    assert out["non_group"]["share_below_2r"] == 0.5
    assert out["non_group"]["share_below_0_45"] == 0.25
    assert out["non_group"]["minimum_centre_distance_m"] == pytest.approx(0.11)
    assert m.pair_overlap(p[:, :1], 0.28)["all"]["share_below_2r"] is None
    with pytest.raises(ValueError, match="group membership"):
        m.pair_overlap(p, 0.28, [[0, 1], [1, 2]])
