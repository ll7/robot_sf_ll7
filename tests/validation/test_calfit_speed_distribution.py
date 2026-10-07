"""Dev-seed probe of real desired-speed assignment and separate integration cap."""

import numpy as np
import pytest

from scripts.validation import pedestrian_validation_10074 as suite


def test_calfit_normal_draw_survives_a_lower_execution_cap():
    cfg = suite.reused.candidate_config(suite.CANDIDATE, 1.3)
    cfg.scene_config.desired_speed_mean = 1.3
    cfg.scene_config.desired_speed_std = 0.2
    cfg.scene_config.desired_speed_seed = 1001
    cfg.scene_config.agent_radius = 0.25
    state = np.array([[0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.5]])
    _, _, before = suite.protocol_simulate(state, [], cfg, 20, speed_cap_m_s=0.2)
    positions, speeds, desired = suite.protocol_simulate(
        state, [], cfg, 20, speed_cap_m_s=0.2, desired_distribution=(1.29, 0.19), desired_seed=1001
    )
    print(
        "DEV1001 desired before/after",
        before.tolist(),
        desired.tolist(),
        "max actual speed",
        speeds.max(),
    )
    assert desired.tolist() == pytest.approx([1.467141262682885])
    assert desired[0] > 0.2
    assert speeds.max() <= 0.2 + 1e-12
    assert positions[-1, 0, 0] > 0
