"""Opt-in real release checkpoints: independent delta-then-release applied-command oracle.

ROBOT_SF_REAL_PPO_PROBE=1 enables artifact hydration and CPU inference.
These tests use seeds 1001 only, including guarded PPO's actual primary/guard path.
"""

import os

import numpy as np
import pytest

from scripts.validation.probe_ppo_release_action_semantics import CONFIGS, real_checkpoint_probe

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("ROBOT_SF_REAL_PPO_PROBE") != "1", reason="opt-in real artifacts"
    ),
]


@pytest.mark.parametrize("arm", CONFIGS)
def test_real_checkpoint_applied_delta_then_release_contract(arm, tmp_path):
    """At steps one and five the real plant matches an independently calculated reference."""
    import json

    # High initial omega exposes absolute/delta divergence immediately for both real models.
    result = real_checkpoint_probe(arm, (0.6, 0.95))
    (tmp_path / f"{arm}-probe.json").write_text(json.dumps(result, indent=2))
    assert result["saved_policy_tensors_equal"]
    contract = result["robot_contract"]
    assert contract == {
        "max_linear_speed": 2.0,
        "max_angular_speed": 1.0,
        "allow_backwards": False,
        "max_linear_accel": 1.0,
        "max_linear_decel": 1.0,
        "max_angular_accel": 1.0,
    }
    # Oracle does not call PPO's conversion, projection, env-action adapter or drive helpers.
    delta_state = np.asarray(result["initial_speed"], dtype=float)
    absolute_state = delta_state.copy()
    raw = np.asarray(result["raw"], dtype=float)
    for row in result["steps"]:
        delta_target = np.clip(delta_state + raw, [0.0, -1.0], [2.0, 1.0])
        delta_accel = np.clip((delta_target - delta_state) / 0.1, -1.0, 1.0)
        delta_state = np.clip(delta_state + 0.1 * delta_accel, [0.0, -1.0], [2.0, 1.0])
        absolute_target = np.clip(raw, [0.0, -1.0], [2.0, 1.0])
        absolute_accel = np.clip((absolute_target - absolute_state) / 0.1, -1.0, 1.0)
        absolute_state = np.clip(absolute_state + 0.1 * absolute_accel, [0.0, -1.0], [2.0, 1.0])
        if row["step"] in (1, 5):
            np.testing.assert_allclose(row["applied"], delta_state, rtol=0, atol=1e-12)
            assert not np.allclose(row["applied"], absolute_state, rtol=0, atol=1e-8)
        if arm == "guarded_ppo":
            assert row["shield"]["decision_label"] == "ppo_clear"
            np.testing.assert_allclose(row["shield"]["proposed_action"], delta_target, atol=1e-12)
