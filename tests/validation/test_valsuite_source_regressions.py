"""Exercise actual campaign rows on synthetic traces; also runnable on pre-fix runner.

The old runner produces numerical values with different definitions or null onset.
The replacements are independent known answers, not import-error witnesses.
"""

import numpy as np
import pytest

from scripts.validation import pedestrian_validation_10074 as suite


def _fake_trace(case):
    def trace(state, segments, config, steps, **kwargs):
        if case == "V1":
            t = np.arange(201) * 0.1
            v = 1.3 * (1 - np.exp(-t / 0.54))
            v[t > 6] = 1.6
            p = np.zeros((201, 1, 2))
            p[1:, 0, 0] = np.cumsum(v[1:]) * 0.1
            speeds = v[1:, None]
        elif case == "V2":
            t = np.arange(51) * 0.1
            p = np.zeros((51, 1, 2))
            p[:, 0, 0] = 4 + t - 0.04 * t * t
            speeds = (1 - 0.08 * t[1:])[:, None]
        elif case == "V4":
            t = np.arange(201) * 0.1
            x = t[:, None] - 0.5 - 0.02 * np.arange(len(state))[None, :]
            p = np.stack([x, np.zeros_like(x)], axis=-1)
            speeds = np.ones((200, len(state)))
        elif case == "V5":
            t = np.arange(161) * 0.1
            p = np.zeros((161, 1, 2))
            p[:, 0, 0] = t
            p[:, 0, 1] = 0.75
            speeds = np.ones((160, 1))
        else:
            speed = 1.15
            t = np.arange(106) * 0.1
            x = t * speed
            p = np.zeros((len(t), len(state), 2))
            p[:, 0, 0] = x
            if len(state) > 1:
                p[:, 0, 1] = np.clip(x - 4, 0, 1) * 0.4
                p[:, 1, 0] = 12 - x
            speeds = np.ones((len(t) - 1, len(state))) * speed
        return p, speeds

    return trace


@pytest.mark.parametrize(
    "case,variant,key,expected",
    [
        ("V1", "native", "fitted_desired_speed_m_s", 1.3),
        ("V2", "0.9", "speed_drop_m_s", 0.32),
        ("V4", "2.4", "all_data_specific_flow_persons_m_s", 350 / (349 * 0.02 * 2.4)),
        ("V5", "diagnostic", "lateral_cm_to_edge_m", 0.5),
        ("V6", "1.15", "onset_m", 2.99),
    ],
)
def test_source_case_known_answer_replaces_wrong_or_missing_estimator(
    monkeypatch, case, variant, key, expected
):
    fake = _fake_trace(case)
    monkeypatch.setattr(suite.reused.harness, "simulate", fake)

    def source_trace(*args, **kwargs):
        p, v = fake(*args, **kwargs)
        return p, v, np.ones(p.shape[1])

    monkeypatch.setattr(suite, "protocol_simulate", source_trace, raising=False)
    row = suite.run_task((case, 1001, variant, 0.4, "baseline"))
    # Fallback fields make the pre-fix failure a concrete incorrect numeric answer.
    fallback = (
        row.get("speed_m_s")
        if case == "V1"
        else row.get("flow_persons_s")
        if case == "V4"
        else None
    )
    assert row.get(key, fallback) == pytest.approx(expected, abs=1e-8), (
        f"{case} source quantity {key}"
    )


def test_all_cases_carry_pair_step_measure_not_initial_overlap_only(monkeypatch):
    fake = _fake_trace("V4")
    monkeypatch.setattr(suite.reused.harness, "simulate", fake)

    def source_trace(*args, **kwargs):
        p, v = fake(*args, **kwargs)
        return p, v, np.ones(p.shape[1])

    monkeypatch.setattr(suite, "protocol_simulate", source_trace, raising=False)
    row = suite.run_task(("V3", 1001, "1.0", 0.4, "baseline"))
    pair = row.get("pair_overlap", {}).get("non_group", {})
    assert pair.get("pair_steps") == 200 * 60 * 59 // 2
    assert pair["minimum_centre_distance_m"] == pytest.approx(0.02)
