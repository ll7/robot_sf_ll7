"""Regression checks for the issue #6158 offline-verdict validator."""

from __future__ import annotations

import numpy as np

from scripts.validation.check_issue_6158_topology_parallel_nmpc_offline_verdict import (
    _assess_pairwise_distinctness,
    _synth_diag,
)


def test_pairwise_distinctness_requires_rollouts_for_every_feasible_pair() -> None:
    """A missing feasible rollout cannot turn partial measurements into proof."""
    diagnostics = [
        _synth_diag("pass_left", feasible=True, objective=1.0),
        _synth_diag("yield_straight", feasible=True, objective=2.0),
        _synth_diag("pass_right", feasible=True, objective=3.0),
    ]
    states = {
        "pass_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
        "yield_straight": np.array([[0.01, 0.0], [0.01, 0.0]]),
    }

    feasible, pairwise, missing_pairs, min_sep, proves_distinctness = _assess_pairwise_distinctness(
        diagnostics, states
    )

    assert feasible == ["pass_left", "yield_straight", "pass_right"]
    assert pairwise == [{"pair": ["pass_left", "yield_straight"], "separation_m": 0.01}]
    assert missing_pairs == [["pass_left", "pass_right"], ["yield_straight", "pass_right"]]
    assert min_sep == 0.01
    assert not proves_distinctness
