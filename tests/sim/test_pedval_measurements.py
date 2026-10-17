"""Source-defined flow plane and censored-run regression contracts; no stepping."""

import numpy as np

from scripts.validation import compare_obstacle_laws_10061 as flow


def test_seyfried_plane_and_supply_match_archived_source(monkeypatch):
    """Table 2 counts at the centre of a 2.8 m channel in a 4 m corridor."""
    captured = {}

    def observe(state, segments, config, steps, **kwargs):
        captured["segments"] = segments
        return np.stack([state[:, :2], state[:, :2]]), np.zeros((1, len(state)))

    monkeypatch.setattr(flow.harness, "simulate", observe)
    row = flow.bottleneck(("legacy_refit", 10.0, -0.57), 1001, 1.0)
    assert row["crossing_plane_m"] == 1.4, (
        "Seyfried y=0.4 is local video coordinate, not entrance offset"
    )
    assert captured["segments"][0][1] == 2.0, "Figure 2 upstream corridor is 4 m wide"
    assert row["persons"] == 60
    assert row["specific_flow_persons_m_s"] is None, (
        "an incomplete 60-person run has no published-estimator value"
    )
