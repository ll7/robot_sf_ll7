"""Tests for the topology score extractor."""

import json
from pathlib import Path

from scripts.analysis.extract_topology_scores import extract_topology

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_topology_extractor_fields():
    """Extract topology score fields from a compact fixture."""
    with (FIXTURES / "topology_sample.jsonl").open() as f:
        rec = json.loads(f.readline())
    out = extract_topology(rec)
    keys = [
        "per_frame_hypothesis_count",
        "alternative_hypothesis_count",
        "selected_hypothesis",
        "rejection_reason",
        "score_margin_to_primary_route",
        "switch_opportunity_count",
        "primary_vs_best_alternative_route_distance",
        "near_parity_gate_reason",
        "selected_static_clearance_min_m",
        "best_alternative_static_clearance_min_m",
    ]
    for k in keys:
        assert k in out
    assert out["selected_hypothesis"] == "primary_route"
    assert out["primary_vs_best_alternative_route_distance"] == 0.2
    assert out["near_parity_gate_reason"] == "eligible_near_parity_alternative"
    assert out["selected_static_clearance_min_m"] == 0.4
    assert out["best_alternative_static_clearance_min_m"] == 0.38


def test_topology_fallback_uses_selected_or_highest_score():
    """Score-component fallback should not blindly choose the first row."""
    selected = extract_topology(
        {
            "topology": "malformed",
            "topology_instrumentation": None,
            "score_components": [
                {"hypothesis": "first", "score": 0.1},
                {"hypothesis": "chosen", "score": 0.2, "selected": True},
            ],
        }
    )
    highest = extract_topology(
        {
            "score_components": [
                {"hypothesis": "first", "score": 0.1},
                {"hypothesis": "highest", "score": 0.2},
            ],
        }
    )

    assert selected["selected_hypothesis"] == "chosen"
    assert highest["selected_hypothesis"] == "highest"
