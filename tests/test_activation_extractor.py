"""Tests for the static-recenter activation extractor."""

import json

from scripts.analysis.extract_activation_from_trace import extract_activation


def test_extract_activation_fields():
    """Extract all required activation fields from a compact fixture."""
    with open("tests/fixtures/static_recenter_sample.jsonl") as f:
        rec = json.loads(f.readline())
    out = extract_activation(rec)
    # required keys exist
    keys = [
        "activation_count",
        "first_activation_step",
        "selected_command_source",
        "command_source_changed",
        "progress_delta_after_activation",
        "trajectory_delta",
        "terminal_outcome_changed",
    ]
    for k in keys:
        assert k in out
    # sample-specific checks
    assert out["activation_count"] == 1
    assert out["first_activation_step"] == 5
