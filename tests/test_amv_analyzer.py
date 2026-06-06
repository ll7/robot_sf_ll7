"""Tests for the AMV timeout trace analyzer."""

import json

from scripts.analysis.amv_timeout_trace_analyzer import analyze_amv


def test_amv_analyzer_classification():
    """Classify the fixture timeout as command-speed constrained."""
    with open("tests/fixtures/amv_timeout_sample.jsonl") as f:
        rec = json.loads(f.readline())
    out = analyze_amv(rec)
    assert "timeout_driver" in out
    assert out["timeout_driver"] == "commands_too_conservative"
