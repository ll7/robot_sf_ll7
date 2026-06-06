"""Tests for the AMV timeout trace analyzer."""

import json
from pathlib import Path

from scripts.analysis.amv_timeout_trace_analyzer import analyze_amv

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_amv_analyzer_classification():
    """Classify the fixture timeout as command-speed constrained."""
    with (FIXTURES / "amv_timeout_sample.jsonl").open() as f:
        rec = json.loads(f.readline())
    out = analyze_amv(rec)
    assert "timeout_driver" in out
    assert out["timeout_driver"] == "commands_too_conservative"
    assert out["whether_actuation_aware_scoring_slowed_progress"] is True


def test_amv_analyzer_rejects_malformed_lists():
    """Malformed timeseries fields should not be treated as valid profiles."""
    out = analyze_amv(
        {
            "progress": [0.0, "bad"],
            "clipping": "not-a-list",
            "saturation": [False],
            "command_speeds": [0.01, {"bad": "value"}],
        }
    )

    assert out["progress_over_time"] is None
    assert out["clipping_over_time"] is None
    assert out["saturation_over_time"] is None
    assert out["command_speed_profile"] is None
    assert out["timeout_driver"] == "other_or_unclassified"
    assert out["whether_actuation_aware_scoring_slowed_progress"] is None


def test_amv_analyzer_preserves_recorded_slowdown_flag():
    """Explicit slowdown fields should override the local heuristic."""
    out = analyze_amv(
        {
            "progress": [0.0, 0.0],
            "command_speeds": [0.01, 0.01],
            "whether_actuation_aware_scoring_slowed_progress": False,
        }
    )

    assert out["whether_actuation_aware_scoring_slowed_progress"] is False
