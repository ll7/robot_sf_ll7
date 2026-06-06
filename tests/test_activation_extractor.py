"""Tests for the static-recenter activation extractor."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.analysis.extract_activation_from_trace import extract_activation

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_extract_activation_fields():
    """Extract all required activation fields from a compact fixture."""
    with (FIXTURES / "static_recenter_sample.jsonl").open() as f:
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


def test_extract_activation_rejects_out_of_bounds_steps():
    """Malformed activation step offsets should produce null deltas, not crashes."""
    out = extract_activation(
        {
            "activations": [{"step": 99}],
            "progress": [0.0, 0.1],
            "trajectory": [[0.0, 0.0], [0.1, 0.0]],
        }
    )

    assert out["progress_delta_after_activation"] is None
    assert out["trajectory_delta"] is None


def test_extract_activation_reads_generic_events():
    """Generic activation events should count when the trace lacks activations."""
    out = extract_activation(
        {
            "events": [{"type": "activation", "step": 2}],
            "progress": [0.0, 0.0, 0.1, 0.2],
            "trajectory": [[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
        }
    )

    assert out["activation_count"] == 1
    assert out["first_activation_step"] == 2


def test_extract_activation_cli_row_id(tmp_path):
    """The --row-id CLI flag should filter matching rows."""
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"row_id": "skip", "activations": []}),
                json.dumps({"row_id": "keep", "activations": [{"step": 1}]}),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/extract_activation_from_trace.py",
            str(trace),
            "--row-id",
            "keep",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["activation_count"] == 1
