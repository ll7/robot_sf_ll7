"""Real dev-seed rehearsal bytes exercise trace schema admission and coverage."""

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.step_trace_invariants import check_episode, invariant_coverage
from scripts.validation.check_step_trace_invariants import main

FIXTURES = Path(__file__).parents[1] / "fixtures/cmpreal"


def _rows():
    with gzip.open(FIXTURES / "rehearsal_traces.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def test_real_v2_trace_schema_is_supported():
    row = _rows()[-1]
    coverage = invariant_coverage(row)
    assert all("unsupported_trace_schema" not in c["issues"] for c in coverage.values())
    assert coverage["a_goal_heading"]["eligible"]
    # Old captured bytes never had the angular reset rate; do not guess it.
    assert coverage["b_drive_limits"]["issues"] == ["initial_angular_acceleration_unavailable"]


@pytest.mark.parametrize("version", [None, "simulation-step-trace.v3", "v2", 2])
def test_unknown_trace_version_does_not_run_numeric_checks(version):
    row = _rows()[0]
    row["algorithm_metadata"]["simulation_step_trace"]["schema_version"] = version
    violations, has_trace = check_episode(row, enabled=["a_goal_heading"])
    assert not violations
    assert not has_trace
    assert invariant_coverage(row)["a_goal_heading"]["issues"] == ["unsupported_trace_schema"]


def test_cli_rejects_mixed_trace_versions(tmp_path):
    row = _rows()[-1]
    other = deepcopy(row)
    other["algorithm_metadata"]["simulation_step_trace"]["schema_version"] = (
        "simulation-step-trace.v1"
    )
    file = tmp_path / "episodes.jsonl"
    file.write_text(json.dumps(row) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError, match="mixed trace schema versions"):
        main([str(file), "--invariants", "a_goal_heading"])
