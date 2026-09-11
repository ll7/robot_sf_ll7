# ruff: isort: skip_file

"""Validate SNQI outputs against the machine-readable JSON Schema.

This test ensures both optimize and recompute flows emit JSON that conforms to
`docs/snqi-weight-tools/snqi_output.schema.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import jsonschema
import pytest


BIN = ["uv", "run", "robot_sf_bench"]


def _write_minimal_inputs(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    """Write synthetic episodes JSONL and baseline stats for SNQI runs.

    Args:
        tmp_path: Directory where the episode and baseline fixtures are written.
        n: Number of synthetic episode records to generate.

    Returns:
        Tuple of the episodes JSONL path and the baseline JSON path.
    """
    episodes = tmp_path / "episodes.jsonl"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "collisions": {"med": 0, "p95": 1},
                "near_misses": {"med": 0, "p95": 2},
                "force_exceed_events": {"med": 0, "p95": 1},
                "jerk_mean": {"med": 0.1, "p95": 0.5},
            },
        ),
        encoding="utf-8",
    )
    lines: list[str] = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "id": i,
                    "metrics": {
                        "collisions": 0 if i % 3 else 1,
                        "near_misses": i % 2,
                        "force_exceed_events": 0,
                        "jerk_mean": 0.1 + 0.05 * i,
                    },
                },
            ),
        )
    episodes.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return episodes, baseline


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run an argv command with the light-test env removed and no check.

    Args:
        args: Command argv passed to ``subprocess.run``.

    Returns:
        Completed process with captured text stdout and stderr.
    """
    env = os.environ.copy()
    # Ensure we do not trip any LIGHT_TEST fast path (want real code paths)
    env.pop("ROBOT_SF_SNQI_LIGHT_TEST", None)
    return subprocess.run(args, capture_output=True, text=True, check=False, env=env)


def _load_schema() -> dict:
    """Load the SNQI output JSON Schema from its repository path.


    Returns:
        Parsed JSON Schema dictionary.
    """
    schema_path = Path("docs/snqi-weight-tools/snqi_output.schema.json")
    assert schema_path.exists(), f"Schema file missing: {schema_path}"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cmd", ["optimize", "recompute"])  # keep runtime tiny
def test_snqi_outputs_conform_to_jsonschema(tmp_path: Path, cmd: str):
    """Assert SNQI optimize/recompute output conforms to the JSON Schema.

    Runs the CLI on minimal fixtures, checks a zero exit code, and validates the
    generated JSON against ``docs/snqi-weight-tools/snqi_output.schema.json``.

    Args:
        tmp_path: Temporary directory for the generated input and output files.
        cmd: Parametrized SNQI subcommand under test (``optimize`` or ``recompute``).
    """
    episodes, baseline = _write_minimal_inputs(tmp_path)
    out = tmp_path / f"snqi_{cmd}.json"
    base = [
        *BIN,
        "snqi",
        cmd,
        "--episodes",
        str(episodes),
        "--baseline",
        str(baseline),
        "--output",
        str(out),
        "--seed",
        "123",
    ]
    # Keep optimize path fast and deterministic
    if cmd == "optimize":
        base += ["--method", "grid", "--grid-resolution", "2", "--sample", "5"]
    else:
        base += ["--sample", "5"]

    cp = _run(base)
    assert cp.returncode == 0, cp.stderr or cp.stdout
    data = json.loads(out.read_text(encoding="utf-8"))

    schema = _load_schema()
    # Raises jsonschema.ValidationError on failure
    jsonschema.validate(instance=data, schema=schema)
