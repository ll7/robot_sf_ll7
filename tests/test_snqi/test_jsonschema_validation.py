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
    """Write minimal inputs.

    Args:
        tmp_path: Auto-generated placeholder description.
        n: Auto-generated placeholder description.

    Returns:
        tuple[Path, Path]: Auto-generated placeholder description.
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
    """Run.

    Args:
        args: Auto-generated placeholder description.

    Returns:
        subprocess.CompletedProcess: Auto-generated placeholder description.
    """
    env = os.environ.copy()
    # Ensure we do not trip any LIGHT_TEST fast path (want real code paths)
    env.pop("ROBOT_SF_SNQI_LIGHT_TEST", None)
    return subprocess.run(args, capture_output=True, text=True, check=False, env=env)


def _load_schema() -> dict:
    """Load schema.

    Returns:
        dict: Auto-generated placeholder description.
    """
    schema_path = Path("docs/snqi-weight-tools/snqi_output.schema.json")
    assert schema_path.exists(), f"Schema file missing: {schema_path}"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("cmd", ["optimize", "recompute"])  # keep runtime tiny
def test_snqi_outputs_conform_to_jsonschema(tmp_path: Path, cmd: str):
    """Test snqi outputs conform to jsonschema.

    Args:
        tmp_path: Auto-generated placeholder description.
        cmd: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
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
