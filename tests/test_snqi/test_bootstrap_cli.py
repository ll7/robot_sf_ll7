"""Bootstrap feature smoke tests for unified SNQI CLI.

These tests exercise the new --bootstrap-samples / --bootstrap-confidence flags in a
very small setting to keep runtime minimal. We generate a handful of synthetic
episodes with simple metric structure plus a baseline file. We then invoke both
`snqi optimize` and `snqi recompute` via the unified `robot_sf_bench` CLI and
confirm that a `bootstrap.recommended_score` block exists in the JSON output
with expected keys.

The optimization uses --method=grid with tiny grid resolution and --sample to
ensure speed. Recompute uses default strategy with sampling. Bootstrap sample
count kept low (5) to minimize runtime while still exercising code paths.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path  # noqa: TC003 - needed at runtime for pytest tmp_path fixture

import pytest

BIN = ["uv", "run", "robot_sf_bench"]


def _write_synthetic_inputs(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    episodes = tmp_path / "episodes.jsonl"
    baseline = tmp_path / "baseline.json"
    # Simple baseline med/p95 approximations
    baseline.write_text(
        json.dumps(
            {
                "collisions": {"med": 0, "p95": 1},
                "near_misses": {"med": 0, "p95": 2},
                "force_exceed_events": {"med": 0, "p95": 1},
                "jerk_mean": {"med": 0.1, "p95": 0.5},
            },
        ),
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
    episodes.write_text("\n".join(lines) + "\n")
    return episodes, baseline


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure we do not trip the LIGHT_TEST fast path (want real code)
    env.pop("ROBOT_SF_SNQI_LIGHT_TEST", None)
    return subprocess.run(args, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("cmd", ["optimize", "recompute"])
def test_bootstrap_block_present(tmp_path: Path, cmd: str):
    episodes, baseline = _write_synthetic_inputs(tmp_path)
    out = tmp_path / f"out_{cmd}.json"
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
        "--bootstrap-samples",
        "5",
        "--bootstrap-confidence",
        "0.90",
        "--seed",
        "42",
        "--validate",
    ]
    if cmd == "optimize":
        # Keep optimization trivial
        base += ["--method", "grid", "--grid-resolution", "2", "--sample", "5"]
    else:
        base += ["--sample", "5"]
    cp = _run(base)
    assert cp.returncode == 0, cp.stderr
    data = json.loads(out.read_text())
    assert "bootstrap" in data, f"bootstrap missing in output keys: {list(data.keys())}"
    rs = data["bootstrap"].get("recommended_score")
    assert isinstance(rs, dict), "recommended_score block missing"
    for key in ["samples", "mean_mean", "std_mean", "ci", "confidence_level"]:
        assert key in rs, f"Missing bootstrap field {key}"
    assert rs["samples"] == 5
    assert isinstance(rs["ci"], list) and len(rs["ci"]) == 2
