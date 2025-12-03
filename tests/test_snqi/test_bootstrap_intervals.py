# ruff: isort: skip_file

"""Bootstrap CI interval assertions via unified CLI.

Note: Import sorting in this small test file triggers a persistent isort
false-positive in our CI task ordering; we disable isort for this file only.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


BIN = ["uv", "run", "robot_sf_bench"]


def _write_synthetic_inputs(tmp_path: Path, n: int = 6) -> tuple[Path, Path]:
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        n: TODO docstring.

    Returns:
        TODO docstring.
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
    """TODO docstring. Document this function.

    Args:
        args: TODO docstring.

    Returns:
        TODO docstring.
    """
    env = os.environ.copy()
    # Ensure we do not trip any LIGHT_TEST fast path (want real code paths)
    env.pop("ROBOT_SF_SNQI_LIGHT_TEST", None)
    return subprocess.run(args, capture_output=True, text=True, check=False, env=env)


@pytest.mark.parametrize("cmd", ["optimize", "recompute"])  # keep runtime tiny
def test_bootstrap_ci_interval_contains_mean(tmp_path: Path, cmd: str):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        cmd: TODO docstring.
    """
    episodes, baseline = _write_synthetic_inputs(tmp_path)
    out = tmp_path / f"out_{cmd}.json"
    base = BIN + [
        "snqi",
        cmd,
        "--episodes",
        str(episodes),
        "--baseline",
        str(baseline),
        "--output",
        str(out),
        "--bootstrap-samples",
        "7",  # small but > 5 for a bit more stability
        "--bootstrap-confidence",
        "0.90",
        "--seed",
        "123",
        "--validate",
    ]
    if cmd == "optimize":
        base += ["--method", "grid", "--grid-resolution", "2", "--sample", "5"]
    else:
        base += ["--sample", "5"]
    cp = _run(base)
    assert cp.returncode == 0, cp.stderr
    data = json.loads(out.read_text())
    rs = data.get("bootstrap", {}).get("recommended_score")
    assert isinstance(rs, dict), "bootstrap.recommended_score block missing"
    # Structure checks
    assert rs.get("samples") == 7
    ci = rs.get("ci")
    assert isinstance(ci, list) and len(ci) == 2, f"bad ci: {ci!r}"
    lower, upper = float(ci[0]), float(ci[1])
    assert np.isfinite(lower) and np.isfinite(upper)
    assert lower <= upper
    mean_mean = float(rs.get("mean_mean"))
    assert np.isfinite(mean_mean)
    std_mean = float(rs.get("std_mean", -1))
    assert std_mean >= 0 and np.isfinite(std_mean)
    # Core assertion: bootstrap mean of means lies within the percentile CI
    assert lower - 1e-12 <= mean_mean <= upper + 1e-12, (
        f"mean_mean {mean_mean} not within CI [{lower}, {upper}]"
    )
    # Confidence level round-trip
    assert pytest.approx(0.90, rel=1e-6) == float(rs.get("confidence_level"))
