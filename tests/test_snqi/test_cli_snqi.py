"""Tests for unified `robot_sf_bench snqi` subcommands.

These run in a lightweight mode using the env var ROBOT_SF_SNQI_LIGHT_TEST=1 so
we avoid the heavy optimization/recompute workflows while still exercising:

* Argparse wiring for nested subcommands
* Dynamic module loader dispatch paths (guarded fast path)
* Expected exit codes (0 on success, 2 on misuse)

We fabricate minimal placeholder JSON/JSONL input files that the SNQI scripts would
normally parse; in LIGHT_TEST mode the early return happens before deep parsing, so
content is irrelevant but we keep them syntactically valid.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

BIN = ["uv", "run", "robot_sf_bench"]  # rely on project script entry via uv


@pytest.fixture()
def snqi_inputs(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    episodes = tmp_path / "episodes.jsonl"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "out.json"
    # Minimal syntactically valid placeholders
    episodes.write_text("{}\n")
    baseline.write_text(json.dumps({"dummy": 1}))
    yield {
        "episodes": episodes,
        "baseline": baseline,
        "output": output,
    }


def _run_cmd(
    args: list[str],
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """TODO docstring. Document this function.

    Args:
        args: TODO docstring.
        env_extra: TODO docstring.

    Returns:
        TODO docstring.
    """
    env = os.environ.copy()
    env["ROBOT_SF_SNQI_LIGHT_TEST"] = "1"  # ensure fast path
    if env_extra:
        env.update(env_extra)
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def test_snqi_optimize_fast_path(snqi_inputs: dict[str, Path]):
    """TODO docstring. Document this function.

    Args:
        snqi_inputs: TODO docstring.
    """
    cp = _run_cmd(
        [
            *BIN,
            "snqi",
            "optimize",
            "--episodes",
            str(snqi_inputs["episodes"]),
            "--baseline",
            str(snqi_inputs["baseline"]),
            "--output",
            str(snqi_inputs["output"]),
        ],
    )
    assert cp.returncode == 0, cp.stderr


def test_snqi_recompute_fast_path(snqi_inputs: dict[str, Path]):
    """TODO docstring. Document this function.

    Args:
        snqi_inputs: TODO docstring.
    """
    cp = _run_cmd(
        [
            *BIN,
            "snqi",
            "recompute",
            "--episodes",
            str(snqi_inputs["episodes"]),
            "--baseline",
            str(snqi_inputs["baseline"]),
            "--output",
            str(snqi_inputs["output"]),
        ],
    )
    assert cp.returncode == 0, cp.stderr


def test_snqi_missing_subcommand():
    """TODO docstring. Document this function."""
    cp = _run_cmd([*BIN, "snqi"])  # no subcommand
    # Argparse prints help and we expect non-zero exit (2 from our dispatcher)
    assert cp.returncode == 2, cp.stderr
    assert "optimize" in cp.stderr or "recompute" in (cp.stdout + cp.stderr)
