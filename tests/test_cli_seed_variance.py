"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 3) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        repeats: TODO docstring.
    """
    scenarios = [
        {
            "id": "sv-smoke",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        },
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


def test_cli_seed_variance(tmp_path: Path, capsys):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=4)
    episodes = tmp_path / "episodes.jsonl"

    rc_run = cli_main(
        [
            "run",
            "--matrix",
            str(matrix_path),
            "--out",
            str(episodes),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--horizon",
            "6",
            "--dt",
            "0.1",
        ],
    )
    cap = capsys.readouterr()
    assert rc_run == 0, f"run failed: {cap.err}"

    out_json = tmp_path / "seed_var.json"
    rc = cli_main(
        [
            "seed-variance",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "scenario_id",
        ],
    )
    cap2 = capsys.readouterr()
    assert rc == 0, f"seed-variance failed: {cap2.err}"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "sv-smoke" in data
    # Pick any metric (e.g., success) and ensure cv key present
    any_metric = next(iter(data["sv-smoke"].keys()))
    stats = data["sv-smoke"][any_metric]
    assert all(k in stats for k in ("mean", "std", "cv", "count"))
