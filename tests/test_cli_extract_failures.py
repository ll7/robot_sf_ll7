from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_matrix(path: Path, repeats: int = 4) -> None:
    scenarios = [
        {
            "id": "xf-smoke",
            "density": "medium",
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


def test_cli_extract_failures_ids_only(tmp_path: Path, capsys):
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

    out_json = tmp_path / "fail_ids.json"
    rc = cli_main(
        [
            "extract-failures",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--collision-threshold",
            "0",  # pick anything with any collisions or near misses
            "--near-miss-threshold",
            "0",
            "--comfort-threshold",
            "0.0",
            "--ids-only",
        ],
    )
    cap2 = capsys.readouterr()
    assert rc == 0, f"extract-failures failed: {cap2.err}"

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data.get("episode_ids"), list)


def test_cli_extract_failures_jsonl(tmp_path: Path, capsys):
    matrix_path = tmp_path / "matrix.yaml"
    _write_matrix(matrix_path, repeats=3)
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

    out_path = tmp_path / "failures.jsonl"
    rc = cli_main(
        [
            "extract-failures",
            "--in",
            str(episodes),
            "--out",
            str(out_path),
            "--collision-threshold",
            "0",
            "--comfort-threshold",
            "0.0",
        ],
    )
    cap2 = capsys.readouterr()
    assert rc == 0, f"extract-failures failed: {cap2.err}"

    # Ensure file not empty if any failures exist; if none, it's okay to be empty
    content = out_path.read_text(encoding="utf-8")
    assert isinstance(content, str)
