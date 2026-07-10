"""CLI tests for the ``flakiness-audit`` benchmark subcommand (issue #4978)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _write_episodes(path: Path) -> None:
    """Write a small episode JSONL with one stable and one knife-edge cell."""
    records = [
        # s1/ppo: unanimous success -> stable
        {"scenario_id": "s1", "algo": "ppo", "seed": 0, "metrics": {"success": 1}},
        {"scenario_id": "s1", "algo": "ppo", "seed": 1, "metrics": {"success": 1}},
        # s2/ppo: 50/50 split across seeds -> knife-edge
        {"scenario_id": "s2", "algo": "ppo", "seed": 0, "metrics": {"success": 1}},
        {"scenario_id": "s2", "algo": "ppo", "seed": 1, "metrics": {"success": 0}},
    ]
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_cli_flakiness_audit_writes_report(tmp_path: Path, capsys):
    """The subcommand loads JSONL, scores cells, and writes a JSON report."""
    episodes = tmp_path / "episodes.jsonl"
    _write_episodes(episodes)
    out_json = tmp_path / "flakiness.json"

    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(episodes),
            "--out",
            str(out_json),
            "--group-by",
            "algo",
        ],
    )
    cap = capsys.readouterr()
    assert rc == 0, f"flakiness-audit failed: {cap.err}"

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "scenario_flakiness.v1"
    assert report["summary"]["n_cells"] == 2
    assert report["summary"]["n_knife_edge_cells"] == 1
    # No exact-repeat data present -> determinism unknown, fail closed.
    assert report["exact_repeat"]["is_deterministic"] is None

    by_key = {c["cell_key"]: c for c in report["cells"]}
    assert by_key["s1::ppo"]["knife_edge"] is False
    assert by_key["s2::ppo"]["knife_edge"] is True


def test_cli_flakiness_audit_missing_input_fails_closed(tmp_path: Path):
    """A missing input path fails closed with a non-zero exit code."""
    out_json = tmp_path / "flakiness.json"
    rc = cli_main(
        [
            "flakiness-audit",
            "--in",
            str(tmp_path / "does_not_exist.jsonl"),
            "--out",
            str(out_json),
        ],
    )
    assert rc == 2
    assert not out_json.exists()
