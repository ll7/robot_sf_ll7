"""Aggregate-summary integration tests for the scenario flakiness audit (#4978)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write benchmark records as newline-delimited JSON."""
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(scenario: str, seed: int, success: bool) -> dict[str, Any]:
    """Build one minimal aggregate and flakiness-audit input record."""
    return {
        "episode_id": f"{scenario}-{seed}-{success}",
        "scenario_id": scenario,
        "seed": seed,
        "scenario_params": {"algo": "planner-a"},
        "metrics": {"success": success, "score": float(success)},
    }


@pytest.mark.parametrize("bootstrap_samples", [0, 20])
def test_aggregate_can_embed_per_cell_flakiness_report(
    tmp_path: Path,
    bootstrap_samples: int,
) -> None:
    """Opt-in aggregate output should expose stability scores and knife-edge flags."""
    episodes = tmp_path / "episodes.jsonl"
    output = tmp_path / "summary.json"
    _write_records(
        episodes,
        [
            _record("stable", 1, True),
            _record("stable", 2, True),
            _record("knife-edge", 1, True),
            _record("knife-edge", 2, False),
        ],
    )

    args = [
        "aggregate",
        "--in",
        str(episodes),
        "--out",
        str(output),
        "--include-flakiness-audit",
    ]
    if bootstrap_samples:
        args.extend(["--bootstrap-samples", str(bootstrap_samples), "--bootstrap-seed", "7"])

    result = cli_main(args)

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    report = payload["_meta"]["scenario_flakiness"]
    assert report["schema_version"] == "scenario_flakiness.v1"
    assert report["summary"]["n_cells"] == 2
    assert report["summary"]["n_knife_edge_cells"] == 1
    cells = {cell["scenario_id"]: cell for cell in report["cells"]}
    assert cells["stable"]["stability_score"] == 1.0
    assert cells["stable"]["knife_edge"] is False
    assert cells["knife-edge"]["stability_score"] == 0.5
    assert cells["knife-edge"]["knife_edge"] is True


def test_aggregate_default_omits_flakiness_report(tmp_path: Path) -> None:
    """Existing aggregate output should remain unchanged unless the audit is requested."""
    episodes = tmp_path / "episodes.jsonl"
    output = tmp_path / "summary.json"
    _write_records(episodes, [_record("stable", 1, True)])

    result = cli_main(["aggregate", "--in", str(episodes), "--out", str(output)])

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "scenario_flakiness" not in payload["_meta"]


def test_aggregate_flakiness_opt_in_fails_without_outcomes(tmp_path: Path) -> None:
    """An opted-in audit should not emit a summary when no outcome evidence exists."""
    episodes = tmp_path / "episodes.jsonl"
    output = tmp_path / "summary.json"
    _write_records(
        episodes,
        [
            {
                "episode_id": "missing-outcome",
                "scenario_id": "scenario-a",
                "seed": 1,
                "scenario_params": {"algo": "planner-a"},
                "metrics": {"score": 1.0},
            }
        ],
    )

    result = cli_main(
        [
            "aggregate",
            "--in",
            str(episodes),
            "--out",
            str(output),
            "--include-flakiness-audit",
        ]
    )

    assert result == 2
    assert not output.exists()
