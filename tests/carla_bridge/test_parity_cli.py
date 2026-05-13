"""Tests for the CARLA oracle replay parity CLI."""

import json

from scripts.carla_bridge.compare_oracle_replay_metrics import main


def test_compare_oracle_replay_metrics_cli_writes_report(monkeypatch, tmp_path, capsys):
    """CLI should compare JSON objects and write a parity report."""
    robot_sf_path = tmp_path / "robot_sf.json"
    carla_path = tmp_path / "carla.json"
    output_path = tmp_path / "report.json"
    robot_sf_path.write_text(json.dumps({"metrics": {"success": True}}), encoding="utf-8")
    carla_path.write_text(json.dumps({"metrics": {"success": True}}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_oracle_replay_metrics.py",
            "--robot-sf",
            str(robot_sf_path),
            "--carla",
            str(carla_path),
            "--output",
            str(output_path),
        ],
    )

    assert main() == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["comparison_schema"] == "carla_oracle_replay_parity_v1"
    assert report["status"] == "comparable"
    assert "wrote CARLA parity report" in capsys.readouterr().out
