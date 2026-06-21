"""Contract tests for Social Mini-Game diagnostic metric documentation."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_social_mini_game_metric_contract_documents_expected_rows() -> None:
    """The Social Mini-Game metric contract should name every emitted diagnostic row."""
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "benchmarks"
        / "social_mini_game_metric_contract_v1.yaml"
    )

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "social-mini-game-metric-contract.v1"
    assert payload["row_schema_version"] == "social-mini-game-metrics.v1"
    assert payload["status_values"] == ["available", "unavailable", "undefined"]
    rows = {row["id"]: row for row in payload["metric_rows"]}
    assert set(rows) == {
        "makespan_ratio",
        "path_deviation_ratio",
        "deadlock_frequency",
        "flow_throughput",
        "distributional_inconvenience",
        "invasiveness",
    }
    assert rows["deadlock_frequency"]["denominator"] == "one episode"
    assert "arrival or exit counts" in rows["flow_throughput"]["missing_data_behavior"]
