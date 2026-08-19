"""Tests for the #7319 predictive-baseline diagnostic contract."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from scripts.validation.run_predictive_baseline_diagnostic import run_diagnostic


def test_predictive_baseline_fixture_is_schema_valid_and_fail_closed() -> None:
    """The same-seed fixture emits method cards and no benchmark metrics."""

    root = Path(__file__).parents[2]
    config = yaml.safe_load(
        (root / "configs/benchmarks/issue_7319_predictive_baselines_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = run_diagnostic(config)
    schema = json.loads(
        (root / "robot_sf/benchmark/schemas/predictive_baseline_diagnostic.v1.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(report)
    assert report["simulator_executed"] is False
    assert report["benchmark_evidence"] is False
    assert report["campaign_approval_required"] is True
    assert {record["status"] for record in report["smoke_records"]} == {"smoke_pass"}
    assert all(record["deterministic"] for record in report["smoke_records"])
    assert len(report["methods"]) == 4
    assert all(len(card["config_digest"]) == 64 for card in report["methods"])
