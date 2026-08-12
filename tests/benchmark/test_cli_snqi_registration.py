"""Characterize benchmark CLI registration and the SNQI loader boundary."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import cli

EXPECTED_CORE_COMMANDS = [
    "baseline",
    "run",
    "summary",
    "aggregate",
    "metric-layers",
    "stress-coverage-report",
    "classify-failure-mechanisms",
    "collision-scenario-similarity",
    "claim",
    "validate-row-claims",
    "export-parquet",
    "analyze-cases",
    "admit-cases",
    "seed-variance",
    "flakiness-audit",
    "extract-failures",
    "snqi-ablate",
    "rank",
    "table",
    "export-canonical-table",
    "debug-seeds",
    "plot-pareto",
    "plot-distributions",
    "plot-planner-tradeoff",
    "plot-scenarios",
    "list-algorithms",
    "list-scenarios",
    "validate-config",
    "preview-scenarios",
    "planner-inclusion-check",
    "doctor",
    "mapf-oracle",
    "snqi",
]


def _root_subparser_action(parser: argparse.ArgumentParser):
    """Return the root parser action that owns benchmark subcommands."""
    return next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )


def _snqi_subparser_action(parser: argparse.ArgumentParser):
    """Return the nested parser action that owns SNQI subcommands."""
    snqi_parser = _root_subparser_action(parser).choices["snqi"]
    return next(
        action for action in snqi_parser._actions if isinstance(action, argparse._SubParsersAction)
    )


def test_core_registration_order_and_snqi_loader_contract() -> None:
    """Keep the refactored dispatcher and SNQI parser boundary stable."""
    parser = cli._configure_parser()

    root_subparsers = _root_subparser_action(parser)
    assert list(root_subparsers.choices) == EXPECTED_CORE_COMMANDS

    snqi_subparsers = _snqi_subparser_action(parser)
    assert list(snqi_subparsers.choices) == ["optimize", "recompute"]

    loader = parser.snqi_loader  # type: ignore[attr-defined]
    assert set(loader) == {"invoke_optimize", "invoke_recompute"}
    assert all(callable(callback) for callback in loader.values())


def test_snqi_registration_preserves_defaults_and_dispatch_metadata() -> None:
    """Preserve the parsed defaults used by the optimize and recompute scripts."""
    parser = cli._configure_parser()

    optimize = parser.parse_args(
        [
            "snqi",
            "optimize",
            "--episodes",
            "episodes.jsonl",
            "--baseline",
            "baseline.json",
            "--output",
            "output.json",
        ],
    )
    assert optimize.cmd == "snqi"
    assert optimize.snqi_cmd == "optimize"
    assert optimize.method == "both"
    assert optimize.grid_resolution == 5
    assert optimize.maxiter == 30
    assert optimize.bootstrap_samples == 0

    recompute = parser.parse_args(
        [
            "snqi",
            "recompute",
            "--episodes",
            "episodes.jsonl",
            "--baseline",
            "baseline.json",
            "--output",
            "output.json",
        ],
    )
    assert recompute.cmd == "snqi"
    assert recompute.snqi_cmd == "recompute"
    assert recompute.strategy == "default"
    assert recompute.export_pareto_front is False
    assert recompute.pareto_front_samples == 600
    assert recompute.decision_preflight is False
    assert recompute.decision_reversal_threshold == 0.0


def test_case_workbench_cli_handler_and_telemetry_config(tmp_path, monkeypatch) -> None:
    """The case command and telemetry config loader keep CLI failures explicit."""

    direct = tmp_path / "telemetry-direct.yaml"
    direct.write_text("analysis_trace: all\nplanner_debug_trace: none\n", encoding="utf-8")
    wrapped = tmp_path / "telemetry-wrapped.yaml"
    wrapped.write_text(
        "telemetry:\n  analysis_trace: all\n  planner_debug_trace: none\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "telemetry-invalid.yaml"
    invalid.write_text("- not-a-mapping\n", encoding="utf-8")

    assert cli._load_telemetry_config(str(direct)) == {
        "analysis_trace": "all",
        "planner_debug_trace": "none",
    }
    assert cli._load_telemetry_config(str(wrapped)) == {
        "analysis_trace": "all",
        "planner_debug_trace": "none",
    }
    assert cli._load_telemetry_config(None) is None
    with pytest.raises(ValueError, match="telemetry config must contain a YAML mapping"):
        cli._load_telemetry_config(str(invalid))

    calls = {}

    def fake_analyze_cases(**kwargs):
        calls.update(kwargs)
        return {"portfolio": [{"case_id": "case-1"}]}

    monkeypatch.setattr(cli, "analyze_cases", fake_analyze_cases)
    args = SimpleNamespace(
        config="config.yaml",
        result_store="episodes.jsonl",
        output=str(tmp_path / "package"),
        check_determinism=True,
    )
    assert cli._handle_analyze_cases(args) == 0
    assert calls["check_determinism"] is True

    def failing_analyze_cases(**_kwargs):
        raise ValueError("bad result store")

    monkeypatch.setattr(cli, "analyze_cases", failing_analyze_cases)
    assert cli._handle_analyze_cases(args) == 2
