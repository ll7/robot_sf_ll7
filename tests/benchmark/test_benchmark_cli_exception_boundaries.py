"""Regression coverage for benchmark CLI exception-boundary contracts."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark import cli

if TYPE_CHECKING:
    from pathlib import Path


def _aggregate_args(input_path: Path, output_path: Path) -> argparse.Namespace:
    """Build the aggregate handler arguments needed by direct boundary tests."""
    return argparse.Namespace(
        in_path=str(input_path),
        out=str(output_path),
        group_by="scenario_params.algo",
        fallback_group_by="scenario_id",
        bootstrap_samples=0,
        bootstrap_confidence=0.95,
        bootstrap_seed=None,
        observation_track_mode="strict",
        snqi_weights=None,
        snqi_baseline=None,
        recompute_snqi=False,
    )


def test_missing_or_malformed_input_returns_cli_error(tmp_path: Path) -> None:
    """JSONL and YAML input errors keep their documented nonzero CLI results."""
    output_path = tmp_path / "summary.json"
    missing_code = cli.cli_main(
        ["aggregate", "--in", str(tmp_path / "missing.jsonl"), "--out", str(output_path)]
    )
    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_text("not-json\n", encoding="utf-8")
    malformed_code = cli.cli_main(
        ["aggregate", "--in", str(malformed_path), "--out", str(output_path)]
    )
    malformed_matrix = tmp_path / "malformed.yaml"
    malformed_matrix.write_text("scenarios: [unterminated\n", encoding="utf-8")
    malformed_yaml_code = cli.cli_main(["validate-config", "--matrix", str(malformed_matrix)])

    assert (missing_code, malformed_code, malformed_yaml_code) == (2, 2, 2)


def test_progress_optional_dependency_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable optional progress display cannot break a benchmark command."""
    original_import = cli.importlib.import_module

    def _missing_tqdm(name: str):
        if name == "tqdm":
            raise ModuleNotFoundError("tqdm unavailable")
        return original_import(name)

    monkeypatch.setattr(cli.importlib, "import_module", _missing_tqdm)

    progress = cli._progress_cb_factory(quiet=False)
    progress(1, 1, {"id": "scenario"}, 7, True, None)


def test_unexpected_collaborator_value_error_is_not_converted_to_cli_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A programmer ValueError propagates instead of becoming CLI exit code 2."""
    args = _aggregate_args(tmp_path / "episodes.jsonl", tmp_path / "summary.json")

    def _unexpected_error(_path: str) -> list[dict[str, object]]:
        raise ValueError("broken collaborator")

    monkeypatch.setattr(cli, "_agg_read_jsonl", _unexpected_error)

    with pytest.raises(ValueError, match="broken collaborator"):
        cli._handle_aggregate(args)


def test_logging_error_does_not_break_successful_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A progress/logging failure remains isolated from a successful aggregate result."""
    output_path = tmp_path / "summary.json"
    args = _aggregate_args(tmp_path / "episodes.jsonl", output_path)
    monkeypatch.setattr(cli, "_agg_read_jsonl", lambda _path: [])
    monkeypatch.setattr(cli, "_agg_compute", lambda *_args, **_kwargs: {"_meta": {}})

    def _logging_failure(*_args: object, **_kwargs: object) -> None:
        raise OSError("logging sink unavailable")

    monkeypatch.setattr(cli.logging, "info", _logging_failure)

    assert cli._handle_aggregate(args) == 0
    assert output_path.exists()
