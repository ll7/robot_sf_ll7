"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark import baseline_stats
from robot_sf.benchmark import cli as benchmark_cli
from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_cli_baseline_subcommand(tmp_path: Path, capsys):
    # Build a minimal scenario matrix YAML
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "cli-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]
    # Write YAML list
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    out_json = tmp_path / "baseline.json"
    out_jsonl = tmp_path / "episodes.jsonl"

    # Run CLI main with arguments
    rc = cli_main(
        [
            "baseline",
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_json),
            "--jsonl",
            str(out_jsonl),
            "--schema",
            SCHEMA_PATH,
            "--base-seed",
            "0",
            "--repeats",
            "1",
            "--horizon",
            "8",
            "--dt",
            "0.1",
        ],
    )
    # Capture output (for sanity, not strictly required)
    captured = capsys.readouterr()
    assert rc == 0, f"CLI returned non-zero: {captured.err}"

    # Check outputs
    assert out_json.exists()
    assert out_jsonl.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    # Should contain at least some baseline keys
    assert "time_to_goal_norm" in data
    assert "collisions" in data


def test_cli_baseline_help_mentions_default_jsonl_path(capsys):
    """Verify baseline help advertises the current default JSONL path."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["baseline", "--help"])

    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert "output/benchmarks/baseline_episodes.jsonl" in captured.out
    assert "output/results/baseline_episodes.jsonl" not in captured.out


def test_run_and_compute_baseline_uses_default_jsonl_path_when_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify the helper uses the default JSONL path when none is provided."""
    seen: dict[str, object] = {}

    def fake_run_batch(*_args, **kwargs):
        seen["out_path"] = kwargs["out_path"]

    def fake_read_jsonl(path):
        seen["read_path"] = path
        return [{"metrics": {"collisions": 0}}]

    def fake_compute(records, metrics=None):
        seen["records"] = records
        seen["metrics"] = metrics
        return {"collisions": {"med": 0.0, "p95": 0.0}}

    monkeypatch.setattr(baseline_stats, "run_batch", fake_run_batch)
    monkeypatch.setattr(baseline_stats, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(baseline_stats, "compute_baseline_stats_from_records", fake_compute)

    out_json = tmp_path / "baseline.json"
    stats = baseline_stats.run_and_compute_baseline(
        [],
        out_json=out_json,
        schema_path=SCHEMA_PATH,
    )

    assert seen["out_path"] == str(baseline_stats.DEFAULT_BASELINE_JSONL_PATH)
    assert seen["read_path"] == str(baseline_stats.DEFAULT_BASELINE_JSONL_PATH)
    assert seen["records"] == [{"metrics": {"collisions": 0}}]
    assert stats == {"collisions": {"med": 0.0, "p95": 0.0}}
    assert json.loads(out_json.read_text(encoding="utf-8")) == stats


def test_cli_baseline_omitted_jsonl_forwards_default_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify the CLI forwards omitted --jsonl as the helper's default sentinel."""
    seen: dict[str, object] = {}

    def fake_run_and_compute_baseline(
        scenarios_or_path,
        *,
        out_json,
        out_jsonl,
        schema_path,
        **_kwargs,
    ):
        seen["scenarios_or_path"] = scenarios_or_path
        seen["out_json"] = out_json
        seen["out_jsonl"] = out_jsonl
        seen["schema_path"] = schema_path
        return {"collisions": {"med": 0.0, "p95": 0.0}}

    monkeypatch.setattr(
        benchmark_cli,
        "run_and_compute_baseline",
        fake_run_and_compute_baseline,
    )

    out_json = tmp_path / "baseline.json"
    rc = cli_main(
        [
            "baseline",
            "--matrix",
            "configs/baselines/example_matrix.yaml",
            "--out",
            str(out_json),
            "--schema",
            SCHEMA_PATH,
        ],
    )

    assert rc == 0
    assert seen["scenarios_or_path"] == "configs/baselines/example_matrix.yaml"
    assert seen["out_json"] == str(out_json)
    assert seen["out_jsonl"] is None
    assert seen["schema_path"] == SCHEMA_PATH


def test_cli_list_scenarios(tmp_path: Path, capsys):
    # Minimal scenario matrix YAML
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "id": "s1",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "repeats": 1,
        },
        {
            "id": "s2",
            "density": "med",
            "flow": "bi",
            "obstacle": "open",
            "repeats": 2,
        },
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    rc = cli_main(["list-scenarios", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()
    assert rc == 0
    # Check output includes both IDs
    assert "s1" in captured.out
    assert "s2" in captured.out


def test_cli_validate_config_success(tmp_path: Path, capsys):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {"id": "s1", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},
        {"id": "s2", "density": "med", "flow": "bi", "obstacle": "open", "repeats": 2},
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    rc = cli_main(["validate-config", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()
    assert rc == 0
    report = json.loads(captured.out)
    assert report["num_scenarios"] == 2
    assert report["errors"] == []


def test_cli_validate_config_marker_spawn_zero_density_is_not_empty_scene(tmp_path: Path, capsys):
    """Marker-spawned zero-density scenarios must not validate as empty scenes."""
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {
            "name": "marker_spawn",
            "map_file": "marker.svg",
            "simulation_config": {"max_episode_steps": 100, "ped_density": 0.0},
            "metadata": {
                "archetype": "bottleneck",
                "density": "low",
                "spawn_mode": "markers",
                "density_advisory": "zero_baseline_route_spawn",
            },
        }
    ]
    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"scenarios": scenarios}, f)

    rc = cli_main(["validate-config", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()

    assert rc == 0
    report = json.loads(captured.out)
    warning_messages = [warning["warning"] for warning in report["warnings"]]
    assert "ped_density=0.0 means no pedestrians spawn" not in warning_messages
    assert not any("ped_density outside recommended" in msg for msg in warning_messages)


def test_cli_validate_config_errors(tmp_path: Path, capsys):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        capsys: TODO docstring.
    """
    matrix_path = tmp_path / "matrix.yaml"
    scenarios = [
        {"id": "dup", "density": "low", "flow": "uni", "obstacle": "open", "repeats": 0},
        {"id": "dup", "density": "med", "flow": "bi", "obstacle": "open", "repeats": 1},
        {"density": "low", "flow": "uni", "obstacle": "open", "repeats": 1},  # missing id
    ]
    import yaml  # type: ignore

    with matrix_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)

    rc = cli_main(["validate-config", "--matrix", str(matrix_path)])
    captured = capsys.readouterr()
    assert rc != 0
    report = json.loads(captured.out)
    assert report["num_scenarios"] == 3
    assert len(report["errors"]) >= 2


def test_cli_list_algorithms_includes_random(capsys):
    """TODO docstring. Document this function.

    Args:
        capsys: TODO docstring.
    """
    rc = cli_main(["list-algorithms"])
    captured = capsys.readouterr()
    assert rc == 0
    # Built-in and baseline registry entries should appear
    assert "simple_policy" in captured.out
    assert "random" in captured.out
