"""Tests for the public API quickstart example (examples/quickstart/05_public_api.py)."""

from __future__ import annotations

import ast
import importlib
import json
import unittest.mock
from pathlib import Path

import pytest

quickstart_mod = importlib.import_module("examples.quickstart.05_public_api")

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "quickstart" / "05_public_api.py"


def test_public_api_quickstart_only_imports_top_level() -> None:
    """Verify the example imports all Robot SF symbols from the supported top-level package only."""
    assert EXAMPLE_SCRIPT.is_file(), f"{EXAMPLE_SCRIPT} must exist"
    tree = ast.parse(EXAMPLE_SCRIPT.read_text(encoding="utf-8"), filename=str(EXAMPLE_SCRIPT))

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)

    # Must import robot_sf
    assert any(mod == "robot_sf" for mod in imported_modules), (
        "Example must import the top-level 'robot_sf' package"
    )

    # Must NOT import internal robot_sf submodules
    internal_imports = [
        mod for mod in imported_modules if mod.startswith("robot_sf.") and mod != "robot_sf"
    ]
    assert not internal_imports, (
        f"Example contains internal imports: {internal_imports}. "
        "It must only import from the supported top-level package 'robot_sf'."
    )


def test_public_api_quickstart_execution_and_episode_record_roundtrip(tmp_path: Path) -> None:
    """Verify quickstart executes headlessly, saves/reloads EpisodeRecord, and computes digest."""
    summary = quickstart_mod.run_quickstart(
        output_dir=tmp_path,
        seed=42,
        horizon=3,
        output_format="friendly",
    )

    assert summary["seed"] == 42
    assert summary["horizon"] == 3
    assert summary["steps"] == 3.0
    assert summary["reloaded_verified"] is True
    assert summary["note"] == "Demonstration output only; not benchmark evidence."
    assert len(summary["record_digest"]) == 16

    saved_path = Path(summary["record_path"])
    assert saved_path.is_file()

    # Verify reload directly from file
    loaded = quickstart_mod.robot_sf.EpisodeRecord.load(saved_path)
    assert loaded.episode_id == summary["episode_id"]
    assert loaded.seed == 42
    assert loaded.horizon == 3
    assert loaded.metrics.get("steps") == 3.0


def test_public_api_quickstart_json_format_matches_facts(tmp_path: Path, capsys) -> None:
    """Friendly and JSON output formats contain the exact same facts."""
    summary = quickstart_mod.run_quickstart(
        output_dir=tmp_path,
        seed=42,
        horizon=3,
        output_format="json",
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["episode_id"] == summary["episode_id"]
    assert payload["scenario_id"] == summary["scenario_id"]
    assert payload["seed"] == summary["seed"]
    assert payload["horizon"] == summary["horizon"]
    assert payload["steps"] == summary["steps"]
    assert payload["total_reward"] == summary["total_reward"]
    assert payload["duration_s"] == summary["duration_s"]
    assert payload["record_path"] == summary["record_path"]
    assert payload["record_digest"] == summary["record_digest"]
    assert payload["reloaded_verified"] is True
    assert payload["note"] == summary["note"]


def test_public_api_quickstart_cli_main(tmp_path: Path, capsys) -> None:
    """The main entry point dispatches CLI arguments and exits cleanly."""
    rc = quickstart_mod.main(
        ["--output-dir", str(tmp_path), "--seed", "123", "--horizon", "2", "--format", "json"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["seed"] == 123
    assert payload["horizon"] == 2


def test_public_api_quickstart_closes_env_on_failure(tmp_path: Path) -> None:
    """The environment is guaranteed to close even when run_episode raises an exception."""
    mock_env = unittest.mock.MagicMock()

    with (
        unittest.mock.patch.object(quickstart_mod.robot_sf, "make_env", return_value=mock_env),
        unittest.mock.patch.object(
            quickstart_mod.robot_sf, "run_episode", side_effect=RuntimeError("simulated boom")
        ),
        pytest.raises(RuntimeError, match="simulated boom"),
    ):
        quickstart_mod.run_quickstart(
            output_dir=tmp_path,
            seed=42,
            horizon=2,
        )

    mock_env.close.assert_called_once()
