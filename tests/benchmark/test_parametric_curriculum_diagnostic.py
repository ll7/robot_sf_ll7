"""Tests for the config-first parametric curriculum diagnostic CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.validation.run_parametric_curriculum_diagnostic import main, run

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/training/ppo/ablations/issue_7316_parametric_curriculum_smoke.yaml"
SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/parametric_curriculum_diagnostic.v1.json"


def test_parametric_curriculum_cli_emits_schema_valid_fixture(tmp_path: Path, monkeypatch) -> None:
    """The smoke command writes a deterministic report without executing training."""

    output = tmp_path / "curriculum.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_parametric_curriculum_diagnostic.py",
            "--config",
            str(CONFIG),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(report)) == []
    assert report["training_executed"] is False


def test_parametric_curriculum_run_is_repeatable() -> None:
    """The config-first report is byte-equivalent across repeated construction."""

    assert run(CONFIG) == run(CONFIG)
