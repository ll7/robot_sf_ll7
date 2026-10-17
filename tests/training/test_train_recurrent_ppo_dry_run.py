"""Dry-run manifest tests for the issue #4014 RecurrentPPO lane."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from scripts.training import train_recurrent_ppo

CONFIG_PATH = Path("configs/training/ppo/issue_4014_ppo_lstm_recurrent_smoke.yaml")


def test_recurrent_ppo_dry_run_writes_manifest(tmp_path: Path) -> None:
    """The real dry-run CLI records its LSTM contract without replacing worker log sinks."""
    handlers_before = logger._core.handlers.copy()
    level_before = logger._core.min_level
    result = subprocess.run(
        [
            sys.executable,
            str(Path(train_recurrent_ppo.__file__).resolve()),
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--run-id",
            "issue-4014-test",
            "--output-dir",
            str(tmp_path),
            "--log-level",
            "WARNING",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert logger._core.handlers == handlers_before
    assert logger._core.min_level == level_before

    manifest_path = tmp_path / "training_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr
    assert payload["schema_version"] == "recurrent-ppo-training-manifest.v2"
    assert payload["issue"] == 7847
    assert payload["algorithm"] == "recurrent_ppo"
    assert payload["policy"] == "MultiInputLstmPolicy"
    assert payload["dry_run"] is True
    assert payload["evidence_tier"] == "dry_run_smoke_prep"
    assert "not a full training comparison" in payload["claim_boundary"]
    assert payload["dependency"]["install_hint"] == "uv sync --extra recurrent"
    assert payload["lstm"] == {
        "enable_critic_lstm": True,
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "shared_lstm": False,
    }
    assert "Slurm or GPU submission" in payload["out_of_scope"]
    assert "full matched comparison campaign (#7846)" in payload["out_of_scope"]


def test_recurrent_ppo_config_uses_multi_input_lstm_policy() -> None:
    """Matched smoke config uses the non-nested dict RecurrentPPO policy."""
    config = train_recurrent_ppo.load_recurrent_ppo_config(
        "configs/training/ppo/issue_4014_recurrent_ppo_lstm_smoke_matched.yaml"
    )

    assert config.recurrent_policy == "MultiInputLstmPolicy"
    assert config.base.policy_id == "recurrent_ppo_lstm_issue_4014_smoke_matched"
