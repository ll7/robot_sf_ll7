"""Config tests for the issue #4014 RecurrentPPO lane."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.training import train_recurrent_ppo

CONFIG_PATH = Path("configs/training/ppo/issue_4014_ppo_lstm_recurrent_smoke.yaml")


def _write_config(tmp_path: Path, updates: dict) -> Path:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(updates)
    target = tmp_path / "recurrent.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def test_recurrent_ppo_config_loads_expected_lstm_contract() -> None:
    """Smoke config should declare the true RecurrentPPO LSTM lane."""
    config = train_recurrent_ppo.load_recurrent_ppo_config(CONFIG_PATH)

    assert config.algorithm == "recurrent_ppo"
    assert config.base.policy_id == "ppo_lstm_recurrent_issue_4014_smoke"
    assert config.policy_kwargs["lstm_hidden_size"] == 128
    assert config.policy_kwargs["n_lstm_layers"] == 1
    assert config.policy_kwargs["enable_critic_lstm"] is True


def test_recurrent_ppo_config_rejects_wrong_algorithm(tmp_path: Path) -> None:
    """The dedicated entry point should fail closed for non-recurrent configs."""
    config_path = _write_config(tmp_path, {"algorithm": "ppo"})

    with pytest.raises(ValueError, match="algorithm must be 'recurrent_ppo'"):
        train_recurrent_ppo.load_recurrent_ppo_config(config_path)


def test_recurrent_ppo_config_rejects_unknown_hyperparameter(tmp_path: Path) -> None:
    """Unsupported RecurrentPPO hyperparameters should be actionable."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["recurrent_ppo_hyperparams"]["unsupported_knob"] = 1
    config_path = tmp_path / "recurrent.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported_knob"):
        train_recurrent_ppo.load_recurrent_ppo_config(config_path)


def test_recurrent_ppo_config_rejects_unknown_policy_kwarg(tmp_path: Path) -> None:
    """Unexpected LSTM policy kwargs should fail before training starts."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["recurrent_ppo_hyperparams"]["policy_kwargs"]["mystery_lstm_flag"] = True
    config_path = tmp_path / "recurrent.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="mystery_lstm_flag"):
        train_recurrent_ppo.load_recurrent_ppo_config(config_path)


def test_missing_sb3_contrib_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-dry-run import gate should name the optional extra."""

    def _missing(_name: str):
        raise ImportError("not installed")

    monkeypatch.setattr(train_recurrent_ppo.importlib, "import_module", _missing)

    with pytest.raises(RuntimeError, match="uv sync --extra recurrent"):
        train_recurrent_ppo._require_sb3_contrib()
