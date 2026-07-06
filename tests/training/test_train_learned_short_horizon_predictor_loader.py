"""Fail-closed tests for the issue #4013 trainer CLI config loader.

These cover the ``_load_config`` hardening: a missing/non-file path, malformed
YAML, or a non-mapping YAML document must raise rather than silently fall back
to all-default values (which would mask a mistyped ``--config``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.training.train_learned_short_horizon_predictor_issue_4013 import _load_config

if TYPE_CHECKING:
    from pathlib import Path


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    """A path that does not exist fails closed instead of using defaults."""
    with pytest.raises(FileNotFoundError):
        _load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_directory_raises(tmp_path: Path) -> None:
    """A directory (not a file) fails closed."""
    with pytest.raises(FileNotFoundError):
        _load_config(tmp_path)


def test_load_config_non_mapping_yaml_raises(tmp_path: Path) -> None:
    """A YAML document that is not a top-level mapping fails closed."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_config(bad)


def test_load_config_malformed_yaml_raises(tmp_path: Path) -> None:
    """Malformed YAML raises a clean ValueError rather than tracebacking."""
    bad = tmp_path / "malformed.yaml"
    bad.write_text("key: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_config(bad)


def test_load_config_valid_mapping_parses_and_ignores_unknown(tmp_path: Path) -> None:
    """A valid mapping parses known keys and drops unknown ones."""
    good = tmp_path / "good.yaml"
    good.write_text("epochs: 3\nunknown_key: 99\n", encoding="utf-8")
    cfg = _load_config(good)
    assert cfg.epochs == 3
    assert not hasattr(cfg, "unknown_key")


def test_load_config_null_value_falls_back_to_default(tmp_path: Path) -> None:
    """An explicit ``null`` value keeps the dataclass default (no nullification)."""
    default = _load_config.__globals__["ShortHorizonTrainerConfig"]()
    cfg_file = tmp_path / "nulls.yaml"
    cfg_file.write_text("output_dir: null\ndevice: null\nepochs: 5\n", encoding="utf-8")
    cfg = _load_config(cfg_file)
    assert cfg.output_dir == default.output_dir
    assert cfg.device == default.device
    assert cfg.epochs == 5
