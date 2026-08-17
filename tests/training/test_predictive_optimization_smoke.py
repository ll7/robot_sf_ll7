"""Tests for the bounded issue #7254 predictive optimization comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.training import run_predictive_optimization_smoke as smoke


def _config() -> dict:
    """Load the committed issue #7254 comparison contract."""
    return smoke._read_mapping(
        Path("configs/training/predictive/predictive_optimization_smoke_issue_7254.yaml"),
        label="config",
    )


def _history(offset: float) -> list[dict[str, float]]:
    """Build a compact three-epoch history with one warm-up epoch."""
    return [
        {
            "epoch": float(epoch),
            "train_loss": 1.0 - 0.1 * epoch + offset,
            "val_loss": 1.2 - 0.08 * epoch + offset,
            "val_ade": 0.7 - 0.03 * epoch + offset,
            "val_fde": 1.1 - 0.04 * epoch + offset,
            "train_runtime_sec": 1.0,
            "val_runtime_sec": 0.2,
            "train_steps": 4.0,
            "val_steps": 1.0,
            "train_examples": 256.0,
            "val_examples": 64.0,
        }
        for epoch in range(1, 4)
    ]


def _result(arm: str, repeat: int, offset: float) -> dict:
    """Build the compact result shape consumed by ``_compare``."""
    return {
        "arm": arm,
        "repeat": repeat,
        "history": _history(offset),
        "throughput": {"examples_per_sec": 256.0, "steps_per_sec": 4.0},
        "peak_cuda_memory_allocated_bytes": 123,
    }


def test_committed_config_has_the_frozen_three_arm_contract() -> None:
    """The comparison config must remain explicit and structurally complete."""
    config = _config()
    smoke._validate_config(config)
    assert {arm["id"] for arm in config["arms"]} == {
        "fp32_control",
        "fp32_loader",
        "amp_loader",
    }
    assert config["comparison"]["warmup_epochs"] == 1


def test_fixture_generation_is_deterministic(tmp_path: Path) -> None:
    """The pinned fixture must have identical arrays and manifest digest across roots."""
    fixture = _config()["fixture"]
    first = smoke._write_fixture(fixture=fixture, output_dir=tmp_path / "first")
    second = smoke._write_fixture(fixture=fixture, output_dir=tmp_path / "second")

    assert first["dataset_sha256"] == second["dataset_sha256"]
    with np.load(first["path"]) as first_npz, np.load(second["path"]) as second_npz:
        for key in ("state", "target", "mask", "target_mask"):
            np.testing.assert_array_equal(first_npz[key], second_npz[key])
    manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "issue_7254_predictive_fixture_v1"


def test_compare_uses_control_repeat_dispersion_and_warmup() -> None:
    """Small arm differences inside the predeclared envelope remain equivalent smoke evidence."""
    config = _config()
    results = [
        _result("fp32_control", 1, 0.0),
        _result("fp32_control", 2, 0.0001),
        _result("fp32_loader", 1, 0.0006),
        _result("amp_loader", 1, 0.0008),
    ]

    comparison = smoke._compare(results=results, config=config)

    assert comparison["warmup_epochs_excluded"] == 1
    assert comparison["result_classification"] == "equivalent_smoke"
    assert comparison["equivalence_envelope"]["val_loss"] == 0.001
    assert comparison["arms"]["fp32_loader"]["equivalent_to_control"] is True
    assert comparison["arms"]["amp_loader"]["equivalent_to_control"] is True


def test_arm_training_args_keep_amp_and_loader_changes_explicit() -> None:
    """The runner must express the permitted arm differences through trainer flags."""
    config = _config()
    control = next(arm for arm in config["arms"] if arm["id"] == "fp32_control")
    amp = next(arm for arm in config["arms"] if arm["id"] == "amp_loader")

    control_args = smoke._arm_training_args(config=config, arm=control)
    amp_args = smoke._arm_training_args(config=config, arm=amp)

    assert "--amp" not in control_args
    assert "--pin-memory" not in control_args
    assert "--amp" in amp_args
    assert "--pin-memory" in amp_args
    assert "--persistent-workers" in amp_args
