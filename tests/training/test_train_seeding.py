"""Tests confirming training scripts route through common.set_global_seed."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_distributional_rl_calls_set_global_seed(monkeypatch) -> None:
    """run_distributional_rl_training calls common.set_global_seed with config.seed."""
    from robot_sf import common as robot_sf_common
    from scripts.training.train_distributional_rl import (
        load_distributional_rl_training_config,
        run_distributional_rl_training,
    )

    calls: list[int] = []
    monkeypatch.setattr(robot_sf_common, "set_global_seed", lambda seed, **kw: calls.append(seed))

    config = load_distributional_rl_training_config(
        "configs/training/distributional_rl/qr_dqn_issue_4016_smoke.yaml"
    )
    run_distributional_rl_training(config, dry_run=True)

    assert calls == [config.seed]


def test_distributional_rl_no_ad_hoc_seeding_in_source() -> None:
    """Old ad-hoc seeding calls are removed from distributional RL trainer source."""
    text = Path("scripts/training/train_distributional_rl.py").read_text()
    assert "common.set_global_seed(config.seed)" in text
    assert "random.seed(config.seed)" not in text
    assert "torch.manual_seed(config.seed)" not in text


def test_predictive_planner_calls_set_global_seed(monkeypatch, tmp_path: Path) -> None:
    """main() calls common.set_global_seed with args.seed."""
    from robot_sf import common as robot_sf_common
    from scripts.training import train_predictive_planner as trainer

    seed = 42
    calls: list[int] = []
    monkeypatch.setattr(robot_sf_common, "set_global_seed", lambda s, **kw: calls.append(s))
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)

    rng = np.random.RandomState(0)
    state = rng.randn(64, 1, 4).astype(np.float32)
    target = rng.randn(64, 1, 2, 2).astype(np.float32)
    mask = np.ones((64, 1), dtype=np.float32)
    target_mask = np.ones((64, 1, 2), dtype=np.float32)
    dataset_path = tmp_path / "dataset.npz"
    np.savez(dataset_path, state=state, target=target, mask=mask, target_mask=target_mask)

    monkeypatch.setattr(trainer, "_run_epoch", lambda **kw: (0.1, 0.2, 0.3))

    rc = trainer.main(
        [
            "--dataset",
            str(dataset_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--epochs",
            "1",
            "--seed",
            str(seed),
            "--hidden-dim",
            "8",
        ]
    )

    assert calls == [seed]
    assert rc == 0


def test_training_imports_seed_python_numpy_and_torch_deterministically() -> None:
    """Both trainers expose the canonical helper with deterministic CPU RNG behavior."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import random

import numpy as np
import torch

from scripts.training import train_distributional_rl as distributional
from scripts.training import train_predictive_planner as predictive

assert distributional.common.set_global_seed is predictive.common.set_global_seed

for trainer in (distributional, predictive):
    trainer.common.set_global_seed(314159)
    first = (random.random(), float(np.random.random()), torch.rand(4))
    trainer.common.set_global_seed(314159)
    second = (random.random(), float(np.random.random()), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )

    assert probe.returncode == 0, probe.stderr


def test_predictive_planner_injected_argv_preserves_missing_dataset_error(
    tmp_path: Path,
) -> None:
    """Programmatic argv keeps the CLI's existing missing-dataset exception."""
    from scripts.training import train_predictive_planner as trainer

    missing_dataset = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        trainer.main(["--dataset", str(missing_dataset)])


def test_predictive_planner_no_ad_hoc_seeding_in_source() -> None:
    """Old ad-hoc seeding calls are removed from predictive planner source."""
    text = Path("scripts/training/train_predictive_planner.py").read_text()
    assert "common.set_global_seed(args.seed)" in text
    assert "np.random.seed(args.seed)" not in text
    assert "torch.manual_seed(args.seed)" not in text
