"""Tests confirming training scripts route through common.set_global_seed."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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


def test_predictive_planner_no_ad_hoc_seeding_in_source() -> None:
    """Old ad-hoc seeding calls are removed from predictive planner source."""
    text = Path("scripts/training/train_predictive_planner.py").read_text()
    assert "common.set_global_seed(args.seed)" in text
    assert "np.random.seed(args.seed)" not in text
    assert "torch.manual_seed(args.seed)" not in text
