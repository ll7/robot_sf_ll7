"""Tests for the issue #4012 offline-online orchestrator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.training import run_offline_online_rl


def test_orchestrator_runs_both_arms_and_writes_diagnostic_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orchestrator preserves diagnostic-only claim boundary."""

    offline_cfg = tmp_path / "offline.yaml"
    scratch_cfg = tmp_path / "scratch.yaml"
    offline_cfg.write_text("offline\n", encoding="utf-8")
    scratch_cfg.write_text("scratch\n", encoding="utf-8")
    experiment_cfg = tmp_path / "experiment.yaml"
    experiment_cfg.write_text(
        f"""offline_online_arm:
  config: {offline_cfg}
scratch_arm:
  config: {scratch_cfg}
output_dir: {tmp_path / "out"}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_offline_online_rl,
        "load_sac_training_config",
        lambda path: _TrainingConfig(enabled=Path(path) == offline_cfg),
    )
    monkeypatch.setattr(
        run_offline_online_rl,
        "run_sac_training",
        lambda config: (
            tmp_path / ("offline.zip" if config.offline_online.enabled else "scratch.zip")
        ),
    )

    summary = run_offline_online_rl.run_offline_online_experiment(experiment_cfg)

    assert summary.evidence_tier == "diagnostic-smoke-only"
    assert not summary.eligible_for_claim
    assert (tmp_path / "out" / "issue_4012_offline_online_summary.json").exists()
    assert (tmp_path / "out" / "issue_4012_offline_online_report.md").exists()


def test_orchestrator_fails_closed_when_scratch_arm_uses_offline_online(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scratch comparator must not use the offline warm-start block."""

    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        f"""offline_online_arm:
  config: {tmp_path / "offline.yaml"}
scratch_arm:
  config: {tmp_path / "scratch.yaml"}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_offline_online_rl,
        "load_sac_training_config",
        lambda _path: _TrainingConfig(enabled=True),
    )

    with pytest.raises(ValueError, match="scratch_arm"):
        run_offline_online_rl.run_offline_online_experiment(cfg)


class _OfflineOnline:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled


class _TrainingConfig:
    def __init__(self, *, enabled: bool) -> None:
        self.seed = 4012
        self.total_timesteps = 32
        self.offline_online = _OfflineOnline(enabled=enabled)


def test_smoke_dataset_materialization_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical smoke orchestrator creates local RLTrajectoryDataset inputs when absent."""

    dataset_path = tmp_path / "issue_4012_offline_online_rl_smoke" / "issue_4012_smoke.jsonl"
    manifest_path = (
        tmp_path / "issue_4012_offline_online_rl_smoke" / "issue_4012_smoke.manifest.json"
    )
    config = SimpleNamespace(
        scenario_config=tmp_path / "scenarios.yaml",
        seed=4012,
        offline_online=SimpleNamespace(
            enabled=True,
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            dataset_split="train",
            min_transitions=1,
        ),
    )

    monkeypatch.setattr(run_offline_online_rl, "load_scenarios", lambda _path: [{}])
    monkeypatch.setattr(run_offline_online_rl, "_build_env", lambda *_args, **_kwargs: _SmokeEnv())

    run_offline_online_rl._materialize_smoke_dataset_if_missing(config)

    assert dataset_path.exists()
    assert manifest_path.exists()
    assert "RLTrajectoryEpisode.v1" in dataset_path.read_text(encoding="utf-8")


class _SmokeSpace:
    shape = (2,)

    def sample(self) -> np.ndarray:
        return np.asarray([0.0, 0.0], dtype=np.float32)


class _SmokeEnv:
    action_space = _SmokeSpace()

    def reset(self) -> np.ndarray:
        return np.asarray([[0.0, 0.0]], dtype=np.float32)

    def step(self, _action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        return (
            np.asarray([[0.1, 0.0]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([True]),
            [{}],
        )

    def close(self) -> None:
        return None
