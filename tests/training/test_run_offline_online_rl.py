"""Tests for the issue #4012 offline-online orchestrator."""

from __future__ import annotations

from pathlib import Path

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
