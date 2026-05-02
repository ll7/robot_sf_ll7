"""Tests for the feature-extractor Optuna sweep launcher helpers."""

from __future__ import annotations

from types import SimpleNamespace

import scripts.training.optuna_feature_extractor as sweep


def test_sampler_seed_varies_by_worker_index() -> None:
    """Distributed workers should not all begin from the same sampler seed."""
    seeds = [sweep._sampler_seed(42, index) for index in range(20)]

    assert seeds[0] == 42
    assert len(set(seeds)) == len(seeds)
    assert sweep._sampler_seed(42, None) == 42


def test_classify_trial_failure_marks_deterministic_cuda_pooling() -> None:
    """The lightweight CNN CUDA determinism failure should be queryable from Optuna attrs."""
    exc = RuntimeError(
        "adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation"
    )

    failure_type, message = sweep._classify_trial_failure(exc)

    assert failure_type == "deterministic_cuda_kernel"
    assert "adaptive_avg_pool2d_backward_cuda" in message


def test_submit_slurm_jobs_passes_distinct_worker_indices(monkeypatch, tmp_path) -> None:
    """Each submitted worker command should carry its worker index and base seed."""
    monkeypatch.chdir(tmp_path)
    commands: list[list[str]] = []

    def fake_run(
        cmd: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> SimpleNamespace:
        commands.append(cmd)
        assert capture_output is True
        assert text is True
        assert check is False
        return SimpleNamespace(returncode=0, stdout="Submitted batch job 123", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)

    sweep._submit_slurm_jobs(
        n_trials=3,
        config="configs/training/ppo/feature_extractor_sweep_base.yaml",
        storage="sqlite:///output/optuna/feat_extractor/test.db",
        study_name="test_study",
        trial_timesteps=32_000,
        eval_every=16_000,
        eval_episodes=5,
        metric="eval_episode_return",
        slurm_time="00:10:00",
        slurm_gpus=1,
        slurm_cpus=2,
        slurm_mem="4G",
        slurm_partition=None,
        disable_wandb=True,
        log_level="WARNING",
        fps_warn_threshold=100.0,
        seed=123,
        extractor_exclude=["attention", "lstm"],
    )

    assert len(commands) == 3
    wrapped = [cmd[cmd.index("--wrap") + 1] for cmd in commands]
    assert [f"--worker-index {index}" in wrap for index, wrap in enumerate(wrapped)] == [
        True,
        True,
        True,
    ]
    assert all("--seed 123" in wrap for wrap in wrapped)
    assert all("--extractor-exclude attention lstm" in wrap for wrap in wrapped)
