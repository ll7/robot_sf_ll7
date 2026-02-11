"""Runtime-path tests for DreamerV3 RLlib launcher orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.training.train_dreamerv3_rllib as dreamer


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class _FakeRay:
    """Minimal ray facade used for run_training unit tests."""

    def __init__(self) -> None:
        self._initialized = False
        self.shutdown_called = False
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs) -> None:
        self._initialized = True
        self.init_kwargs = dict(kwargs)

    def is_initialized(self) -> bool:
        return self._initialized

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._initialized = False


class _FakeAlgo:
    """Simple algorithm stub exposing train/save/stop lifecycle hooks."""

    def __init__(self, *, fail_on_train: bool = False) -> None:
        self.fail_on_train = fail_on_train
        self.train_calls = 0
        self.stop_called = False

    def train(self) -> dict[str, object]:
        if self.fail_on_train:
            raise RuntimeError("train failed")
        self.train_calls += 1
        return {
            "episode_return_mean": float(self.train_calls),
            "num_env_steps_sampled_lifetime": self.train_calls * 10,
        }

    def save_to_path(self, checkpoint_dir: str) -> str:
        path = Path(checkpoint_dir) / f"checkpoint_{self.train_calls}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def stop(self) -> None:
        self.stop_called = True


class _FrozenDateTime:
    """Deterministic datetime replacement for reproducible run directories."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)
        return datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)


def test_apply_config_method_warns_when_payload_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing AlgorithmConfig hooks should emit warnings instead of silent drops."""
    warnings: list[str] = []
    monkeypatch.setattr(
        dreamer.logger,
        "warning",
        lambda msg, *args: warnings.append(str(msg).format(*args)),
    )

    cfg = object()
    result = dreamer._apply_config_method(cfg, "resources", {"num_gpus": 1})

    assert result is cfg
    assert warnings
    assert "algorithm.resources" in warnings[0]


def test_apply_env_runner_settings_warns_when_payload_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing env_runner/rollout hooks should emit warnings."""
    warnings: list[str] = []
    monkeypatch.setattr(
        dreamer.logger,
        "warning",
        lambda msg, *args: warnings.append(str(msg).format(*args)),
    )

    cfg = object()
    result = dreamer._apply_env_runner_settings(cfg, {"num_env_runners": 4})

    assert result is cfg
    assert warnings
    assert "algorithm.env_runners payload" in warnings[0]


def test_run_training_writes_summary_and_result_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """run_training should emit summary/result artifacts and clean up ray/algo lifecycle."""
    config_path = _write_yaml(
        tmp_path / "dreamer.yaml",
        f"""
experiment:
  run_id: smoke
  output_root: {tmp_path.as_posix()}/output
  train_iterations: 2
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
ray:
  num_cpus: auto
  num_gpus: auto
algorithm:
  framework: torch
  env_runners:
    num_env_runners: auto
  resources:
    num_gpus: auto
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgo()
    detect_calls = {"count": 0}
    register_calls: list[str] = []

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (
            fake_ray,
            object,
            lambda env_name, _creator: register_calls.append(env_name),
        ),
    )
    monkeypatch.setattr(
        dreamer,
        "detect_hardware_capacity",
        lambda **_kwargs: (
            detect_calls.__setitem__("count", detect_calls["count"] + 1)
            or SimpleNamespace(
                usable_cpus=8,
                visible_gpus=1,
                logical_cpus=8,
                allocated_cpus=8,
                allocated_gpus=1,
            )
        ),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dreamer, "_build_algorithm_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)

    exit_code = dreamer.run_training(run_config)

    assert exit_code == 0
    assert detect_calls["count"] == 1  # one shared probe for ray+algorithm auto settings
    assert register_calls == ["robot_sf_dreamerv3_smoke"]
    assert fake_algo.stop_called is True
    assert fake_ray.shutdown_called is True
    assert fake_ray.init_kwargs is not None
    assert fake_ray.init_kwargs["num_cpus"] == 8
    assert fake_ray.init_kwargs["num_gpus"] == 1

    run_dir = run_config.experiment.output_root / "smoke_20260211T120000Z"
    summary_path = run_dir / "run_summary.json"
    result_path = run_dir / "result.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    assert summary_path.exists()
    assert result_path.exists()
    assert checkpoint_dir.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == "smoke"
    assert summary["train_iterations"] == 2
    assert len(summary["history"]) == 2

    result_lines = result_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 2
    assert json.loads(result_lines[0])["iteration"] == 1
    assert json.loads(result_lines[1])["iteration"] == 2


def test_run_training_cleans_up_ray_and_algo_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lifecycle cleanup must happen even when algorithm training fails."""
    config_path = _write_yaml(
        tmp_path / "dreamer_fail.yaml",
        f"""
experiment:
  run_id: smoke_fail
  output_root: {tmp_path.as_posix()}/output
  train_iterations: 1
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
algorithm:
  framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgo(fail_on_train=True)

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (fake_ray, object, lambda _env_name, _creator: None),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dreamer, "_build_algorithm_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)

    with pytest.raises(RuntimeError, match="train failed"):
        dreamer.run_training(run_config)

    assert fake_algo.stop_called is True
    assert fake_ray.shutdown_called is True
