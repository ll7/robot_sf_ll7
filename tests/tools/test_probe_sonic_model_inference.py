"""Tests for the SoNIC model-only inference probe."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from scripts.tools.probe_sonic_model_inference import _render_markdown, run_model_probe

if TYPE_CHECKING:
    from pathlib import Path


def test_run_model_probe_missing_checkpoint(tmp_path: Path) -> None:
    """Missing checkpoint files should fail fast."""
    repo_root = tmp_path / "repo"
    (repo_root / "trained_models" / "SoNIC_GST" / "checkpoints").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        run_model_probe(repo_root=repo_root, model_name="SoNIC_GST", checkpoint="05207.pt")


def test_run_model_probe_reports_direct_and_shimmed_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe should distinguish direct import failure from shimmed success."""
    repo_root = tmp_path / "repo"
    checkpoint_path = repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fake-checkpoint")

    args = SimpleNamespace(
        env_name="CrowdSimPredRealGST-v0", num_processes=1, no_cuda=True, cuda=False
    )
    config = SimpleNamespace(
        sim=SimpleNamespace(
            human_num=2, human_num_range=1, predict_steps=5, predict_method="inferred"
        ),
        robot=SimpleNamespace(policy="selfAttn_merge_srnn", sensor="coordinates"),
        humans=SimpleNamespace(policy="orca"),
        action_space=SimpleNamespace(kinematics="holonomic"),
        env=SimpleNamespace(use_wrapper=True),
        policy=SimpleNamespace(constant_std=True),
    )

    class FakePolicy:
        def __init__(self, *_args, **_kwargs):
            if config.policy.constant_std:
                raise AssertionError("Torch not compiled with CUDA enabled")
            self.base = SimpleNamespace(human_node_rnn_size=128, human_human_edge_rnn_size=256)

        def load_state_dict(self, _state, strict=False):
            assert strict is False
            return ["dist.logstd._bias"], []

        def act(self, _obs, rnn_hxs, _masks, deterministic=False):
            assert deterministic is True
            return (
                torch.zeros((1, 1), dtype=torch.float32),
                torch.tensor([[-2.7, -0.28]], dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                rnn_hxs,
            )

    def fake_import_module(name: str):
        if name.endswith(".arguments"):
            return SimpleNamespace(get_args=lambda: args)
        if name.endswith(".configs.config"):
            return SimpleNamespace(Config=lambda: config)
        if name == "rl.networks.model":
            return SimpleNamespace(Policy=FakePolicy)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "scripts.tools.probe_sonic_model_inference.importlib.import_module", fake_import_module
    )
    monkeypatch.setattr(
        "scripts.tools.probe_sonic_model_inference.torch.load",
        lambda *_args, **_kwargs: {"weights": torch.tensor([1.0])},
    )

    report = run_model_probe(repo_root=repo_root, model_name="SoNIC_GST", checkpoint="05207.pt")

    assert report.direct_verdict == "direct model import blocked"
    assert "Torch not compiled with CUDA enabled" in (report.direct_failure_summary or "")
    assert report.shimmed_verdict == "model-only inference reproducible with shims"
    assert report.shimmed_failure_summary is None
    assert report.missing_state_keys == ["dist.logstd._bias"]
    assert report.unexpected_state_keys == []
    assert report.action_shape == [1, 2]
    assert report.value_shape == [1, 1]
    assert report.source_contract["action_kinematics"] == "holonomic"


def test_render_markdown_includes_shimmed_result() -> None:
    """Markdown output should surface the shimmed inference verdict."""
    report = SimpleNamespace(
        issue=626,
        repo_remote_url="https://github.com/tasl-lab/SoNIC-Social-Nav",
        model_name="SoNIC_GST",
        checkpoint="05207.pt",
        direct_verdict="direct model import blocked",
        direct_failure_summary="ModuleNotFoundError: No module named 'gym'",
        shimmed_verdict="model-only inference reproducible with shims",
        shimmed_failure_summary=None,
        shims_applied=["gymnasium as gym module alias"],
        missing_state_keys=["dist.logstd._bias"],
        unexpected_state_keys=[],
        action_sample=[-2.7, -0.28],
        action_shape=[1, 2],
        value_shape=[1, 1],
        source_contract={
            "robot_policy": "selfAttn_merge_srnn",
            "human_policy": "orca",
            "robot_sensor": "coordinates",
            "predict_method": "inferred",
            "action_kinematics": "holonomic",
            "env_use_wrapper": True,
            "env_name": "CrowdSimPredRealGST-v0",
        },
    )

    markdown = _render_markdown(report)

    assert "model-only inference reproducible with shims" in markdown
    assert "gymnasium as gym module alias" in markdown
    assert "holonomic" in markdown


def test_run_model_probe_restores_import_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe must not leak shimmed modules or repo-root imports into the caller state."""
    repo_root = tmp_path / "repo"
    checkpoint_path = repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fake-checkpoint")

    original_path = sys.path[:]
    sentinel_gym = object()
    sentinel_envs = object()
    sys.modules["gym"] = sentinel_gym  # type: ignore[assignment]
    sys.modules["rl.networks.envs"] = sentinel_envs  # type: ignore[assignment]

    args = SimpleNamespace(
        env_name="CrowdSimPredRealGST-v0", num_processes=1, no_cuda=True, cuda=False
    )
    config = SimpleNamespace(
        sim=SimpleNamespace(
            human_num=2, human_num_range=1, predict_steps=5, predict_method="inferred"
        ),
        robot=SimpleNamespace(policy="selfAttn_merge_srnn", sensor="coordinates"),
        humans=SimpleNamespace(policy="orca"),
        action_space=SimpleNamespace(kinematics="holonomic"),
        env=SimpleNamespace(use_wrapper=True),
        policy=SimpleNamespace(constant_std=True),
    )

    class FakePolicy:
        def __init__(self, *_args, **_kwargs):
            if config.policy.constant_std:
                raise AssertionError("Torch not compiled with CUDA enabled")
            self.base = SimpleNamespace(human_node_rnn_size=128, human_human_edge_rnn_size=256)

        def load_state_dict(self, _state, strict=False):
            return [], []

        def act(self, _obs, rnn_hxs, _masks, deterministic=False):
            return (
                torch.zeros((1, 1), dtype=torch.float32),
                torch.tensor([[-2.7, -0.28]], dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.float32),
                rnn_hxs,
            )

    def fake_import_module(name: str):
        if name.endswith(".arguments"):
            return SimpleNamespace(get_args=lambda: args)
        if name.endswith(".configs.config"):
            return SimpleNamespace(Config=lambda: config)
        if name == "rl.networks.model":
            return SimpleNamespace(Policy=FakePolicy)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "scripts.tools.probe_sonic_model_inference.importlib.import_module", fake_import_module
    )
    monkeypatch.setattr(
        "scripts.tools.probe_sonic_model_inference.torch.load",
        lambda *_args, **_kwargs: {"weights": torch.tensor([1.0])},
    )

    try:
        run_model_probe(repo_root=repo_root, model_name="SoNIC_GST", checkpoint="05207.pt")
        assert sys.modules["gym"] is sentinel_gym
        assert sys.modules["rl.networks.envs"] is sentinel_envs
        assert str(repo_root) not in sys.path
        assert sys.path == original_path
    finally:
        sys.modules["gym"] = sentinel_gym  # type: ignore[assignment]
        sys.modules["rl.networks.envs"] = sentinel_envs  # type: ignore[assignment]
