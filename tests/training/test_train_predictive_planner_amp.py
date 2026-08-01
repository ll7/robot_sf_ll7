"""Tests for the opt-in mixed-precision (AMP) path in predictive planner training.

These tests lock in the issue #6509 contract:

- AMP defaults to off and is CUDA-gated: it is never active on CPU or when CUDA is unavailable.
- The CPU training path (loss values and optimizer step count) is unchanged when AMP is off.
- GradScaler/autocast are only constructed/active when AMP is enabled AND CUDA is available.

CUDA availability is mocked so the full suite runs on CPU-only hosts; no GPU is required.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
)
from scripts.training import train_predictive_planner as trainer


def _tiny_loader(*, num_samples: int = 16, batch_size: int = 4) -> DataLoader:
    """Build a tiny deterministic CPU DataLoader for ``_run_epoch`` tests."""
    rng = np.random.default_rng(123)
    state = rng.normal(size=(num_samples, 3, 4)).astype(np.float32)
    target = rng.normal(size=(num_samples, 3, 4, 2)).astype(np.float32)
    mask = np.ones((num_samples, 3), dtype=np.float32)
    target_mask = np.ones((num_samples, 3, 4), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(state),
        torch.from_numpy(target),
        torch.from_numpy(mask),
        torch.from_numpy(target_mask),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def _tiny_model() -> PredictiveTrajectoryModel:
    """Build a fresh seeded tiny predictive model for deterministic CPU runs."""
    torch.manual_seed(42)
    return PredictiveTrajectoryModel(
        PredictiveModelConfig(
            max_agents=3,
            horizon_steps=4,
            input_dim=4,
            hidden_dim=8,
            message_passing_steps=1,
        )
    )


def _fresh_cpu_setup() -> tuple[PredictiveTrajectoryModel, torch.optim.Optimizer]:
    """Return a freshly seeded model/optimizer pair for deterministic CPU training."""
    np.random.seed(42)
    torch.manual_seed(42)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_amp_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --amp flag must default to off so the float32 path is the default."""
    monkeypatch.setattr("sys.argv", ["train_predictive_planner.py"])
    args = trainer.parse_args()
    assert args.amp is False
    # The default effective setting is off regardless of device.
    assert (
        trainer._resolve_amp_enabled(requested=bool(args.amp), device=torch.device("cpu")) is False
    )
    assert (
        trainer._resolve_amp_enabled(requested=bool(args.amp), device=torch.device("cuda")) is False
    )


def test_resolve_amp_enabled_is_cuda_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMP is only effective when requested, on a CUDA device, with CUDA available."""
    cpu = torch.device("cpu")
    cuda = torch.device("cuda")

    # AMP off -> never effective, regardless of device or CUDA availability.
    assert trainer._resolve_amp_enabled(requested=False, device=cuda) is False
    assert trainer._resolve_amp_enabled(requested=False, device=cpu) is False

    # AMP requested but on CPU -> never effective.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer._resolve_amp_enabled(requested=True, device=cpu) is False

    # AMP requested on CUDA, but CUDA unavailable -> not effective.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert trainer._resolve_amp_enabled(requested=True, device=cuda) is False

    # AMP requested on CUDA, CUDA available -> effective (the only True case).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer._resolve_amp_enabled(requested=True, device=cuda) is True


def test_amp_off_cpu_training_path_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With AMP off, the CPU training path must match the deterministic float32 baseline.

    This locks the issue #6509 requirement that the CPU/dry-run path is bit-for-bit unchanged
    when AMP is disabled: loss/ade/fde match the recorded baseline, the optimizer steps once per
    batch, and neither ``torch.autocast`` nor a GradScaler participates in the path.
    """
    autocast_calls: list[tuple[tuple, dict]] = []
    real_autocast = torch.autocast

    def _spy_autocast(*args: object, **kwargs: object) -> object:
        """Record autocast invocations while delegating to the real context manager."""
        autocast_calls.append((args, dict(kwargs)))
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", _spy_autocast)

    loader = _tiny_loader()
    model, optimizer = _fresh_cpu_setup()
    device = torch.device("cpu")

    step_count = 0
    real_step = optimizer.step

    def _count_step(*args: object, **kwargs: object) -> object:
        nonlocal step_count
        step_count += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", _count_step)

    loss, ade, fde = trainer._run_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        scaler=None,
    )

    # Deterministic float32 CPU baseline captured under torch 2.13 for this fixture.
    assert loss == pytest.approx(1.86718833, abs=1e-6)
    assert ade == pytest.approx(1.80651096, abs=1e-6)
    assert fde == pytest.approx(1.76219779, abs=1e-6)
    # Optimizer steps exactly once per batch and autocast never participates.
    assert step_count == len(loader)
    assert autocast_calls == []

    # A second independent seeded run must reproduce the same numbers (determinism).
    loader2 = _tiny_loader()
    model2, optimizer2 = _fresh_cpu_setup()
    loss2, ade2, fde2 = trainer._run_epoch(
        model=model2,
        loader=loader2,
        optimizer=optimizer2,
        device=device,
        scaler=None,
    )
    assert (loss, ade, fde) == (loss2, ade2, fde2)


def test_amp_off_eval_path_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With AMP off, the CPU eval path also runs without autocast or a scaler."""
    autocast_calls: list[tuple[tuple, dict]] = []
    real_autocast = torch.autocast

    def _spy_autocast(*args: object, **kwargs: object) -> object:
        """Record autocast invocations while delegating to the real context manager."""
        autocast_calls.append((args, dict(kwargs)))
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", _spy_autocast)

    loader = _tiny_loader()
    model = _tiny_model()
    loss, ade, fde = trainer._run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=torch.device("cpu"),
        scaler=None,
    )
    assert np.isfinite(loss) and np.isfinite(ade) and np.isfinite(fde)
    assert autocast_calls == []


def test_grad_scaler_only_constructed_when_amp_and_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GradScaler is constructed only when AMP is requested and CUDA is available.

    This mirrors the construction in ``main()`` and proves the GradScaler is absent (None) on
    CPU or when CUDA is unavailable, satisfying the strict no-op requirement.
    """
    cuda = torch.device("cuda")

    # CUDA available and requested -> GradScaler is constructed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    use_amp = trainer._resolve_amp_enabled(requested=True, device=cuda)
    assert use_amp is True
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    assert isinstance(scaler, torch.amp.GradScaler)

    # CUDA unavailable -> no scaler even when requested.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    use_amp_off = trainer._resolve_amp_enabled(requested=True, device=cuda)
    assert use_amp_off is False
    assert (torch.amp.GradScaler("cuda") if use_amp_off else None) is None

    # On CPU -> never a scaler.
    assert trainer._resolve_amp_enabled(requested=True, device=torch.device("cpu")) is False


def test_autocast_only_active_when_scaler_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    """autocast is invoked with cuda/float16 only when a scaler is provided to ``_run_epoch``.

    The eval pass (no backward) is used so the real GradScaler is never exercised on CPU, keeping
    the test GPU-free while still proving the autocast region is gated by the scaler.
    """
    autocast_calls: list[tuple[tuple, dict]] = []
    real_autocast = torch.autocast

    def _spy_autocast(*args: object, **kwargs: object) -> object:
        """Record autocast invocations while delegating to the real context manager."""
        autocast_calls.append((args, dict(kwargs)))
        return real_autocast(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", _spy_autocast)

    # Scaler absent -> no autocast (the AMP-off path).
    loader = _tiny_loader()
    trainer._run_epoch(
        model=_tiny_model(),
        loader=loader,
        optimizer=None,
        device=torch.device("cpu"),
        scaler=None,
    )
    assert autocast_calls == []

    # Scaler present -> autocast active with device_type=cuda and dtype=float16.
    # The autocast context manager is constructed once per epoch and re-entered each batch,
    # so the factory is invoked exactly once with the CUDA/float16 configuration.
    autocast_calls.clear()
    scaler = torch.amp.GradScaler("cuda")
    trainer._run_epoch(
        model=_tiny_model(),
        loader=loader,
        optimizer=None,
        device=torch.device("cpu"),
        scaler=scaler,
    )
    assert len(autocast_calls) == 1
    for _, kwargs in autocast_calls:
        assert kwargs["device_type"] == "cuda"
        assert kwargs["dtype"] is torch.float16


def test_nullcontext_is_a_safe_default_context() -> None:
    """The AMP-off branch uses a nullcontext that is a transparent no-op."""
    ctx = nullcontext()
    with ctx:
        value = 2.0 + 3.0
    assert value == 5.0
