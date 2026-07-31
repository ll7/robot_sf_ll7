"""CPU tests for configurable DataLoader worker and pinned-memory settings.

These tests cover the predictive-planner loader-settings contract from issue
#6488 without requiring a GPU, a real dataset, or any external service:

- defaults preserve the legacy single-process, blocking CPU path;
- a configured positive worker count is deterministic and seeded;
- non-blocking transfer is enabled only on the pinned-memory CUDA path, never
  on CPU;
- the effective loader settings are recorded in ``training_summary.json``.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from scripts.training import train_predictive_planner as trainer

if TYPE_CHECKING:
    from pathlib import Path


def _resolve(
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    device: torch.device | str = "cpu",
) -> trainer.LoaderSettings:
    """Resolve loader settings with convenient test defaults."""
    torch_device = torch.device(device) if isinstance(device, str) else device
    return trainer._resolve_loader_settings(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        device=torch_device,
    )


def _make_arrays(
    *, n: int = 96, agents: int = 3, horizon: int = 6, seed: int = 0
) -> dict[str, np.ndarray]:
    """Build small non-degenerate predictive numpy arrays for loader tests."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, agents, 4)).astype(np.float32)
    base = rng.normal(size=(n, agents, 1, 2)).astype(np.float32)
    delta = rng.normal(size=(n, agents, horizon, 2)).astype(np.float32) * 0.3
    target = (base + np.cumsum(delta, axis=2)).astype(np.float32)
    mask = np.ones((n, agents), dtype=np.float32)
    target_mask = np.ones((n, agents, horizon), dtype=np.float32)
    return {
        "state": state,
        "target": target,
        "mask": mask,
        "target_mask": target_mask,
    }


def test_resolve_defaults_preserve_single_process_blocking_cpu_path() -> None:
    """Defaults must keep num_workers=0, pin_memory False, and blocking transfer on CPU."""
    settings = _resolve(device="cpu")

    assert settings.num_workers == 0
    assert settings.pin_memory is False
    assert settings.persistent_workers is False
    assert settings.prefetch_factor is None
    assert settings.non_blocking is False
    assert settings.device == "cpu"


@pytest.mark.parametrize(
    ("pin_memory", "device", "expected_non_blocking"),
    [
        (False, "cpu", False),
        (True, "cpu", False),
        (False, "cuda", False),
        (True, "cuda", True),
    ],
)
def test_non_blocking_gated_on_pinned_memory_and_cuda(
    pin_memory: bool, device: str, expected_non_blocking: bool
) -> None:
    """Pinning and non-blocking transfer must be enabled only for CUDA."""
    settings = _resolve(num_workers=2, pin_memory=pin_memory, device=device)

    assert settings.non_blocking is expected_non_blocking
    assert settings.pin_memory is (pin_memory and device == "cuda")
    if device == "cpu":
        assert settings.non_blocking is False


def test_positive_workers_enable_persistent_and_prefetch() -> None:
    """num_workers>0 must keep persistent_workers/prefetch_factor effective."""
    settings = _resolve(
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=4,
        device="cpu",
    )

    assert settings.num_workers == 2
    assert settings.persistent_workers is True
    assert settings.prefetch_factor == 4


def test_positive_workers_resolve_default_prefetch_for_manifest_provenance() -> None:
    """Positive-worker loaders must record PyTorch's default prefetch factor when omitted."""
    settings = _resolve(num_workers=2, prefetch_factor=None, device="cpu")

    assert settings.prefetch_factor == 2

    arrays = _make_arrays(n=16)
    train_loader, _val_loader = trainer._prepare_loaders(
        val_split=0.2,
        batch_size=4,
        seed=7,
        settings=settings,
        **arrays,
    )
    assert train_loader.prefetch_factor == 2


def test_zero_workers_force_off_persistent_and_prefetch() -> None:
    """num_workers=0 must drop persistent_workers/prefetch_factor to safe values."""
    settings = _resolve(
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=4,
        device="cpu",
    )

    assert settings.num_workers == 0
    assert settings.persistent_workers is False
    assert settings.prefetch_factor is None


def test_resolve_rejects_invalid_inputs() -> None:
    """Negative worker counts and prefetch factors below one must fail closed."""
    with pytest.raises(ValueError, match="num-workers"):
        _resolve(num_workers=-1)
    with pytest.raises(ValueError, match="prefetch-factor"):
        _resolve(num_workers=2, prefetch_factor=0, device="cpu")


def test_loader_settings_manifest_is_json_serializable() -> None:
    """The manifest view must serialize to JSON with the expected keys."""
    settings = _resolve(num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=3)

    manifest = settings.as_manifest()
    serialized = json.loads(json.dumps(manifest))

    assert set(serialized) == {
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "non_blocking",
        "device",
    }
    assert serialized == {
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 3,
        "non_blocking": False,
        "device": "cpu",
    }


def test_prepare_loaders_defaults_are_blocking_single_process() -> None:
    """Default loaders must be num_workers=0 with no persistent workers or generator."""
    arrays = _make_arrays()
    settings = _resolve(device="cpu")

    train_loader, val_loader = trainer._prepare_loaders(
        val_split=0.2,
        batch_size=16,
        seed=42,
        settings=settings,
        **arrays,
    )

    for loader in (train_loader, val_loader):
        assert loader.num_workers == 0
        assert loader.pin_memory is False
        assert loader.persistent_workers is False
        assert loader.prefetch_factor is None
        assert loader.generator is None
    # The default path must iterate without error (blocking CPU transfer).
    assert sum(len(state_b) for state_b, *_ in train_loader) > 0


def test_prepare_loaders_positive_workers_attach_seeded_init_and_generator() -> None:
    """num_workers>0 must attach the seeded worker_init_fn, generator, and kwargs."""
    arrays = _make_arrays()
    settings = _resolve(
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        device="cpu",
    )

    train_loader, val_loader = trainer._prepare_loaders(
        val_split=0.2,
        batch_size=16,
        seed=7,
        settings=settings,
        **arrays,
    )

    for loader in (train_loader, val_loader):
        assert loader.num_workers == 2
        assert loader.pin_memory is False
        assert loader.persistent_workers is True
        assert loader.prefetch_factor == 3
        assert loader.worker_init_fn is trainer._seeded_worker_init_fn
        assert loader.generator is not None


def test_prepare_loaders_positive_workers_are_deterministic() -> None:
    """Two loaders with the same worker count and seed must yield the same batches."""
    arrays = _make_arrays(n=32)
    settings = _resolve(num_workers=2, persistent_workers=False, prefetch_factor=2, device="cpu")

    def _collect() -> list[torch.Tensor]:
        """Materialize the train-loader batch order for one full pass."""
        train_loader, _val_loader = trainer._prepare_loaders(
            val_split=0.2,
            batch_size=8,
            seed=123,
            settings=settings,
            **arrays,
        )
        return [state_b.clone() for state_b, *_ in train_loader]

    first_pass = _collect()
    second_pass = _collect()

    assert len(first_pass) == len(second_pass)
    assert all(torch.equal(a, b) for a, b in zip(first_pass, second_pass, strict=True))


def test_run_epoch_forwards_non_blocking_flag_on_cpu_without_cuda() -> None:
    """_run_epoch must accept non_blocking and run on CPU regardless of the flag."""
    arrays = _make_arrays(n=48)
    settings = _resolve(device="cpu")
    train_loader, _val_loader = trainer._prepare_loaders(
        val_split=0.2,
        batch_size=16,
        seed=42,
        settings=settings,
        **arrays,
    )

    cfg = trainer.PredictiveModelConfig(
        max_agents=int(arrays["state"].shape[1]),
        horizon_steps=int(arrays["target"].shape[2]),
        input_dim=int(arrays["state"].shape[2]),
        hidden_dim=16,
        message_passing_steps=1,
        feature_schema_name="predictive_legacy_v1",
    )
    device = torch.device("cpu")
    model = trainer.PredictiveTrajectoryModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    blocking = trainer._run_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        non_blocking=False,
    )
    non_blocking = trainer._run_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        non_blocking=True,
    )

    for value, name in [(*blocking[:1], "blocking"), (*non_blocking[:1], "non_blocking")]:
        assert np.isfinite(value), f"{name} epoch loss must be finite on CPU"


def _write_non_degenerate_dataset(path: Path) -> None:
    """Write a small non-degenerate predictive NPZ dataset for end-to-end runs."""
    arrays = _make_arrays(n=96)
    np.savez(path, **arrays)


def test_training_summary_manifest_records_effective_loader_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """main() must record effective loader settings in training_summary.json."""
    dataset = tmp_path / "predictive_rollouts.npz"
    _write_non_degenerate_dataset(dataset)
    output_dir = tmp_path / "training_run"

    argv = [
        "train_predictive_planner.py",
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--seed",
        "42",
        "--hidden-dim",
        "16",
        "--message-passing-steps",
        "1",
        # Permissive gates keep the run green on random data without a quality claim.
        "--max-val-ade",
        "1e9",
        "--max-val-fde",
        "1e9",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    # Force the deterministic CPU path so the test does not require or depend on CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    return_code = trainer.main()

    assert return_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["data_loader"] == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "non_blocking": False,
        "device": "cpu",
    }
