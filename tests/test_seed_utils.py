from __future__ import annotations

import os
import random

import pytest

from robot_sf.common.seed import _import_torch, get_seed_state_sample, set_global_seed


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    """Reset torch deterministic knobs after each test to avoid bleed-over."""

    torch = _import_torch()
    if torch is None:
        yield
        return

    prev_algos = None
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        try:
            prev_algos = bool(torch.are_deterministic_algorithms_enabled())
        except Exception:  # pragma: no cover - defensive guard
            prev_algos = None

    prev_cudnn_det = getattr(getattr(torch, "backends", None), "cudnn", None)
    prev_det_flag = getattr(prev_cudnn_det, "deterministic", None) if prev_cudnn_det else None
    prev_bench_flag = getattr(prev_cudnn_det, "benchmark", None) if prev_cudnn_det else None

    yield

    try:
        if prev_algos is not None and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(prev_algos)
        if prev_cudnn_det is not None:
            if prev_det_flag is not None:
                prev_cudnn_det.deterministic = prev_det_flag
            if prev_bench_flag is not None:
                prev_cudnn_det.benchmark = prev_bench_flag
    except Exception:  # pragma: no cover - best effort cleanup
        pass


def test_set_global_seed_determinism():
    rep1 = set_global_seed(123, deterministic=True)
    s1 = get_seed_state_sample(n=5)
    rep2 = set_global_seed(123, deterministic=True)
    s2 = get_seed_state_sample(n=5)

    assert rep1.seed == rep2.seed == 123
    assert s1["random"] == s2["random"]
    assert s1["numpy"] == s2["numpy"]


essential = object()


def test_seed_changes_sequences():
    set_global_seed(111, deterministic=True)
    a = [random.random() for _ in range(3)]
    set_global_seed(222, deterministic=True)
    b = [random.random() for _ in range(3)]
    assert a != b


def test_matplotlib_headless_default():
    set_global_seed(0, deterministic=True)
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_torch_optional_behavior():
    torch = _import_torch()
    rep = set_global_seed(7, deterministic=True)
    if torch is not None:
        assert rep.has_torch is True
    else:
        assert rep.has_torch is False
