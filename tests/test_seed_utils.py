from __future__ import annotations

import os
import random

from robot_sf.utils.seed_utils import _import_torch, get_seed_state_sample, set_global_seed


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
