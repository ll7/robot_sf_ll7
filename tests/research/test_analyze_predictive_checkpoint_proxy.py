"""Tests for the #3204 proxy-vs-ADE checkpoint-selection analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / "scripts" / "research" / "analyze_predictive_checkpoint_proxy.py"
_spec = importlib.util.spec_from_file_location("analyze_predictive_checkpoint_proxy", _TOOL)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _hist(pairs):
    return [
        {"epoch": float(i + 1), "val_ade": a, "success_rate": s} for i, (a, s) in enumerate(pairs)
    ]


def test_spearman_monotonic():
    """Spearman is 1.0 for a strictly increasing relation."""
    assert abs(mod.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9


def test_proxy_better_when_ade_min_has_low_success():
    """If the lowest-val_ade epoch has poor hard-success but another epoch is better, proxy wins."""
    # epoch 4 has the best (lowest) val_ade but low success; epoch 2 has higher success.
    hist = _hist([(1.0, 0.10), (0.9, 0.30), (0.8, 0.10), (0.7, 0.10)])
    rep = mod.analyze_proxy_history(hist)
    assert rep["verdict"] == "proxy_better"
    assert rep["proxy_selected"]["epoch"] == 2.0
    assert rep["ade_selected"]["epoch"] == 4.0
    assert rep["proxy_minus_ade_success"] > 0


def test_ade_adequate_when_it_picks_best_success():
    """If the lowest-val_ade epoch also has the highest success, ADE selection is adequate."""
    hist = _hist([(1.0, 0.10), (0.9, 0.20), (0.7, 0.40), (0.8, 0.30)])
    rep = mod.analyze_proxy_history(hist)
    assert rep["verdict"] == "ade_adequate"


def test_inconclusive_without_success_spread():
    """No spread in proxy success -> inconclusive (cannot distinguish selectors), fail-closed honesty."""
    hist = _hist([(1.0, 0.1), (0.9, 0.1), (0.8, 0.1), (0.7, 0.1)])
    rep = mod.analyze_proxy_history(hist)
    assert rep["verdict"] == "inconclusive"


def test_low_confidence_with_few_epochs():
    """Fewer than the correlation threshold of usable epochs is flagged low-confidence."""
    hist = _hist([(0.9, 0.30), (0.7, 0.10)])
    rep = mod.analyze_proxy_history(hist)
    assert rep["verdict"].endswith("low_confidence")


def test_analyze_summary_wraps_history():
    """analyze_summary reads proxy.history and adds report metadata."""
    summary = {
        "model_id": "m",
        "proxy": {
            "enabled": True,
            "history": _hist([(1.0, 0.1), (0.8, 0.3), (0.7, 0.1), (0.6, 0.1)]),
        },
    }
    rep = mod.analyze_summary(summary)
    assert rep["model_id"] == "m"
    assert rep["proxy_enabled"] is True
    assert rep["verdict"] == "proxy_better"
