"""Proxy-vs-ADE checkpoint-selection analysis for the predictive planner (#3204).

`train_predictive_planner.py` can evaluate a hard-seed proxy every N epochs and records, per
proxy epoch, the trajectory-error metric (`val_ade`) and the proxy benchmark success
(`success_rate`). This tool consumes that training summary and answers the #3204 question:

    Does selecting a checkpoint by a rollout-derived proxy pick a higher hard-case benchmark
    success than selecting by trajectory error (val_ade)?

It reports, across the proxy-evaluated epochs:
- the val_ade-selected epoch (min val_ade) and its proxy hard-success;
- the proxy-selected epoch (max proxy success) and its proxy hard-success;
- the Spearman rank correlation between val_ade and proxy success (no scipy dependency);
- a verdict: whether proxy selection beats ADE selection on hard-case success.

It is honest about insufficient evidence: if there are too few proxy epochs or no spread in
proxy success, it returns an `inconclusive` verdict rather than a misleading correlation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_MIN_EPOCHS_FOR_CORRELATION = 4


def _rank(values: list[float]) -> list[float]:
    """Return average ranks (1-based) of ``values``, ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Return the Pearson correlation of two equal-length sequences, or None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx**0.5 * vy**0.5)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Return the Spearman rank correlation (Pearson on ranks), or None if undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def analyze_proxy_history(
    history: list[dict[str, Any]], *, success_margin: float = 1e-9
) -> dict[str, Any]:
    """Analyze proxy-vs-ADE selection from a list of per-epoch proxy records."""
    usable = [
        h for h in history if h.get("val_ade") is not None and h.get("success_rate") is not None
    ]
    if len(usable) < 2:
        return {
            "verdict": "inconclusive",
            "reason": "fewer than 2 usable proxy epochs",
            "n": len(usable),
        }

    ades = [float(h["val_ade"]) for h in usable]
    succ = [float(h["success_rate"]) for h in usable]

    ade_sel = min(usable, key=lambda h: float(h["val_ade"]))
    proxy_sel = max(usable, key=lambda h: float(h["success_rate"]))

    rho = spearman(ades, succ)
    success_spread = max(succ) - min(succ)
    beats = float(proxy_sel["success_rate"]) > float(ade_sel["success_rate"]) + success_margin

    if success_spread <= success_margin:
        verdict = "inconclusive"
        reason = "no spread in proxy hard-success across epochs; cannot distinguish selectors"
    elif len(usable) < _MIN_EPOCHS_FOR_CORRELATION:
        verdict = "proxy_better_low_confidence" if beats else "ade_adequate_low_confidence"
        reason = f"only {len(usable)} proxy epochs; correlation not robust"
    elif beats:
        verdict = "proxy_better"
        reason = "proxy-selected epoch has higher hard-success than the val_ade-selected epoch"
    else:
        verdict = "ade_adequate"
        reason = "val_ade-selected epoch already matches or beats proxy selection on hard-success"

    return {
        "verdict": verdict,
        "reason": reason,
        "n_proxy_epochs": len(usable),
        "spearman_val_ade_vs_success": rho,
        "success_spread": success_spread,
        "ade_selected": {
            "epoch": ade_sel.get("epoch"),
            "val_ade": ade_sel.get("val_ade"),
            "success_rate": ade_sel.get("success_rate"),
        },
        "proxy_selected": {
            "epoch": proxy_sel.get("epoch"),
            "val_ade": proxy_sel.get("val_ade"),
            "success_rate": proxy_sel.get("success_rate"),
        },
        "proxy_minus_ade_success": float(proxy_sel["success_rate"])
        - float(ade_sel["success_rate"]),
    }


def analyze_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Analyze a training summary's proxy history and return the #3204 report."""
    proxy = summary.get("proxy", {})
    history = proxy.get("history", []) if isinstance(proxy, dict) else []
    report = analyze_proxy_history(history)
    report["model_id"] = summary.get("model_id")
    report["proxy_enabled"] = bool(proxy.get("enabled")) if isinstance(proxy, dict) else False
    report["schema_version"] = "predictive-checkpoint-proxy-report.v1"
    report["claim_boundary"] = (
        "Compares proxy-based vs val_ade-based checkpoint selection on the recorded hard-seed proxy. "
        "Not a paper-grade benchmark claim; the proxy is a hard-seed subset, not the full matrix."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the #3204 analysis CLI over a training summary JSON and return a POSIX exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = json.loads(args.training_summary.read_text())
    report = analyze_summary(summary)
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
