"""Fail-closed readiness gate for promoting the advisory compat-matrix (issue #5039).

Background
----------
PR #5037 adds an *advisory*, non-blocking ``compat-matrix`` job to ``.github/workflows/ci.yml``
that exercises the declared Python 3.11/3.13 support on ``ubuntu-latest`` and ``macos-latest``.
Issue #5039 asks to promote that matrix to a *required* CI gate -- but only "after a reliable
set of hosted runs is green", and to record the evidence used to judge it stable.

That precondition is genuine hosted-CI evidence which cannot be manufactured locally. This tool
turns the "is the matrix proven enough to promote?" judgement into a reusable, machine-checkable
gate instead of an ad-hoc call. It reads a manifest of recorded hosted runs and reports whether
every required OS/Python cell has accumulated enough green runs within the runtime budget.

It fails closed: an empty or incomplete manifest yields ``status: blocked`` so the promotion PR
cannot claim readiness without evidence. It submits no jobs, changes no CI gate, and asserts no
benchmark or paper claim -- it only maps recorded run evidence to a ``ready``/``blocked`` verdict.

Exit codes
----------
* ``0`` -- evaluation completed (``ready`` or, without ``--require-ready``, ``blocked``).
* ``2`` -- ``--require-ready`` was set and the matrix is not yet promotable.
* ``1`` -- hard error (missing/malformed manifest); fail closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    _REPO_ROOT / "docs" / "context" / "issue_5039_compat_matrix_promotion_manifest.yaml"
)

STATUS_READY = "ready"
STATUS_BLOCKED = "blocked"

_REQUIRED_TOP_KEYS = ("schema_version", "promotion_gate", "recorded_runs")
_REQUIRED_GATE_KEYS = (
    "required_cells",
    "min_green_runs_per_cell",
    "runtime_budget_minutes",
)


class ManifestError(ValueError):
    """Raised when the manifest is missing or structurally invalid (fail closed)."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and structurally validate the promotion manifest, failing closed on error."""
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ManifestError(f"manifest is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a mapping")
    missing = [key for key in _REQUIRED_TOP_KEYS if key not in data]
    if missing:
        raise ManifestError(f"manifest missing required keys: {sorted(missing)}")
    gate = data["promotion_gate"]
    if not isinstance(gate, dict):
        raise ManifestError("promotion_gate must be a mapping")
    gate_missing = [key for key in _REQUIRED_GATE_KEYS if key not in gate]
    if gate_missing:
        raise ManifestError(f"promotion_gate missing keys: {sorted(gate_missing)}")
    if not isinstance(data["recorded_runs"], list):
        raise ManifestError("recorded_runs must be a list (use [] when no evidence exists)")
    return data


def _cell_key(entry: dict[str, Any]) -> tuple[str, str]:
    """Return the ``(os, python)`` identity of a cell or recorded run."""
    return str(entry.get("os")), str(entry.get("python"))


def _is_green_within_budget(run: dict[str, Any], budget_minutes: float) -> bool:
    """A run counts only when it succeeded and stayed within the runtime budget."""
    if str(run.get("conclusion", "")).lower() != "success":
        return False
    duration = run.get("duration_minutes")
    if duration is None:
        # Missing duration cannot be proven within budget -> fail closed.
        return False
    try:
        return float(duration) <= float(budget_minutes)
    except (TypeError, ValueError):
        return False


def evaluate(manifest: dict[str, Any]) -> dict[str, Any]:
    """Evaluate promotion readiness from recorded hosted-run evidence.

    A cell is satisfied when at least ``min_green_runs_per_cell`` recorded runs for that
    ``(os, python)`` succeeded within ``runtime_budget_minutes``. The matrix is ``ready``
    only when every required cell is satisfied.
    """
    gate = manifest["promotion_gate"]
    required_cells: list[dict[str, Any]] = list(gate["required_cells"])
    min_green = int(gate["min_green_runs_per_cell"])
    budget = float(gate["runtime_budget_minutes"])
    runs: list[dict[str, Any]] = list(manifest["recorded_runs"])

    green_by_cell: dict[tuple[str, str], int] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        if _is_green_within_budget(run, budget):
            key = _cell_key(run)
            green_by_cell[key] = green_by_cell.get(key, 0) + 1

    cell_reports: list[dict[str, Any]] = []
    unmet: list[str] = []
    for cell in required_cells:
        key = _cell_key(cell)
        count = green_by_cell.get(key, 0)
        satisfied = count >= min_green
        cell_reports.append(
            {
                "os": key[0],
                "python": key[1],
                "green_runs": count,
                "required": min_green,
                "satisfied": satisfied,
            }
        )
        if not satisfied:
            unmet.append(f"{key[0]} / py{key[1]}: {count}/{min_green} green runs")

    ready = len(required_cells) > 0 and not unmet
    return {
        "status": STATUS_READY if ready else STATUS_BLOCKED,
        "ready": ready,
        "min_green_runs_per_cell": min_green,
        "runtime_budget_minutes": budget,
        "cells": cell_reports,
        "unmet_cells": unmet,
    }


def _render_text(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        f"compat-matrix promotion readiness: {report['status'].upper()}",
        f"  gate: >= {report['min_green_runs_per_cell']} green runs/cell "
        f"within {report['runtime_budget_minutes']:g} min",
    ]
    for cell in report["cells"]:
        mark = "PASS" if cell["satisfied"] else "MISS"
        lines.append(
            f"  [{mark}] {cell['os']} / py{cell['python']}: "
            f"{cell['green_runs']}/{cell['required']} green runs"
        )
    if not report["ready"]:
        state = manifest.get("state", {})
        nxt = state.get("next_empirical_action")
        if nxt:
            lines.append(f"  next: {nxt}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate the manifest and report readiness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the promotion manifest (default: the issue #5039 manifest).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero (2) when the matrix is not yet promotable.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = evaluate(manifest)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_text(report, manifest))

    if args.require_ready and not report["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
