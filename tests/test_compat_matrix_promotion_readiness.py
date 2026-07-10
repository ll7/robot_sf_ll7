"""Contract + logic tests for the compat-matrix promotion readiness gate (issue #5039)."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "ci" / "check_compat_matrix_promotion_readiness.py"
MANIFEST = ROOT / "docs" / "context" / "issue_5039_compat_matrix_promotion_manifest.yaml"

CANONICAL_CELLS = [
    {"os": "ubuntu-latest", "python": "3.11"},
    {"os": "ubuntu-latest", "python": "3.13"},
    {"os": "macos-latest", "python": "3.11"},
    {"os": "macos-latest", "python": "3.13"},
]


def _load_checker():
    spec = importlib.util.spec_from_file_location("_compat_promotion_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _green_runs(cell: dict[str, str], count: int, *, duration: float = 12.0):
    return [
        {
            "os": cell["os"],
            "python": cell["python"],
            "run_url": f"https://example/run/{cell['os']}/{cell['python']}/{i}",
            "conclusion": "success",
            "duration_minutes": duration,
            "sha": "deadbeef",
        }
        for i in range(count)
    ]


def test_shipped_manifest_parses_with_canonical_gate() -> None:
    """The tracked manifest keeps the four canonical cells and a coverage-floor decision."""
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema_version"] == "compat_matrix_promotion_manifest.v1"
    assert data["issue"] == 5039
    gate = data["promotion_gate"]
    assert gate["required_cells"] == CANONICAL_CELLS
    assert gate["min_green_runs_per_cell"] >= 1
    assert gate["runtime_budget_minutes"] > 0
    assert data["recorded_runs"] == []
    assert data["coverage_floor"]["status"] == "split_recommended"


def test_shipped_manifest_is_currently_blocked() -> None:
    """With no recorded evidence the gate must fail closed as blocked."""
    checker = _load_checker()
    manifest = checker.load_manifest(MANIFEST)
    report = checker.evaluate(manifest)
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert len(report["unmet_cells"]) == len(CANONICAL_CELLS)


def test_full_green_evidence_promotes_to_ready() -> None:
    """Enough in-budget green runs across every cell flips the gate to ready."""
    checker = _load_checker()
    manifest = checker.load_manifest(MANIFEST)
    min_green = manifest["promotion_gate"]["min_green_runs_per_cell"]
    runs = []
    for cell in CANONICAL_CELLS:
        runs.extend(_green_runs(cell, min_green))
    manifest = copy.deepcopy(manifest)
    manifest["recorded_runs"] = runs
    report = checker.evaluate(manifest)
    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["unmet_cells"] == []


def test_over_budget_and_failed_runs_do_not_count() -> None:
    """A failing run or an over-budget run is not green evidence."""
    checker = _load_checker()
    manifest = checker.load_manifest(MANIFEST)
    min_green = manifest["promotion_gate"]["min_green_runs_per_cell"]
    budget = manifest["promotion_gate"]["runtime_budget_minutes"]
    runs = []
    for cell in CANONICAL_CELLS:
        runs.extend(_green_runs(cell, min_green))
    # Degrade one cell: swap its green runs for one over-budget and one failed run.
    target = CANONICAL_CELLS[0]
    runs = [r for r in runs if (r["os"], r["python"]) != (target["os"], target["python"])]
    runs.extend(_green_runs(target, 1, duration=budget + 1))  # over budget
    runs.append(
        {
            "os": target["os"],
            "python": target["python"],
            "conclusion": "failure",
            "duration_minutes": 5.0,
        }
    )
    manifest = copy.deepcopy(manifest)
    manifest["recorded_runs"] = runs
    report = checker.evaluate(manifest)
    assert report["status"] == "blocked"
    assert any(target["os"] in cell for cell in report["unmet_cells"])


def test_malformed_manifest_fails_closed() -> None:
    """Missing required keys raise ManifestError rather than silently passing."""
    checker = _load_checker()
    with pytest.raises(checker.ManifestError):
        checker.load_manifest(ROOT / "does_not_exist.yaml")


def test_cli_require_ready_blocks_on_shipped_manifest() -> None:
    """The CLI exits 0 by default but 2 under --require-ready while blocked."""
    default = subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(MANIFEST)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert default.returncode == 0, default.stderr
    assert "BLOCKED" in default.stdout

    required = subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(MANIFEST), "--require-ready"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert required.returncode == 2, required.stdout
