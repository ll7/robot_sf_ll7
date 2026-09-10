"""Fixture-mode tests for the pareto plotting example (issue #8732)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "examples" / "plotting" / "fixtures" / "pareto_dominated_tie.json"

sys.path.insert(0, str(_REPO_ROOT / "examples" / "plotting"))

from plot_pareto import (  # noqa: E402
    _load_fixture_records,
    _validate_fixture_records,
    main,
)

from robot_sf.benchmark.plots import compute_pareto_points  # noqa: E402


def _front_labels() -> set[str]:
    records = _load_fixture_records()
    points, labels = compute_pareto_points(records, "collisions", "comfort_exposure")
    front = __import__("robot_sf.benchmark.plots", fromlist=["pareto_front_indices"])
    indices = front.pareto_front_indices(points)
    return {labels[i] for i in indices}


def test_fixture_front_identities() -> None:
    """Nondominated groups plus the exact tie are on the front; E is dominated."""
    assert _front_labels() == {"A", "B", "C", "D"}


def test_fixture_exact_tie_shared_point() -> None:
    """Groups A and D evaluate to the identical mean point."""
    records = _load_fixture_records()
    points, labels = compute_pareto_points(records, "collisions", "comfort_exposure")
    by_label = dict(zip(labels, points, strict=True))
    assert by_label["A"] == by_label["D"]


def test_reversed_objective_changes_front() -> None:
    """Flipping the objective direction must change the nondominated set."""
    records = _load_fixture_records()
    points, labels = compute_pareto_points(records, "collisions", "comfort_exposure")
    front = __import__("robot_sf.benchmark.plots", fromlist=["pareto_front_indices"])
    normal = {labels[i] for i in front.pareto_front_indices(points)}
    flipped = {
        labels[i]
        for i in front.pareto_front_indices(points, x_higher_better=True, y_higher_better=True)
    }
    assert normal != flipped


def test_validation_rejects_missing_metric() -> None:
    records = [{"scenario_id": "x", "metrics": {"collisions": 1.0}}]
    with pytest.raises(SystemExit):
        _validate_fixture_records(records, "collisions", "comfort_exposure")


def test_validation_rejects_non_finite_metric() -> None:
    records = [
        {"scenario_id": "x", "metrics": {"collisions": float("nan"), "comfort_exposure": 0.5}}
    ]
    with pytest.raises(SystemExit):
        _validate_fixture_records(records, "collisions", "comfort_exposure")


def test_validation_rejects_duplicate_identity() -> None:
    records = [
        {"scenario_id": "dup", "metrics": {"collisions": 1.0, "comfort_exposure": 0.5}},
        {"scenario_id": "dup", "metrics": {"collisions": 0.5, "comfort_exposure": 0.5}},
    ]
    with pytest.raises(SystemExit):
        _validate_fixture_records(records, "collisions", "comfort_exposure")


def test_fixture_mode_writes_png_svg_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixture mode writes all three artifacts with a stable summary."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out_dir = tmp_path / "smoke"
    assert main(["--fixture", "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "pareto_fixture.png").stat().st_size > 0
    assert (out_dir / "pareto_fixture.svg").stat().st_size > 0
    first = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert first["front_labels"] == ["A", "B", "C", "D"]
    assert first["mechanics_only"] is True
    assert main(["--fixture", "--out-dir", str(out_dir)]) == 0
    second = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert second == first


def test_fixture_svg_bytes_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated independent fixture renders produce identical SVG bytes."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    command = [sys.executable, str(_REPO_ROOT / "examples/plotting/plot_pareto.py"), "--fixture"]
    subprocess.run([*command, "--out-dir", str(first_dir)], check=True, cwd=_REPO_ROOT)
    subprocess.run([*command, "--out-dir", str(second_dir)], check=True, cwd=_REPO_ROOT)
    assert (first_dir / "pareto_fixture.svg").read_bytes() == (
        second_dir / "pareto_fixture.svg"
    ).read_bytes()


def test_fixture_json_path_must_stay_inside_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An escaping fixture summary path fails before writing outside ``--out-dir``."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out_dir = tmp_path / "smoke"
    escaped = tmp_path / "escape.json"
    with pytest.raises(SystemExit, match="inside --out-dir"):
        main(["--fixture", "--out-dir", str(out_dir), "--out-json", "../escape.json"])
    assert not escaped.exists()


def test_bare_invocation_defaults_to_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running with no input source executes the fixture demo, not an error."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    assert main([]) == 0
    summary = tmp_path / "output" / "example-pareto-smoke" / "summary.json"
    assert summary.is_file()
    assert json.loads(summary.read_text(encoding="utf-8"))["front_labels"] == ["A", "B", "C", "D"]


def test_synthetic_path_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-existing synthetic demo path keeps working."""
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out = tmp_path / "demo.png"
    assert main(["--synthetic", "--out", str(out)]) == 0
    assert out.stat().st_size > 0
