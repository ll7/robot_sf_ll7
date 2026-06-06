"""Tests for the one-command mechanism-aware reproduction quickstart."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.demo import reproduce_mechanism_report as quickstart

if TYPE_CHECKING:
    from pathlib import Path


def test_diagnostic_args_for_default_case_use_disposable_output(tmp_path: Path) -> None:
    """The default case should route to a local output directory."""
    case = quickstart.CASES["topology-primary-route"]

    argv = quickstart.diagnostic_args_for_case(case, tmp_path)

    assert "--candidate" in argv
    assert "hybrid_rule_v3_waypoint2_route_lookahead8" in argv
    assert "--stage" in argv
    assert "full_matrix" in argv
    assert "--scenario-name" in argv
    assert "classic_realworld_double_bottleneck_high" in argv
    assert "--output-dir" in argv
    assert str(tmp_path / "topology_primary_route") in argv


def test_run_case_preserves_underlying_exit_code(monkeypatch, tmp_path: Path, capsys) -> None:
    """The wrapper should not hide fail-closed diagnostic exits."""
    calls: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        calls.append(argv)
        return 2

    monkeypatch.setattr(quickstart.run_topology_hypothesis_diagnostics, "main", fake_main)

    exit_code = quickstart.run_case(
        case_name="topology-primary-route",
        output_root=tmp_path,
    )

    assert exit_code == 2
    assert len(calls) == 1
    assert "--output-dir" in calls[0]
    stdout = capsys.readouterr().out
    assert quickstart.CLAIM_BOUNDARY in stdout
    assert "preserving fail-closed exit code" in stdout


def test_main_defaults_to_topology_primary_route(monkeypatch, tmp_path: Path) -> None:
    """The documented one-command path should default to the supported case."""
    calls: list[tuple[str, Path]] = []

    def fake_run_case(*, case_name: str, output_root: Path) -> int:
        calls.append((case_name, output_root))
        return 0

    monkeypatch.setattr(quickstart, "run_case", fake_run_case)

    assert quickstart.main(["--output-root", str(tmp_path)]) == 0
    assert calls == [("topology-primary-route", tmp_path)]
