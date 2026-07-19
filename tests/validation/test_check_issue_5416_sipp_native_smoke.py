"""Contract tests for the issue #5416 native SIPP smoke validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation import check_issue_5416_sipp_native_smoke as smoke_validator
from scripts.validation.check_issue_5416_sipp_native_smoke import SmokeError, validate_smoke


def test_smoke_arguments_are_pinned_before_expensive_execution(tmp_path: Path) -> None:
    """The reusable validator cannot be repurposed into an unreviewed campaign runner."""
    with pytest.raises(SmokeError, match="must stay pinned"):
        validate_smoke(
            packet_path=Path(
                "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"
            ),
            native_config_path=Path("configs/algos/sipp_lattice_native_command.yaml"),
            scenario_id="classic_head_on_corridor_low",
            seed=112,
            horizon=500,
            dt=0.1,
            workers=1,
            output_dir=tmp_path,
        )


def test_main_reports_unexpected_standard_exception_as_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Unexpected validator errors still produce the fail-closed JSON contract."""

    def raise_assertion_error(**_: object) -> dict[str, object]:
        raise AssertionError("unexpected validation failure")

    monkeypatch.setattr(smoke_validator, "validate_smoke", raise_assertion_error)

    result = smoke_validator.main(
        [
            "--scenario-id",
            "classic_head_on_corridor_low",
            "--seed",
            "111",
            "--horizon",
            "500",
            "--dt",
            "0.1",
            "--workers",
            "1",
            "--output-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert result == 1
    assert capsys.readouterr().out == (
        '{"error": "unexpected validation failure", "status": "blocked"}\n'
    )


def test_five_planner_mode_omits_single_row_arguments(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The frozen five-planner mode needs only its own packet and output inputs."""

    monkeypatch.setattr(
        smoke_validator,
        "validate_five_planner_smoke",
        lambda **_: {"status": "ready", "eligible_rows": 5, "excluded_rows": 0},
    )

    result = smoke_validator.main(["--five-planner-smoke", "--output-dir", str(tmp_path), "--json"])

    assert result == 0
    assert capsys.readouterr().out == (
        '{"eligible_rows": 5, "excluded_rows": 0, "status": "ready"}\n'
    )


def test_standard_mode_still_requires_single_row_arguments(tmp_path: Path) -> None:
    """Relaxing the five-planner parser must not broaden the standard smoke contract."""

    with pytest.raises(SystemExit, match="2"):
        smoke_validator.main(["--output-dir", str(tmp_path)])
