"""Contract tests for the issue #5416 native SIPP smoke validator."""

from __future__ import annotations

from pathlib import Path

import pytest

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
