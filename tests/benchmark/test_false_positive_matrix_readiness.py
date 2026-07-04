"""Tests for issue #3300 stronger false-positive matrix readiness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.false_positive_matrix_readiness import (
    REQUIRED_OBSERVATION_MODE,
    STATUS_BLOCKED,
    STATUS_READY,
    check_false_positive_matrix_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOMINAL_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_3300_false_positive_stronger_nominal_smoke.yaml"
)
PERTURBED_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_3300_false_positive_stronger_perturbed_smoke.yaml"
)
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/check_false_positive_matrix_readiness_issue_3300.py"


def test_stronger_matrix_configs_are_ready() -> None:
    """The preregistered stronger matrix has multiple scenarios, seeds, and injection."""

    readiness = check_false_positive_matrix_readiness(NOMINAL_CONFIG, PERTURBED_CONFIG)

    assert readiness.status == STATUS_READY
    assert len(set(readiness.scenario_ids)) == 3
    assert len(set(readiness.seeds)) == 2
    assert readiness.planner_observation_modes == [REQUIRED_OBSERVATION_MODE]
    assert len(readiness.pedestrian_scenario_ids) == 3
    assert readiness.injection_probe["pedestrians_added"] == 1
    assert readiness.injection_probe["pedestrians_count"] == 2


def test_one_scenario_or_one_seed_matrix_blocks() -> None:
    """The checker rejects another one-scenario or one-seed smoke as too weak."""

    readiness = check_false_positive_matrix_readiness(
        REPO_ROOT / "configs/benchmarks/issue_3300_false_positive_nominal_smoke.yaml",
        REPO_ROOT / "configs/benchmarks/issue_3300_false_positive_perturbed_smoke.yaml",
    )

    assert readiness.status == STATUS_BLOCKED
    assert any("at least 2 distinct scenarios" in blocker for blocker in readiness.blockers)
    assert any("at least 2 distinct fixed seeds" in blocker for blocker in readiness.blockers)


def test_non_structured_planner_observation_blocks(tmp_path: Path) -> None:
    """The matrix must keep planner observations structured-pedestrian compatible."""

    nominal = tmp_path / "nominal.yaml"
    perturbed = tmp_path / "perturbed.yaml"
    nominal.write_text(
        NOMINAL_CONFIG.read_text(encoding="utf-8").replace(
            "observation_mode: socnav_state",
            "observation_mode: goal_state",
        ),
        encoding="utf-8",
    )
    perturbed.write_text(PERTURBED_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    readiness = check_false_positive_matrix_readiness(nominal, perturbed)

    assert readiness.status == STATUS_BLOCKED
    assert any(REQUIRED_OBSERVATION_MODE in blocker for blocker in readiness.blockers)


@pytest.mark.parametrize("min_scenarios,min_seeds", [(3, 2), (2, 2)])
def test_cli_ready_contract(min_scenarios: int, min_seeds: int) -> None:
    """CLI emits JSON readiness evidence and exits zero for the stronger matrix."""

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--nominal-config",
            str(NOMINAL_CONFIG),
            "--perturbed-config",
            str(PERTURBED_CONFIG),
            "--min-scenarios",
            str(min_scenarios),
            "--min-seeds",
            str(min_seeds),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == STATUS_READY
    assert payload["injection_probe"]["pedestrians_added"] == 1


def test_cli_blocks_too_weak_threshold() -> None:
    """CLI fails closed when the requested matrix threshold is not met."""

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--nominal-config",
            str(NOMINAL_CONFIG),
            "--perturbed-config",
            str(PERTURBED_CONFIG),
            "--min-scenarios",
            "4",
            "--min-seeds",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 3, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == STATUS_BLOCKED
    assert payload["blockers"]
