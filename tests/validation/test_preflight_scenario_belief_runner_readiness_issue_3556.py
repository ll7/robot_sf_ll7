"""Tests for the ScenarioBelief drop-vs-retain runner-readiness gate (#3556).

These exercise the fail-closed readiness contract with synthetic YAML fixtures only; they never
run the benchmark matrix or roll episodes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validation.preflight_scenario_belief_runner_readiness_issue_3556 import (
    DEFAULT_CONFIG,
    ISSUE,
    SCHEMA_VERSION,
    check_runner_readiness,
    main,
)

# A minimal, fully-pinned config that should pass every readiness check.
_VALID_CONFIG: dict = {
    "schema_version": "scenario-belief-episode-safety-config.v1",
    "issue": 3471,
    "belief_modes": ["oracle", "uncertain_retained", "uncertain_dropped"],
    "seeds": [101, 102, 103],
    "params": {
        "max_steps": 40,
        "dt": 0.1,
        "start_x": 2.0,
        "path_y": 5.0,
        "goal_x": 9.0,
        "corridor_x": 5.0,
        "robot_radius": 0.4,
        "ped_radius": 0.3,
        "near_miss_margin": 0.6,
        "ped_cross_speed": 0.7,
    },
}


def _write(tmp_path: Path, data: dict) -> Path:
    """Write a YAML config fixture and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_committed_default_config_is_ready():
    """The committed #3471 config the runner promotes must already pass the readiness gate."""
    report = check_runner_readiness(DEFAULT_CONFIG)
    assert report.ready, [c.as_dict() for c in report.failures]


def test_valid_synthetic_config_is_ready(tmp_path):
    """A fully-pinned synthetic config passes every check."""
    report = check_runner_readiness(_write(tmp_path, _VALID_CONFIG))
    assert report.ready
    # The fail-closed planner contract checks are present and passing.
    names = {c.name for c in report.checks}
    assert {"dropped_mode_consumes_uncertainty", "unsupported_planner_fails_closed"} <= names


def test_missing_mode_blocks_readiness(tmp_path):
    """Dropping a required contrast mode makes the gate fail closed."""
    data = {**_VALID_CONFIG, "belief_modes": ["oracle", "uncertain_retained"]}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    failed = {c.name for c in report.failures}
    assert "belief_modes_pinned" in failed
    assert "contrast_modes_present" in failed


def test_unknown_mode_blocks_readiness(tmp_path):
    """An unknown belief mode is rejected rather than silently accepted."""
    data = {**_VALID_CONFIG, "belief_modes": [*_VALID_CONFIG["belief_modes"], "made_up_mode"]}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "belief_modes_pinned" in {c.name for c in report.failures}


def test_missing_seeds_blocks_readiness(tmp_path):
    """Seeds must be explicitly pinned; an absent matrix fails closed."""
    data = {k: v for k, v in _VALID_CONFIG.items() if k != "seeds"}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "seeds_pinned" in {c.name for c in report.failures}


def test_duplicate_seeds_block_readiness(tmp_path):
    """A seed matrix with duplicates is not a valid pinned matrix."""
    data = {**_VALID_CONFIG, "seeds": [101, 101, 102]}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "seeds_pinned" in {c.name for c in report.failures}


def test_unpinned_param_blocks_readiness(tmp_path):
    """Leaving an episode parameter to the runner default fails the pinning check."""
    params = {k: v for k, v in _VALID_CONFIG["params"].items() if k != "near_miss_margin"}
    data = {**_VALID_CONFIG, "params": params}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "params_pinned" in {c.name for c in report.failures}


def test_invalid_geometry_blocks_readiness(tmp_path):
    """A corridor outside the start->goal span cannot contest the path and is rejected."""
    params = {**_VALID_CONFIG["params"], "corridor_x": 20.0}
    data = {**_VALID_CONFIG, "params": params}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "params_valid" in {c.name for c in report.failures}


def test_nonpositive_radius_blocks_readiness(tmp_path):
    """A non-positive robot radius is geometrically invalid."""
    params = {**_VALID_CONFIG["params"], "robot_radius": 0.0}
    data = {**_VALID_CONFIG, "params": params}
    report = check_runner_readiness(_write(tmp_path, data))
    assert not report.ready
    assert "params_valid" in {c.name for c in report.failures}


def test_missing_config_raises():
    """A missing config path raises (CLI maps this to the exit-2 error path)."""
    with pytest.raises(FileNotFoundError):
        check_runner_readiness(Path("/nonexistent/config.yaml"))


def test_report_schema_shape(tmp_path):
    """The structured report carries the schema/issue metadata and a claim boundary."""
    report = check_runner_readiness(_write(tmp_path, _VALID_CONFIG))
    payload = report.as_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["issue"] == ISSUE
    assert "does not run the benchmark matrix" in payload["claim_boundary"]
    assert payload["failed_checks"] == []


def test_cli_exit_codes(tmp_path, capsys):
    """The CLI returns 0 when ready, 1 when not ready, and 2 on an unevaluable config."""
    ready_cfg = _write(tmp_path, _VALID_CONFIG)
    assert main(["--config", str(ready_cfg), "--json"]) == 0

    bad = {**_VALID_CONFIG, "seeds": []}
    not_ready_cfg = tmp_path / "bad.yaml"
    not_ready_cfg.write_text(yaml.safe_dump(bad))
    assert main(["--config", str(not_ready_cfg)]) == 1

    assert main(["--config", str(tmp_path / "missing.yaml"), "--json"]) == 2
