"""CPU tests for the MAP-Elites QD campaign integration (issue #5308).

Validates the campaign config, script path, and archive schema without running
the full simulator-backed evaluation pipeline. The heavy production wiring is
already exercised by ``test_issue_5308_qd_production.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.cma_me import CMaMeEmitter
from robot_sf.adversarial.qd import (
    QD_ARCHIVE_SCHEMA_VERSION,
    GridSpec,
    QDArchive,
    write_qd_archive,
)

_CAMPAIGN_CONFIG_PATH = "configs/adversarial/issue_5308_qd_campaign.yaml"


def _load_campaign_payload() -> dict:
    """Load the campaign YAML config and return its payload."""
    path = Path(_CAMPAIGN_CONFIG_PATH)
    if not path.exists():
        pytest.skip(f"Campaign config not found at {_CAMPAIGN_CONFIG_PATH}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        pytest.fail("Campaign config must be a YAML mapping")
    return payload


def test_campaign_config_has_valid_schema() -> None:
    """Campaign config must carry the expected schema version."""
    payload = _load_campaign_payload()
    assert payload.get("schema_version") == "adversarial_qd_campaign.v1"
    assert payload.get("issue") == 5308
    assert payload.get("scenario_template", "").endswith(".yaml")


def test_campaign_config_has_search_space() -> None:
    """The campaign search space must define all required variables."""
    payload = _load_campaign_payload()
    variables = payload["search_space"]["variables"]
    for key in ("start_x", "start_y", "goal_x", "goal_y", "scenario_seed"):
        assert key in variables, f"Missing search space variable: {key}"
        for bound in ("min", "max"):
            assert bound in variables[key], f"Missing {bound} in {key}"


def test_campaign_config_has_behavior_grid() -> None:
    """The behavior grid spec must define valid bounds."""
    payload = _load_campaign_payload()
    bg = payload["behavior_grid"]
    for key in ("x_min", "x_max", "y_min", "y_max", "bins"):
        assert key in bg, f"Missing behavior_grid key: {key}"
    assert bg["x_max"] > bg["x_min"]
    assert bg["y_max"] > bg["y_min"]
    assert bg["bins"] >= 2


def test_campaign_config_has_emitters() -> None:
    """At least one valid emitter must be configured."""
    payload = _load_campaign_payload()
    emitters = payload.get("emitters", [])
    assert len(emitters) >= 1
    valid = {"random", "coordinate", "cma_me"}
    for emitter in emitters:
        assert emitter in valid, f"Unknown emitter: {emitter}"


def test_campaign_budget_is_reasonable() -> None:
    """Campaign budget must be positive and capped for a bounded CPU run."""
    payload = _load_campaign_payload()
    budget = payload.get("budget", 0)
    assert 1 <= budget <= 512, f"Budget {budget} out of expected range"


def test_campaign_config_wires_grid_to_qd_config() -> None:
    """The campaign behavior grid must produce a valid QDSearchConfig."""
    payload = _load_campaign_payload()
    bg = payload["behavior_grid"]
    grid = GridSpec(
        x_min=float(bg["x_min"]),
        x_max=float(bg["x_max"]),
        y_min=float(bg["y_min"]),
        y_max=float(bg["y_max"]),
        bins=int(bg["bins"]),
    )
    assert grid.cell_count >= 4
    assert grid.cell_index((grid.x_min, grid.y_min)) is not None
    assert grid.cell_index((grid.x_max, grid.y_max)) is not None


def test_campaign_emitter_names_map_to_classes() -> None:
    """Ensure every configured emitter name maps to a loadable class."""
    from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig
    from robot_sf.adversarial.samplers import (
        CoordinateRefinementSampler,
        RandomCandidateSampler,
    )

    space = SearchSpaceConfig(
        start_x=RangeConfig(0.0, 4.0),
        start_y=RangeConfig(0.0, 0.0),
        goal_x=RangeConfig(0.0, 4.0),
        goal_y=RangeConfig(0.0, 0.0),
        spawn_time_s=RangeConfig(0.0, 0.0),
        pedestrian_speed_mps=RangeConfig(1.0, 1.0),
        pedestrian_delay_s=RangeConfig(0.0, 0.0),
        scenario_seed=RangeConfig(1.0, 1.0),
    )
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    archive = QDArchive(grid=grid)

    payload = _load_campaign_payload()
    for name in payload.get("emitters", []):
        name_lower = name.strip().lower()
        if name_lower == "random":
            assert RandomCandidateSampler(space, seed=0) is not None
        elif name_lower == "coordinate":
            assert CoordinateRefinementSampler(space, seed=0) is not None
        elif name_lower == "cma_me":
            assert CMaMeEmitter(space, archive, seed=0) is not None
        else:
            pytest.fail(f"Unknown emitter: {name}")


def test_campaign_script_dry_run(tmp_path: Path) -> None:
    """The campaign script must exit cleanly in dry-run mode."""
    from scripts.adversarial.run_qd_campaign import run_campaign

    config_path = Path(_CAMPAIGN_CONFIG_PATH)
    if not config_path.exists():
        pytest.skip(f"Campaign config not found at {_CAMPAIGN_CONFIG_PATH}")

    report = run_campaign(
        str(config_path),
        str(tmp_path / "dry_run_out"),
        dry_run=True,
    )
    assert report["status"] == "dry_run"


def test_campaign_script_rejects_bad_config(tmp_path: Path) -> None:
    """The campaign script must fail closed on a missing config."""
    from scripts.adversarial.run_qd_campaign import load_campaign_config

    missing = tmp_path / "missing_config.yaml"
    with pytest.raises(FileNotFoundError):
        load_campaign_config(missing)


def test_campaign_script_rejects_wrong_schema(tmp_path: Path) -> None:
    """The campaign script must reject a config with an unsupported schema."""
    from scripts.adversarial.run_qd_campaign import load_campaign_config

    bad = tmp_path / "bad_schema.yaml"
    bad.write_text(yaml.safe_dump({"schema_version": "unsupported.v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported campaign schema"):
        load_campaign_config(bad)


def test_campaign_archive_artifact_schema(tmp_path: Path) -> None:
    """The QD archive artifact written by a campaign must match the schema."""
    from robot_sf.adversarial.attribution import attribution_from_episode_record
    from robot_sf.adversarial.certification import not_available_status
    from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
    from robot_sf.adversarial.qd import QDSearchResult, default_behavior_descriptor

    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    archive = QDArchive(grid=grid)

    for i in range(12):
        record = {
            "status": "completed",
            "termination_reason": "collision" if i % 2 == 0 else "timeout",
            "outcome": {
                "route_complete": False,
                "collision": i % 2 == 0,
                "timeout": i % 2 != 0,
            },
            "metrics": {
                "distance_to_human_min": 2.5 * ((i % 4) / 3.0),
                "time_to_collision_min": 3.0 * ((i % 3) / 2.0),
            },
        }
        episode_path = tmp_path / f"episode_{i:04d}.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        candidate = CandidateSpec(
            start=Pose2D(float(i) * 0.4, 0.0),
            goal=Pose2D(4.0 - float(i) * 0.4, 0.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
        evaluation = CandidateEvaluation(
            candidate=candidate,
            certification_status=not_available_status("advisory"),
            objective_value=float(i),
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=tmp_path / "scenario_dummy.yaml",
            bundle_path=tmp_path,
        )
        descriptor = default_behavior_descriptor(evaluation)
        assert descriptor is not None
        archive.try_insert(
            descriptor=descriptor,
            evaluation=evaluation,
            certification_status=None,
        )

    assert archive.filled_cell_count() > 0

    result_obj = QDSearchResult(
        archive=archive,
        num_evaluated=12,
        num_admitted=archive.filled_cell_count(),
    )

    artifact_path = write_qd_archive(result_obj, tmp_path / "archive.json")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == QD_ARCHIVE_SCHEMA_VERSION
    assert payload["behavior_axes"] == ["distance_to_human_min", "time_to_collision_min"]
    assert payload["summary"]["filled_cell_count"] > 0
    assert payload["summary"]["distinct_failure_modes"] is not None
    assert payload["summary"]["coverage_fraction"] > 0.0
    assert payload["summary"]["qd_score"] > 0.0
    assert len(payload["cells"]) >= 2
