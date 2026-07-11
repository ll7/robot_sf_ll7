"""Tests for the issue #4973 Zanlungo corridor acceptance harness."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_sf.research.zanlungo_corridor_acceptance import (
    AcceptanceConfig,
    CorridorFixtureConfig,
    CorridorTrace,
    classify_corridor_trace,
    load_acceptance_config,
    render_acceptance_markdown,
    run_acceptance,
    write_acceptance_report,
)
from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    SOCIAL_FORCE_DEFAULT,
)
from robot_sf.sim.sim_config import SimulationSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/research/issue_4973_zanlungo_corridor_acceptance.yaml"


@pytest.fixture(scope="module")
def config() -> AcceptanceConfig:
    """Load the canonical predeclared acceptance packet once."""
    return load_acceptance_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def report(config: AcceptanceConfig) -> dict:
    """Run the bounded CPU acceptance matrix once for contract assertions."""
    return run_acceptance(config)


def test_canonical_config_predeclares_fixture_and_parameter_bookkeeping(
    config: AcceptanceConfig,
) -> None:
    """The packet fixes geometry, seed, reference row, and one-at-a-time variants."""
    assert config.issue == 4973
    assert config.benchmark_evidence is False
    assert config.fixture.seed == 4973
    assert config.fixture.dt_s == pytest.approx(0.1)
    assert config.reference_case_id == "zanlungo_paper_reference"
    assert config.cases[0].pedestrian_model == SOCIAL_FORCE_DEFAULT
    assert {case.varied_parameter for case in config.cases[1:]} == {
        "reference",
        "interaction_strength",
        "interaction_range_m",
    }
    assert all(case.relative_to_reference is not None for case in config.cases[1:])


def test_cpu_harness_labels_reference_yielding_and_control_collision_proxy(
    report: dict,
) -> None:
    """The predeclared reference yields while the reactive control misses clearance."""
    by_id = {row["case_id"]: row for row in report["rows"]}
    assert by_id["zanlungo_paper_reference"]["metrics"]["outcome_label"] == "yielding"
    assert by_id["social_force_control"]["metrics"]["outcome_label"] == "collision_proxy"
    assert report["acceptance_checks"]["reference_outcome_is_yielding"] is True
    assert report["acceptance_met"] is True


def test_every_parameter_row_has_exact_replay_and_explicit_non_benchmark_metadata(
    report: dict,
) -> None:
    """Every row replays byte-identically and remains outside benchmark evidence."""
    assert report["benchmark_evidence"] is False
    assert report["evidence_tier"] == "diagnostic-only"
    assert report["status"] == {
        "cpu_only": True,
        "slurm_or_gpu_used": False,
        "full_campaign_run": False,
    }
    assert report["acceptance_checks"]["all_rows_replay_deterministic"] is True
    for row in report["rows"]:
        assert row["replay_deterministic"] is True
        assert row["trace_sha256"] == row["replay_trace_sha256"]


def test_default_pedestrian_force_path_remains_social_force(report: dict) -> None:
    """Running the opt-in harness does not mutate the simulator default selector."""
    assert SimulationSettings().pedestrian_model == SOCIAL_FORCE_DEFAULT
    assert report["acceptance_checks"]["default_force_path_unchanged"] is True
    assert any(
        row["pedestrian_model"] == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1 for row in report["rows"]
    )


def test_classifier_labels_sustained_conflict_slowdown_as_freezing(
    config: AcceptanceConfig,
) -> None:
    """A synthetic stalled conflict hits the named freezing outcome branch."""
    positions = np.asarray(
        [
            [[4.4, 1.9], [5.6, 2.1]],
            [[4.5, 1.9], [5.5, 2.1]],
            [[4.5, 1.9], [5.5, 2.1]],
            [[4.5, 1.9], [5.5, 2.1]],
        ],
        dtype=float,
    )
    thresholds = replace(config.thresholds, freeze_window_s=0.3)
    metrics = classify_corridor_trace(
        CorridorTrace(positions=positions, velocities=np.zeros_like(positions), dt_s=0.1),
        config.fixture,
        thresholds,
    )
    assert metrics["outcome_label"] == "freezing"
    assert metrics["freeze_criterion_met"] is True


def test_classifier_rejects_invalid_replay_timestep(config: AcceptanceConfig) -> None:
    """Replay classification fails closed when timing provenance is invalid."""
    positions = np.zeros((1, 2, 2), dtype=float)
    with pytest.raises(ValueError, match="trace.dt_s"):
        classify_corridor_trace(
            CorridorTrace(positions=positions, velocities=positions.copy(), dt_s=0.0),
            config.fixture,
            config.thresholds,
        )


def test_config_loader_fails_closed_if_benchmark_evidence_is_promoted(tmp_path: Path) -> None:
    """The acceptance packet cannot silently become benchmark evidence."""
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "benchmark_evidence: false", "benchmark_evidence: true", 1
    )
    path = tmp_path / "promoted.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_evidence must be false"):
        load_acceptance_config(path)


@pytest.mark.parametrize(
    ("source", "replacement", "match"),
    [
        ("duration_s: 10.0", "duration_s: true", "fixture.duration_s"),
        (
            "interaction_strength: 1.13",
            "interaction_strength: true",
            "parameters.interaction_strength",
        ),
    ],
)
def test_config_loader_rejects_boolean_numeric_values(
    tmp_path: Path, source: str, replacement: str, match: str
) -> None:
    """YAML booleans cannot silently alter numeric fixture or force parameters."""
    path = tmp_path / "boolean-value.yaml"
    path.write_text(CONFIG_PATH.read_text(encoding="utf-8").replace(source, replacement, 1))
    with pytest.raises(ValueError, match=match):
        load_acceptance_config(path)


def test_config_loader_rejects_evidence_boundary_drift(tmp_path: Path) -> None:
    """Packet provenance must match the report's diagnostic-only evidence boundary."""
    path = tmp_path / "drifted-evidence-tier.yaml"
    path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            "evidence_tier: diagnostic-only", "evidence_tier: nominal-benchmark", 1
        )
    )
    with pytest.raises(ValueError, match="metadata.evidence_tier"):
        load_acceptance_config(path)


def test_config_path_is_independent_of_the_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The test's repository fixture path works from non-root test invocations."""
    monkeypatch.chdir(tmp_path)
    assert load_acceptance_config(CONFIG_PATH).issue == 4973


def test_config_loader_fails_closed_on_dishonest_parameter_ratio(tmp_path: Path) -> None:
    """Sensitivity bookkeeping cannot claim a multiplier that the parameters do not encode."""
    text = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "relative_to_reference: 1.7699115044247788",
        "relative_to_reference: 1.0",
        1,
    )
    path = tmp_path / "dishonest-ratio.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="exact ratio"):
        load_acceptance_config(path)


def test_report_writers_preserve_claim_boundary_and_rows(report: dict, tmp_path: Path) -> None:
    """Local artifacts preserve the machine-readable and reviewer-readable contracts."""
    paths = write_acceptance_report(report, tmp_path)
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8")) == report
    markdown = paths["readme"].read_text(encoding="utf-8")
    assert markdown == render_acceptance_markdown(report)
    assert "benchmark_evidence=false" in markdown
    assert "zanlungo_paper_reference" in markdown
    assert "Acceptance met: **True**" in markdown


@pytest.mark.parametrize(
    "changes",
    [
        {"duration_s": 0.0},
        {"corridor_min_y_m": 3.0, "corridor_max_y_m": 2.0},
    ],
)
def test_fixture_configuration_rejects_invalid_runtime_or_geometry(
    config: AcceptanceConfig,
    changes: dict[str, float],
) -> None:
    """Malformed CPU fixture controls fail before simulator construction."""
    with pytest.raises(ValueError):
        CorridorFixtureConfig(**{**config.fixture.__dict__, **changes})
