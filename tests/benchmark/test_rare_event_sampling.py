"""Tests for issue #4163 rare-event importance-sampling harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.rare_event_sampling import (
    RareEventSamplingError,
    RareEventSamplingSpec,
    SampledScenarioRow,
    apply_sampled_scenario_mutation,
    build_sampling_summary,
    estimate_failure_probability,
    parameter_vector_hash,
    proposal_support_status,
    sample_scenario_rows,
)

_SCRIPT_PATH = Path("scripts/benchmark/run_rare_event_estimation_issue_4163.py")
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_rare_event_estimation_issue_4163",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC is not None
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
assert _SCRIPT_SPEC.loader is not None
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
run_smoke = _SCRIPT_MODULE.main


def _toy_spec_payload(samples: int = 2000) -> dict:
    return {
        "schema_version": "rare_event_sampling.v1",
        "proposal": "tilted_distribution",
        "parameters": {
            "x": {
                "base": "uniform",
                "low": 0.0,
                "high": 1.0,
                "proposal_low": 0.8,
                "proposal_high": 1.0,
            }
        },
        "objective_event": "x_exceeds_0_95",
        "samples": samples,
        "seed": 4163,
    }


def test_importance_sampling_recovers_analytic_uniform_tail_probability() -> None:
    """Likelihood ratios recover the known P_base[x >= 0.95] = 0.05 toy rate."""

    spec = RareEventSamplingSpec.from_payload(_toy_spec_payload(samples=2000))
    rows = sample_scenario_rows(spec)
    estimate = estimate_failure_probability(
        rows,
        [row.parameters["x"] >= 0.95 for row in rows],
        objective_event=spec.objective_event,
    )
    assert estimate.estimate == pytest.approx(0.05, abs=0.008)
    assert estimate.confidence_interval[0] <= 0.05 <= estimate.confidence_interval[1]
    assert estimate.naive_monte_carlo_estimate > estimate.estimate
    assert estimate.importance_sampling_variance >= 0.0
    assert estimate.naive_monte_carlo_variance >= 0.0
    assert estimate.variance_ratio_vs_naive is not None
    assert all(row.likelihood_ratio == pytest.approx(0.2) for row in rows)


def test_malformed_proposal_distribution_fails_closed() -> None:
    """Invalid proposal bounds are rejected before sampling."""

    payload = _toy_spec_payload(samples=1)
    payload["parameters"]["x"]["proposal_low"] = 1.0
    payload["parameters"]["x"]["proposal_high"] = 0.8
    with pytest.raises(RareEventSamplingError, match="proposal uniform bounds"):
        RareEventSamplingSpec.from_payload(payload)


def test_scenario_mutation_is_deterministic_and_does_not_edit_input() -> None:
    """Supported sampled knobs are applied to a copy and recorded with a stable hash."""

    parameters = {
        "ped_density": 0.07,
        "crossing_time_offset_s": -0.5,
        "pedestrian_speed_multiplier": 1.2,
    }
    row = SampledScenarioRow(
        sample_index=0,
        seed=4163,
        parameters=parameters,
        base_probability=2.0,
        proposal_probability=4.0,
        likelihood_ratio=0.5,
        parameter_vector_hash=parameter_vector_hash(parameters),
    )
    scenario = {
        "name": "crossing",
        "simulation_config": {"ped_density": 0.01},
        "single_pedestrians": [{"speed": 1.0}],
        "metadata": {"archetype": "crossing"},
    }
    mutated = apply_sampled_scenario_mutation(scenario, row)
    assert scenario["simulation_config"]["ped_density"] == 0.01
    assert mutated["simulation_config"]["ped_density"] == pytest.approx(0.07)
    assert mutated["simulation_config"]["crossing_time_offset_s"] == pytest.approx(-0.5)
    assert mutated["single_pedestrians"][0]["speed"] == pytest.approx(1.2)
    assert (
        mutated["metadata"]["rare_event_sampling"]["parameter_vector_hash"]
        == row.parameter_vector_hash
    )


def test_estimator_rejects_non_finite_weights() -> None:
    """Estimator fails closed when likelihood-ratio bookkeeping is unusable."""

    row = SampledScenarioRow(
        sample_index=0,
        seed=4163,
        parameters={"x": 1.0},
        base_probability=1.0,
        proposal_probability=1.0,
        likelihood_ratio=float("nan"),
        parameter_vector_hash="bad",
    )
    with pytest.raises(RareEventSamplingError, match="likelihood ratios"):
        estimate_failure_probability([row], [True])


def test_smoke_runner_writes_summary_with_provenance(tmp_path: Path) -> None:
    """CLI smoke writes estimator, sample rows, and scenario provenance."""

    config = tmp_path / "rare_event.yaml"
    payload = _toy_spec_payload(samples=8)
    payload["parameters"] = {
        "ped_density": {
            "base": "uniform",
            "low": 0.01,
            "high": 0.08,
            "proposal_low": 0.04,
            "proposal_high": 0.08,
        }
    }
    payload["objective_event"] = "collision_or_severe_intrusion"
    payload["scenario_matrix"] = "configs/scenarios/planner_sanity_matrix_v1.yaml"
    payload["synthetic_toy_model"] = {"parameter": "ped_density", "threshold": 0.065}
    config.write_text(json.dumps(payload), encoding="utf-8")

    output_dir = tmp_path / "output"
    evidence_dir = tmp_path / "evidence"
    assert (
        run_smoke(
            [
                "--config",
                str(config),
                "--output-dir",
                str(output_dir),
                "--evidence-dir",
                str(evidence_dir),
            ]
        )
        == 0
    )

    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "rare_event_sampling.v1"
    assert summary["estimator"]["objective_event"] == "collision_or_severe_intrusion"
    assert summary["scenario_provenance"]["scenario_name"] == "planner_sanity_simple"
    assert summary["sample_provenance"]["row_count"] == 8
    assert len(summary["sample_provenance"]["parameter_vector_hashes"]) == 8
    assert summary["planner_arms"] == ["synthetic_planner"]
    assert summary["episode_budget"] == 8
    assert summary["empirical_reference"]["status"] == "blocked"
    assert summary["empirical_reference"]["comparison"] == "larger_bruteforce_reference"
    assert "larger brute-force reference" in summary["empirical_reference"]["next_action"]
    assert "synthetic_planner" in summary["arm_estimates"]
    assert "variance_ratio_vs_naive" in summary["arm_estimates"]["synthetic_planner"]
    assert (output_dir / "episodes.jsonl").exists()


def test_static_constriction_config_reports_two_arm_diagnostic_summary(tmp_path: Path) -> None:
    """Static-constriction smoke config records family and per-arm estimator provenance."""

    output_dir = tmp_path / "output"
    evidence_dir = tmp_path / "evidence"
    assert (
        run_smoke(
            [
                "--config",
                "configs/benchmarks/rare_event/issue_4163_static_constriction_smoke.yaml",
                "--output-dir",
                str(output_dir),
                "--evidence-dir",
                str(evidence_dir),
            ]
        )
        == 0
    )

    summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["planner_arms"] == ["ppo_frozen", "ppo_frozen_wrapper_on"]
    assert summary["episode_budget"] == 64
    assert set(summary["arm_estimates"]) == {"ppo_frozen", "ppo_frozen_wrapper_on"}
    assert summary["static_constriction_family"]["family_id"] == "static_constriction"
    assert summary["static_constriction_family"]["scenario_ids"] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    assert summary["empirical_reference"]["status"] == "blocked"
    assert summary["empirical_reference"]["claim_boundary"].startswith("Diagnostic-only")
    for estimate in summary["arm_estimates"].values():
        assert estimate["objective_event"] == "collision_or_severe_intrusion"
        assert "confidence_interval" in estimate
        assert "variance_ratio_vs_naive" in estimate

    episode_rows = (output_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(episode_rows) == 64


def test_smoke_runner_rejects_empty_planner_arms(tmp_path: Path) -> None:
    """An explicitly empty planner arm list fails instead of falling back silently."""

    config_path = tmp_path / "rare_event_empty_arms.yaml"
    config_path.write_text(
        """
schema_version: rare_event_sampling.v1
proposal: tilted_distribution
parameters:
  ped_density:
    base: uniform
    low: 0.01
    high: 0.08
    proposal_low: 0.045
    proposal_high: 0.08
objective_event: collision_or_severe_intrusion
samples: 2
seed: 4163
scenario_matrix: configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml
planner_arms: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="planner_arms must be a non-empty list"):
        run_smoke(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--evidence-dir",
                str(tmp_path / "evidence"),
            ]
        )


def test_smoke_runner_rejects_available_reference_without_evidence(tmp_path: Path) -> None:
    """Available empirical references must point at the reference evidence."""

    config_path = tmp_path / "rare_event_bad_reference.yaml"
    payload = _toy_spec_payload(samples=2)
    payload["parameters"] = {
        "ped_density": {
            "base": "uniform",
            "low": 0.01,
            "high": 0.08,
            "proposal_low": 0.04,
            "proposal_high": 0.08,
        }
    }
    payload["objective_event"] = "collision_or_severe_intrusion"
    payload["empirical_reference"] = {
        "status": "available",
        "comparison": "larger_bruteforce_reference",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="empirical_reference.evidence_path"):
        run_smoke(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--evidence-dir",
                str(tmp_path / "evidence"),
            ]
        )


@pytest.mark.parametrize("empty_path", [None, "", "   "])
def test_smoke_runner_rejects_available_reference_with_empty_evidence_path(
    tmp_path: Path, empty_path: object
) -> None:
    """A present-but-empty evidence_path must not bypass the available-reference gate."""

    config_path = tmp_path / "rare_event_empty_reference.yaml"
    payload = _toy_spec_payload(samples=2)
    payload["parameters"] = {
        "ped_density": {
            "base": "uniform",
            "low": 0.01,
            "high": 0.08,
            "proposal_low": 0.04,
            "proposal_high": 0.08,
        }
    }
    payload["objective_event"] = "collision_or_severe_intrusion"
    payload["empirical_reference"] = {
        "status": "available",
        "comparison": "larger_bruteforce_reference",
        "evidence_path": empty_path,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="empirical_reference.evidence_path"):
        run_smoke(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--evidence-dir",
                str(tmp_path / "evidence"),
            ]
        )


def _restricted_spec_payload() -> dict:
    """Base uniform[0, 1] with a proposal that cannot see the event mass."""
    return {
        "schema_version": "rare_event_sampling.v1",
        "proposal": "tilted_distribution",
        "parameters": {
            "x": {
                "base": "uniform",
                "low": 0.0,
                "high": 1.0,
                "proposal_low": 0.8,
                "proposal_high": 1.0,
            }
        },
        "objective_event": "x_below_0_1",
        "samples": 500,
        "seed": 4163,
    }


def test_uncovered_event_estimates_zero_with_restricted_support_warning() -> None:
    """A proposal missing the event mass estimates 0 and says so loudly."""
    spec = RareEventSamplingSpec.from_payload(_restricted_spec_payload())
    rows = sample_scenario_rows(spec)
    assert rows and all(0.8 <= row.parameters["x"] <= 1.0 for row in rows)
    events = [row.parameters["x"] < 0.1 for row in rows]
    assert not any(events)
    estimate = estimate_failure_probability(rows, events, objective_event=spec.objective_event)
    assert estimate.estimate == 0.0
    assert estimate.confidence_interval == (0.0, 0.0)
    assert estimate.variance_ratio_vs_naive is None
    assert any(w.startswith("zero_weighted_events") for w in estimate.warnings)
    assert any(w.startswith("variance_ratio_undefined") for w in estimate.warnings)
    status, detail = proposal_support_status(spec)
    assert status == "restricted_support"
    assert "x" in detail
    summary = build_sampling_summary(spec=spec, rows=rows, events=events)
    assert summary["support_status"]["status"] == "restricted_support"
    assert summary["estimator"]["warnings"] == list(estimate.warnings)


def test_zero_observed_events_flag_collapsed_interval() -> None:
    """Zero events under full support still warn instead of certifying impossibility."""
    payload = _restricted_spec_payload()
    payload["parameters"] = {"x": {"base": "uniform", "low": 0.0, "high": 1.0}}
    payload["objective_event"] = "x_below_0_001"
    spec = RareEventSamplingSpec.from_payload(payload)
    assert proposal_support_status(spec)[0] == "full_support"
    rows = sample_scenario_rows(spec)
    events = [False] * len(rows)
    estimate = estimate_failure_probability(rows, events, objective_event=spec.objective_event)
    assert estimate.estimate == 0.0
    assert estimate.confidence_interval == (0.0, 0.0)
    assert any(w.startswith("zero_weighted_events") for w in estimate.warnings)


def test_single_dominant_weight_flags_degeneracy() -> None:
    """One nonzero weight is a single effective sample, not a measurement."""
    rows = [
        SampledScenarioRow(
            sample_index=index,
            seed=7,
            parameters={"x": 0.9},
            base_probability=1.0,
            proposal_probability=1.0,
            likelihood_ratio=weight,
            parameter_vector_hash=f"row-{index}",
        )
        for index, weight in enumerate([1.0, 0.0, 0.0, 0.0])
    ]
    estimate = estimate_failure_probability(
        rows, [True, False, False, False], objective_event="x_below_0_1"
    )
    assert estimate.estimate == pytest.approx(0.25)
    assert estimate.effective_sample_size == pytest.approx(1.0)
    assert len(estimate.warnings) == 1
    assert estimate.warnings[0].startswith("weight_degeneracy")


def test_proposal_event_frequency_names_the_naive_value_honestly() -> None:
    """The honestly named field matches the legacy value on a covered event."""
    spec = RareEventSamplingSpec.from_payload(_toy_spec_payload(samples=500))
    rows = sample_scenario_rows(spec)
    estimate = estimate_failure_probability(
        rows,
        [row.parameters["x"] >= 0.95 for row in rows],
        objective_event=spec.objective_event,
    )
    assert estimate.estimate == pytest.approx(0.05, abs=0.015)
    assert estimate.proposal_event_frequency == estimate.naive_monte_carlo_estimate
    assert estimate.warnings == ()
