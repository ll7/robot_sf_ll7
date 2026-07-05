"""Tests for the forecast-risk closed-loop coupling gate (issue #2916).

Covers:
- adapter risk mapping (proximity -> bounded scalar risk),
- fail-closed on degraded / missing / oracle batches,
- verdict logic for all three branches (continue | revise | stop),
- seed-identity across rows in the runner.

All inputs are small synthetic fixtures; no heavy simulator is started.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys

import numpy as np
import pytest

from robot_sf.benchmark.forecast_batch import (
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
)
from robot_sf.benchmark.forecast_risk_adapter import (
    FORECAST_RISK_ADAPTER_SCHEMA_VERSION,
    compute_forecast_risk,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts/benchmark/run_forecast_risk_coupling_gate.py"


def _load_runner():
    """Import the runner script as a module for verdict/seed-identity tests.

    Returns:
        The imported runner module.
    """
    module_name = "run_forecast_risk_coupling_gate"
    spec = importlib.util.spec_from_file_location(module_name, _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass type resolution can find the module.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_batch(
    deterministic: np.ndarray | None,
    *,
    fallback_status: str = "native",
    degraded_status: str = "none",
    oracle_state: bool = False,
    actor_mask: list[bool] | None = None,
    include_forecast: bool = True,
) -> ForecastBatch:
    """Build a minimal synthetic ForecastBatch for adapter tests.

    Returns:
        A validated ForecastBatch with one actor.
    """
    mask = actor_mask if actor_mask is not None else [True]
    forecasts = []
    if include_forecast and mask[0]:
        forecasts.append(ActorForecast(actor_id="0", deterministic=deterministic))
    provenance = ForecastBatchProvenance(
        predictor_id="cv-test",
        predictor_family="constant_velocity",
        observation_tier="tracked_agents",
        frame=CoordinateFrame(name="world", units="m", axes=("x", "y")),
        dt_s=0.1,
        horizons_s=[0.5, 1.0],
        scenario_id="synthetic",
        seed=7,
        fallback_status=fallback_status,
        degraded_status=degraded_status,
        actor_ids=["0"],
        actor_mask=mask,
        actor_mask_metadata={"semantics": "true means available"},
        feature_schema={"position": "xy_m"},
        timestamp="1970-01-01T00:00:00+00:00",
        oracle_state=oracle_state,
    )
    return ForecastBatch(
        provenance=provenance,
        forecasts=forecasts,
        metadata={"artifact_role": "test"},
    )


# --------------------------------------------------------------------------- #
# Adapter risk mapping
# --------------------------------------------------------------------------- #


def test_adapter_close_forecast_gives_high_risk():
    """A forecast mean very near the robot yields high, available risk."""
    deterministic = np.array([[0.2, 0.0], [0.3, 0.0]], dtype=float)
    batch = _make_batch(deterministic)
    signal = compute_forecast_risk(batch, [0.0, 0.0], influence_radius_m=3.0)
    assert signal.available is True
    assert signal.reason == "ok"
    assert signal.risk > 0.9  # 1 - 0.2/3.0 ~= 0.933
    assert signal.contributing_actor_count == 1
    assert signal.nearest_predicted_distance_m == pytest.approx(0.2)


def test_adapter_far_forecast_gives_zero_risk():
    """A forecast mean beyond the influence radius yields zero (but available) risk."""
    deterministic = np.array([[10.0, 0.0], [11.0, 0.0]], dtype=float)
    batch = _make_batch(deterministic)
    signal = compute_forecast_risk(batch, [0.0, 0.0], influence_radius_m=3.0)
    assert signal.available is True
    assert signal.risk == 0.0


def test_adapter_risk_is_bounded_and_monotone():
    """Risk increases as the predicted occupancy approaches the robot."""
    near = compute_forecast_risk(
        _make_batch(np.array([[0.5, 0.0], [0.5, 0.0]])), [0.0, 0.0], influence_radius_m=3.0
    )
    far = compute_forecast_risk(
        _make_batch(np.array([[2.0, 0.0], [2.0, 0.0]])), [0.0, 0.0], influence_radius_m=3.0
    )
    assert 0.0 <= far.risk <= near.risk <= 1.0


def test_adapter_to_dict_has_schema_version():
    """The signal serializes with the adapter schema version."""
    signal = compute_forecast_risk(_make_batch(np.array([[1.0, 0.0], [1.0, 0.0]])), [0.0, 0.0])
    payload = signal.to_dict()
    assert payload["schema_version"] == FORECAST_RISK_ADAPTER_SCHEMA_VERSION
    assert payload["available"] is True


def test_adapter_rejects_non_positive_influence_radius():
    """A non-positive influence radius is a hard error."""
    with pytest.raises(ValueError, match="influence_radius_m"):
        compute_forecast_risk(
            _make_batch(np.array([[1.0, 0.0], [1.0, 0.0]])), [0.0, 0.0], influence_radius_m=0.0
        )


# --------------------------------------------------------------------------- #
# Fail-closed behavior
# --------------------------------------------------------------------------- #


def test_adapter_fails_closed_on_degraded_batch():
    """A degraded batch is unavailable with risk 0.0 (never a low-risk success)."""
    batch = _make_batch(np.array([[0.1, 0.0], [0.1, 0.0]]), degraded_status="degraded_observation")
    signal = compute_forecast_risk(batch, [0.0, 0.0])
    assert signal.available is False
    assert signal.risk == 0.0
    assert signal.reason.startswith("degraded:")


def test_adapter_fails_closed_on_fallback_batch():
    """A non-native fallback status fails closed."""
    batch = _make_batch(np.array([[0.1, 0.0], [0.1, 0.0]]), fallback_status="fallback")
    signal = compute_forecast_risk(batch, [0.0, 0.0])
    assert signal.available is False
    assert signal.reason.startswith("fallback:")


def test_adapter_fails_closed_on_oracle_batch():
    """An oracle-sourced batch fails closed even when geometry is close."""
    batch = _make_batch(np.array([[0.1, 0.0], [0.1, 0.0]]), oracle_state=True)
    signal = compute_forecast_risk(batch, [0.0, 0.0])
    assert signal.available is False
    assert signal.reason == "oracle_state"


def test_adapter_fails_closed_on_missing_actor():
    """A batch whose only actor is masked out (no forecast) fails closed."""
    batch = _make_batch(None, actor_mask=[False], include_forecast=False)
    signal = compute_forecast_risk(batch, [0.0, 0.0])
    assert signal.available is False
    assert signal.risk == 0.0


# --------------------------------------------------------------------------- #
# Verdict logic (all three branches)
# --------------------------------------------------------------------------- #


def _verdict_cfg() -> dict:
    """Return a minimal verdict config block.

    Returns:
        Config mapping with the verdict thresholds.
    """
    return {
        "verdict": {
            "min_safety_event_reduction": 1,
            "max_false_positive_stop_ratio": 3.0,
            "max_runtime_s": 5.0,
        }
    }


def _row(
    name: str, risk_source: str, *, safety_events: int, fp_stops: int, classification: str = "ok"
):
    """Build a synthetic per-row result for verdict tests.

    Returns:
        A row result mapping shaped like ``evaluate_row`` output.
    """
    return {
        "row": name,
        "risk_source": risk_source,
        "classification": classification,
        "classification_reason": "synthetic",
        "metrics": {
            "collision": safety_events >= 2,
            "near_miss": safety_events >= 1,
            "safety_events": safety_events,
            "false_positive_stops": fp_stops,
            "runtime_s": 0.001,
            "snqi": 0.0,
        },
    }


def test_verdict_continue_on_safety_benefit():
    """A forecast row that cuts safety events with no regression -> continue."""
    module = _load_runner()
    results = [
        _row("no_forecast", "none", safety_events=2, fp_stops=0),
        _row("cv_risk", "constant_velocity", safety_events=0, fp_stops=0),
        _row("semantic_risk", "semantic_cv", safety_events=1, fp_stops=0),
        _row("interaction_risk", "interaction_aware_cv", safety_events=1, fp_stops=0),
    ]
    verdict = module.emit_verdict(results, _verdict_cfg())
    assert verdict["decision"] == "continue"


def test_verdict_stop_on_regression_without_benefit():
    """Forecast rows that worsen safety with no benefit -> stop."""
    module = _load_runner()
    results = [
        _row("no_forecast", "none", safety_events=0, fp_stops=0),
        _row("cv_risk", "constant_velocity", safety_events=2, fp_stops=0),
        _row("semantic_risk", "semantic_cv", safety_events=2, fp_stops=0),
        _row("interaction_risk", "interaction_aware_cv", safety_events=2, fp_stops=0),
    ]
    verdict = module.emit_verdict(results, _verdict_cfg())
    assert verdict["decision"] == "stop"


def test_verdict_revise_when_inconclusive():
    """No safety benefit and no regression -> revise (conservative default)."""
    module = _load_runner()
    results = [
        _row("no_forecast", "none", safety_events=1, fp_stops=0),
        _row("cv_risk", "constant_velocity", safety_events=1, fp_stops=0),
        _row("semantic_risk", "semantic_cv", safety_events=1, fp_stops=0),
        _row("interaction_risk", "interaction_aware_cv", safety_events=1, fp_stops=0),
    ]
    verdict = module.emit_verdict(results, _verdict_cfg())
    assert verdict["decision"] == "revise"


def test_verdict_blocked_rows_do_not_count_as_benefit():
    """A blocked forecast row never contributes a safety benefit."""
    module = _load_runner()
    results = [
        _row("no_forecast", "none", safety_events=2, fp_stops=0),
        _row("cv_risk", "constant_velocity", safety_events=0, fp_stops=0, classification="blocked"),
        _row("semantic_risk", "semantic_cv", safety_events=2, fp_stops=0),
        _row("interaction_risk", "interaction_aware_cv", safety_events=2, fp_stops=0),
    ]
    verdict = module.emit_verdict(results, _verdict_cfg())
    assert "cv_risk" in verdict["blocked_rows"]
    # Blocked benefit row removed; remaining rows match control -> not a benefit.
    assert verdict["decision"] in {"revise", "stop"}


def test_verdict_false_positive_explosion_blocks_continue():
    """A safety benefit nullified by a false-positive-stop explosion -> not continue."""
    module = _load_runner()
    results = [
        _row("no_forecast", "none", safety_events=2, fp_stops=1),
        _row("cv_risk", "constant_velocity", safety_events=0, fp_stops=10),
        _row("semantic_risk", "semantic_cv", safety_events=2, fp_stops=1),
        _row("interaction_risk", "interaction_aware_cv", safety_events=2, fp_stops=1),
    ]
    verdict = module.emit_verdict(results, _verdict_cfg())
    # cv_risk cut events but exploded FP stops (ratio 10 > 3) -> regression, no clean benefit.
    assert verdict["decision"] != "continue"


# --------------------------------------------------------------------------- #
# Seed identity across rows
# --------------------------------------------------------------------------- #


def test_seed_identity_enforced_by_config():
    """The runner requires a single shared seed and scenario across rows."""
    module = _load_runner()
    good = {
        "fixture": {"seed": 111, "scenario_id": "s", "trace_path": "x", "dt_s": 0.1},
        "rows": [
            {"name": "no_forecast", "risk_source": "none"},
            {"name": "cv_risk", "risk_source": "constant_velocity"},
            {"name": "semantic_risk", "risk_source": "semantic_cv"},
            {"name": "interaction_risk", "risk_source": "interaction_aware_cv"},
        ],
    }
    module._verify_seed_identity(good)  # does not raise

    bad = {"fixture": {"scenario_id": "s"}, "rows": [{"name": "a", "risk_source": "none"}]}
    with pytest.raises(ValueError, match="seed"):
        module._verify_seed_identity(bad)


def test_seed_identity_requires_four_row_matrix():
    """Config validation should fail before evaluation when any required row is absent."""
    module = _load_runner()
    bad = {
        "fixture": {"seed": 111, "scenario_id": "s", "trace_path": "x", "dt_s": 0.1},
        "rows": [
            {"name": "no_forecast", "risk_source": "none"},
            {"name": "cv_risk", "risk_source": "constant_velocity"},
        ],
    }

    with pytest.raises(ValueError, match="exactly the four forecast-risk rows"):
        module._verify_seed_identity(bad)


def test_observed_pedestrian_non_finite_position_is_unavailable():
    """Malformed observed positions should make forecast risk unavailable, not crash."""
    module = _load_runner()
    frame = {"observed_pedestrians": [{"id": 1, "position": [float("nan"), 0.0]}]}

    assert module._ped_state_from_frame(frame) is None


def test_ground_truth_non_finite_position_fails_clearly():
    """Outcome scoring should reject non-finite ground-truth pedestrian positions."""
    module = _load_runner()
    state = module._RowState(robot_pos=np.array([0.0, 0.0], dtype=float))
    frame = {"pedestrians": [{"position": [float("inf"), 0.0]}]}

    with pytest.raises(ValueError, match="finite xy vector"):
        module._score_outcome_step(
            frame,
            action="go",
            risk_source="constant_velocity",
            conflict_distance_m=1.5,
            state=state,
        )


def test_unavailable_signal_in_conflict_fails_closed():
    """Unavailable forecast signals during a conflict window should block the row."""
    module = _load_runner()
    state = module._RowState(robot_pos=np.array([0.0, 0.0], dtype=float))
    state.risk_unavailable_this_step = True
    frame = {"pedestrians": [{"position": [0.5, 0.0]}]}

    module._score_outcome_step(
        frame,
        action="go",
        risk_source="constant_velocity",
        conflict_distance_m=1.5,
        state=state,
    )

    assert state.risk_unavailable_in_conflict == 1
    classification, reason = module._classify_row(
        risk_source="constant_velocity",
        conflict_steps=1,
        risk_available_steps=0,
        risk_unavailable_in_conflict=1,
    )
    assert classification == "blocked"
    assert "fail-closed" in reason


def test_run_accepts_absolute_config_outside_repo(tmp_path: pathlib.Path):
    """Repro command formatting should not crash for config paths outside the repo."""
    module = _load_runner()
    source_config = REPO_ROOT / "configs/research/forecast_risk_coupling_issue_2916.yaml"
    config_path = tmp_path / "forecast_risk_coupling_issue_2916.yaml"
    shutil.copy(source_config, config_path)

    report = module.run(config_path, tmp_path / "out")

    assert report["reproducibility"]["config_path"] == str(config_path)
    assert f"--config {config_path}" in report["reproducibility"]["command"]


def test_runner_rows_share_seed_and_scenario(tmp_path):
    """End-to-end: all rows in the report carry the same seed/scenario denominator."""
    module = _load_runner()
    config_path = REPO_ROOT / "configs/research/forecast_risk_coupling_issue_2916.yaml"
    report = module.run(config_path, tmp_path)
    fixture = report["config"]["fixture"]
    # Every row was evaluated against the same fixture seed/scenario.
    assert fixture["seed"] == 111
    assert fixture["scenario_id"] == "issue_2756_occluded_emergence"
    row_names = {row["row"] for row in report["rows"]}
    assert row_names == {"no_forecast", "cv_risk", "semantic_risk", "interaction_risk"}
    assert {row["seed"] for row in report["rows"]} == {fixture["seed"]}
    assert {row["scenario_id"] for row in report["rows"]} == {fixture["scenario_id"]}
    # The verdict is one of the three allowed branches.
    assert report["verdict"]["decision"] in {"continue", "revise", "stop"}
