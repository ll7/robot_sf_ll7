"""Regression coverage for pedestrian integration-scheme selection (issue #4979)."""

from __future__ import annotations

import numpy as np
import pytest
from pysocialforce.config import SceneConfig
from pysocialforce.scene import PedState, normalize_integration_scheme

from robot_sf.sim.pedestrian_model_variants import step_hsfm_total_force
from robot_sf.sim.sim_config import SimulationSettings
from scripts.analysis.measure_pedestrian_integrator_scheme_sensitivity import (
    SCHEMA_VERSION,
    measure_scheme_sensitivity,
)


def _ped_state(scheme: str) -> PedState:
    """Build a one-pedestrian fixture with a known acceleration."""
    state = np.array([[0.0, 0.0, 1.0, 0.0, 10.0, 0.0]], dtype=float)
    return PedState(state, [], SceneConfig(dt_secs=0.1, integration_scheme=scheme))


def test_explicit_and_semi_implicit_euler_use_different_position_velocities() -> None:
    """Explicit uses pre-step velocity while semi-implicit uses the updated velocity."""
    explicit = _ped_state("explicit_euler")
    semi_implicit = _ped_state("semi_implicit_euler")

    explicit.step(np.array([[-2.0, 0.0]], dtype=float))
    semi_implicit.step(np.array([[-2.0, 0.0]], dtype=float))

    assert explicit.pos()[0] == pytest.approx([0.1, 0.0])
    assert semi_implicit.pos()[0] == pytest.approx([0.08, 0.0])
    assert explicit.vel()[0] == pytest.approx(semi_implicit.vel()[0])


def test_default_scheme_preserves_historical_semi_implicit_update_order() -> None:
    """The default names and retains the repository's existing position update order."""
    settings = SimulationSettings()

    assert settings.pedestrian_integration_scheme == "semi_implicit_euler"
    default = PedState(
        np.array([[0.0, 0.0, 1.0, 0.0, 10.0, 0.0]], dtype=float), [], SceneConfig(dt_secs=0.1)
    )
    default.step(np.array([[-2.0, 0.0]], dtype=float))
    assert default.pos()[0] == pytest.approx([0.08, 0.0])


def test_invalid_integration_scheme_fails_closed() -> None:
    """Unknown scheme names are rejected before simulation construction."""
    with pytest.raises(ValueError, match="Unsupported integration_scheme"):
        SimulationSettings(pedestrian_integration_scheme="rk4")


def test_none_integration_scheme_preserves_compatibility_default() -> None:
    """A nullable configuration seam retains the historical semi-implicit behavior."""

    assert normalize_integration_scheme(None) == "semi_implicit_euler"


def test_hsfm_total_force_honors_explicit_position_update() -> None:
    """Opt-in headed variants share the same selected position-update contract."""
    state = np.array([[0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    next_state, _ = step_hsfm_total_force(
        state,
        np.array([[2.0, 0.0]], dtype=float),
        np.array([0.0], dtype=float),
        dt=0.1,
        max_speeds=np.array([10.0], dtype=float),
        integration_scheme="explicit_euler",
    )

    assert next_state[0, :4] == pytest.approx([0.1, 0.0, 1.2, 0.0])


def test_scheme_sensitivity_report_covers_all_speed_archetypes() -> None:
    """The diagnostic emits finite divergence metrics for the three configured archetypes."""
    report = measure_scheme_sensitivity(dt=0.1, steps=12)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["evidence_status"] == "diagnostic-only"
    assert report["schemes"] == {
        "baseline": "semi_implicit_euler",
        "comparison": "explicit_euler",
    }
    rows = report["trajectory_divergence"]
    assert [row["archetype"] for row in rows] == ["cautious", "standard", "hurried"]
    assert all(np.isfinite(row["max_position_divergence_m"]) for row in rows)
    assert any(row["max_position_divergence_m"] > 0 for row in rows)


def test_scheme_sensitivity_fails_closed_for_non_finite_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diagnostic report cannot silently summarize non-finite trajectories."""

    import scripts.analysis.measure_pedestrian_integrator_scheme_sensitivity as diagnostic

    monkeypatch.setattr(diagnostic, "_trajectory", lambda **_kwargs: np.array([[[np.nan, 0.0]]]))

    with pytest.raises(ValueError, match="Non-finite trajectory divergence"):
        diagnostic.measure_scheme_sensitivity(steps=1)
