"""Tests for the Issue #2227 contrastive recenter/topology mechanism panels.

These tests run the real planners (short horizons) and assert the contractual
properties of the deliverable:

* both mechanisms' exported traces validate against ``simulation_trace_export.v1``;
* the activation diagnostic is reported per arm;
* contrastive panels (PNG + PDF) are produced for both arms of each mechanism;
* runs are deterministic for a fixed seed/horizon;
* the honest-null path (mechanism expected but does not activate, or activates
  without an observable delta) is representable and labelled.

The runs are deliberately short to keep the suite affordable while still
exercising the activation paths.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

import scripts.analysis.build_recenter_topology_panels_issue_2227 as panels
from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)

if TYPE_CHECKING:
    from pathlib import Path

_NULL_CLASSIFICATIONS = {
    "expected_here_did_not_activate",
    "activated_no_observable_change",
}
_REAL_DELTA_CLASSIFICATIONS = {
    "activated_outcome_changed",
    "activated_trace_changed_outcome_unchanged",
}


def _spec(mechanism_id: str, *, horizon: int) -> panels.MechanismSpec:
    """Return the named mechanism spec with a reduced horizon for tests."""
    base = next(spec for spec in panels.MECHANISMS if spec.mechanism_id == mechanism_id)
    return dataclasses.replace(base, horizon=horizon)


@pytest.fixture(scope="module")
def static_recenter_result(tmp_path_factory: pytest.TempPathFactory) -> panels.MechanismResult:
    """Run the static-recenter mechanism once (reduced horizon) for the module."""
    output_dir = tmp_path_factory.mktemp("static_recenter")
    spec = _spec("static_recenter", horizon=60)
    return panels._run_mechanism(
        spec,
        registry_path=panels.DEFAULT_REGISTRY,
        output_dir=output_dir,
        commit="test-commit",
        command="pytest",
    )


@pytest.fixture(scope="module")
def topology_result(tmp_path_factory: pytest.TempPathFactory) -> panels.MechanismResult:
    """Run the topology-command mechanism once (reduced horizon) for the module."""
    output_dir = tmp_path_factory.mktemp("topology")
    spec = _spec("topology_command", horizon=60)
    return panels._run_mechanism(
        spec,
        registry_path=panels.DEFAULT_REGISTRY,
        output_dir=output_dir,
        commit="test-commit",
        command="pytest",
    )


@pytest.mark.parametrize("fixture_name", ["static_recenter_result", "topology_result"])
def test_both_arm_traces_validate_against_schema(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Both arms export traces that load cleanly against the v1 schema."""
    result: panels.MechanismResult = request.getfixturevalue(fixture_name)
    for arm in (result.off, result.on):
        assert arm.trace_path.exists()
        trace = load_simulation_trace_export(arm.trace_path)  # raises on schema violation
        assert trace.schema_version == "simulation_trace_export.v1"
        assert trace.source.planner_id == arm.planner_id
        assert len(trace.frames) == arm.frame_count > 0


@pytest.mark.parametrize("fixture_name", ["static_recenter_result", "topology_result"])
def test_activation_diagnostic_reported_per_arm(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Each arm carries an explicit activation diagnostic with an ``active`` flag."""
    result: panels.MechanismResult = request.getfixturevalue(fixture_name)
    for arm in (result.off, result.on):
        assert "active" in arm.activation
        assert isinstance(arm.activation["active"], bool)
        assert arm.activation.get("diagnostic")
    # The disabled arm must never report the mechanism as active.
    assert result.off.activation["active"] is False


@pytest.mark.parametrize("fixture_name", ["static_recenter_result", "topology_result"])
def test_panels_png_and_pdf_produced(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Both arms render a panel with PNG and PDF artifacts."""
    result: panels.MechanismResult = request.getfixturevalue(fixture_name)
    assert len(result.panel_artifacts) == 2
    panel_dir = result.on.trace_path.parents[2] / "panels" / result.spec.mechanism_id
    pngs = sorted(panel_dir.rglob("*.png"))
    pdfs = sorted(panel_dir.rglob("*.pdf"))
    assert len(pngs) == 2
    assert len(pdfs) == 2
    caption = (panel_dir / "mechanism_caption.md").read_text(encoding="utf-8")
    assert panels.CLAIM_BOUNDARY in caption
    assert "Activated?" in caption


def test_isolation_only_flag_differs() -> None:
    """Toggling builds configs that differ only by the mechanism flag."""
    spec = _spec("static_recenter", horizon=10)
    base = panels._runtime_config(panels.DEFAULT_REGISTRY, spec.candidate)
    on_config = dict(base)
    on_config[spec.flag] = True
    off_config = dict(base)
    off_config[spec.flag] = False
    differing = {
        key for key in set(on_config) | set(off_config) if on_config.get(key) != off_config.get(key)
    }
    assert differing == {spec.flag}


def test_static_recenter_determinism(tmp_path: Path) -> None:
    """Re-running the static-recenter mechanism yields identical exported frames."""
    spec = _spec("static_recenter", horizon=40)
    traces: list[list] = []
    for run in range(2):
        out = tmp_path / f"run_{run}"
        result = panels._run_mechanism(
            spec,
            registry_path=panels.DEFAULT_REGISTRY,
            output_dir=out,
            commit="test-commit",
            command="pytest",
        )
        trace = load_simulation_trace_export(result.on.trace_path)
        traces.append([(f.step, tuple(f.robot["position"])) for f in trace.frames])
    assert traces[0] == traces[1]


@pytest.mark.parametrize("fixture_name", ["static_recenter_result", "topology_result"])
def test_classification_matches_observed_delta(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Classification is internally consistent and honest (null vs real delta)."""
    result: panels.MechanismResult = request.getfixturevalue(fixture_name)
    if not result.activated:
        assert result.classification == "expected_here_did_not_activate"
        assert result.null_reason is not None
    elif result.classification in _NULL_CLASSIFICATIONS:
        # Activated but no observable change -> must be reported as an honest null.
        assert result.null_reason is not None
        assert not result.outcome_changed
    else:
        assert result.classification in _REAL_DELTA_CLASSIFICATIONS
        assert result.command_source_changed or result.trajectory_delta_m > 1e-6


def test_honest_null_path_is_representable(tmp_path: Path) -> None:
    """A non-activating arm is classified as an honest null, not a fabricated delta."""
    spec = next(spec for spec in panels.MECHANISMS if spec.mechanism_id == "static_recenter")
    # Construct two synthetic arms where the mechanism never activates.
    inactive_activation = {
        "diagnostic": "static_recenter_term_positive_in_decision_trace",
        "active": False,
        "recenter_term_activation_count": 0,
        "first_activation_step": None,
    }
    trace_doc = {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": "null_case",
        "source": {
            "scenario_id": spec.scenario_id,
            "seed": spec.seed,
            "planner_id": "p",
            "episode_id": "e",
            "generated_by": "test",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": [
            {
                "step": 0,
                "time_s": 0.1,
                "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
                "pedestrians": [],
                "planner": {"selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0}},
            }
        ],
    }
    null_path = tmp_path / "issue_2227_null_case.json"
    null_path.write_text(json.dumps(trace_doc), encoding="utf-8")
    arm = panels.ArmResult(
        arm="on",
        planner_id="p",
        enabled=True,
        trace_path=null_path,
        frame_count=1,
        activation=inactive_activation,
        selected_source_counts={"dynamic_window": 1},
        terminal={"success": False, "steps": 1},
    )
    off = panels.ArmResult(
        arm="off",
        planner_id="p_off",
        enabled=False,
        trace_path=null_path,
        frame_count=1,
        activation={"active": False},
        selected_source_counts={"dynamic_window": 1},
        terminal={"success": False, "steps": 1},
    )
    result = panels._classify(spec, arm, off)
    assert result.activated is False
    assert result.classification == "expected_here_did_not_activate"
    assert result.null_reason is not None
