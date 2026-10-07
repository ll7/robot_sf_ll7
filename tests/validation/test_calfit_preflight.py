"""No-stepping controls for the CALFIT admission and geometry audit."""

import json

import pytest

from scripts.validation import calfit_preflight_10074 as preflight
from scripts.validation import pedestrian_validation_10074 as suite


def test_known_trajectories_make_every_primary_estimator_observable():
    controls = preflight.estimator_controls()
    assert {c["case"] for c in controls} == {f"V{i}" for i in range(1, 7)}
    assert len(controls) == 9
    assert all(c["known_answer_pass"] for c in controls)
    assert [c["observed"] for c in controls[:7]] == pytest.approx(
        [1.29, 0.54, 0.40, 1.90, 6 / 11, 0.50, 0.50], abs=1e-8
    )
    assert controls[7]["observed"] is None
    assert controls[8]["observed"] == pytest.approx(5.0, abs=0.2)


def test_target_records_pass_author_policy_before_model_search():
    result = preflight.audit()
    assert result["ideal_gate"]["measurement_missing"] == []
    assert result["ideal_gate"]["physical_violations"] == []
    assert result["ideal_gate"]["exit_code"] == 0
    assert {g["exit_code"] for g in result["per_case_gate"].values()} == {0}
    assert result["search_admissible"] is True
    assert result["experiment_episodes"] == 0


def test_narrowest_aperture_requires_penetration_at_every_requested_radius():
    rows = [
        g for g in preflight.audit()["rigid_disc_aperture_geometry"] if g["shoulder_ratio"] == 0.9
    ]
    assert [g["aperture_width_m"] for g in rows] == pytest.approx([0.414] * 3)
    assert [g["centre_slack_m"] for g in rows] == pytest.approx([-0.086, -0.146, -0.186])
    assert [g["minimum_continuous_penetration_m"] for g in rows] == pytest.approx(
        [0.043, 0.073, 0.093]
    )


def test_actual_gate_keeps_missingness_and_physics_separate_from_policy():
    rows = preflight.ideal_gate_records()
    rows[0]["fitted_tau_s"] = None
    rows[2]["wall_penetration_m"] = 0.043
    gate = suite.acceptance_gate(rows)
    assert gate["exit_code"] == 3
    assert gate["measurement_missing"] == ["V1/native/1001"]
    assert gate["physical_violations"] == ["V2/0.61/1001"]


def test_cli_persists_admissible_receipt_with_source_identity(tmp_path):
    out = tmp_path / "preflight.json"
    assert preflight.main(["--out", str(out)]) == 0
    receipt = json.loads(out.read_text())
    assert receipt["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert receipt["all_estimators_known_answer_pass"] is True
    assert len(receipt["source_sha256"]) == 7
    assert "robot_sf/research/pedestrian_acceptance.py" in receipt["source_sha256"]
    assert all(len(digest) == 64 for digest in receipt["source_sha256"].values())


def test_feasible_geometry_roundoff_does_not_block_preflight(monkeypatch):
    monkeypatch.setattr(suite, "acceptance_gate", lambda *args, **kwargs: {"exit_code": 0})
    assert preflight.audit()["search_admissible"] is True
