"""Tests for the issue #3067 clean/noisy/partial observation-robustness slice."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py"
FIXTURE_PATH = (
    REPO_ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/"
    "dense_pedestrian_stress_episode_0000.json"
)
_LOADED = None


def _load_module():
    """Load the slice runner as an importable module."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    spec = importlib.util.spec_from_file_location(
        "run_sensor_noise_robustness_slice_issue_3067", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _LOADED = mod
    return _LOADED


def _frames():
    """Return the fixture frames."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["frames"]


def _eval_all():
    """Evaluate every matrix row against the fixture."""
    mod = _load_module()
    frames = _frames()
    return mod, [mod.evaluate_row(name, cfg, frames) for name, cfg in mod.MATRIX.items()]


def test_clean_noisy_partial_rows_present_with_metadata():
    """Matrix yields clean, noisy, and partial rows, each with perturbation metadata."""
    _mod, rows = _eval_all()
    families = {r["family"] for r in rows}
    assert {"clean", "noisy", "partial"} <= families
    for row in rows:
        meta = row["perturbation_metadata"]
        assert meta["metadata_complete"] is True
        assert "observation_quality" in meta
        assert row["observation_level"]["key"]  # observation-level metadata present
        assert "noise_profile" in row["spec"]


def test_clean_vs_perturbed_delta_computed():
    """Deltas are computed per metric against the clean reference row."""
    mod, rows = _eval_all()
    deltas = mod.compute_deltas(rows)
    assert deltas["available"] is True
    assert deltas["reference_row"] == "clean"
    # Every non-clean row gets a delta entry with the core metrics.
    perturbed = [r for r in rows if r["family"] != "clean"]
    assert set(deltas["rows"]) == {r["row"] for r in perturbed}
    for entry in deltas["rows"].values():
        assert "min_observed_distance_m" in entry["deltas"]
        assert "observation_continuity" in entry["deltas"]


def test_perturbations_move_a_metric_non_null():
    """The pedestrian-dominated fixture yields a non-null clean-vs-perturbed delta."""
    mod, rows = _eval_all()
    deltas = mod.compute_deltas(rows)
    assert deltas["any_perturbed_metric_moved"] is True


def test_incomplete_metadata_blocks_overall_classification():
    """Incomplete per-row metadata fails closed to a blocked overall classification."""
    mod, rows = _eval_all()
    deltas = mod.compute_deltas(rows)
    # Corrupt metadata on a perturbed row.
    rows[1]["perturbation_metadata"]["metadata_complete"] = False
    overall = mod.classify_overall(rows, deltas)
    assert overall["label"] == "blocked"
    assert overall["label"] in mod.OVERALL_CLASSIFICATIONS


def test_unavailable_deltas_block():
    """A missing/unusable clean reference fails closed to blocked."""
    mod, _rows = _eval_all()
    overall = mod.classify_overall(_rows, {"available": False, "rows": {}})
    assert overall["label"] == "blocked"


def test_degraded_row_never_counts_as_success():
    """A fully-suppressed perturbed row is degraded and not counted as success."""
    mod = _load_module()
    frames = _frames()
    row = mod.evaluate_row(
        "all_missed",
        {
            "family": "partial",
            "observation_level": "occluded_partial_state",
            "description": "All actors missed.",
            "spec_kw": {"missed_detection_probability": 1.0, "seed": mod.SEED},
        },
        frames,
    )
    assert row["status"] == "degraded"
    deltas = mod.compute_deltas([r for r in _eval_all()[1] if r["family"] == "clean"] + [row])
    assert deltas["rows"]["all_missed"]["counts_as_success"] is False


def test_all_perturbed_degraded_yields_non_claim():
    """If every perturbed row is degraded, the overall is a non-claim."""
    mod = _load_module()
    frames = _frames()
    clean = mod.evaluate_row("clean", mod.MATRIX["clean"], frames)
    degraded = mod.evaluate_row(
        "all_missed",
        {
            "family": "partial",
            "observation_level": "occluded_partial_state",
            "description": "All actors missed.",
            "spec_kw": {"missed_detection_probability": 1.0, "seed": mod.SEED},
        },
        frames,
    )
    rows = [clean, degraded]
    deltas = mod.compute_deltas(rows)
    overall = mod.classify_overall(rows, deltas)
    assert overall["label"] == "non-claim"


def test_determinism_same_seed_same_metrics():
    """Two evaluations with the same seed produce identical metrics."""
    mod = _load_module()
    frames = _frames()
    a = mod.evaluate_row("noisy_medium", mod.MATRIX["noisy_medium"], frames)
    b = mod.evaluate_row("noisy_medium", mod.MATRIX["noisy_medium"], frames)
    # Drop runtime (wall-clock, non-deterministic) before comparing.
    a["metrics"].pop("perturbation_runtime_s")
    b["metrics"].pop("perturbation_runtime_s")
    assert a["metrics"] == b["metrics"]


def test_classification_vocabulary_stable():
    """Overall and row-status vocabularies are the stable expected sets."""
    mod = _load_module()
    assert mod.OVERALL_CLASSIFICATIONS == ("benchmark", "diagnostic", "blocked", "non-claim")
    assert mod.ROW_STATUSES == ("ok", "degraded", "invalid", "not-available")


def test_full_run_writes_report(tmp_path):
    """run_slice emits a JSON report with the diagnostic claim boundary."""
    mod = _load_module()
    report = mod.run_slice(tmp_path, command="pytest")
    assert report["paper_grade"] is False
    assert report["evidence_tier"] == "smoke"
    assert report["overall_classification"]["label"] in mod.OVERALL_CLASSIFICATIONS
    assert "sim-to-real" in report["claim_boundary"].lower()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "README.md").exists()


def test_occlusion_mask_adapts_to_actor_count():
    """A larger occlusion mask is truncated to the frame actor count."""
    mod = _load_module()
    spec = mod.ObservationPerturbationSpec(occlusion_mask=np.array([True, False, False, True]))
    adapted = mod._spec_for_actor_count(spec, 3)
    assert np.asarray(adapted.occlusion_mask).size == 3
