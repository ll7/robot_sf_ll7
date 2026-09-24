"""SNQI-v2 contract properties, strict assets and campaign/offline parity."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from robot_sf.benchmark.snqi.compute import (
    compute_snqi,
    compute_snqi_v0,
    compute_snqi_v1,
    compute_snqi_v2,
    normalize_snqi_v2_terms,
)
from robot_sf.benchmark.snqi.v2_reports import (
    build_family_report,
    enrich_campaign_v2,
    family_vectors,
    score_episode,
)
from robot_sf.benchmark.snqi.v2_spec import (
    QUALITY_TERMS,
    SIMULATED_FORCE,
    SOURCES,
    WEIGHTS,
    SnqiV2Spec,
    load_snqi_v2_spec,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "configs/benchmarks/snqi_v2"


def fixture_spec() -> SnqiV2Spec:
    """Return synthetic anchors for algorithm tests, never benchmark evidence."""
    return SnqiV2Spec(
        WEIGHTS,
        {"T": 3, "N": 0.25, "F": 10, "J": 2, "K": 4},
        SIMULATED_FORCE,
        "synthetic-test-only",
        (101, 102),
        0.8,
        {},
        {},
    )


def metrics(**overrides):
    """Return complete synthetic metrics with an independently retained legacy score."""
    return {
        "success": 1,
        "total_collision_count": 0,
        "time_to_goal_ideal_ratio": 1,
        "near_misses": 0,
        "executed_steps": 100,
        SIMULATED_FORCE: 0,
        "jerk_mean": 0,
        "curvature_mean": 0,
        "snqi": -0.12345678901234567,
        **overrides,
    }


def anchor_document():
    """Return clearly synthetic, full-schema calibration metadata for loader tests."""
    spec = fixture_spec()
    return {
        "version": "SNQI-v2.0",
        "status": "frozen",
        "anchors": {
            key: {
                "lower": 0,
                "upper": value,
                "type": "normative" if key in ("T", "N") else "calibration_p95",
            }
            for key, value in spec.upper_anchors.items()
        },
        "force_decision": {"source": SIMULATED_FORCE, "spearman_rho_F_N": 0.8},
        "calibration": {
            "episode_count": 1344,
            "arms": [f"synthetic-{i}" for i in range(14)],
            "scenarios": [f"synthetic-{i}" for i in range(48)],
            "execution_mode": "native",
            "source_commit": "a" * 40,
            "episodes_sha256": "b" * 64,
            "run_id": "synthetic-test-only",
            "split_id": "synthetic-test-only",
            "seeds": [101, 102],
        },
    }


@pytest.fixture
def spec_files(tmp_path):
    """Write versioned test assets with synthetic anchors."""
    paths = [tmp_path / name for name in ("weights.json", "anchors.json", "family.yaml")]
    paths[0].write_bytes((ASSETS / "weights.v2.0.json").read_bytes())
    paths[1].write_text(json.dumps(anchor_document()))
    paths[2].write_bytes((ASSETS / "family.v2.0.yaml").read_bytes())
    return paths


def test_spec_load_and_immutable(spec_files):
    spec = load_snqi_v2_spec(*spec_files)
    assert spec.weights == WEIGHTS
    assert all(len(value) == 64 for value in spec.hashes.values())
    with pytest.raises(TypeError):
        spec.weights["K"] = 0
    with pytest.raises(ValueError, match="overlap"):
        spec.validate_evaluation_seeds([101, 120])


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "strata", "negative"])
def test_weight_loader_fail_closed(spec_files, mutation):
    doc = json.loads(spec_files[0].read_text())
    if mutation == "missing":
        del doc["weights"]["w_K"]
    elif mutation == "extra":
        doc["weights"]["w_extra"] = {"value": 0, "rationale": "undeclared"}
    else:
        doc["weights"]["w_C"]["value"] = {"nan": float("nan"), "strata": 1, "negative": -1}[
            mutation
        ]
    spec_files[0].write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("mutation", ["pending", "zero", "nan", "lower", "decision", "seeds"])
def test_anchor_loader_fail_closed(spec_files, mutation):
    doc = anchor_document()
    if mutation == "pending":
        doc["status"] = "pending_calibration"
    elif mutation == "decision":
        doc["force_decision"]["spearman_rho_F_N"] = 0.90
    elif mutation == "seeds":
        doc["calibration"]["seeds"] = [111, 112]
    elif mutation == "lower":
        doc["anchors"]["F"]["lower"] = 1
    else:
        doc["anchors"]["F"]["upper"] = 0 if mutation == "zero" else float("nan")
    spec_files[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_snqi_v2_spec(*spec_files)


def test_duplicate_and_derived_source_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_sources({**SOURCES, "K": SOURCES["J"]})
    with pytest.raises(ValueError, match="derived"):
        validate_sources({**SOURCES, "F": "comfort_exposure", "K": "force_exceed_events"})
    with pytest.raises(ValueError, match="derived"):
        validate_sources({**SOURCES, "K": "min_distance"})


vector = st.tuples(
    st.floats(0, 20, allow_nan=False),
    st.integers(0, 100),
    st.floats(0, 100, allow_nan=False),
    st.floats(0, 100, allow_nan=False),
    st.floats(0, 100, allow_nan=False),
)


def from_vector(values, *, success=1, collision=0):
    return metrics(
        time_to_goal_ideal_ratio=values[0],
        near_misses=values[1],
        robot_force_impulse_total=values[2],
        jerk_mean=values[3],
        curvature_mean=values[4],
        success=success,
        total_collision_count=collision,
    )


@given(vector, vector, st.integers(0, 1), st.integers(0, 1))
def test_collision_strata(a, b, success_a, success_b):
    spec = fixture_spec()
    assert compute_snqi_v2(from_vector(a, success=success_a, collision=1), spec) < compute_snqi_v2(
        from_vector(b, success=success_b), spec
    )


@given(vector, vector)
def test_success_strata(a, b):
    spec = fixture_spec()
    assert compute_snqi_v2(from_vector(a), spec) > compute_snqi_v2(from_vector(b, success=0), spec)


@given(vector, st.integers(0, 1), st.integers(0, 1))
def test_ranges_and_monotonicity(values, success, collision):
    spec = fixture_spec()
    original = from_vector(values, success=success, collision=collision)
    score = compute_snqi_v2(original, spec)
    lower, upper = (-2.70, -1) if collision else ((0.05, 1) if success else (-0.70, 0))
    assert lower - 1e-12 <= score <= upper + 1e-12
    for source in (
        "time_to_goal_ideal_ratio",
        "near_misses",
        SIMULATED_FORCE,
        "jerk_mean",
        "curvature_mean",
    ):
        raised = min(original[source] + 1, 100) if source == "near_misses" else original[source] + 1
        assert compute_snqi_v2({**original, source: raised}, spec) <= score + 1e-12


def test_normalization_endpoints_failure_time_and_dispatch():
    spec = fixture_spec()
    base = metrics()
    assert normalize_snqi_v2_terms(base, spec) == {
        "S": 1,
        "C": 0,
        "T": 0,
        "N": 0,
        "F": 0,
        "J": 0,
        "K": 0,
    }
    full = metrics(
        time_to_goal_ideal_ratio=3,
        near_misses=25,
        robot_force_impulse_total=10,
        jerk_mean=2,
        curvature_mean=4,
    )
    assert all(normalize_snqi_v2_terms(full, spec)[key] == 1 for key in QUALITY_TERMS)
    full["success"] = 0
    full["time_to_goal_ideal_ratio"] = float("nan")
    assert normalize_snqi_v2_terms(full, spec)["T"] == 0
    assert compute_snqi(full, {}, {}, score_version="SNQI-v2", spec=spec) == compute_snqi_v2(
        full, spec
    )


@pytest.mark.parametrize(
    "source", [SIMULATED_FORCE, "jerk_mean", "curvature_mean", "near_misses", "executed_steps"]
)
def test_no_missing_or_nan_imputation(source):
    for value in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            compute_snqi_v2(metrics(**{source: value}), fixture_spec())


def records():
    return [
        {
            "algo": key,
            "seed": seed,
            "scenario_id": "fixture",
            "steps": 100,
            "metrics": metrics(success=success),
        }
        for key, success in (("a", 1), ("b", 0))
        for seed in (111, 112)
    ]


def test_legacy_values_preserved_and_legacy_functions_unmodified():
    ep = records()[0]
    before = json.dumps(ep["metrics"], sort_keys=True)
    enriched = score_episode(ep, fixture_spec())
    filtered = {k: v for k, v in enriched["metrics"].items() if not k.startswith("snqi_v2")}
    assert json.dumps(filtered, sort_keys=True) == before
    baseline = {
        key: {"med": 0, "p95": 2}
        for key in (
            "time_to_goal_norm",
            "collisions",
            "near_misses",
            "comfort_exposure",
            "force_exceed_events",
            "jerk_mean",
        )
    }
    for version, func in (("SNQI-v0", compute_snqi_v0), ("SNQI-v1", compute_snqi_v1)):
        assert compute_snqi(ep["metrics"], {}, baseline, score_version=version) == func(
            ep["metrics"], {}, baseline
        )


def test_family_determinism_grid_and_scaling():
    vectors = family_vectors()
    assert vectors == family_vectors()
    assert len(vectors) == 2013
    assert {"equal", "heavy_K", "leave_one_out_F", "relaxed_S0.5_C0.5"} <= {
        v["name"] for v in vectors
    }
    assert all(min(v["weights"][t] for t in QUALITY_TERMS) >= 0.02 for v in vectors)
    report = build_family_report(records(), fixture_spec(), bootstrap_samples=10)
    assert report == build_family_report(records(), fixture_spec(), bootstrap_samples=10)
    scaled = replace(fixture_spec(), weights={key: 7 * value for key, value in WEIGHTS.items()})
    other = build_family_report(records(), scaled, bootstrap_samples=10)
    assert [r["planner"] for r in report["declared_ranking"]] == [
        r["planner"] for r in other["declared_ranking"]
    ]
    assert report["top1_frequency"] == other["top1_frequency"]


def test_campaign_writes_both_reports_and_fields(tmp_path):
    entries = []
    for planner in ("a", "b"):
        path = tmp_path / f"{planner}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records() if r["algo"] == planner))
        entries.append({"status": "ok", "planner": {"key": planner}, "episodes_path": str(path)})
    artifacts = enrich_campaign_v2(
        entries, fixture_spec(), tmp_path / "reports", repo_root=tmp_path, bootstrap_samples=10
    )
    assert len(artifacts) == 4
    assert all(Path(path).exists() for path in artifacts.values())
    raw = json.loads((tmp_path / "a.jsonl").read_text().splitlines()[0])
    assert raw["metrics"]["snqi_v2"] == 1
    assert raw["metrics"]["snqi"] == records()[0]["metrics"]["snqi"]
    assert len(raw["metrics"]["snqi_v2_terms"]) == 7


def calibration_records():
    """Generate a full synthetic development grid with independent F and N."""
    import numpy as np

    rng = np.random.default_rng(8)
    arms = [f"arm{i}" for i in range(14)]
    scenarios = [f"scenario{i}" for i in range(48)]
    rows = [
        {
            "planner_key": arm,
            "scenario_id": scenario,
            "seed": seed,
            "status": "success",
            "steps": 100,
            "metrics": metrics(
                near_misses=int(rng.integers(0, 25)),
                robot_force_impulse_total=float(rng.uniform(1, 30)),
                jerk_mean=2,
                curvature_mean=4,
            ),
        }
        for arm in arms
        for scenario in scenarios
        for seed in (101, 102)
    ]
    return rows, {
        "arms": arms,
        "scenarios": scenarios,
        "run_id": "synthetic",
        "source_commit": "a" * 40,
        "episodes_sha256": "b" * 64,
    }


def test_calibration_exact_grid_and_no_imputation():
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors

    rows, kwargs = calibration_records()
    result = derive_calibration_anchors(rows, **kwargs)
    assert result["force_decision"]["source"] == SIMULATED_FORCE
    assert result["anchors"]["J"]["upper"] == 2
    with pytest.raises(ValueError, match="1344"):
        derive_calibration_anchors(rows[:-1], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        derive_calibration_anchors(rows[:-1] + rows[:1], **kwargs)
    rows[0]["seed"] = 111
    with pytest.raises(ValueError, match="out-of-split"):
        derive_calibration_anchors(rows, **kwargs)


def test_calibration_switch_requires_full_pp_coverage():
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors
    from robot_sf.benchmark.snqi.v2_spec import PP_EQUIV_FORCE

    rows, kwargs = calibration_records()
    for row in rows:
        row["metrics"][SIMULATED_FORCE] = row["metrics"]["near_misses"]
        row["metrics"][PP_EQUIV_FORCE] = 3
    result = derive_calibration_anchors(rows, **kwargs)
    assert result["force_decision"]["source"] == PP_EQUIV_FORCE
    assert result["anchors"]["F"]["upper"] == 3
    rows[0]["metrics"][PP_EQUIV_FORCE] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        derive_calibration_anchors(rows, **kwargs)


def test_config_hash_preserves_disabled_v2_and_serializes_enabled(tmp_path):
    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload

    cfg = CampaignConfig("fixture", tmp_path / "scenarios.yaml", ())
    assert "snqi_v2_spec" not in _config_hash_payload(cfg)
    with_spec = replace(cfg, snqi_v2_spec=fixture_spec())
    assert _config_hash_payload(with_spec)["snqi_v2_spec"]["snqi_v2_version"] == "SNQI-v2"


def test_offline_cli_emits_mandatory_pair(spec_files, tmp_path):
    from scripts.tools.analyze_snqi_contract import main

    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records()))
    report_dir = tmp_path / "reports"
    assert (
        main(
            [
                "--score-version",
                "SNQI-v2",
                "--episodes",
                str(path),
                "--weights",
                str(spec_files[0]),
                "--anchors",
                str(spec_files[1]),
                "--family",
                str(spec_files[2]),
                "--reports-dir",
                str(report_dir),
            ]
        )
        == 0
    )
    assert (report_dir / "snqi_v2_family.json").exists()
    assert (report_dir / "snqi_v2_diagnostics.json").exists()


def test_unavailable_optional_metric_does_not_mark_planner_degraded():
    row = {
        **records()[0],
        "algorithm_metadata": {
            "status": "ok",
            "paired_effect_metric_producer": {
                "fields": {"false_positive_stop_rate": {"status": "unavailable"}}
            },
        },
    }
    assert score_episode(row, fixture_spec())["metrics"]["snqi_v2"] == 1
    row["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
    with pytest.raises(ValueError, match="fallback"):
        score_episode(row, fixture_spec())


def test_real_small_campaign_v2_outputs(tmp_path):
    from robot_sf.benchmark.camera_ready._config_types import (
        CampaignConfig,
        PlannerSpec,
        SeedPolicy,
        SnqiContractConfig,
    )
    from robot_sf.benchmark.camera_ready_campaign import run_campaign

    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "- name: v2_smoke\n  map_file: "
        + str(ROOT / "maps/svg_maps/classic_crossing.svg")
        + "\n  seeds: [201, 202]\n  simulation_config:\n    ped_density: 0.0\n"
    )
    cfg = CampaignConfig(
        name="v2_smoke",
        scenario_matrix_path=scenario_path,
        planners=(PlannerSpec(key="goal", algo="goal"),),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(201, 202)),
        horizon=4,
        dt=0.1,
        workers=1,
        export_publication_bundle=False,
        bootstrap_samples=10,
        snqi_contract=SnqiContractConfig(calibration_trials=10),
        snqi_v2_spec=fixture_spec(),
    )
    result = run_campaign(
        cfg, output_root=tmp_path / "out", campaign_id="v2-smoke", skip_publication_bundle=True
    )
    root = Path(result["campaign_root"])
    assert (root / "reports/snqi_v2_family.json").exists()
    assert (root / "reports/snqi_v2_diagnostics.json").exists()
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    assert manifest["metrics"]["snqi_v2_version"] == "SNQI-v2"
    summary = json.loads((root / "reports/campaign_summary.json").read_text())
    assert summary["campaign"]["snqi_v2_calibration_split_id"] == "synthetic-test-only"
    paths = list((root / "runs").glob("*/episodes.jsonl"))
    assert len(paths) == 1
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert len(rows) == 2
    assert all("snqi_v2" in row["metrics"] and "snqi" in row["metrics"] for row in rows)
