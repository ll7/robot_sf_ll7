"""SNQI-v2 contract properties, strict assets and campaign/offline parity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
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
    TERMS,
    WEIGHTS,
    SnqiV2Spec,
    load_snqi_v2_spec,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "configs/benchmarks/snqi_v2"
# From the 0.0.7 publication bundle's resolved manifest (source 07f7e8d43084).
# Keep these pins independent of the current checkout so calibration fails on input drift.
# Exact historical template bytes; the live release template can select newer planners.
FROZEN_007_CAMPAIGN = ROOT / "tests/fixtures/snqi_v2/frozen_007_campaign.yaml"
FROZEN_007_CAMPAIGN_SHA256 = "095331329b06673dc165109c8523579549f769c98542b207a712f6e2bf9ed6ad"
FROZEN_007_SCENARIO_SHA256 = "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"
FROZEN_007_SEED_SETS_SHA256 = "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6"
FROZEN_007_PLANNER_CONFIG_SHA256 = {
    "prediction_planner": "a5f3775110b7c1351e183016afe8a17348a66d35088eeb6ff55a182dc924a481",
    "social_force": "bcd785fff7753dd31bcb899b1bab0cfec8d0c706b053ff28cd2988ca58761afb",
    "ppo": "51ccfbf4400a306b355e2c3f0f46eda3489d5ce3bc85beaa023a6a1da9c9fb41",
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield": (
        "895cd46ff26c0b03fd51d5e5fb6e77f16ad2c6f6cd1fcead7089c8dda8c7966d"
    ),
    "scenario_adaptive_hybrid_orca_v2_collision_guard": (
        "90bc08cffe40e7f5b38c371db7c2a22790970c6817792dc7865686686c0a7dc3"
    ),
    "hybrid_rule_v3_fast_progress_static_escape": (
        "905ec25e24b3d5cedee1508ac6ad20a09f913d368c99d4329c839bc359640935"
    ),
    "hybrid_rule_v3_fast_progress_static_escape_continuous": (
        "8ed0a4048cba78f70abf37effaf8b93fe7cbb9de25897d4f65008adcb6e91ecd"
    ),
    "guarded_ppo": "69f273f311590009a344f3a88592cb19f54524d252f469ab5fc66f7cf2c9e772",
    "predictive_mppi": "5213a9e44c74cb79f78466d414645f6ca233d761e8fa5b77e5cfc3161ed94adb",
    "risk_dwa": "1351439539ed02334891f6a1ef6ac652745791b1b830a5f8616850fbd5ccb37c",
}


def test_development_calibration_matches_frozen_007_campaign_identity():
    """Only development seeds and execution/publication metadata may differ from 0.0.7."""
    frozen_bytes = FROZEN_007_CAMPAIGN.read_bytes()
    assert hashlib.sha256(frozen_bytes).hexdigest() == FROZEN_007_CAMPAIGN_SHA256
    frozen = yaml.safe_load(frozen_bytes)
    calibration = yaml.safe_load((ASSETS / "calibration.dev101_102.yaml").read_bytes())

    assert len(calibration["planners"]) == 14
    assert calibration["seed_policy"] == {
        "mode": "fixed-list",
        "seeds": [101, 102],
        "seed_sets_path": frozen["seed_policy"]["seed_sets_path"],
    }
    assert calibration["name"] == "snqi_v2_calibration_dev101_102"
    assert calibration["paper_facing"] is False
    assert calibration["workers"] == 16
    assert calibration["export_publication_bundle"] is False
    assert calibration["arm_isolation"] == "subprocess"

    allowed_deviations = {
        "name",
        "paper_facing",
        "seed_policy",
        "workers",
        "export_publication_bundle",
        "arm_isolation",
    }
    assert {key: value for key, value in calibration.items() if key not in allowed_deviations} == {
        key: value for key, value in frozen.items() if key not in allowed_deviations
    }
    assert hashlib.sha256((ROOT / frozen["scenario_matrix"]).read_bytes()).hexdigest() == (
        FROZEN_007_SCENARIO_SHA256
    )
    assert hashlib.sha256(
        (ROOT / frozen["seed_policy"]["seed_sets_path"]).read_bytes()
    ).hexdigest() == (FROZEN_007_SEED_SETS_SHA256)
    assert {
        arm["key"]: hashlib.sha256((ROOT / arm["algo_config"]).read_bytes()).hexdigest()
        for arm in calibration["planners"]
        if "algo_config" in arm
    } == FROZEN_007_PLANNER_CONFIG_SHA256


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
            "benchmark_execution": "nonfallback",
            "command_mode_counts": {f"synthetic-{i}": {"native": 96} for i in range(14)},
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


@pytest.mark.parametrize("prefix", ["version: forged\n", "seed: 0\n", "<<: {version: forged}\n"])
def test_family_loader_rejects_duplicate_yaml_keys(spec_files, prefix):
    family = spec_files[2]
    family.write_text(prefix + family.read_text())
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("term", TERMS)
def test_weight_loader_rejects_boolean_numbers(spec_files, term, value):
    """JSON booleans cannot exploit Python equality with declared numeric weights."""
    document = json.loads(spec_files[0].read_text())
    document["weights"][f"w_{term}"]["value"] = value
    spec_files[0].write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("term", QUALITY_TERMS)
@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_anchor_loader_rejects_boolean_numbers(spec_files, term, bound, value):
    """Neither physical zero nor calibrated anchors may be encoded as booleans."""
    document = anchor_document()
    document["anchors"][term][bound] = value
    spec_files[1].write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("value", [False, True])
def test_anchor_loader_rejects_boolean_correlation(spec_files, value):
    """A boolean cannot select the force variant through a numeric rho comparison."""
    from robot_sf.benchmark.snqi.v2_spec import PP_EQUIV_FORCE

    document = anchor_document()
    document["force_decision"].update(
        spearman_rho_F_N=value, source=PP_EQUIV_FORCE if value else SIMULATED_FORCE
    )
    spec_files[1].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="rho"):
        load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "total_collision_count",
        "time_to_goal_ideal_ratio",
        "near_misses",
        "executed_steps",
        SIMULATED_FORCE,
        "jerk_mean",
        "curvature_mean",
    ],
)
def test_scoring_rejects_boolean_numbers(field, value):
    """Every active numeric scoring input rejects JSON booleans before normalization."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compute_snqi_v2(metrics(**{field: value}), fixture_spec())


@pytest.mark.parametrize("success", [False, True])
def test_scoring_preserves_declared_boolean_success(success):
    """The canonical producer's binary outcome has explicit 0/1 score semantics."""
    boolean_metrics = metrics(success=success)
    assert compute_snqi_v2(boolean_metrics, fixture_spec()) == compute_snqi_v2(
        metrics(success=int(success)), fixture_spec()
    )
    assert boolean_metrics["success"] is success


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


@pytest.mark.parametrize("term", TERMS)
def test_versioned_loader_rejects_alternate_stratum_safe_weights(spec_files, term):
    """The fixed version cannot silently admit another safety-stratified score."""
    doc = json.loads(spec_files[0].read_text())
    doc["weights"][f"w_{term}"]["value"] *= 1.01
    mutated = {key: doc["weights"][f"w_{key}"]["value"] for key in TERMS}
    quality = sum(mutated[key] for key in QUALITY_TERMS)
    assert mutated["S"] > quality
    assert mutated["C"] > mutated["S"] + quality
    spec_files[0].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="exact declared weight"):
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


@pytest.mark.parametrize("mutation", ["missing", "arm", "fallback", "short", "bool", "claim"])
def test_anchor_loader_rejects_unproven_command_mode_census(spec_files, mutation):
    doc = anchor_document()
    calibration = doc["calibration"]
    if mutation == "missing":
        del calibration["command_mode_counts"]
    elif mutation == "arm":
        del calibration["command_mode_counts"]["synthetic-0"]
    elif mutation == "claim":
        calibration["benchmark_execution"] = "native"
    else:
        calibration["command_mode_counts"]["synthetic-0"] = {
            "fallback": {"fallback": 96},
            "short": {"native": 95},
            "bool": {"native": True, "adapter": 95},
        }[mutation]
    spec_files[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="calibration"):
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
    assert report["bootstrap"]["confidence_intervals_paired"] is True
    assert report["bootstrap"]["stability"]["paired"] is False
    assert report == build_family_report(records(), fixture_spec(), bootstrap_samples=10)
    scaled = replace(fixture_spec(), weights={key: 7 * value for key, value in WEIGHTS.items()})
    other = build_family_report(records(), scaled, bootstrap_samples=10)
    assert [r["planner"] for r in report["declared_ranking"]] == [
        r["planner"] for r in other["declared_ranking"]
    ]
    assert report["top1_frequency"] == other["top1_frequency"]


def write_campaign_arm(path, rows):
    """Create real producer custody for synthetic episode rows."""
    from robot_sf.benchmark.result_provenance import (
        build_result_provenance_manifest,
        manifest_path_for_result_jsonl,
        write_result_provenance_manifest,
    )

    rows = [
        {
            **row,
            "episode_id": f"{row['scenario_id']}-{row['seed']}",
            "config_hash": "fixture",
            "git_hash": "a" * 40,
        }
        for row in rows
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload = build_result_provenance_manifest(
        out_path=path,
        episode_records=rows,
        schema_path=ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=ROOT / "configs/scenarios/classic_interactions_francis2023.yaml",
        scenarios=[{"name": "fixture"}],
        algo=rows[0]["algo"],
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="fixture",
        total_jobs=len(rows),
        written=len(rows),
        horizon=600,
        dt=0.1,
        record_forces=True,
        active_observation_mode="native",
        active_observation_level="full",
    )
    write_result_provenance_manifest(manifest_path_for_result_jsonl(path), payload)
    return rows


def test_campaign_writes_both_reports_and_fields(tmp_path):
    entries = []
    for planner in ("a", "b"):
        path = tmp_path / f"{planner}.jsonl"
        write_campaign_arm(path, [r for r in records() if r["algo"] == planner])
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
    before = {
        str(path): path.read_bytes()
        for entry in entries
        for path in (
            Path(entry["episodes_path"]),
            Path(entry["episodes_path"] + ".provenance.json"),
        )
    }
    enrich_campaign_v2(
        entries, fixture_spec(), tmp_path / "reports", repo_root=tmp_path, bootstrap_samples=10
    )
    assert all(Path(path).read_bytes() == content for path, content in before.items())


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
            "horizon": 600,
            "scenario_params": {"run_horizon": 600, "run_dt": 0.1, "record_forces": True},
            "algorithm_metadata": {"execution_mode": "native"},
            "steps": 100,
            "metrics": metrics(
                near_misses=int(rng.integers(0, 25)),
                robot_force_impulse_total=float(rng.uniform(1, 30)),
                jerk_mean=2,
                curvature_mean=4,
                robot_force_metadata={"sample_timing": "pre_integration"},
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


@pytest.mark.parametrize("field", ["algorithm_metadata", "horizon", "scenario_params"])
def test_calibration_rejects_missing_execution_contract(field):
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors

    rows, kwargs = calibration_records()
    rows[0].pop(field)
    with pytest.raises(ValueError, match="calibration requires"):
        derive_calibration_anchors(rows, **kwargs)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize(
    "field", ["steps", "near_misses", SIMULATED_FORCE, "jerk_mean", "curvature_mean", "pp_force"]
)
def test_calibration_rejects_boolean_numbers(field, value):
    """No malformed numeric sample can affect the F switch or a p95 anchor."""
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors
    from robot_sf.benchmark.snqi.v2_spec import PP_EQUIV_FORCE

    rows, kwargs = calibration_records()
    if field == "pp_force":
        for row in rows:
            row["metrics"][SIMULATED_FORCE] = row["metrics"]["near_misses"]
            row["metrics"][PP_EQUIV_FORCE] = 3
        field = PP_EQUIV_FORCE
    container = rows[0] if field == "steps" else rows[0]["metrics"]
    container[field] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
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


@pytest.mark.parametrize("adapter_count", [1, 1344])
@pytest.mark.parametrize("command_mode", ["adapter", "mixed"])
def test_calibration_records_actual_command_modes_without_relabeling(
    adapter_count, command_mode, spec_files
):
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors

    rows, kwargs = calibration_records()
    for row in rows[:adapter_count]:
        row["algorithm_metadata"]["planner_kinematics"] = {"execution_mode": command_mode}
    anchors = derive_calibration_anchors(rows, **kwargs)
    document = anchors["calibration"]
    assert "execution_mode" not in document
    assert document["benchmark_execution"] == "nonfallback"
    census = document["command_mode_counts"]
    assert sum(counts.get(command_mode, 0) for counts in census.values()) == adapter_count
    assert sum(counts.get("native", 0) for counts in census.values()) == 1344 - adapter_count
    assert all(sum(counts.values()) == 96 for counts in census.values())
    spec_files[1].write_text(json.dumps(anchors))
    load_snqi_v2_spec(*spec_files)


@pytest.mark.parametrize("marker", ["fallback_used", "fallback_triggered", "degraded"])
def test_calibration_rejects_fallback_or_degraded_adapter(marker):
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors

    rows, kwargs = calibration_records()
    rows[0]["algorithm_metadata"] = {
        "planner_kinematics": {"execution_mode": "adapter"},
        "planner_runtime": {marker: True},
    }
    with pytest.raises(ValueError, match="fallback/degraded"):
        derive_calibration_anchors(rows, **kwargs)


@pytest.fixture
def calibration_archive(tmp_path, guarded_episode, request):
    """Build a complete synthetic archive using real producer custody constructors."""
    from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
    from robot_sf.benchmark.camera_ready._preflight import _scenario_matrix_hash
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload
    from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, PlannerSpec, SeedPolicy
    from robot_sf.benchmark.map_runner.map_runner_identity import (
        compute_map_episode_id,
        scenario_identity_payload,
        scenario_with_episode_seed_defaults,
    )
    from robot_sf.benchmark.release_acceptance import _result_provenance_scenarios
    from robot_sf.benchmark.result_provenance import (
        build_result_provenance_manifest,
        build_simulator_settings_provenance,
        manifest_path_for_result_jsonl,
        write_result_provenance_manifest,
    )
    from robot_sf.benchmark.utils import _config_hash

    rows, kwargs = calibration_records()
    kwargs["expected_algorithms"] = {
        arm: "guarded_ppo" if i == 0 else "goal" for i, arm in enumerate(kwargs["arms"])
    }
    scenario_path = tmp_path / "matrix.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            [
                {"name": name, "map_file": str(ROOT / "maps/svg_maps/classic_crossing.svg")}
                for name in kwargs["scenarios"]
            ]
        )
    )
    cfg = CampaignConfig(
        name="custody-test",
        scenario_matrix_path=scenario_path,
        planners=tuple(
            PlannerSpec(key=arm, algo=kwargs["expected_algorithms"][arm]) for arm in kwargs["arms"]
        ),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(101, 102)),
        horizon=600,
        dt=0.1,
        **getattr(request, "param", {}),
    )
    resolved = _load_campaign_scenarios(cfg)
    effective = _result_provenance_scenarios(cfg, resolved, kinematics="differential_drive")
    by_name = {row["name"]: row for row in effective}
    (tmp_path / "reports").mkdir()
    (tmp_path / "preflight").mkdir()
    manifest = {
        "campaign_id": "synthetic-archive",
        "git": {"commit": kwargs["source_commit"]},
        "config_hash": _config_hash(_config_hash_payload(cfg)),
        "scenario_matrix_hash": _scenario_matrix_hash(resolved),
        "kinematics_matrix": ["differential_drive"],
        "seed_policy": {"resolved_seeds": [101, 102]},
        "planners": [
            {"key": arm, "algo": kwargs["expected_algorithms"][arm], "enabled": True}
            for arm in kwargs["arms"]
        ],
    }
    (tmp_path / "campaign_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "preflight/preview_scenarios.json").write_text(
        json.dumps({"scenarios": [{"name": name} for name in kwargs["scenarios"]]})
    )
    runs = []
    for arm in kwargs["arms"]:
        path = tmp_path / "runs" / f"{arm}__differential_drive" / "episodes.jsonl"
        path.parent.mkdir(parents=True)
        arm_rows = [row for row in rows if row["planner_key"] == arm]
        algo = kwargs["expected_algorithms"][arm]
        mode, level = (
            ("sensor_fusion_state", "lidar_2d")
            if algo == "guarded_ppo"
            else ("goal_state", "oracle_full_state")
        )
        for row in arm_rows:
            row["algo"] = algo
            row["git_hash"] = kwargs["source_commit"]
            row["scenario_params"] = scenario_identity_payload(
                scenario_with_episode_seed_defaults(by_name[row["scenario_id"]], seed=row["seed"]),
                algo=algo,
                algo_config={},
                horizon=600,
                dt=0.1,
                record_forces=True,
                observation_mode=mode,
                observation_level=level,
                safety_wrapper=cfg.safety_wrapper,
                record_planner_decision_trace=cfg.record_planner_decision_trace,
                record_simulation_step_trace=cfg.record_simulation_step_trace,
            )
            row["episode_id"] = compute_map_episode_id(row["scenario_params"], row["seed"])
            row["config_hash"] = _config_hash(row["scenario_params"])
            row["algorithm_metadata"] = (
                json.loads(json.dumps(guarded_episode["algorithm_metadata"]))
                if algo == "guarded_ppo"
                else {
                    "execution_mode": "native",
                    "algorithm": algo,
                    "canonical_algorithm": algo,
                    "planner_contract": {"planner_id": algo},
                }
            )
            row["result_provenance"] = {
                "schema_version": "benchmark_row_provenance.v1",
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "config_hash": row["config_hash"],
                "repo_commit": row["git_hash"],
                "simulator_settings": build_simulator_settings_provenance(
                    horizon=600,
                    dt=0.1,
                    record_forces=True,
                    active_observation_mode=mode,
                    active_observation_level=level,
                ),
            }
        path.write_text("".join(json.dumps(row) + "\n" for row in arm_rows))
        payload = build_result_provenance_manifest(
            out_path=path,
            episode_records=arm_rows,
            schema_path=ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
            scenario_path=scenario_path,
            scenarios=effective,
            algo=algo,
            algo_config_path=None,
            benchmark_profile="baseline-safe",
            suite_key="fixture",
            total_jobs=96,
            written=96,
            horizon=600,
            dt=0.1,
            record_forces=True,
            active_observation_mode=mode,
            active_observation_level=level,
        )
        payload["run"]["repo_commit"] = kwargs["source_commit"]
        write_result_provenance_manifest(manifest_path_for_result_jsonl(path), payload)
        runs.append(
            {
                "status": "ok",
                "planner": {"key": arm, "algo": algo, "kinematics": "differential_drive"},
                "episodes_path": str(path),
                "summary": {
                    "status": "ok",
                    "total_jobs": 96,
                    "written": 96,
                    "algorithm_metadata_contract": {"execution_mode": "native"},
                },
            }
        )
    (tmp_path / "reports/campaign_summary.json").write_text(json.dumps({"runs": runs}))
    return cfg, rows, kwargs


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_sidecar",
        "stale_raw",
        "misrouted_summary",
        "sidecar_row",
        "sidecar_source",
        "forged_planner",
        "forged_algorithm",
        "coherent_scenario_config",
        "forged_sidecar_algorithm",
        "manifest_config",
        "sidecar_input",
        "nested_fallback",
    ],
)
def test_calibration_freeze_rejects_unbound_custody(  # noqa: C901, PLR0915
    tmp_path, calibration_archive, mutation
):
    from robot_sf.benchmark.result_provenance import (
        _canonical_input_bundle_sha256,
        manifest_path_for_result_jsonl,
        validate_result_provenance_manifest,
    )
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors
    from robot_sf.benchmark.utils import _config_hash

    cfg, _, kwargs = calibration_archive
    path = tmp_path / "runs" / f"{kwargs['arms'][0]}__differential_drive" / "episodes.jsonl"
    sidecar = manifest_path_for_result_jsonl(path)
    payload = json.loads(sidecar.read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_sidecar":
        sidecar.unlink()
    elif mutation == "misrouted_summary":
        target = tmp_path / "reports/campaign_summary.json"
        summary = json.loads(target.read_text())
        summary["runs"][0]["episodes_path"] = summary["runs"][1]["episodes_path"]
        target.write_text(json.dumps(summary))
    elif mutation == "manifest_config":
        target = tmp_path / "campaign_manifest.json"
        manifest = json.loads(target.read_text())
        manifest["config_hash"] = "f" * 64
        target.write_text(json.dumps(manifest))
    elif mutation == "sidecar_row":
        payload["rows"][0]["config_hash"] = "f" * 64
    elif mutation == "sidecar_source":
        payload["run"]["repo_commit"] = "f" * 40
    elif mutation == "forged_sidecar_algorithm":
        identity = payload["campaign_identity"]
        identity["algorithm"] = "goal"
        identity["config_hash"] = _config_hash(
            {
                "schema_path": payload["inputs"]["schema_path"]["path"],
                "algo": "goal",
                "algo_config_path": None,
            }
        )
        identity["input_bundle_sha256"] = _canonical_input_bundle_sha256(
            inputs=payload["inputs"],
            algo="goal",
            protocol_version=payload["run"]["protocol_version"],
            suite_key=identity["suite_key"],
        )
        validate_result_provenance_manifest(payload)
    elif mutation == "sidecar_input":
        forged_input = tmp_path / "forged-matrix.yaml"
        forged_input.write_text("- name: forged-scenario\n")
        payload["inputs"]["scenario_matrix"].update(
            path=str(forged_input), sha256=hashlib.sha256(forged_input.read_bytes()).hexdigest()
        )
        identity = payload["campaign_identity"]
        identity["input_bundle_sha256"] = _canonical_input_bundle_sha256(
            inputs=payload["inputs"],
            algo=identity["algorithm"],
            protocol_version=payload["run"]["protocol_version"],
            suite_key=identity["suite_key"],
        )
        validate_result_provenance_manifest(payload)
    else:
        if mutation == "stale_raw":
            rows[0]["metrics"][SIMULATED_FORCE] += 1
        elif mutation == "forged_planner":
            rows[0]["planner_key"] = "forged-other-arm"
        elif mutation == "forged_algorithm":
            rows[0]["algo"] = "goal"
        elif mutation == "nested_fallback":
            rows[0]["algorithm_metadata"]["planner_runtime"]["fallback_triggered"] = True
        else:
            rows[0]["scenario_params"]["map_file"] = "forged.svg"
            digest = _config_hash(rows[0]["scenario_params"])
            rows[0]["config_hash"] = digest
            rows[0]["result_provenance"]["config_hash"] = digest
            payload["rows"][0]["config_hash"] = digest
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if mutation != "stale_raw":
            payload["raw_artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_result_provenance_manifest(payload)
    if mutation != "missing_sidecar":
        sidecar.write_text(json.dumps(payload))
    output = tmp_path / "anchors.json"
    output.write_bytes(b"existing frozen anchor must survive rejection\n")
    before = output.read_bytes()
    with pytest.raises((ValueError, OSError)):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "safety_wrapper",
        "cbf_safety_filter",
        "observation_mode",
        "algo_config_hash",
        "extra",
        "duplicate_id",
    ],
)
def test_calibration_freeze_rejects_coherent_identity_forgery(
    tmp_path, calibration_archive, mutation
):
    from robot_sf.benchmark.result_provenance import (
        manifest_path_for_result_jsonl,
        validate_result_provenance_manifest,
    )
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors
    from robot_sf.benchmark.utils import _config_hash

    cfg, _, kwargs = calibration_archive
    path = tmp_path / "runs" / f"{kwargs['arms'][0]}__differential_drive" / "episodes.jsonl"
    sidecar = manifest_path_for_result_jsonl(path)
    payload = json.loads(sidecar.read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[1]
    if mutation == "duplicate_id":
        row["episode_id"] = rows[0]["episode_id"]
    else:
        row["scenario_params"][mutation] = {
            "safety_wrapper": {"enabled": True, "mode": "other-policy"},
            "cbf_safety_filter": {"enabled": True},
            "observation_mode": "forged-observation",
            "algo_config_hash": "forged-policy",
            "extra": {"behavior": "changed"},
        }[mutation]
        digest = _config_hash(row["scenario_params"])
        row["config_hash"] = digest
        row["result_provenance"]["config_hash"] = digest
        payload["rows"][1]["config_hash"] = digest
        from robot_sf.benchmark.map_runner.map_runner_identity import compute_map_episode_id

        row["episode_id"] = compute_map_episode_id(row["scenario_params"], row["seed"])
    payload["rows"][1]["episode_id"] = row["episode_id"]
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    payload["raw_artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    validate_result_provenance_manifest(payload)
    sidecar.write_text(json.dumps(payload))
    output = tmp_path / "anchors.json"
    output.write_bytes(b"prior anchor\n")
    with pytest.raises(ValueError, match="canonical|episode identity"):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == b"prior anchor\n"


@pytest.mark.parametrize("prefix", ["horizon: 1\n", "seed_policy: {mode: forged}\n", None])
def test_calibration_freeze_rejects_duplicate_acquisition_yaml(
    tmp_path, calibration_archive, prefix
):
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors
    from robot_sf.benchmark.utils import _config_hash

    cfg, _, _ = calibration_archive
    path = tmp_path / "acquisition.yaml"
    path.write_text(
        prefix + "horizon: 600\nseed_policy: {mode: fixed-list, seeds: [101, 102]}\n"
        if prefix is not None
        else "horizon: 600\nseed_policy: {mode: forged, mode: fixed-list, seeds: [101, 102]}\n"
    )
    cfg = replace(
        cfg,
        source_config_path=path,
        source_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_hash"] = _config_hash(_config_hash_payload(cfg))
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "anchors.json"
    output.write_bytes(b"prior anchor\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == b"prior anchor\n"


@pytest.mark.parametrize("phase", ["before_input_snapshot", "before_publish"])
def test_calibration_freeze_rejects_acquisition_source_race(
    tmp_path, calibration_archive, monkeypatch, phase
):
    from robot_sf.benchmark.camera_ready._util import _config_hash_payload
    from robot_sf.benchmark.snqi import v2_calibration
    from robot_sf.benchmark.utils import _config_hash

    cfg, _, _ = calibration_archive
    source = tmp_path / "acquisition.yaml"
    source.write_text("horizon: 600\n")
    cfg = replace(
        cfg,
        source_config_path=source,
        source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config_hash"] = _config_hash(_config_hash_payload(cfg))
    manifest_path.write_text(json.dumps(manifest))
    method = (
        "_snapshot_calibration_files"
        if phase == "before_input_snapshot"
        else "derive_calibration_anchors"
    )
    original = getattr(v2_calibration, method)
    changed = []

    def mutate_after_validation(*args, **kwargs):
        result = original(*args, **kwargs)
        source.write_text("horizon: 1\n")
        changed.append(True)
        return result

    monkeypatch.setattr(v2_calibration, method, mutate_after_validation)
    output = tmp_path / "anchors.json"
    output.write_bytes(b"prior anchor\n")
    with pytest.raises(ValueError, match="config source changed|custody changed"):
        v2_calibration.freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert changed == [True]
    assert output.read_bytes() == b"prior anchor\n"


def test_calibration_acquisition_yaml_is_strict_and_hash_bound(tmp_path):
    from robot_sf.benchmark.snqi.v2_calibration import _validated_calibration_config

    config = _validated_calibration_config(None)
    source = ASSETS / "calibration.dev101_102.yaml"
    assert config.source_config_path == source
    assert config.source_config_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert config.horizon == 600
    assert config.seed_policy.seeds == (101, 102)
    path = tmp_path / "changed.yaml"
    path.write_bytes(source.read_bytes() + b"\n# bytes changed after canonical load\n")
    with pytest.raises(ValueError, match="config source changed"):
        _validated_calibration_config(replace(config, source_config_path=path))


@pytest.mark.parametrize("location", ["metadata", "nested_metadata", "seed", "manifest", "sidecar"])
def test_calibration_freeze_rejects_duplicate_json_keys(tmp_path, calibration_archive, location):
    from robot_sf.benchmark.result_provenance import (
        manifest_path_for_result_jsonl,
        validate_result_provenance_manifest,
    )
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors

    cfg, _, kwargs = calibration_archive
    path = tmp_path / "runs" / f"{kwargs['arms'][0]}__differential_drive" / "episodes.jsonl"
    sidecar = manifest_path_for_result_jsonl(path)
    payload = json.loads(sidecar.read_text())
    if location in {"manifest", "sidecar"}:
        target = tmp_path / "campaign_manifest.json" if location == "manifest" else sidecar
        prefix = '"config_hash":"forged",' if location == "manifest" else '"run":{},'
        target.write_text("{" + prefix + target.read_text()[1:])
    else:
        lines = path.read_text().splitlines(keepends=True)
        if location == "nested_metadata":
            lines[0] = lines[0].replace(
                '"algorithm_metadata": {',
                '"algorithm_metadata": {"fallback_triggered":true,"fallback_triggered":false,',
                1,
            )
        else:
            prefix = (
                '"algorithm_metadata":{"fallback_triggered":true},'
                if location == "metadata"
                else '"seed":999,'
            )
            lines[0] = "{" + prefix + lines[0][1:]
        path.write_text("".join(lines))
        payload["raw_artifacts"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        validate_result_provenance_manifest(payload)
        sidecar.write_text(json.dumps(payload))
    output = tmp_path / "anchors.json"
    output.write_bytes(b"prior anchor\n")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == b"prior anchor\n"


@pytest.mark.parametrize(
    "calibration_archive",
    [
        {},
        {
            "safety_wrapper": {"enabled": True, "arm_key": "wrapper_on"},
            "record_planner_decision_trace": True,
        },
    ],
    indirect=True,
)
def test_freeze_calibration_archive_binds_files_and_source(
    tmp_path, monkeypatch, calibration_archive
):
    import hashlib
    import weakref

    from robot_sf.benchmark.snqi import v2_calibration
    from robot_sf.benchmark.snqi.v2_calibration import freeze_campaign_anchors

    class Payload(list):
        """Weak-referenceable decoded force payload for retention proof."""

    refs = []
    original_reader = v2_calibration.read_episode_files
    original_derive = v2_calibration.derive_calibration_anchors

    def read_with_payload(paths):
        for record in original_reader(paths):
            payload = Payload([1.0] * 1000)
            refs.append(weakref.ref(payload))
            record["metrics"]["robot_force_samples"] = payload
            yield record

    def derive_without_retained_payloads(records, **kwargs):
        assert refs and not any(ref() is not None for ref in refs)
        assert all("robot_force_samples" not in row["metrics"] for row in records)
        return original_derive(records, **kwargs)

    monkeypatch.setattr(v2_calibration, "read_episode_files", read_with_payload)
    monkeypatch.setattr(
        v2_calibration, "derive_calibration_anchors", derive_without_retained_payloads
    )
    cfg, rows, kwargs = calibration_archive
    output = tmp_path / "anchors.json"
    original_read_bytes = Path.read_bytes

    def reject_whole_episode_file_read(path):
        assert path.suffix != ".jsonl", "episode file hashes must stream"
        return original_read_bytes(path)

    with monkeypatch.context() as scope:
        scope.setattr(Path, "read_bytes", reject_whole_episode_file_read)
        document = freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    expected = original_derive(rows, **kwargs)
    assert len({row["episode_id"] for row in rows}) < len(rows)
    assert document["anchors"] == expected["anchors"]
    assert document["force_decision"] == expected["force_decision"]
    assert (
        document["calibration"]["command_mode_counts"]
        == expected["calibration"]["command_mode_counts"]
    )
    assert json.loads(output.read_text()) == document
    hashes = document["calibration"]["episode_files_sha256"]
    assert len(hashes) == 14
    assert all(
        hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() == value
        for path, value in hashes.items()
    )
    path = tmp_path / "runs" / f"{kwargs['arms'][0]}__differential_drive" / "episodes.jsonl"
    path.write_text(path.read_text().replace(kwargs["source_commit"], "c" * 40))
    before = output.read_bytes()
    with pytest.raises(ValueError, match="stale/misrouted"):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == before
    corrupted = [json.loads(line) for line in path.read_text().splitlines()]
    for row in corrupted:
        row["git_hash"] = kwargs["source_commit"]
    corrupted[0]["algorithm_metadata"]["planner_runtime"] = {"fallback_triggered": True}
    path.write_text("".join(json.dumps(row) + "\n" for row in corrupted))
    with pytest.raises(ValueError, match="stale/misrouted"):
        freeze_campaign_anchors(tmp_path, output, campaign_config=cfg)
    assert output.read_bytes() == before


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


@pytest.fixture
def guarded_episode():
    """Use real shield serialization with the composite's native safe counters."""
    from robot_sf.planner.safety_shield import ShieldDecision

    decision = ShieldDecision(
        proposed_action=(1.0, 0.0),
        filtered_action=(0.1, 0.0),
        decision_label="fallback_safe",
        intervention_reason="safe_risk_dwa",
        intervened=True,
        fallback_controller_state={
            "policy": "RiskDWAPlannerAdapter",
            "selected_safe": True,
            "action_adaptation": {"mode": "guard_selected_command"},
        },
    ).to_metadata()
    return {
        **records()[0],
        "planner_key": "declared-arm",
        "algorithm_metadata": {
            "status": "ok",
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "planner_kinematics": {"execution_mode": "mixed"},
            "guard_stats": {"fallback_safe": 1},
            "planner_runtime": {"last_decision": decision},
            "shield_stats": {"last_decision": decision, "decision_counts": {"fallback_safe": 1}},
        },
    }


@pytest.mark.parametrize("expected", [None, "goal", "guarded_ppo"])
@pytest.mark.parametrize("nested_failure", [False, True])
def test_guarded_execution_matches_release_classifier(guarded_episode, expected, nested_failure):
    from robot_sf.benchmark.release_acceptance import _status_markers
    from robot_sf.benchmark.snqi.v2_reports import validate_episode_execution

    if nested_failure:
        guarded_episode["algorithm_metadata"]["shield_stats"]["last_decision"][
            "fallback_controller_state"
        ]["fallback_triggered"] = True
    rejected = expected != "guarded_ppo" or nested_failure
    assert bool(_status_markers(guarded_episode, "row", expected_algorithm=expected)) is rejected
    if rejected:
        with pytest.raises(ValueError, match="fallback/degraded"):
            validate_episode_execution(guarded_episode, expected_algorithm=expected)
        with pytest.raises(ValueError, match="fallback/degraded"):
            build_family_report(
                [guarded_episode],
                fixture_spec(),
                bootstrap_samples=2,
                expected_algorithms={"declared-arm": expected},
            )
    else:
        validate_episode_execution(guarded_episode, expected_algorithm=expected)
        report = build_family_report(
            [guarded_episode],
            fixture_spec(),
            bootstrap_samples=2,
            expected_algorithms={"declared-arm": expected},
        )
        assert report["episode_count"] == 1


@pytest.mark.parametrize("expected", [None, "goal", "guarded_ppo"])
def test_guarded_campaign_entry_binding(tmp_path, guarded_episode, expected):
    path = tmp_path / "episodes.jsonl"
    write_campaign_arm(path, [guarded_episode])
    original = path.read_bytes()
    entry = {
        "status": "ok",
        "episodes_path": str(path),
        "planner": {"key": "declared-arm", "algo": expected},
    }
    if expected != "guarded_ppo":
        with pytest.raises(ValueError, match="fallback/degraded"):
            enrich_campaign_v2(
                [entry],
                fixture_spec(),
                tmp_path / "reports",
                repo_root=tmp_path,
                bootstrap_samples=2,
            )
        assert path.read_bytes() == original
    else:
        result = enrich_campaign_v2(
            [entry], fixture_spec(), tmp_path / "reports", repo_root=tmp_path, bootstrap_samples=2
        )
        assert Path(result["snqi_v2_family_json"]).exists()


@pytest.mark.parametrize("expected", [None, "goal", "guarded_ppo"])
def test_guarded_offline_requires_independent_file_map(
    tmp_path, spec_files, guarded_episode, expected
):
    from scripts.tools.analyze_snqi_contract import main

    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(guarded_episode) + "\n")
    args = [
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
        str(tmp_path / "reports"),
    ]
    if expected is not None:
        declaration = tmp_path / "execution.json"
        declaration.write_text(
            json.dumps({path.name: {"key": "independent-arm", "algo": expected}})
        )
        args += ["--execution-map", str(declaration)]
    if expected != "guarded_ppo":
        with pytest.raises(ValueError, match="fallback/degraded"):
            main(args)
    else:
        assert main(args) == 0
        family = json.loads((tmp_path / "reports/snqi_v2_family.json").read_text())
        assert family["declared_ranking"][0]["planner"] == "independent-arm"


@pytest.mark.parametrize("expected", [None, "goal", "guarded_ppo"])
def test_guarded_calibration_uses_declared_algorithm(guarded_episode, expected):
    from robot_sf.benchmark.snqi.v2_calibration import derive_calibration_anchors

    rows, kwargs = calibration_records()
    rows[0]["algorithm_metadata"] = guarded_episode["algorithm_metadata"]
    kwargs["expected_algorithms"] = {rows[0]["planner_key"]: expected}
    if expected != "guarded_ppo":
        with pytest.raises(ValueError, match="fallback/degraded"):
            derive_calibration_anchors(rows, **kwargs)
    else:
        assert derive_calibration_anchors(rows, **kwargs)["calibration"]["episode_count"] == 1344


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


def test_real_small_campaign_v2_outputs(tmp_path, monkeypatch):
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
        + "\n  seeds: [201]\n  simulation_config:\n    ped_density: 0.0\n"
    )
    cfg = CampaignConfig(
        name="v2_smoke",
        scenario_matrix_path=scenario_path,
        planners=(PlannerSpec(key="goal", algo="goal"),),
        seed_policy=SeedPolicy(mode="fixed-list", seeds=(201,)),
        horizon=4,
        dt=0.1,
        workers=1,
        export_publication_bundle=False,
        snqi_weights_path=ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json",
        snqi_baseline_path=ROOT / "configs/benchmarks/snqi_baseline_camera_ready_v3.json",
        bootstrap_samples=10,
        snqi_contract=SnqiContractConfig(calibration_trials=10),
        snqi_v2_spec=fixture_spec(),
    )
    from robot_sf.benchmark.camera_ready import campaign
    from robot_sf.benchmark.identity.hash_utils import sha256_file
    from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest

    originals = {}
    original_enrich = campaign.enrich_campaign_v2

    def capture_enrichment(entries, *args, **kwargs):
        for entry in entries:
            path = Path(entry["episodes_path"])
            originals[str(path)] = {
                "hash": sha256_file(path),
                "rows": [json.loads(line) for line in path.read_text().splitlines()],
            }
        return original_enrich(entries, *args, **kwargs)

    monkeypatch.setattr(campaign, "enrich_campaign_v2", capture_enrichment)
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
    assert len(rows) == 1
    assert all("snqi_v2" in row["metrics"] and "snqi" in row["metrics"] for row in rows)
    assert not summary["campaign_integrity"]["blockers"]
    sidecar = json.loads(paths[0].with_suffix(".jsonl.provenance.json").read_text())
    validate_result_provenance_manifest(sidecar)
    artifact = next(item for item in sidecar["raw_artifacts"] if item["kind"] == "episodes_jsonl")
    assert artifact["sha256"] == sha256_file(paths[0])
    assert sidecar["snqi_v2_enrichment"]["input_sha256"] == originals[str(paths[0])]["hash"]
    assert sidecar["snqi_v2_enrichment"]["output_sha256"] == artifact["sha256"]
    for before, after in zip(originals[str(paths[0])]["rows"], rows, strict=True):
        for key, value in before.items():
            if key == "metrics":
                assert {k: after[key][k] for k in value} == value
            else:
                assert after[key] == value


def test_family_rejects_unpaired_or_duplicate_cells():
    rows = records()
    with pytest.raises(ValueError, match="paired scenario/seed"):
        build_family_report(rows[:-1], fixture_spec(), bootstrap_samples=2)
    with pytest.raises(ValueError, match="paired scenario/seed"):
        build_family_report(rows + rows[:1], fixture_spec(), bootstrap_samples=2)


def test_streaming_retains_only_compact_records_and_distinguishes_same_algo_arms(
    tmp_path, monkeypatch
):
    import weakref

    from robot_sf.benchmark.snqi import v2_reports

    class Payload(list):
        """Weak-referenceable decoded trace payload for a retention assertion."""

    entries, expected, refs = [], [], []
    for arm in ("variant-a", "variant-b"):
        rows = [
            {
                **row,
                "algo": "shared",
                "metrics": {
                    **row["metrics"],
                    "robot_force_samples": [[1.0, 2.0]] * 3000,
                },
                "algorithm_metadata": {"simulation_step_trace": {"steps": ["large"] * 3000}},
            }
            for row in records()
            if row["algo"] == "a"
        ]
        path = tmp_path / f"{arm}.jsonl"
        written = write_campaign_arm(path, rows)
        entries.append({"status": "ok", "planner": {"key": arm}, "episodes_path": str(path)})
        expected.extend({**row, "planner_key": arm, "kinematics": None} for row in written)
    expected_family = build_family_report(expected, fixture_spec(), bootstrap_samples=2)
    original_reader = v2_reports.read_episode_files
    original_writer = v2_reports.write_v2_reports

    def watched_reader(paths):
        for row in original_reader(paths):
            payload = Payload(row["metrics"]["robot_force_samples"])
            refs.append(weakref.ref(payload))
            row["metrics"]["robot_force_samples"] = payload
            assert sum(ref() is not None for ref in refs) <= 2
            yield row

    def checked_reports(rows, *args, **kwargs):
        assert all(ref() is None for ref in refs)
        assert all("algorithm_metadata" not in row for row in rows)
        assert all(set(row["metrics"]) == set(SOURCES.values()) for row in rows)
        return original_writer(rows, *args, **kwargs)

    monkeypatch.setattr(v2_reports, "read_episode_files", watched_reader)
    monkeypatch.setattr(v2_reports, "write_v2_reports", checked_reports)
    artifacts = enrich_campaign_v2(
        entries, fixture_spec(), tmp_path / "reports", repo_root=tmp_path, bootstrap_samples=2
    )
    actual = json.loads(Path(artifacts["snqi_v2_family_json"]).read_text())
    assert actual == expected_family
    assert {row["planner"] for row in actual["declared_ranking"]} == {"variant-a", "variant-b"}
    for entry in entries:
        row = json.loads(Path(entry["episodes_path"]).read_text().splitlines()[0])
        assert len(row["metrics"]["robot_force_samples"]) == 3000
        assert len(row["algorithm_metadata"]["simulation_step_trace"]["steps"]) == 3000


@pytest.mark.parametrize(
    "defect",
    ["unpaired", "stale_sidecar", "missing_sidecar", "duplicate_row_key", "duplicate_sidecar_key"],
)
def test_streaming_failure_keeps_originals_and_removes_staged_files(tmp_path, defect):
    entries = []
    for arm in ("a", "b"):
        path = tmp_path / f"{arm}.jsonl"
        rows = [row for row in records() if row["algo"] == arm]
        if defect == "unpaired" and arm == "b":
            rows = rows[:1]
        write_campaign_arm(path, rows)
        entries.append({"status": "ok", "planner": {"key": arm}, "episodes_path": str(path)})
    path = Path(entries[-1]["episodes_path"] + ".provenance.json")
    if defect == "stale_sidecar":
        payload = json.loads(path.read_text())
        payload["raw_artifacts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
    elif defect == "missing_sidecar":
        path.unlink()
    elif defect == "duplicate_sidecar_key":
        path.write_text('{"run":{},' + path.read_text()[1:])
    elif defect == "duplicate_row_key":
        raw_path = Path(entries[-1]["episodes_path"])
        raw_path.write_text('{"seed":999,' + raw_path.read_text()[1:])
        payload = json.loads(path.read_text())
        payload["raw_artifacts"][0]["sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="paired scenario/seed|sidecar|duplicate JSON key"):
        enrich_campaign_v2(
            entries, fixture_spec(), tmp_path / "reports", repo_root=tmp_path, bootstrap_samples=2
        )
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(tmp_path.glob(".*.snqi-v2.tmp"))


def test_offline_execution_map_rejects_duplicate_planner_key(tmp_path):
    import argparse

    from scripts.tools.analyze_snqi_contract import _v2_execution_declarations

    path = tmp_path / "execution.json"
    path.write_text('{"episodes.jsonl":{"key":"arm","algo":"goal","algo":"guarded_ppo"}}')
    with pytest.raises(ValueError, match="duplicate JSON key: algo"):
        _v2_execution_declarations(argparse.Namespace(execution_map=path))
