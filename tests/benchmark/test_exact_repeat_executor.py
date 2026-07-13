"""Contract tests for the exact-repeat campaign executor (issue #5375)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robot_sf.baselines import is_runnable_algo
from robot_sf.benchmark.exact_repeat_campaign import (
    HOST_REPORT_SCHEMA_VERSION,
    RESOLVED_DEFINITIONS_SCHEMA_VERSION,
    UNRUNNABLE_DISPOSITION,
    _check_manifest_from_bundle,
    _compute_trajectory_hash,
    _get_environment_fingerprint,
    _safe_json_value,
    canonical_sha256,
    execute_campaign,
    resolve_runnable_definitions,
    verify_host_report,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "benchmark"
    / "scenario_flakiness_issue_4978"
)
CAMPAIGN_CONFIG = (
    REPOSITORY_ROOT
    / "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """Build the retained exact-repeat manifest once for executor tests."""
    episodes = [
        json.loads(line)
        for line in (FIXTURE_DIR / "real_campaign_episodes.jsonl").read_text().splitlines()
    ]
    from robot_sf.benchmark.exact_repeat_campaign import (
        build_manifest,
    )

    return build_manifest(_load(FIXTURE_DIR / "real_campaign_flakiness_report.json"), episodes)


@pytest.fixture(scope="module")
def resolved_bundle(manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve the runnable definitions used by executor tests."""
    return resolve_runnable_definitions(manifest, CAMPAIGN_CONFIG)


# --- _safe_json_value -------------------------------------------------------


def test_safe_json_value_passes_through_primitive_types():
    """Primitive JSON values retain their exact representation."""
    assert _safe_json_value(True) is True
    assert _safe_json_value(False) is False
    assert _safe_json_value(42) == 42
    assert _safe_json_value(3.14) == 3.14
    assert _safe_json_value("hello") == "hello"


def test_safe_json_value_converts_nan_and_inf_to_null():
    """Non-finite floats become canonical JSON null values."""
    assert _safe_json_value(float("nan")) is None
    assert _safe_json_value(float("inf")) is None
    assert _safe_json_value(float("-inf")) is None


def test_safe_json_value_converts_nested_structures():
    """Nested collections receive recursive finite-value cleanup."""
    value = _safe_json_value({"a": [1.0, float("nan"), True], "b": {"c": float("inf")}})
    assert value == {"a": [1.0, None, True], "b": {"c": None}}


def test_safe_json_value_converts_numpy_scalars_and_arrays():
    """NumPy values become JSON-safe Python values before hashing."""
    value = _safe_json_value({"scalar": np.float32(1.5), "array": np.array([1, 2])})
    assert value == {"scalar": 1.5, "array": [1, 2]}


# --- _compute_trajectory_hash -----------------------------------------------


def test_trajectory_hash_is_valid_sha256_hex():
    """A complete record produces a conventional SHA-256 digest."""
    record = {
        "outcome": {"success": True, "collision": False, "timeout": False},
        "metrics": {"success": 1.0, "collisions": 0.0, "near_misses": 0.0},
    }
    h = _compute_trajectory_hash(record)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_trajectory_hash_is_deterministic_for_same_record():
    """Equal records always produce equal trajectory hashes."""
    record = {
        "outcome": {"success": False, "collision": True, "timeout": False},
        "metrics": {"success": 0.0, "collisions": 1.0, "near_misses": 2.0},
    }
    h1 = _compute_trajectory_hash(record)
    h2 = _compute_trajectory_hash(record)
    assert h1 == h2


def test_trajectory_hash_differs_when_outcome_changes():
    """Changing the outcome changes the trajectory hash."""
    r1 = {
        "outcome": {"success": True, "collision": False, "timeout": False},
        "metrics": {"success": 1.0, "collisions": 0.0},
    }
    r2 = {
        "outcome": {"success": False, "collision": True, "timeout": False},
        "metrics": {"success": 0.0, "collisions": 1.0},
    }
    assert _compute_trajectory_hash(r1) != _compute_trajectory_hash(r2)


def test_trajectory_hash_handles_nan_in_metrics():
    """A NaN metric is canonicalized before hashing."""
    record = {
        "outcome": {"success": False, "collision": False, "timeout": True},
        "metrics": {
            "success": 0.0,
            "collisions": 0.0,
            "time_to_goal_success_only": float("nan"),
        },
    }
    h = _compute_trajectory_hash(record)
    assert len(h) == 64


def test_trajectory_hash_rejects_missing_outcome():
    """Malformed records without outcomes fail closed."""
    with pytest.raises(ValueError, match="outcome"):
        _compute_trajectory_hash({"metrics": {}})


def test_trajectory_hash_rejects_missing_metrics():
    """Malformed records without metrics fail closed."""
    with pytest.raises(ValueError, match="metrics"):
        _compute_trajectory_hash({"outcome": {"success": True}})


# --- _get_environment_fingerprint -------------------------------------------


def test_environment_fingerprint_has_required_fields():
    """The generated environment fingerprint satisfies host schema fields."""
    env = _get_environment_fingerprint()
    for field in (
        "machine_id",
        "cpu_only",
        "workers",
        "numpy_version",
        "numba_version",
        "python_version",
        "git_commit",
        "lockfile_sha256",
    ):
        assert field in env, f"missing environment field: {field}"
    assert env["cpu_only"] is True
    assert env["workers"] == 1
    assert len(env["lockfile_sha256"]) == 64


# --- _check_manifest_from_bundle --------------------------------------------


def test_check_manifest_from_bundle_validates_resolved_bundle(manifest, resolved_bundle):
    """A fresh resolved bundle supplies all exact-repeat targets."""
    indexed, repeats = _check_manifest_from_bundle(resolved_bundle)
    assert repeats == 3
    assert len(indexed) == 140


def test_check_manifest_from_bundle_rejects_missing_execution_contract():
    """A missing execution contract is rejected before execution."""
    bad = {
        "execution_contract": None,
        "targets": [
            {
                "scenario_id": "s",
                "planner": "p",
                "seed": 1,
                "source_config_hash": "abc",
                "horizon": 50,
            }
        ],
    }
    with pytest.raises(ValueError, match="must be an object"):
        _check_manifest_from_bundle(bad)


# --- execute_campaign with mocked runner ------------------------------------


def _build_mock_record(seed: int) -> dict[str, Any]:
    """Build a stable mock episode record whose hash depends only on seed."""
    return {
        "outcome": {"success": True, "collision": False, "timeout": False},
        "metrics": {
            "success": 1.0,
            "collisions": 0.0,
            "near_misses": 0.0,
            "route_distance": 10.0 + seed * 0.01,
            "min_distance": 1.5,
        },
        "scenario_id": "test-scenario",
        "seed": seed,
    }


def _deterministic_mock_runner(
    scenario_params: dict[str, Any],
    seed: int,
    *,
    algo: str = "goal",
    **kwargs: Any,
) -> dict[str, Any]:
    """Mock run_episode that returns identical data for the same seed."""
    return _build_mock_record(seed)


def test_execute_campaign_produces_valid_host_report(tmp_path, manifest, resolved_bundle):
    """Execute with mock runner produces a host_result.json that passes verify_host_report."""
    output_dir = tmp_path / "host_run"
    host_result = execute_campaign(
        resolved_bundle, output_dir=output_dir, run_episode=_deterministic_mock_runner
    )

    assert host_result["schema_version"] == HOST_REPORT_SCHEMA_VERSION
    assert host_result["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(host_result["results"]) == len(manifest["targets"])
    assert (output_dir / "host_result.json").exists()

    # Verify the host report passes the verifier
    verified = verify_host_report(manifest, host_result)
    assert verified["summary"]["n_targets"] == 140
    assert verified["summary"]["n_cells"] == 7


def test_execute_campaign_dispositions_unrunnable_orca_cells(tmp_path, resolved_bundle):
    """orca cells are recorded as an explicit disposition instead of crashing.

    The #5263 bundle contains ``algo: orca`` cells, which the ``run_episode``
    baseline registry cannot construct on current main. Rather than raising
    ``ValueError: Unknown algorithm 'orca'``, execute must emit a schema-compatible
    per-cell disposition and continue running the other cells.
    """

    def guarded_runner(scenario_params, seed, *, algo="goal", **kwargs):
        # The runner must never be invoked for an unrunnable planner.
        assert is_runnable_algo(algo), f"runner invoked for unrunnable algo {algo!r}"
        return _build_mock_record(seed)

    host_result = execute_campaign(
        resolved_bundle, output_dir=tmp_path / "orca_disposition", run_episode=guarded_runner
    )

    by_planner = {}
    for result in host_result["results"]:
        by_planner.setdefault(result["planner"], []).append(result)

    orca_results = by_planner["orca"]
    assert orca_results, "expected orca targets in the bundle"
    for result in orca_results:
        assert result["disposition"] == UNRUNNABLE_DISPOSITION
        assert result["repeats"] == []
        assert "orca" in result["disposition_reason"]

    # Runnable planners still execute their repeats normally.
    for planner in ("ppo", "goal"):
        for result in by_planner[planner]:
            assert "disposition" not in result
            assert len(result["repeats"]) == 3


def test_verify_host_report_scopes_repeat_claim_to_runnable_cells(
    tmp_path, manifest, resolved_bundle
):
    """verify-host accepts orca dispositions and scopes the claim to runnable cells."""
    host_result = execute_campaign(
        resolved_bundle,
        output_dir=tmp_path / "orca_verify",
        run_episode=_deterministic_mock_runner,
    )
    verified = verify_host_report(manifest, host_result)
    summary = verified["summary"]

    assert summary["n_targets"] == 140
    assert summary["n_cells"] == 7
    # orca contributes 3 unrunnable cells (60 targets); ppo(3) + goal(1) remain.
    assert summary["n_unrunnable_cells"] == 3
    assert summary["n_runnable_cells"] == 4
    assert summary["n_unrunnable_targets"] == 60
    assert summary["n_runnable_targets"] == 80
    # The deterministic mock runner keeps the runnable-cell claim green.
    assert summary["all_cells_bitwise_identical"] is True

    unrunnable_cells = [cell for cell in verified["cells"] if cell.get("unrunnable")]
    assert {cell["planner"] for cell in unrunnable_cells} == {"orca"}
    for cell in unrunnable_cells:
        assert cell["disposition"] == UNRUNNABLE_DISPOSITION
        assert cell["exact_repeat_determinism"] is None


def test_execute_campaign_all_repeats_identical_for_deterministic_runner(
    tmp_path, manifest, resolved_bundle
):
    """A deterministic mock runner produces identical repeats."""
    output_dir = tmp_path / "host_run_identical"
    host_result = execute_campaign(
        resolved_bundle, output_dir=output_dir, run_episode=_deterministic_mock_runner
    )

    verified = verify_host_report(manifest, host_result)
    assert verified["summary"]["all_cells_bitwise_identical"] is True


def test_execute_campaign_writes_host_result_json(tmp_path, resolved_bundle):
    """The executor writes host_result.json to the output directory."""
    output_dir = tmp_path / "host_run_json"

    execute_campaign(resolved_bundle, output_dir=output_dir, run_episode=_deterministic_mock_runner)

    host_path = output_dir / "host_result.json"
    assert host_path.exists()
    loaded = json.loads(host_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == HOST_REPORT_SCHEMA_VERSION


def test_execute_campaign_target_filter_executes_only_requested_targets(
    tmp_path, manifest, resolved_bundle
):
    """target_filter rejects incomplete reports instead of writing unverifiable evidence."""
    # Pick the first target's scenario_definition_id
    first_def_id = resolved_bundle["targets"][0]["scenario_definition_id"]
    second_def_id = resolved_bundle["targets"][1]["scenario_definition_id"]

    output_dir = tmp_path / "host_run_filtered"
    with pytest.raises(ValueError, match="requires compatible cached results"):
        execute_campaign(
            resolved_bundle,
            output_dir=output_dir,
            run_episode=_deterministic_mock_runner,
            target_filter=[first_def_id, second_def_id],
        )
    assert not (output_dir / "host_result.json").exists()


def test_execute_campaign_resume_skips_cached_results(tmp_path, resolved_bundle):
    """On second run, cached targets are skipped and produce same results."""
    call_log: list[int] = []

    def counting_runner(scenario_params, seed, **kwargs):
        call_log.append(seed)
        return _deterministic_mock_runner(scenario_params, seed, **kwargs)

    output_dir = tmp_path / "host_run_resume"
    execute_campaign(resolved_bundle, output_dir=output_dir, run_episode=counting_runner)
    first_run_count = len(call_log)

    call_log.clear()
    execute_campaign(resolved_bundle, output_dir=output_dir, run_episode=counting_runner)
    second_run_count = len(call_log)

    assert second_run_count == 0, f"resume should skip all targets, but ran {second_run_count}"
    assert first_run_count > 0, "first run should have executed targets"


def test_execute_campaign_reexecutes_a_malformed_cache_entry(tmp_path, resolved_bundle):
    """A non-mapping cache entry is ignored instead of crashing resume execution."""
    output_dir = tmp_path / "host_run_malformed_cache"
    execute_campaign(resolved_bundle, output_dir=output_dir, run_episode=_deterministic_mock_runner)
    # Pick a runnable target: unrunnable planners (e.g. orca) are dispositioned
    # without invoking the runner, so they never re-execute on a corrupt cache.
    target = next(t for t in resolved_bundle["targets"] if is_runnable_algo(t["planner"]))
    cache_file = (
        output_dir
        / "target_cache"
        / (f"{target['scenario_id']}_{target['planner']}_{target['seed']}.json")
    )
    cache_file.write_text("null\n", encoding="utf-8")
    calls: list[int] = []

    def counting_runner(scenario_params, seed, **kwargs):
        calls.append(seed)
        return _deterministic_mock_runner(scenario_params, seed, **kwargs)

    execute_campaign(resolved_bundle, output_dir=output_dir, run_episode=counting_runner)
    assert len(calls) == 3


def test_execute_campaign_preserves_numpy_near_miss_metrics(tmp_path, resolved_bundle):
    """Finite NumPy near-miss values are recorded instead of silently becoming zero."""

    def numpy_metric_runner(scenario_params, seed, **kwargs):
        record = _deterministic_mock_runner(scenario_params, seed, **kwargs)
        record["metrics"]["near_misses"] = np.float32(2)
        return record

    report = execute_campaign(
        resolved_bundle,
        output_dir=tmp_path / "host_run_numpy_near_misses",
        run_episode=numpy_metric_runner,
    )
    assert {
        repeat["near_misses"] for result in report["results"] for repeat in result["repeats"]
    } == {2}


def test_execute_campaign_rejects_missing_scenario_definition(tmp_path, resolved_bundle):
    """Missing definitions fail closed with a clear error before dictionary conversion."""
    tampered = copy.deepcopy(resolved_bundle)
    scenario_id = tampered["targets"][0]["scenario_definition_id"]
    del tampered["scenario_definitions"][scenario_id]
    tampered["bundle_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "bundle_sha256"}
    )
    with pytest.raises(ValueError, match="missing scenario_definition"):
        execute_campaign(tampered, output_dir=tmp_path / "missing_definition")


def test_execute_campaign_rejects_planner_definition_mismatch(tmp_path, resolved_bundle):
    """A target cannot execute under a planner different from its manifest identity."""
    tampered = copy.deepcopy(resolved_bundle)
    target = tampered["targets"][0]
    planner_definition = tampered["planner_definitions"][target["planner_definition_id"]]
    planner_definition["algo"] = "orca" if target["planner"] != "orca" else "goal"
    tampered["bundle_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "bundle_sha256"}
    )

    with pytest.raises(ValueError, match="does not match.*planner_definition"):
        execute_campaign(tampered, output_dir=tmp_path / "planner_mismatch")


def test_execute_campaign_rejects_invalid_bundle_schema():
    """Bundle with wrong schema_version raises."""
    bad_bundle = {"schema_version": "wrong_version"}
    with pytest.raises(ValueError, match="schema_version"):
        execute_campaign(bad_bundle, output_dir=Path("/tmp/no_such_dir"))


def test_execute_campaign_bundle_sha_validation():
    """Bundle with tampered sha256 is rejected."""

    # Create a valid-looking bundle with a wrong bundle_sha256
    bundle = {
        "schema_version": RESOLVED_DEFINITIONS_SCHEMA_VERSION,
        "manifest_sha256": "a" * 64,
        "execution_contract": {"repeats_per_target": 3},
        "targets": [],
        "bundle_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="bundle_sha256"):
        execute_campaign(bundle, output_dir=Path("/tmp/no_such_dir"))
