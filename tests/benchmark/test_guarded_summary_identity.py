"""Contributor identity must survive worker aggregation and campaign admission."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.fallback_policy import (
    is_verified_guarded_ppo,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.map_runner.map_runner_batch_runner import _initial_feasibility_totals
from robot_sf.benchmark.map_runner.map_runner_batch_summary import (
    apply_worker_metadata_bridge,
    merge_runtime_algorithm_contract,
)


def _records(kind, bad_first):
    """Return one unbound contributor and one verified contributor with native telemetry."""
    valid = {
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "guard_stats": {"fallback_safe": 1},
        }
    }
    bad = {
        "missing": {},
        "null": {"algorithm_metadata": None},
        "list": {"algorithm_metadata": []},
        "nonempty-list": {"algorithm_metadata": ["invalid"]},
        "empty": {"algorithm_metadata": {}},
    }[kind]
    return [bad, valid] if bad_first else [valid, bad]


def _summary(records, separate_workers):
    """Use the production bridge and merger, including serialized independent worker contracts."""
    runtime = None
    for record in records:
        bridge = apply_worker_metadata_bridge(
            record,
            feasibility_totals=_initial_feasibility_totals(),
            runtime_algorithm_contract=None if separate_workers else runtime,
        )
        if separate_workers:
            payload = json.loads(json.dumps(bridge.runtime_algorithm_contract))
            runtime = merge_runtime_algorithm_contract(runtime or {}, payload)
        else:
            runtime = bridge.runtime_algorithm_contract
    contract = merge_runtime_algorithm_contract(
        enrich_algorithm_metadata(algo="guarded_ppo"), runtime
    )
    return {
        "status": "ok",
        "written": len(records),
        "total_jobs": len(records),
        "failed_jobs": 0,
        "failures": [],
        "algorithm_readiness": {"name": "guarded_ppo"},
        "algorithm_metadata_contract": contract,
    }


@pytest.mark.parametrize("kind", ["missing", "null", "list", "nonempty-list", "empty"])
@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("separate_workers", [False, True])
def test_every_contributing_episode_must_bind_identity(kind, bad_first, separate_workers):
    """No missing metadata may borrow another episode's verified guarded identity."""
    records = _records(kind, bad_first)
    before = deepcopy(records)
    summary = _summary(records, separate_workers)
    contract = summary["algorithm_metadata_contract"]
    assert not is_verified_guarded_ppo(contract, expected_algorithm="guarded_ppo")
    availability = summarize_benchmark_availability(summary)
    assert not availability.benchmark_success
    assert "guard_stats.fallback_safe" in availability.availability_reason
    assert contract["guard_stats"]["fallback_safe"] == 1
    assert records == before


@pytest.mark.parametrize("kind", ["missing", "null", "list", "empty"])
@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("separate_workers", [False, True])
def test_unbound_contributor_halts_guarded_campaign(
    tmp_path, monkeypatch, kind, bad_first, separate_workers
):
    """The real campaign halts subsequent arms on the aggregate's unverified shield telemetry."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [1001]\n"
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        f"name: identity_smoke\nscenario_matrix: {scenario}\n"
        "seed_policy:\n  mode: fixed-list\n  seeds: [1001]\n"
        "stop_on_failure: true\nexport_publication_bundle: false\n"
        "planners:\n  - key: guarded_ppo\n    algo: guarded_ppo\n"
        "    planner_group: experimental\n    benchmark_profile: experimental\n"
        "  - key: goal\n    algo: goal\n    planner_group: core\n"
        "    benchmark_profile: baseline-safe\n"
    )
    calls = []

    def run_batch(scenarios_or_path, out_path, schema_path, *, algo, **kwargs):
        """Supply adversarial episodes at the existing campaign batch dependency boundary."""
        del scenarios_or_path, schema_path, kwargs
        calls.append(algo)
        assert algo == "guarded_ppo", "Campaign executed an arm after unbound shield telemetry"
        records = _records(kind, bad_first)
        for index, record in enumerate(records):
            record.update(
                episode_id=f"smoke-{index}",
                scenario_id="smoke",
                config_hash="identity-smoke-config",
                seed=1001,
                scenario_params={"algo": "guarded_ppo"},
                metrics={"success": 0.0, "collisions": 0.0},
            )
        out_path.write_text("".join(json.dumps(record) + "\n" for record in records))
        return _summary(records, separate_workers)

    monkeypatch.setattr("robot_sf.benchmark.camera_ready_campaign.run_batch", run_batch)
    result = run_campaign(load_campaign_config(config), output_root=tmp_path / "out")
    assert calls == ["guarded_ppo"]
    assert result["benchmark_success"] is False
    assert result["exit_code"] != 0
    payload = json.loads(Path(result["summary_json"]).read_text())
    row = payload["planner_rows"][0]
    assert "guard_stats.fallback_safe" in row["availability_reason"]
