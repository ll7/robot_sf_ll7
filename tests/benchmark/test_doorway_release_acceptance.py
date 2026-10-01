"""Static, byte-backed acceptance witnesses; never reset or step sealed seeds."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from robot_sf.benchmark.seed_bands import EVAL_SEEDS_0_0_8
from tests.benchmark import test_release_acceptance as fixtures

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml"
MATRIX = ROOT / "configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml"


@pytest.fixture
def doorway(tmp_path, monkeypatch):
    """Build producer-format JSONLs and sidecars without instantiating any planner."""
    payload = yaml.safe_load(TEMPLATE.read_text())
    scenarios = yaml.safe_load(MATRIX.read_text())["scenarios"]
    # seed-holdout: synthetic-fixture begin
    for scenario in scenarios:
        scenario["seeds"] = list(EVAL_SEEDS_0_0_8)
        scenario["map_file"] = (
            (MATRIX.parent / scenario["map_file"]).resolve().relative_to(ROOT).as_posix()
        )
    roster = tuple(payload["planners"]["keys"])
    algorithms = {key: ("hybrid_rule_local_planner" if "hybrid" in key else key) for key in roster}
    monkeypatch.setattr(fixtures, "_PLANNER_KEYS", roster)
    monkeypatch.setattr(fixtures, "_PLANNER_ALGORITHMS", algorithms)
    monkeypatch.setattr(fixtures, "_SCENARIO_IDS", tuple(s["name"] for s in scenarios))
    monkeypatch.setattr(fixtures, "_SEEDS", EVAL_SEEDS_0_0_8)
    campaign, cfg = fixtures._write_provenance_bound_full_campaign(
        tmp_path,
        monkeypatch,
        horizon=400,
        scenarios=scenarios,
    )
    manifest = fixtures._full_manifest()
    manifest.release_kind = "benchmark-doorway-width-slice.v1"
    manifest.width_slice_contract = deepcopy(payload["width_slice_contract"])
    manifest.expected_episode_cells = 1260
    # Tracked template requests 600, but authored scenario caps and rows are 400.
    manifest.expected_horizon_steps = 600
    # seed-holdout: synthetic-fixture end
    return campaign, cfg, manifest, scenarios


def acceptance(doorway):
    """Invoke production admission against real JSONLs and provenance sidecars."""
    campaign, cfg, manifest, _ = doorway
    return validate_full_benchmark_release_acceptance(
        campaign,
        manifest=manifest,
        campaign_config=cfg,
        source_repository_root=cfg.source_repository_root,
    )


def test_bound_doorway_slice_accepted(doorway):
    """The complete 1260-cell authored slice must have its own denominator."""
    report = acceptance(doorway)
    assert report["status"] == "valid", report["blockers"]
    assert report["expected_episode_cells"] == 1260
    assert report["observed_episode_rows"] == report["unique_episode_identities"] == 1260
    assert report["expected_scenario_count"] == 3


def test_tracked_legacy_kind_requires_bound_slice_contract(doorway):
    """The unchanged template kind is compatible only with its exact v1 binding."""
    doorway[2].release_kind = "benchmark-width-slice"
    assert acceptance(doorway)["status"] == "valid"
    doorway[2].width_slice_contract = None
    assert acceptance(doorway)["status"] == "invalid"


@pytest.mark.parametrize(
    "mutation", ["width", "scenario", "roster", "seeds", "horizon", "count", "cap", "contract"]
)
def test_slice_refuses_mutated_axes(doorway, mutation):
    """A self-declared width kind must not admit any altered frozen axis."""
    _, _, manifest, scenarios = doorway
    if mutation == "width":
        scenarios[0]["metadata"]["width_slice_m"] = 2.4
    elif mutation == "scenario":
        scenarios[0]["name"] = "another_doorway"
    elif mutation == "roster":
        manifest.planner_keys = ("substitute",) + manifest.planner_keys[1:]
    elif mutation == "seeds":
        manifest.resolved_seeds = tuple(range(1001, 1031))
    elif mutation == "horizon":
        manifest.expected_horizon_steps = 601
    elif mutation == "count":
        manifest.expected_episode_cells = 1259
    elif mutation == "cap":
        scenarios[0]["simulation_config"]["max_episode_steps"] = 399
    else:
        manifest.width_slice_contract["widths_m"] = [2.2, 2.8, 3.7]
    report = acceptance(doorway)
    assert report["status"] == "invalid"
    assert report["benchmark_success"] is False


def test_main_refuses_slice_denominator(doorway):
    """A 1260-cell campaign never inherits the main release gate."""
    doorway[2].release_kind = "benchmark-data"
    report = acceptance(doorway)
    assert report["status"] == "invalid"
    assert report["expected_episode_cells"] == 20160
    assert any("20160" in b for b in report["blockers"])


def test_slice_loader_preserves_template_bytes_and_uses_h400():
    """Only the v0.2 bound slice receives the runtime cap; legacy data stay unchanged."""
    from dataclasses import replace

    from robot_sf.benchmark.release_protocol import (
        load_release_campaign_config,
        load_release_manifest,
    )

    legacy = load_release_manifest(TEMPLATE.with_name("three_width_doorway_release_0_0_8_v1.yaml"))
    before = legacy.canonical_campaign_config_path.read_bytes()
    assert load_release_campaign_config(legacy).horizon == 600
    bound = replace(legacy, schema_version="benchmark-release-manifest.v0.2")
    assert load_release_campaign_config(bound).horizon == 400
    assert legacy.canonical_campaign_config_path.read_bytes() == before
