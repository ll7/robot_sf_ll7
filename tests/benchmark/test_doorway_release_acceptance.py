"""Static, byte-backed acceptance witnesses; never reset or step sealed seeds."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from tests.benchmark import test_release_acceptance as fixtures

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml"
MATRIX = ROOT / "configs/scenarios/francis2023_narrow_doorway_three_width_release_0_0_8_v1.yaml"


@pytest.fixture(autouse=True)
def forbid_environment_construction(monkeypatch):
    """Static witnesses must fail immediately if they ever attempt an environment."""

    def abort(*args, **kwargs):
        pytest.fail("environment construction attempted by static acceptance witness")

    monkeypatch.setattr("robot_sf.gym_env.environment_factory.make_robot_env", abort)
    monkeypatch.setattr("robot_sf.gym_env.robot_env.RobotEnv.__init__", abort)
    monkeypatch.setattr("robot_sf.gym_env.robot_env.RobotEnv.reset", abort)
    monkeypatch.setattr("robot_sf.gym_env.robot_env.RobotEnv.step", abort)


def build_doorway(tmp_path, monkeypatch, *, seeds=None, width=None, cap=None):
    """Build producer-format JSONLs and sidecars without instantiating any planner."""
    payload = yaml.safe_load(TEMPLATE.read_text())
    scenarios = yaml.safe_load(MATRIX.read_text())["scenarios"]
    inventory = tuple(payload["seed_policy"]["resolved_seeds"]) if seeds is None else seeds
    # seed-holdout: synthetic-fixture begin
    for scenario in scenarios:
        scenario["seeds"] = list(inventory)
        scenario["map_file"] = (
            (MATRIX.parent / scenario["map_file"]).resolve().relative_to(ROOT).as_posix()
        )
    if width is not None:
        scenarios[0]["metadata"]["width_slice_m"] = width
    if cap is not None:
        scenarios[0]["simulation_config"]["max_episode_steps"] = cap
    roster = tuple(payload["planners"]["keys"])
    algorithms = {key: ("hybrid_rule_local_planner" if "hybrid" in key else key) for key in roster}
    monkeypatch.setattr(fixtures, "_PLANNER_KEYS", roster)
    monkeypatch.setattr(fixtures, "_PLANNER_ALGORITHMS", algorithms)
    monkeypatch.setattr(fixtures, "_SCENARIO_IDS", tuple(s["name"] for s in scenarios))
    monkeypatch.setattr(fixtures, "_SEEDS", inventory)
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
    manifest.expected_horizon_steps = 400
    # seed-holdout: synthetic-fixture end
    return campaign, cfg, manifest, scenarios


@pytest.fixture
def doorway(tmp_path, monkeypatch):
    """Build the correctly bound metadata-only slice."""
    return build_doorway(tmp_path, monkeypatch)


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
    doorway[2].expected_horizon_steps = 600
    assert acceptance(doorway)["status"] == "valid"
    doorway[2].width_slice_contract = None
    report = acceptance(doorway)
    assert report["status"] == "invalid"
    assert (
        "doorway slice requires the exact benchmark-doorway-width-slice.v1 binding"
        in report["blockers"]
    )


@pytest.mark.parametrize(
    "mutation,blocker",
    [
        ("width", "doorway slice scenario width, map or H400 cap differs from its binding"),
        ("scenario", "doorway slice requires exactly the authored 2.2/2.8/3.6 m scenarios"),
        ("roster", "doorway slice requires the exact main-campaign 14-arm roster"),
        ("seeds", "doorway slice requires the exact sealed 30-seed inventory"),
        ("horizon", "doorway slice must declare H400"),
        ("horizon600", "doorway slice must declare H400"),
        ("count", "doorway slice requires exactly 1260 cells"),
        ("cap", "doorway slice scenario width, map or H400 cap differs from its binding"),
        ("contract", "doorway slice requires the exact benchmark-doorway-width-slice.v1 binding"),
    ],
)
def test_slice_refuses_mutated_axes(doorway, mutation, blocker):
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
    elif mutation == "horizon600":
        manifest.expected_horizon_steps = 600
    elif mutation == "count":
        manifest.expected_episode_cells = 1259
    elif mutation == "cap":
        scenarios[0]["simulation_config"]["max_episode_steps"] = 399
    else:
        manifest.width_slice_contract["widths_m"] = [2.2, 2.8, 3.7]
    report = acceptance(doorway)
    assert blocker in report["blockers"], report
    assert report["status"] == "invalid"
    assert report["benchmark_success"] is False


def test_slice_refuses_consistent_wrong_seed_inventory(tmp_path, monkeypatch):
    """Consistent dev-seed metadata must be refused solely by the sealed binding."""
    campaign = build_doorway(tmp_path, monkeypatch, seeds=tuple(range(1001, 1031)))
    report = acceptance(campaign)
    assert report["blockers"] == ["doorway slice requires the exact sealed 30-seed inventory"]
    assert report["status"] == "invalid"
    assert report["observed_episode_rows"] == report["unique_episode_identities"] == 1260


@pytest.mark.parametrize("axis,value", [("width", 2.4), ("cap", 399)])
def test_slice_refuses_consistent_wrong_width_or_cap(tmp_path, monkeypatch, axis, value):
    """Rehashed matrix and sidecars leave only the width/cap binding to refuse them."""
    campaign = build_doorway(tmp_path, monkeypatch, **{axis: value})
    report = acceptance(campaign)
    assert report["blockers"] == [
        "doorway slice scenario width, map or H400 cap differs from its binding"
    ]
    assert report["status"] == "invalid"
    assert report["observed_episode_rows"] == report["unique_episode_identities"] == 1260


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
    legacy_cfg = load_release_campaign_config(legacy)
    assert legacy_cfg.horizon is None
    bound = replace(
        legacy,
        schema_version="benchmark-release-manifest.v0.2",
        scenario_horizons_path=legacy_cfg.scenario_horizons_path,
        scenario_horizons_sha256=legacy_cfg.scenario_horizons_sha256,
    )
    from robot_sf.benchmark.release_protocol import resolve_release_horizon_budgets

    cfg = load_release_campaign_config(bound)
    assert cfg.horizon is None
    assert set(resolve_release_horizon_budgets(bound, cfg).values()) == {400}
    assert legacy.canonical_campaign_config_path.read_bytes() == before


@pytest.mark.parametrize(
    "kind,horizon",
    [("benchmark-width-slice", 401), ("benchmark-doorway-width-slice.v1", 600)],
)
def test_materialization_refuses_wrong_template_horizon(kind, horizon):
    """Identity normalization must not silently repair a tampered declaration."""
    from robot_sf.benchmark import release_protocol as protocol

    payload = yaml.safe_load(TEMPLATE.read_text())
    payload["release_kind"] = kind
    payload["matrix"]["horizon_steps"] = horizon
    with pytest.raises(ValueError, match="template horizon"):
        protocol._materialize_release_template_payload(
            payload,
            template_path=TEMPLATE,
            metadata_path=TEMPLATE,
            metadata_sha256="0" * 64,
            source_commit="0" * 40,
            latest_main_base_commit="0" * 40,
            release_tag="diagnostic-static-fixture",
            concept_doi="10.5281/zenodo.99000001",
            version_doi="10.5281/zenodo.99000002",
            repository_root=ROOT,
        )
