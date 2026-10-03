"""Source-bound 0.0.8 scoring and smoke contracts for #10112; no episodes.

Protect real authored/template bytes and public smoke admission. Removing v2
bindings or selecting historical smoke makes these tests fail. Prior authority
and v0_5 tests verify seeds/parity without post-freeze anchors or current smoke
admission. No test-only production seam is needed.
"""

from pathlib import Path

import yaml

from robot_sf.benchmark import runtime_smoke_admission as smoke

ROOT = Path(__file__).resolve().parents[2]
AUTHORED = "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"


def test_authored_campaign_binds_pending_acquisition():
    """Freeze source must name scoring assets and real dev acquisition, without fake anchors."""
    config = yaml.safe_load((ROOT / AUTHORED).read_bytes())
    binding = config.get("snqi_v2_spec")
    assert binding is not None, "D-083 campaign has no SNQI-v2 acquisition binding"
    assert binding["acquisition_config_path"] == (
        "configs/benchmarks/snqi_v2/calibration.dev1001_1002_scheduled_acquisition.yaml"
    )
    assert binding["anchors_path"] == "configs/benchmarks/snqi_v2/anchors.v2.0.json"
    assert binding["anchor_freeze"] == "same-source-custody"
    assert config["snqi_weights"] is None and config["snqi_baseline"] is None
    assert not config["snqi_contract"]["enabled"]


def test_both_release_templates_pin_v2_acquisition():
    """Both identities must hash scoring and acquisition inputs instead of legacy assets."""
    for name in (
        "benchmark_data_release_s30_h600.template.yaml",
        "three_width_doorway_release_0_0_8_v1.template.yaml",
    ):
        payload = yaml.safe_load((ROOT / "configs/benchmarks/releases" / name).read_bytes())
        binding = payload["metrics"].get("snqi_v2_binding")
        assert binding is not None, f"{name} has no source-bound v2 asset pins"
        assert set(binding) == {"weights", "anchors", "family", "acquisition_config"}
        assert "snqi_weights_path" not in payload["metrics"]


def test_canonical_smoke_admits_current_dev_contract():
    """Actual public admission must select tracked dev1003 smoke and the D-083 roster."""
    authored = yaml.safe_load((ROOT / AUTHORED).read_bytes())
    keys = tuple(planner["key"] for planner in authored["planners"])
    assert smoke.RUNTIME_SMOKE_PLANNER_KEYS == keys, "canonical smoke still uses the v3 roster"
    contract = smoke._canonical_smoke_contract(repo_root=ROOT, expected_planner_keys=keys)
    assert contract[4] == 1003, "canonical smoke does not bind development seed 1003"
    assert contract[0].name.endswith("v0_6.yaml")
    config = yaml.safe_load(contract[1].read_bytes())
    assert config["derived_from"]["config"] == AUTHORED
    assert config["planners"] == authored["planners"]
