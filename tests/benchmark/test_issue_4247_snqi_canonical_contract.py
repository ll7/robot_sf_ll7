"""Regression tests for issue #4247 canonical SNQI weight governance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES
from robot_sf.benchmark.snqi.weights_validation import validate_weights_mapping

ROOT = Path(__file__).resolve().parents[2]
WEIGHT_SET_REGISTRY = ROOT / "configs/benchmarks/snqi_weight_sets_camera_ready.yaml"
CANONICAL_WEIGHTS = ROOT / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
EXPECTED_V3_SHA256 = "71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2"
CANONICAL_CAMPAIGN_CONFIGS = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1.yaml",
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_release_smoke.yaml",
    ROOT / "configs/benchmarks/paper_experiment_matrix_v1_extended_seeds_s10.yaml",
)


def _load_weight_set_registry() -> dict[str, object]:
    """Load the camera-ready SNQI weight-set registry."""
    return yaml.safe_load(WEIGHT_SET_REGISTRY.read_text(encoding="utf-8"))


def test_camera_ready_v3_canonical_weight_vector_hash_is_frozen() -> None:
    """The author-ruling canonical vector remains the checked-in v3 file."""
    digest = hashlib.sha256(CANONICAL_WEIGHTS.read_bytes()).hexdigest()
    weights = validate_weights_mapping(json.loads(CANONICAL_WEIGHTS.read_text(encoding="utf-8")))

    assert digest == EXPECTED_V3_SHA256
    assert set(weights) == set(WEIGHT_NAMES)


def test_camera_ready_weight_registry_labels_v1_v2_as_sensitivity_probes() -> None:
    """Historical camera-ready vectors stay loadable only as labeled probes."""
    registry = _load_weight_set_registry()
    assert registry["canonical_set"] == "camera_ready_v3"

    weight_sets = registry["weight_sets"]
    for set_id in ("camera_ready_v1", "camera_ready_v2"):
        entry = weight_sets[set_id]
        assert entry["role"] == "sensitivity_probe"
        assert entry["label"].endswith("_sensitivity_probe")
        weights_path = ROOT / entry["path"]
        weights = validate_weights_mapping(json.loads(weights_path.read_text(encoding="utf-8")))
        assert set(weights) == set(WEIGHT_NAMES)

    canonical = weight_sets["camera_ready_v3"]
    assert canonical["role"] == "canonical"
    assert canonical["path"] == "configs/benchmarks/snqi_weights_camera_ready_v3.json"
    assert canonical["sha256"] == EXPECTED_V3_SHA256


@pytest.mark.parametrize("config_path", CANONICAL_CAMPAIGN_CONFIGS)
def test_canonical_camera_ready_configs_pin_v3_and_enforce_snqi_contract(
    config_path: Path,
) -> None:
    """Canonical camera-ready benchmark gates use v3 weights and hard SNQI enforcement."""
    cfg = load_campaign_config(config_path)

    assert cfg.snqi_weights_path == CANONICAL_WEIGHTS
    assert cfg.snqi_contract.enforcement == "enforce"


def test_snqi_contract_enforce_mode_loads_as_hard_enforcement(tmp_path: Path) -> None:
    """Config loader accepts the author-ruling enforce mode."""
    scenario_path = tmp_path / "matrix.yaml"
    scenario_path.write_text(
        "- name: smoke\n  map_file: maps/svg_maps/classic_crossing.svg\n  seeds: [111]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: issue_4247_enforce",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "snqi_contract:",
                "  enabled: true",
                "  enforcement: enforce",
                "planners:",
                "  - key: goal",
                "    algo: goal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_campaign_config(config_path)

    assert cfg.snqi_contract.enforcement == "enforce"
