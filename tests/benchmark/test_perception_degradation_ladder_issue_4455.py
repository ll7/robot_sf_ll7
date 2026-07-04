"""Tests for issue #4455 perception-degradation ladder preregistration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)

_SCRIPT_PATH = Path("scripts/benchmark/build_perception_degradation_ladder_issue_4455.py")
_SPEC = importlib.util.spec_from_file_location("issue_4455_ladder_builder", _SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

DEFAULT_MANIFEST = _MODULE.DEFAULT_MANIFEST
build_ladder_configs = _MODULE.build_ladder_configs

PROFILE_CATALOG = Path("configs/benchmarks/perception_degradation_profiles_v1.yaml")
PREREGISTRATION_WRAPPER = Path(
    "configs/benchmarks/issue_4455_perception_degradation_ladder_preregistration.yaml"
)


def test_issue_4455_ladder_generates_loadable_profile_configs(tmp_path: Path) -> None:
    """The preregistered ladder expands into camera-ready smoke configs."""

    out_dir = tmp_path / "ladder"

    generated = build_ladder_configs(DEFAULT_MANIFEST, out_dir)

    assert len(generated) == 5
    loaded = [load_campaign_config(path) for path in generated]
    profiles = {cfg.observation_noise["profile"]: cfg for cfg in loaded if cfg.observation_noise}
    assert "none" in profiles
    assert profiles["fixed_detection_delay_1"].observation_noise["observation_delay_steps"] == 1
    assert (
        profiles["occlusion_range_4m"].observation_noise["pedestrian_occlusion_max_range_m"] == 4.0
    )
    assert all(cfg.seed_policy.seeds == (111, 112) for cfg in loaded)
    assert all(len(cfg.planners) == 3 for cfg in loaded)


def test_issue_4455_ladder_validate_only_does_not_write_configs(tmp_path: Path) -> None:
    """Validate-only mode checks the manifest without producing local campaign artifacts."""

    generated = build_ladder_configs(DEFAULT_MANIFEST, tmp_path / "ladder", validate_only=True)

    assert generated == []
    assert not (tmp_path / "ladder").exists()


def test_issue_4455_profile_catalog_pins_unique_ordered_hashes() -> None:
    """The canonical profile catalog pins ordered unique normalized hashes."""

    catalog = yaml.safe_load(PROFILE_CATALOG.read_text(encoding="utf-8"))
    profiles = catalog["profiles"]
    keys = [profile["key"] for profile in profiles]

    assert keys == catalog["ordered_profile_keys"]
    assert len(keys) == len(set(keys))
    for profile in profiles:
        normalized = normalize_observation_noise_spec(profile["observation_noise"])
        assert profile["profile_hash"] == observation_noise_hash(normalized)


def test_issue_4455_preregistration_wrapper_points_to_detailed_manifest() -> None:
    """The maintainer-requested preregistration path remains discoverable."""

    wrapper = yaml.safe_load(PREREGISTRATION_WRAPPER.read_text(encoding="utf-8"))

    assert wrapper["include"] == "perception_degradation/issue_4455_ladder_v1.yaml"
    assert wrapper["submit_in_this_pr"] is False
