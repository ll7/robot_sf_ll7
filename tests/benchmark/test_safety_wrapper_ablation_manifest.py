"""Tests for the dry-run safety-wrapper factorial-ablation manifest (issue #3501)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    SAFETY_WRAPPER_ABLATION_SCHEMA,
    WRAPPER_OFF_ARM,
    WRAPPER_ON_ARM,
    ManifestOptions,
    build_safety_wrapper_ablation_manifest,
    check_factorial_ablation,
    load_safety_wrapper_ablation_config,
    write_safety_wrapper_ablation_manifest,
)
from robot_sf.robot.safety_wrapper import SAFETY_WRAPPER_SCHEMA

_CONFIG_PATH = "configs/research/safety_wrapper_ablation_v1.yaml"

if TYPE_CHECKING:
    from pathlib import Path


def _repo_config() -> dict[str, object]:
    return load_safety_wrapper_ablation_config(_CONFIG_PATH)


def _options() -> ManifestOptions:
    return ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234")


def test_manifest_factorizes_planners_over_wrapper_off_on() -> None:
    """Every planner gets exactly the off and on arm; cells carry the shared paired seeds."""
    config = _repo_config()
    manifest = build_safety_wrapper_ablation_manifest(config, options=_options())

    assert manifest["schema_version"] == SAFETY_WRAPPER_ABLATION_SCHEMA
    assert manifest["safety_wrapper_schema"] == SAFETY_WRAPPER_SCHEMA
    assert manifest["status"] == "manifest_dry_run_only"
    assert manifest["evidence_status"] == "not_benchmark_evidence"

    planners = manifest["planner_groups"]
    assert manifest["cell_count"] == len(planners) * 2
    # Each planner appears once per arm.
    for planner in planners:
        arms_for_planner = sorted(
            cell["wrapper_arm"] for cell in manifest["cells"] if cell["planner"] == planner
        )
        assert arms_for_planner == sorted([WRAPPER_OFF_ARM, WRAPPER_ON_ARM])
    # Paired seeds are applied identically to every cell.
    for cell in manifest["cells"]:
        assert cell["seeds"] == manifest["seeds"]

    check = manifest["factorial_check"]
    assert check["complete"] is True
    assert check["arms_are_off_on"] is True
    assert check["off_on_enabled"] is True
    assert check["seeds_paired_across_arms"] is True
    assert check["expected_cell_count"] == len(planners) * 2


def test_manifest_echoes_predeclared_wrapper_thresholds_as_provenance() -> None:
    """The on arm echoes the fixed, predeclared SafetyWrapperConfig thresholds."""
    manifest = build_safety_wrapper_ablation_manifest(_repo_config(), options=_options())

    arms = {arm["key"]: arm for arm in manifest["wrapper_arms"]}
    off_arm = arms[WRAPPER_OFF_ARM]
    on_arm = arms[WRAPPER_ON_ARM]

    assert off_arm["enabled"] is False
    assert off_arm["baseline"] is True
    assert off_arm["wrapper_config"] is None

    assert on_arm["enabled"] is True
    assert on_arm["baseline"] is False
    assert on_arm["thresholds_source"] == "predeclared_fixed_no_per_planner_tuning"
    assert on_arm["wrapper_config"] == {
        "pedestrian_caution_radius_m": 2.0,
        "capped_speed_m_s": 0.5,
        "ttc_veto_threshold_s": 1.0,
        "clearance_veto_m": 0.3,
    }
    assert on_arm["runtime_binding_status"] == "unresolved_runtime_binding"
    assert manifest["event_ledger_target"] == 3482


def test_manifest_claim_boundary_prevents_benchmark_or_paper_claims() -> None:
    """The dry-run manifest must not be worded as evidence."""
    manifest = build_safety_wrapper_ablation_manifest(_repo_config(), options=_options())

    claim_boundary = manifest["claim_boundary"]
    assert "dry-run factorial-ablation manifest only" in claim_boundary
    assert "not benchmark evidence" in claim_boundary
    assert "not a mitigation-effectiveness result" in claim_boundary
    assert "not paper-facing evidence" in claim_boundary
    assert manifest["dry_run"] is True


def test_config_rejects_arm_that_breaks_off_on_factorization() -> None:
    """An enabled off arm breaks the wrapper off/on contrast and must be rejected."""
    config = _repo_config()
    arms = {arm["key"]: arm for arm in config["wrapper_arms"]}
    arms[WRAPPER_OFF_ARM]["enabled"] = True

    with pytest.raises(ValueError, match="wrapper_off.*enabled: false"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_non_positive_wrapper_threshold() -> None:
    """On-arm thresholds must construct a real SafetyWrapperConfig (positive thresholds)."""
    config = _repo_config()
    arms = {arm["key"]: arm for arm in config["wrapper_arms"]}
    arms[WRAPPER_ON_ARM]["config"]["capped_speed_m_s"] = 0.0

    with pytest.raises(ValueError, match="capped_speed_m_s must be > 0"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_unpaired_seeds() -> None:
    """Duplicate seeds break the paired-seed contract across arms."""
    config = _repo_config()
    config["fixed_scope"]["seeds"] = [111, 111, 112]

    with pytest.raises(ValueError, match="seeds must be unique"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_config_rejects_non_mapping_wrapper_arm() -> None:
    """A non-mapping entry in wrapper_arms is rejected cleanly (no raw TypeError)."""
    config = _repo_config()
    config["wrapper_arms"].append("not-a-mapping")

    with pytest.raises(ValueError, match="each entry in wrapper_arms must be a mapping"):
        build_safety_wrapper_ablation_manifest(config, options=_options())


def test_check_factorial_ablation_flags_missing_arm() -> None:
    """The checker reports incomplete factorization when an arm is missing."""
    only_off = [{"key": WRAPPER_OFF_ARM, "enabled": False, "baseline": True}]
    report = check_factorial_ablation(["orca"], only_off, [111, 112])

    assert report["complete"] is False
    assert report["arms_are_off_on"] is False


def test_manifest_json_output_is_deterministic(tmp_path: Path) -> None:
    """Repeated builds and writes with the same inputs produce byte-identical JSON."""
    config = _repo_config()
    options = _options()

    first = build_safety_wrapper_ablation_manifest(config, options=options)
    second = build_safety_wrapper_ablation_manifest(config, options=options)

    assert first == second
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )

    first_path = write_safety_wrapper_ablation_manifest(first, tmp_path / "first")
    second_path = write_safety_wrapper_ablation_manifest(second, tmp_path / "second")

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
