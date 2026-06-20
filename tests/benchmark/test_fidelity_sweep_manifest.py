"""Tests for dry-run fidelity sweep manifests for issue #3207."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config
from robot_sf.benchmark.fidelity_sweep_manifest import (
    FIDELITY_SWEEP_MANIFEST_SCHEMA,
    ManifestOptions,
    build_fidelity_sweep_manifest,
    write_fidelity_sweep_manifest,
)

_CONFIG_PATH = "configs/research/fidelity_sensitivity_v1.yaml"

if TYPE_CHECKING:
    from pathlib import Path


def _repo_config() -> dict[str, object]:
    return load_fidelity_sensitivity_config(_CONFIG_PATH)


def test_manifest_uses_axes_surface_and_preserves_payloads() -> None:
    """The manifest contract is sourced from canonical `axes`, not `fidelity_axes`."""
    config = _repo_config()

    manifest = build_fidelity_sweep_manifest(
        config,
        options=ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234"),
    )

    assert manifest["schema_version"] == FIDELITY_SWEEP_MANIFEST_SCHEMA
    assert manifest["status"] == "manifest_dry_run_only"
    assert manifest["evidence_status"] == "not_benchmark_evidence"
    assert manifest["fixed_scope"] == config["fixed_scope"]
    assert manifest["seeds"] == [111, 112, 113]
    assert manifest["planner_groups"] == ["orca", "default_social_force", "hybrid_rule_v0_minimal"]
    assert manifest["config_surface_relationship"]["canonical_surface"] == "axes"
    assert manifest["config_surface_relationship"]["secondary_surface"] == "fidelity_axes"
    assert manifest["config_surface_relationship"]["manifest_source"] == "axes"
    assert [axis["key"] for axis in manifest["axes"]] == [axis["key"] for axis in config["axes"]]

    timestep_axis = manifest["axes"][0]
    assert timestep_axis["baseline_variant"] == "dt_0_10_nominal"
    assert timestep_axis["variants"][0]["patch"] == {"dt": 0.05}
    assert timestep_axis["variants"][0]["observation_noise"] is None
    assert timestep_axis["variants"][0]["runtime_binding_status"] == ("unresolved_runtime_binding")

    noise_axis = next(axis for axis in manifest["axes"] if axis["key"] == "observation_noise")
    assert noise_axis["baseline_variant"] == "none_nominal"
    assert noise_axis["variants"][1]["patch"] is None
    assert noise_axis["variants"][1]["observation_noise"]["profile"] == "pose_heading_low"
    assert noise_axis["variants"][1]["runtime_binding_status"] == ("unresolved_runtime_binding")


def test_manifest_rejects_axes_without_exactly_one_baseline() -> None:
    """Each canonical axis must mark exactly one baseline variant."""
    config = _repo_config()
    config["axes"][0]["variants"][0]["baseline"] = True

    with pytest.raises(ValueError, match="exactly one baseline variant"):
        build_fidelity_sweep_manifest(
            config,
            options=ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234"),
        )


def test_manifest_claim_boundary_prevents_benchmark_or_paper_claims() -> None:
    """The dry-run manifest must not be worded as evidence."""
    manifest = build_fidelity_sweep_manifest(
        _repo_config(),
        options=ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234"),
    )

    claim_boundary = manifest["claim_boundary"]
    assert "dry-run manifest only" in claim_boundary
    assert "not benchmark evidence" in claim_boundary
    assert "not simulator-realism evidence" in claim_boundary
    assert "not sim-to-real evidence" in claim_boundary
    assert "not paper-facing evidence" in claim_boundary
    assert manifest["dry_run"] is True


def test_manifest_json_output_is_deterministic(tmp_path: Path) -> None:
    """Repeated builds and writes with the same inputs should produce byte-identical JSON."""
    config = _repo_config()
    options = ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234")

    first = build_fidelity_sweep_manifest(config, options=options)
    second = build_fidelity_sweep_manifest(config, options=options)

    assert first == second
    assert json.dumps(first, indent=2, sort_keys=True) == json.dumps(
        second, indent=2, sort_keys=True
    )

    first_path = write_fidelity_sweep_manifest(first, tmp_path / "first")
    second_path = write_fidelity_sweep_manifest(second, tmp_path / "second")

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_manifest_reports_fidelity_axes_as_secondary_risk_not_source() -> None:
    """Conflicting `fidelity_axes` names must not change the manifest axis source."""
    config = _repo_config()
    config["fidelity_axes"] = {
        "unrelated_runtime_surface": {
            "description": "must not drive the dry-run manifest",
            "nominal": 1.0,
            "values": [1.0, 2.0],
        }
    }

    manifest = build_fidelity_sweep_manifest(
        config,
        options=ManifestOptions(config_path=_CONFIG_PATH, git_head="abc1234"),
    )

    assert "unrelated_runtime_surface" not in {axis["key"] for axis in manifest["axes"]}
    assert manifest["config_surface_relationship"]["manifest_source"] == "axes"
    assert manifest["config_surface_relationship"]["secondary_axis_keys"] == [
        "unrelated_runtime_surface"
    ]
    assert (
        manifest["config_surface_relationship"]["relationship"]
        == "axes is canonical for this dry-run manifest; fidelity_axes is retained as a "
        "secondary analysis-contract surface and is not used to enumerate variants."
    )
