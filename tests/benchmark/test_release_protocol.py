"""Tests for benchmark release protocol helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark import release_protocol
from robot_sf.benchmark.release_protocol import (
    BENCHMARK_PROTOCOL_VERSION,
    build_release_provenance,
    build_resolved_release_manifest,
    load_release_manifest,
    validate_release_manifest,
)


def test_smoke_release_manifest_validates_against_campaign_config() -> None:
    """The checked-in smoke release manifest should validate cleanly."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml")
    )

    validation = validate_release_manifest(manifest)
    resolved = build_resolved_release_manifest(manifest)

    assert validation["status"] == "valid"
    assert validation["problem_count"] == 0
    assert resolved["benchmark_protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    assert resolved["canonical_campaign_name"] == "paper_experiment_matrix_v1_release_smoke"
    assert resolved["planners"]["keys"][0] == "prediction_planner"


def test_load_release_manifest_rejects_invalid_protocol_version(tmp_path: Path) -> None:
    """Protocol versions must be pinned to the supported benchmark protocol."""
    payload = {
        "schema_version": "benchmark-release-manifest.v0.1",
        "benchmark_protocol_version": "1.0.0",
        "release_id": "bad",
        "release_tag": "bad",
        "maturity": "pre-1.0",
        "canonical_campaign_config": "campaign.yaml",
        "campaign_config_sha256": "abc",
        "expected_paper_profile_version": "paper-matrix-v1",
        "scenario": {"matrix_path": "scenario.yaml", "matrix_sha256": "def"},
        "seed_policy": {"mode": "fixed-list", "seed_set": None, "seeds": [111]},
        "metrics": {},
        "planners": {"keys": ["goal"], "groups": {"goal": "core"}},
        "kinematics": {"matrix": ["differential_drive"]},
        "artifacts": {"required_paths": ["reports/campaign_summary.json"]},
        "provenance": {
            "repository_url": "https://github.com/ll7/robot_sf_ll7",
            "doi": "10.5281/zenodo.<record-id>",
        },
        "citation_path": "CITATION.cff",
        "release_checklist_path": "RELEASE.md",
    }
    for filename in (
        "campaign.yaml",
        "scenario.yaml",
        "CITATION.cff",
        "RELEASE.md",
    ):
        (tmp_path / filename).write_text("placeholder\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark_protocol_version"):
        load_release_manifest(manifest_path)


def test_load_release_manifest_rejects_missing_file() -> None:
    """Missing manifests should fail with a path-specific error."""
    with pytest.raises(FileNotFoundError, match="Benchmark release manifest not found"):
        load_release_manifest(Path("configs/benchmarks/releases/does_not_exist.yaml"))


def test_load_release_manifest_rejects_json_non_mapping(tmp_path: Path) -> None:
    """JSON manifests must deserialize to mappings."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('["not", "a", "mapping"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping payload"):
        load_release_manifest(manifest_path)


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("schema_version", "wrong", "schema_version"),
        ("release_id", "", "release_id"),
        ("release_tag", "", "release_tag"),
        ("scenario", [], "scenario must be a mapping"),
        ("seed_policy", [], "seed_policy must be a mapping"),
        ("planners", [], "planners must be a mapping"),
        ("kinematics", [], "kinematics must be a mapping"),
        ("artifacts", [], "artifacts must be a mapping"),
        ("provenance", [], "provenance must be a mapping"),
    ],
)
def test_load_release_manifest_rejects_invalid_top_level_fields(
    tmp_path: Path,
    field: str,
    value: object,
    pattern: str,
) -> None:
    """Manifest loader should reject malformed required fields."""
    payload = {
        "schema_version": "benchmark-release-manifest.v0.1",
        "benchmark_protocol_version": "0.1.0",
        "release_id": "rid",
        "release_tag": "tag",
        "maturity": "pre-1.0",
        "canonical_campaign_config": "campaign.yaml",
        "campaign_config_sha256": "abc",
        "expected_paper_profile_version": "paper-matrix-v1",
        "scenario": {"matrix_path": "scenario.yaml", "matrix_sha256": "def"},
        "seed_policy": {"mode": "fixed-list", "seed_set": None, "seeds": [111]},
        "metrics": {},
        "planners": {"keys": ["goal"], "groups": {"goal": "core"}},
        "kinematics": {"matrix": ["differential_drive"]},
        "artifacts": {"required_paths": ["reports/campaign_summary.json"]},
        "provenance": {
            "repository_url": "https://github.com/ll7/robot_sf_ll7",
            "doi": "10.5281/zenodo.<record-id>",
        },
        "citation_path": "CITATION.cff",
        "release_checklist_path": "RELEASE.md",
    }
    payload[field] = value
    for filename in ("campaign.yaml", "scenario.yaml", "CITATION.cff", "RELEASE.md"):
        (tmp_path / filename).write_text("placeholder\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises((ValueError, FileNotFoundError), match=pattern):
        load_release_manifest(manifest_path)


def test_load_release_manifest_rejects_missing_required_path_and_hash_fields(
    tmp_path: Path,
) -> None:
    """Path-backed and hash-backed required fields should fail clearly when absent."""
    (tmp_path / "campaign.yaml").write_text("name: t\n", encoding="utf-8")
    (tmp_path / "scenario.yaml").write_text("- name: t\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\n", encoding="utf-8")
    (tmp_path / "RELEASE.md").write_text("# release\n", encoding="utf-8")
    payload = {
        "schema_version": "benchmark-release-manifest.v0.1",
        "benchmark_protocol_version": "0.1.0",
        "release_id": "rid",
        "release_tag": "tag",
        "maturity": "pre-1.0",
        "canonical_campaign_config": "campaign.yaml",
        "campaign_config_sha256": "",
        "scenario": {"matrix_path": "scenario.yaml", "matrix_sha256": ""},
        "seed_policy": {"mode": "fixed-list", "seed_set": None, "seeds": [111]},
        "metrics": {
            "snqi_weights_path": "missing.json",
            "snqi_weights_sha256": "",
        },
        "planners": {"keys": ["goal"], "groups": {"goal": "core"}},
        "kinematics": {"matrix": ["differential_drive"]},
        "artifacts": {"required_paths": ["reports/campaign_summary.json"]},
        "provenance": {
            "repository_url": "",
            "doi": "",
        },
        "citation_path": "CITATION.cff",
        "release_checklist_path": "RELEASE.md",
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario.matrix_sha256"):
        load_release_manifest(manifest_path)

    payload["scenario"]["matrix_sha256"] = "def"
    payload["campaign_config_sha256"] = "abc"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metrics.snqi_weights_path"):
        load_release_manifest(manifest_path)

    payload["metrics"] = {}
    payload["campaign_config_sha256"] = ""
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="campaign_config_sha256"):
        load_release_manifest(manifest_path)


def test_validate_release_manifest_reports_mismatches(monkeypatch) -> None:
    """Validation should surface config, seed, planner, and asset drift explicitly."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml")
    )
    cfg = release_protocol.load_campaign_config(manifest.canonical_campaign_config_path)
    drifted_cfg = cfg.__class__(
        **{
            **cfg.__dict__,
            "paper_facing": False,
            "paper_profile_version": "wrong-profile",
            "paper_interpretation_profile": "wrong-interpretation",
            "kinematics_matrix": ("holonomic",),
        }
    )
    drifted_manifest = release_protocol.BenchmarkReleaseManifest(
        **{
            **manifest.__dict__,
            "campaign_config_sha256": "wrong",
            "scenario_matrix_sha256": "wrong",
            "snqi_weights_sha256": "wrong" if manifest.snqi_weights_sha256 else None,
            "snqi_baseline_sha256": "wrong" if manifest.snqi_baseline_sha256 else None,
            "seed_policy": {"mode": "fixed-list", "seed_set": None, "seeds": [999]},
            "planner_keys": ("goal",),
            "planner_groups": {"goal": "experimental"},
            "expected_paper_profile_version": "other-profile",
            "expected_paper_interpretation_profile": "other-interpretation",
            "expected_kinematics_matrix": ("holonomic",),
            "expected_holonomic_command_mode": "vw",
        }
    )

    report = validate_release_manifest(drifted_manifest, campaign_config=drifted_cfg)

    assert report["status"] == "invalid"
    assert "campaign_config_sha256 does not match canonical_campaign_config" in report["problems"]
    assert "scenario.matrix_sha256 does not match scenario.matrix_path" in report["problems"]
    assert "metrics.snqi_weights_sha256 does not match snqi_weights_path" in report["problems"]
    assert "metrics.snqi_baseline_sha256 does not match snqi_baseline_path" in report["problems"]
    assert "canonical campaign config must be paper_facing: true" in report["problems"]
    assert "expected_paper_profile_version does not match campaign config" in report["problems"]
    assert (
        "expected_paper_interpretation_profile does not match campaign config" in report["problems"]
    )
    assert "seed_policy does not match campaign config" in report["problems"]
    assert "planners.keys does not match enabled planners in campaign config" in report["problems"]
    assert "planners.groups does not match campaign config" in report["problems"]


def test_validate_release_manifest_reports_optional_asset_presence_mismatch(monkeypatch) -> None:
    """Validation should flag when manifest asset presence diverges from the campaign config."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml")
    )
    cfg = release_protocol.load_campaign_config(manifest.canonical_campaign_config_path)
    manifest_without_assets = release_protocol.BenchmarkReleaseManifest(
        **{
            **manifest.__dict__,
            "snqi_weights_path": None,
            "snqi_weights_sha256": None,
            "snqi_baseline_path": None,
            "snqi_baseline_sha256": None,
        }
    )

    report = validate_release_manifest(manifest_without_assets, campaign_config=cfg)

    assert report["status"] == "invalid"
    assert "metrics.snqi_weights_path presence does not match campaign config" in report["problems"]
    assert (
        "metrics.snqi_baseline_path presence does not match campaign config" in report["problems"]
    )


def test_build_release_provenance_and_helpers_cover_repo_relative_fallback(tmp_path: Path) -> None:
    """Release provenance should include stable hashes and tolerate paths outside the repo."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml")
    )
    outside_path = tmp_path / "elsewhere"
    outside_path.mkdir(parents=True, exist_ok=True)
    (outside_path / "payload.json").write_text("{}", encoding="utf-8")
    (tmp_path / "mapping.yaml").write_text("key: value\n", encoding="utf-8")

    repo_relative = release_protocol._repo_relative(outside_path / "payload.json")
    payload = release_protocol._load_mapping(tmp_path / "mapping.yaml")

    provenance = build_release_provenance(
        manifest,
        campaign_root=Path("output/benchmarks/camera_ready/example"),
        invoked_command="uv run python scripts/tools/run_benchmark_release.py ...",
    )
    args = release_protocol.parse_release_args(
        [
            "--manifest",
            "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml",
        ]
    )

    assert repo_relative == str((outside_path / "payload.json").resolve())
    assert payload == {"key": "value"}
    assert provenance["benchmark_protocol_version"] == "0.1.0"
    assert provenance["manifest_sha256"]
    assert args.mode == "run"
    assert args.manifest.name == "paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
