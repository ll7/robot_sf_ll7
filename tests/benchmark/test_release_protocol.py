"""Tests for benchmark release protocol helpers."""

from __future__ import annotations

import copy
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

STRESS_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml"
)


def _stress_manifest_payload() -> dict[str, object]:
    """Load an isolated mutable copy of the canonical diagnostic stress manifest."""
    payload = yaml.safe_load(STRESS_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return copy.deepcopy(payload)


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


def test_release_campaign_config_runs_single_worker() -> None:
    """The canonical camera-ready release config should stay sequential, and the manifest must remain valid."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml")
    )
    cfg = release_protocol.load_campaign_config(manifest.canonical_campaign_config_path)

    assert cfg.workers == 1

    # Validate manifest still passes, including pinned campaign_config_sha256 check.
    validation = validate_release_manifest(manifest)
    assert validation["status"] == "valid", (
        f"Canonical release manifest failed validation after config change: "
        f"{validation.get('problems', [])}"
    )


def test_diagnostic_trace_pin_validates_against_non_paper_config() -> None:
    """The #7086 trace pin validates without admitting a paper-facing release."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/issue_7086_trace_dossier_diagnostic_v0_1.yaml")
    )

    validation = validate_release_manifest(manifest)

    assert manifest.maturity == "diagnostic"
    assert validation["status"] == "valid"
    assert validation["problem_count"] == 0


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


@pytest.mark.parametrize(
    ("case", "pattern"),
    (
        ("contract_type", "stress_smoke_contract must be a mapping"),
        ("schema", "schema_version"),
        ("review_base", "review_base_commit"),
        ("source_policy", "source_commit_policy"),
        ("episode_cells", "expected_episode_cells"),
        ("dt_type", "expected_dt"),
        ("dt_nonfinite", "expected_dt"),
        ("kinematics", "expected_kinematics"),
        ("hybrid_arms_empty", "required_hybrid_arms"),
        ("hybrid_arms_duplicate", "required_hybrid_arms"),
        ("scenario_type", "scenario must be a mapping"),
        ("seed_policy_type", "seed_policy must be a mapping"),
        ("suite_hash", "suite_policy_sha256"),
        ("seed_hash", "seed_sets_sha256"),
        ("route_hash", "route_certification_sha256"),
        ("pinned_assets_type", "pinned_assets must be a mapping"),
        ("nested_seed_hash", "pinned_assets.seed_sets_sha256"),
        ("scenario_sources_empty", "scenario_sources must be a non-empty list"),
        ("scenario_source_type", r"scenario_sources\[0\] must be a mapping"),
        ("scenario_source_duplicate", "duplicate asset paths"),
        ("scenario_source_hash", r"scenario_sources\[0\].sha256"),
        ("hybrid_planner_missing", r"hybrid_configs\[0\].planner_key"),
        ("hybrid_planner_duplicate", "duplicate planner keys"),
        ("branch_witnesses_empty", "branch_witnesses must be a non-empty list"),
        ("branch_witness_key", r"branch_witnesses\[0\].branch_key"),
        ("branch_witness_field", "does not match branch_key"),
        ("branch_witness_config", "must match a pinned hybrid config"),
        ("branch_witness_hash", "does not match its pinned hybrid config"),
        ("branch_witness_kind", "kind has unsupported value"),
    ),
)
def test_diagnostic_stress_contract_rejects_malformed_pins(  # noqa: C901, PLR0912, PLR0915
    case: str, pattern: str
) -> None:
    """Every fail-closed stress-contract parser branch has a malformed fixture."""
    payload = _stress_manifest_payload()
    contract = payload["stress_smoke_contract"]
    assert isinstance(contract, dict)
    scenario = payload["scenario"]
    assert isinstance(scenario, dict)
    seed_policy = payload["seed_policy"]
    assert isinstance(seed_policy, dict)

    if case == "contract_type":
        payload["stress_smoke_contract"] = []
    elif case == "schema":
        contract["schema_version"] = "wrong"
    elif case == "review_base":
        contract["review_base_commit"] = "not-a-sha"
    elif case == "source_policy":
        contract["source_commit_policy"] = "floating"
    elif case == "episode_cells":
        contract["expected_episode_cells"] = True
    elif case == "dt_type":
        contract["expected_dt"] = "0.1"
    elif case == "dt_nonfinite":
        contract["expected_dt"] = float("nan")
    elif case == "kinematics":
        contract["expected_kinematics"] = "holonomic"
    elif case == "hybrid_arms_empty":
        contract["required_hybrid_arms"] = []
    elif case == "hybrid_arms_duplicate":
        arm = contract["required_hybrid_arms"][0]
        contract["required_hybrid_arms"] = [arm, arm]
    elif case == "scenario_type":
        payload["scenario"] = []
    elif case == "seed_policy_type":
        payload["seed_policy"] = []
    elif case == "suite_hash":
        scenario["suite_policy_sha256"] = "bad"
    elif case == "seed_hash":
        seed_policy["seed_sets_sha256"] = "bad"
    elif case == "route_hash":
        scenario["route_certification_sha256"] = "bad"
    elif case == "pinned_assets_type":
        contract["pinned_assets"] = []
    elif case == "nested_seed_hash":
        pins = contract["pinned_assets"]
        assert isinstance(pins, dict)
        pins["seed_sets"] = {"path": pins["seed_sets_path"], "sha256": "bad"}
    elif case == "scenario_sources_empty":
        contract["scenario_sources"] = []
    elif case == "scenario_source_type":
        contract["scenario_sources"] = ["not-a-mapping"]
    elif case == "scenario_source_duplicate":
        pins = contract["scenario_sources"]
        assert isinstance(pins, list)
        pins.append(copy.deepcopy(pins[0]))
    elif case == "scenario_source_hash":
        contract["scenario_sources"][0]["sha256"] = "bad"
    elif case == "hybrid_planner_missing":
        contract["hybrid_configs"][0].pop("planner_key")
    elif case == "branch_witnesses_empty":
        contract["branch_witnesses"] = []
    elif case == "branch_witness_key":
        contract["branch_witnesses"][0]["branch_key"] = "not-a-branch-key"
    elif case == "branch_witness_field":
        contract["branch_witnesses"][0]["algorithm"] = "hybrid_rule_local_planner"
    elif case == "branch_witness_config":
        contract["branch_witnesses"][0]["config_path"] = contract["scenario_sources"][0]["path"]
    elif case == "branch_witness_hash":
        contract["branch_witnesses"][0]["config_sha256"] = "0" * 64
    elif case == "branch_witness_kind":
        contract["branch_witnesses"][0]["kind"] = "unsupported"
    else:
        pins = contract["hybrid_configs"]
        assert isinstance(pins, list)
        duplicate = copy.deepcopy(pins[1])
        duplicate["planner_key"] = pins[0]["planner_key"]
        pins[1] = duplicate

    with pytest.raises(ValueError, match=pattern):
        release_protocol._load_stress_smoke_contract(STRESS_MANIFEST, payload)


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


def test_load_release_manifest_rejects_non_file_required_path(tmp_path: Path) -> None:
    """Required manifest paths should fail fast when they point at directories."""
    template_path = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
    )
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    manifest = load_release_manifest(template_path)

    payload["canonical_campaign_config"] = str(manifest.canonical_campaign_config_path)
    payload["scenario"]["matrix_path"] = str(manifest.scenario_matrix_path)
    payload["citation_path"] = str(tmp_path / "citation_dir")
    payload["release_checklist_path"] = str(manifest.release_checklist_path)
    if manifest.snqi_weights_path is not None:
        payload["metrics"]["snqi_weights_path"] = str(manifest.snqi_weights_path)
    if manifest.snqi_baseline_path is not None:
        payload["metrics"]["snqi_baseline_path"] = str(manifest.snqi_baseline_path)

    directory_target = tmp_path / "citation_dir"
    directory_target.mkdir(parents=True, exist_ok=True)

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="citation_path must be a file path"):
        load_release_manifest(manifest_path)


def test_manifest_side_inputs_are_repository_contained_and_not_symlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every manifest-side input must be a real file in the repository scope."""
    repo = tmp_path / "repo"
    manifest_dir = repo / "configs" / "releases"
    manifest_dir.mkdir(parents=True)
    monkeypatch.setattr(release_protocol, "get_repository_root", lambda: repo)
    manifest_path = manifest_dir / "manifest.yaml"
    safe_path = manifest_dir / "safe.yaml"
    safe_path.write_text("safe\n", encoding="utf-8")

    assert release_protocol._resolve_required_file(manifest_path, "safe.yaml", "input") == safe_path

    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="repository"):
        release_protocol._resolve_required_file(manifest_path, "../../../outside.yaml", "input")
    with pytest.raises(ValueError, match="repository"):
        release_protocol._resolve_required_file(manifest_path, str(outside), "input")

    escaped = repo / "inputs" / "escaped.yaml"
    escaped.parent.mkdir()
    escaped.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        release_protocol._resolve_required_file(manifest_path, "../../inputs/escaped.yaml", "input")


def test_manifest_required_artifact_paths_are_campaign_relative() -> None:
    """Manifest artifact declarations cannot escape the runtime campaign root."""
    for value in ("/tmp/campaign/report.json", "../report.json", "reports/../report.json"):
        with pytest.raises(ValueError, match="campaign-relative"):
            release_protocol._load_manifest_artifacts_section(
                {"artifacts": {"required_paths": [value]}}
            )


def test_load_release_manifest_rejects_empty_required_artifact_path(tmp_path: Path) -> None:
    """Required artifact lists should reject empty or whitespace-only entries."""
    template_path = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
    )
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    manifest = load_release_manifest(template_path)

    payload["canonical_campaign_config"] = str(manifest.canonical_campaign_config_path)
    payload["scenario"]["matrix_path"] = str(manifest.scenario_matrix_path)
    payload["citation_path"] = str(manifest.citation_path)
    payload["release_checklist_path"] = str(manifest.release_checklist_path)
    if manifest.snqi_weights_path is not None:
        payload["metrics"]["snqi_weights_path"] = str(manifest.snqi_weights_path)
    if manifest.snqi_baseline_path is not None:
        payload["metrics"]["snqi_baseline_path"] = str(manifest.snqi_baseline_path)

    payload["artifacts"]["required_paths"] = ["reports/campaign_summary.json", "   "]

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts.required_paths must not contain empty values"):
        load_release_manifest(manifest_path)


def test_v02_release_manifest_pins_full_s30_publication_contract(tmp_path: Path) -> None:
    """v0.2 binds base, matrix, suite/certification, seeds, SNQI policy, and fresh DOI."""
    source_path = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
    )
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    loaded = load_release_manifest(source_path)
    payload.update(
        {
            "schema_version": "benchmark-release-manifest.v0.2",
            "latest_main_base_commit": "a" * 40,
            "matrix": {"expected_episode_cells": 20160, "horizon_steps": 600},
            "publication": {
                "channel": "direct_zenodo_benchmark_dataset",
                "concept_doi": "10.5281/zenodo.99999990",
                "version_doi": "10.5281/zenodo.99999991",
            },
        }
    )
    payload["canonical_campaign_config"] = str(loaded.canonical_campaign_config_path)
    payload["campaign_config_sha256"] = release_protocol._sha256_file(
        loaded.canonical_campaign_config_path
    )
    payload["scenario"]["matrix_path"] = str(loaded.scenario_matrix_path)
    suite_policy = Path(
        "configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1_suite_policy.yaml"
    ).resolve()
    route_certification = Path(
        "configs/benchmarks/route_clearance_certifications_v1.yaml"
    ).resolve()
    payload["scenario"].update(
        {
            "suite_policy_path": str(suite_policy),
            "suite_policy_sha256": release_protocol._sha256_file(suite_policy),
            "route_certification_path": str(route_certification),
            "route_certification_sha256": release_protocol._sha256_file(route_certification),
        }
    )
    payload["seed_policy"].update(
        {
            "seed_sets_path": str(Path("configs/benchmarks/seed_sets_v1.yaml").resolve()),
            "seed_sets_sha256": release_protocol._sha256_file(
                Path("configs/benchmarks/seed_sets_v1.yaml")
            ),
            "resolved_seeds": list(range(111, 141)),
        }
    )
    payload["metrics"]["snqi_weights_path"] = str(loaded.snqi_weights_path)
    payload["metrics"]["snqi_baseline_path"] = str(loaded.snqi_baseline_path)
    payload["metrics"]["snqi_claim_policy"] = "advisory_no_ranking"
    payload["provenance"]["doi"] = "10.5281/zenodo.99999991"
    payload["citation_path"] = str(loaded.citation_path)
    payload["release_checklist_path"] = str(loaded.release_checklist_path)
    manifest_path = tmp_path / "release-v0.2.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    manifest = load_release_manifest(manifest_path)
    result = validate_release_manifest(manifest)
    assert result["status"] == "valid"
    assert manifest.expected_episode_cells == 20160
    assert manifest.expected_horizon_steps == 600
    assert manifest.snqi_claim_policy == "advisory_no_ranking"
    assert manifest.concept_doi != manifest.version_doi


def test_validate_release_manifest_reports_mismatches() -> None:
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


def test_validate_release_manifest_reports_optional_asset_presence_mismatch() -> None:
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
    assert args.resume_receipt is None
    assert args.resume_receipt_max_age_hours == 24.0
