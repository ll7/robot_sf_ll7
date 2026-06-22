"""Tests for adversarial scenario manifest generation, validation, and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D, SearchSpaceConfig
from robot_sf.adversarial.scenario_manifest import (
    MANIFEST_SCHEMA_VERSION,
    NATURALISTIC_PRIOR_SCHEMA_VERSION,
    AdversarialScenarioManifest,
    GeneratorInfo,
    ManifestCategory,
    SourceLineage,
    ValidationRecord,
    build_manifest,
    compute_control_hash,
    evaluate_naturalistic_prior,
    generate_manifests,
    validate_candidate_manifest,
    validate_manifest_payload,
    write_manifest_yaml,
)
from robot_sf.benchmark.manifest_lineage import validate_lineage_contract
from scripts.tools.generate_adversarial_scenario_manifests import (
    _load_template_info,
)
from scripts.tools.generate_adversarial_scenario_manifests import (
    main as cli_main,
)

_LLM_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "issue_2529"


def _search_space(*, min_distance: float = 0.5) -> SearchSpaceConfig:
    """Build a fixed search-space fixture."""
    return SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 1.0},
                "start_y": {"min": 2.0, "max": 2.0},
                "goal_x": {"min": 5.0, "max": 5.0},
                "goal_y": {"min": 2.0, "max": 2.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 7, "max": 7},
            },
            "constraints": {"min_start_goal_distance_m": min_distance},
        }
    )


def _wide_search_space() -> SearchSpaceConfig:
    return SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 3.0},
                "start_y": {"min": 2.0, "max": 4.0},
                "goal_x": {"min": 7.0, "max": 9.0},
                "goal_y": {"min": 2.0, "max": 4.0},
                "spawn_time_s": {"min": 0.0, "max": 2.0},
                "pedestrian_speed_mps": {"min": 0.8, "max": 1.4},
                "pedestrian_delay_s": {"min": 0.0, "max": 2.0},
                "scenario_seed": {"min": 100, "max": 999},
            },
            "constraints": {"min_start_goal_distance_m": 2.0},
        }
    )


def _valid_candidate() -> CandidateSpec:
    return CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )


def _source() -> SourceLineage:
    return SourceLineage(
        scenario_template="crossing_ttc.yaml",
        search_space="crossing_ttc_space.yaml",
        map_id="classic_cross_trap",
        scenario_name="crossing_ttc_template",
        config_path="configs/scenarios/templates/crossing_ttc.yaml",
        search_space_path="configs/adversarial/crossing_ttc_space.yaml",
    )


def _generator() -> GeneratorInfo:
    return GeneratorInfo(
        family="test_family",
        generator_id="TestSampler",
        seed=77,
        candidate_index=3,
    )


def _valid_manifest_payload() -> dict[str, object]:
    return build_manifest(_valid_candidate(), source=_source(), generator=_generator()).to_dict()


def test_compute_control_hash_is_deterministic() -> None:
    c = _valid_candidate()
    h1 = compute_control_hash(c)
    h2 = compute_control_hash(c)
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_compute_control_hash_changes_with_fields() -> None:
    base = _valid_candidate()
    base_hash = compute_control_hash(base)
    modified = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.5,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    assert compute_control_hash(modified) != base_hash


def test_compute_control_hash_normalizes_signed_zero() -> None:
    positive_zero = CandidateSpec(
        start=Pose2D(0.0, 0.0),
        goal=Pose2D(5.0, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    negative_zero = CandidateSpec(
        start=Pose2D(-0.0, -0.0),
        goal=Pose2D(5.0, -0.0),
        spawn_time_s=-0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=-0.0,
        scenario_seed=7,
    )

    assert compute_control_hash(positive_zero) == compute_control_hash(negative_zero)


def test_validate_candidate_no_search_space() -> None:
    c = _valid_candidate()
    errors, warnings = validate_candidate_manifest(c, search_space=None)
    assert errors == []
    assert warnings == []


def test_validate_candidate_with_search_space() -> None:
    ss = _search_space()
    c = _valid_candidate()
    errors, warnings = validate_candidate_manifest(c, search_space=ss)
    assert errors == []
    assert warnings == []


def test_validate_candidate_out_of_bounds() -> None:
    ss = _search_space()
    c = CandidateSpec(
        start=Pose2D(10.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    errors, _warnings = validate_candidate_manifest(c, search_space=ss)
    assert any("start.x outside" in e for e in errors)


def test_validate_candidate_non_finite() -> None:
    c = CandidateSpec(
        start=Pose2D(float("nan"), 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    errors, _warnings = validate_candidate_manifest(c)
    assert any("start.x must be finite" in e for e in errors)


def test_validate_candidate_non_positive_speed() -> None:
    c = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=0.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    errors, _warnings = validate_candidate_manifest(c)
    assert any("pedestrian_speed_mps must be positive" in e for e in errors)


def test_validate_candidate_negative_timing() -> None:
    c = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=-1.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    errors, _warnings = validate_candidate_manifest(c)
    assert any("spawn_time_s must be non-negative" in e for e in errors)


def test_validate_candidate_too_short_route() -> None:
    ss = _search_space(min_distance=2.0)
    c = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(1.5, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    errors, _warnings = validate_candidate_manifest(c, search_space=ss)
    assert any("min_start_goal_distance_m" in e for e in errors)


def test_validate_candidate_duplicate_hash() -> None:
    c = _valid_candidate()
    existing: set[str] = {compute_control_hash(c)}
    _errors, warnings = validate_candidate_manifest(c, existing_hashes=existing)
    assert any("duplicate" in w for w in warnings)


def test_classify_valid() -> None:
    c = _valid_candidate()
    manifest = build_manifest(c, source=_source())
    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.VALID


def test_classify_invalid_out_of_bounds() -> None:
    ss = _search_space()
    c = CandidateSpec(
        start=Pose2D(10.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    manifest = build_manifest(c, search_space=ss)
    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.INVALID


def test_classify_degenerate_non_finite() -> None:
    c = CandidateSpec(
        start=Pose2D(float("inf"), 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    manifest = build_manifest(c)
    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.DEGENERATE


def test_classify_degenerate_non_positive_speed() -> None:
    c = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=-1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    manifest = build_manifest(c)
    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.DEGENERATE


def test_manifest_serializes_to_yaml(tmp_path: Path) -> None:
    c = _valid_candidate()
    manifest = build_manifest(c, source=_source())
    text = manifest.to_yaml()
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert loaded["candidate_controls"]["spawn_time_s"] == 0.0
    assert loaded["generator_id"] == "RandomCandidateSampler"
    assert loaded["validator_version"] == "adversarial_scenario_manifest_validator.v1"
    assert loaded["evidence_tier"] == "diagnostic-only"
    assert loaded["denominator_policy"] == "generated_candidates_not_benchmark_denominator"
    assert loaded["execution_gate"] == "generated_only"
    assert loaded["claim_boundary"] == loaded["evidence_boundary"]
    assert loaded["execution_status"] == "generated_only"
    assert "diagnostic-only" in loaded["evidence_boundary"]
    assert loaded["source"]["map_id"] == "classic_cross_trap"
    assert loaded["naturalistic_prior"]["schema_version"] == NATURALISTIC_PRIOR_SCHEMA_VERSION
    assert loaded["naturalistic_prior"]["profile"] == "urban_vru_default_v1"
    assert loaded["naturalistic_prior"]["passed"] is True
    assert loaded["naturalistic_prior"]["violation_flags"] == []
    assert validate_lineage_contract(loaded) == []


def test_manifest_round_trips_through_yaml(tmp_path: Path) -> None:
    c = _valid_candidate()
    original = build_manifest(c, source=_source())
    text = original.to_yaml()
    restored = AdversarialScenarioManifest.from_yaml(text)
    assert restored.schema_version == original.schema_version
    assert restored.execution_status == original.execution_status
    assert restored.evidence_boundary == original.evidence_boundary
    assert restored.candidate_controls == original.candidate_controls
    assert restored.naturalistic_prior is not None
    assert original.naturalistic_prior is not None
    assert restored.naturalistic_prior.passed == original.naturalistic_prior.passed
    assert restored.validation is not None
    assert original.validation is not None
    assert restored.validation.status == original.validation.status
    assert restored.validation.errors == original.validation.errors


def test_validate_manifest_payload_accepts_valid_payload() -> None:
    manifest = build_manifest(
        _valid_candidate(),
        source=_source(),
        generator=_generator(),
        search_space=_search_space(),
    )

    record = validate_manifest_payload(manifest.to_dict(), search_space=_search_space())

    assert record.status == ManifestCategory.VALID
    assert record.errors == ()
    assert record.normalized_control_hash == manifest.validation.normalized_control_hash


def test_adversarial_manifest_payload_satisfies_shared_lineage_contract() -> None:
    """Adversarial manifests expose the shared lineage/evidence-boundary fields."""
    payload = _valid_manifest_payload()

    assert validate_lineage_contract(payload) == []
    assert payload["generator_id"] == "TestSampler"
    assert payload["validator_version"] == "adversarial_scenario_manifest_validator.v1"
    assert payload["claim_boundary"] == payload["evidence_boundary"]


def test_validate_manifest_payload_rejects_bad_schema() -> None:
    payload = _valid_manifest_payload()
    payload["schema_version"] = "wrong.v1"

    record = validate_manifest_payload(payload)

    assert record.status == ManifestCategory.INVALID
    assert record.errors == ("schema_version must be adversarial_scenario_manifest.v1",)


def test_validate_manifest_payload_rejects_missing_controls() -> None:
    payload = _valid_manifest_payload()
    payload.pop("candidate_controls")

    record = validate_manifest_payload(payload)

    assert record.status == ManifestCategory.INVALID
    assert record.errors == ("candidate_controls must be a mapping",)


def test_validate_manifest_payload_classifies_degenerate_controls() -> None:
    payload = _valid_manifest_payload()
    payload["candidate_controls"]["pedestrian_speed_mps"] = 0.0
    payload.pop("naturalistic_prior")

    record = validate_manifest_payload(payload)

    assert record.status == ManifestCategory.DEGENERATE
    assert record.errors == ("pedestrian_speed_mps must be positive",)


def test_naturalistic_prior_flags_unrealistic_speed_without_invalidating_controls() -> None:
    candidate = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=3.5,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )

    manifest = build_manifest(candidate, source=_source(), generator=_generator())

    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.VALID
    assert any(
        "naturalistic prior violation" in warning for warning in manifest.validation.warnings
    )
    assert manifest.naturalistic_prior is not None
    assert manifest.naturalistic_prior.passed is False
    assert manifest.naturalistic_prior.violation_flags == (
        "pedestrian_speed_mps_outside_urban_vru_default_v1",
    )


def test_validate_manifest_payload_rejects_inconsistent_naturalistic_prior_metadata() -> None:
    candidate = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=3.5,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    payload = build_manifest(candidate, source=_source(), generator=_generator()).to_dict()
    payload["naturalistic_prior"]["passed"] = True
    payload["naturalistic_prior"]["violation_flags"] = []

    record = validate_manifest_payload(payload)

    assert record.status == ManifestCategory.INVALID
    assert "naturalistic_prior.passed does not match candidate controls" in record.errors
    assert "naturalistic_prior.violation_flags do not match candidate controls" in record.errors


def test_evaluate_naturalistic_prior_inclusive_bounds() -> None:
    low = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=0.4,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    high = CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=10.0,
        pedestrian_speed_mps=2.2,
        pedestrian_delay_s=3.0,
        scenario_seed=7,
    )

    assert evaluate_naturalistic_prior(low).passed is True
    assert evaluate_naturalistic_prior(high).passed is True


def test_validate_manifest_payload_rejects_fractional_seed() -> None:
    payload = _valid_manifest_payload()
    payload["candidate_controls"]["scenario_seed"] = 7.5

    record = validate_manifest_payload(payload)

    assert record.status == ManifestCategory.INVALID
    assert record.errors == ("candidate_controls.scenario_seed must be an integer",)


def test_validate_manifest_payload_classifies_duplicates() -> None:
    manifest = _valid_manifest_payload()
    existing = {compute_control_hash(_valid_candidate())}

    record = validate_manifest_payload(manifest, existing_hashes=existing)

    assert record.status == ManifestCategory.DEGENERATE
    assert record.warnings == (f"duplicate normalized control hash: {next(iter(existing))}",)


def test_generate_manifests_is_deterministic() -> None:
    ss = _wide_search_space()
    source = _source()
    manifests_a, summary_a = generate_manifests(
        ss, seed=42, count=5, source=source, generator_family="random"
    )
    manifests_b, summary_b = generate_manifests(
        ss, seed=42, count=5, source=source, generator_family="random"
    )
    assert summary_a == summary_b
    for a, b in zip(manifests_a, manifests_b, strict=True):
        assert a.candidate_controls == b.candidate_controls
        assert a.validation is not None
        assert b.validation is not None
        assert a.validation.status == b.validation.status


def test_generate_manifests_different_seeds_produce_different_candidates() -> None:
    ss = _wide_search_space()
    _manifests_a, _ = generate_manifests(ss, seed=1, count=3)
    _manifests_b, _ = generate_manifests(ss, seed=2, count=3)
    controls_a = [m.candidate_controls for m in _manifests_a]
    controls_b = [m.candidate_controls for m in _manifests_b]
    assert any(a != b for a, b in zip(controls_a, controls_b, strict=True))


def test_generate_manifests_summary_shape() -> None:
    ss = _wide_search_space()
    _manifests, summary = generate_manifests(ss, seed=0, count=8)
    assert summary["total_candidates"] == 8
    assert summary["valid"] + summary["invalid"] + summary["degenerate"] == 8
    assert summary["naturalistic_prior"]["profile"] == "urban_vru_default_v1"
    assert (
        summary["naturalistic_prior"]["pass"]
        + summary["naturalistic_prior"]["fail"]
        + summary["naturalistic_prior"]["unavailable"]
        == 8
    )
    assert isinstance(summary["rejection_reasons"], dict)


def test_write_manifest_yaml(tmp_path: Path) -> None:
    c = _valid_candidate()
    manifest = build_manifest(c)
    path = write_manifest_yaml(manifest, tmp_path / "candidate_0000.yaml")
    assert path.exists()
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION


def test_manifest_schema_version_constant() -> None:
    assert MANIFEST_SCHEMA_VERSION == "adversarial_scenario_manifest.v1"


def test_manifest_with_generator_info() -> None:
    c = _valid_candidate()
    gen = GeneratorInfo(
        family="test_family", generator_id="TestSampler", seed=99, candidate_index=3
    )
    manifest = build_manifest(c, generator=gen)
    assert manifest.generator is not None
    assert manifest.generator.family == "test_family"
    assert manifest.generator.seed == 99
    assert manifest.generator.candidate_index == 3


def test_manifest_with_duplicate_detection() -> None:
    ss = _search_space()
    c = _valid_candidate()
    existing: set[str] = {compute_control_hash(c)}
    manifest = build_manifest(c, search_space=ss, existing_hashes=existing)
    assert manifest.validation is not None
    assert manifest.validation.status == ManifestCategory.DEGENERATE
    assert any("duplicate" in w for w in manifest.validation.warnings)


def test_generate_manifests_classifies_duplicates_as_degenerate() -> None:
    ss = _search_space()

    _manifests, summary = generate_manifests(ss, seed=42, count=4)

    assert summary["valid"] == 1
    assert summary["degenerate"] == 3
    assert summary["rejection_reasons"] == {"duplicate normalized control hash": 3}


def test_cli_generates_expected_files(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    template_path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "test_template",
                        "map_id": "classic_cross_trap",
                        "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                        "robot_config": {},
                        "metadata": {"archetype": "test"},
                        "seeds": [1],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    exit_code = cli_main(
        [
            "--search-space",
            str(space_path),
            "--scenario-template",
            str(template_path),
            "--count",
            "4",
            "--seed",
            "42",
            "--output-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "adversarial_scenario_manifest_generation_summary.v1"
    assert summary["source"]["scenario_template"] == "template.yaml"
    assert summary["source"]["search_space"] == "space.yaml"
    assert summary["generator"] == {
        "family": "random",
        "generator_id": "RandomCandidateSampler",
        "seed": 42,
    }
    assert summary["total_candidates"] == 4

    search_space = SearchSpaceConfig.from_file(space_path)
    for i in range(4):
        cand_path = out_dir / f"candidate_{i:04d}.yaml"
        assert cand_path.exists(), f"missing {cand_path}"
        loaded = yaml.safe_load(cand_path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert loaded["naturalistic_prior"]["passed"] is True
        record = validate_manifest_payload(loaded, search_space=search_space)
        assert record.status == ManifestCategory.VALID


def test_cli_rejects_missing_files(tmp_path: Path) -> None:
    exit_code = cli_main(
        [
            "--search-space",
            str(tmp_path / "nonexistent.yaml"),
            "--scenario-template",
            str(tmp_path / "nonexistent.yaml"),
            "--count",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


def test_cli_rejects_invalid_count(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    template_path.write_text("scenarios: [{name: t, map_id: m}]", encoding="utf-8")
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        "variables: {start_x: {min: 0, max: 1}, start_y: {min: 0, max: 1}, "
        "goal_x: {min: 2, max: 3}, goal_y: {min: 2, max: 3}}",
        encoding="utf-8",
    )
    exit_code = cli_main(
        [
            "--search-space",
            str(space_path),
            "--scenario-template",
            str(template_path),
            "--count",
            "0",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1


def test_load_template_info_supports_dict_scenarios(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    template_path.write_text(
        yaml.safe_dump(
            {"scenarios": {"crossing": {"name": "dict_template", "map_id": "classic_cross_trap"}}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    source = _load_template_info(template_path)

    assert source.map_id == "classic_cross_trap"
    assert source.scenario_name == "dict_template"


def test_load_template_info_ignores_scalar_scenarios(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    template_path.write_text("scenarios: 7\n", encoding="utf-8")

    source = _load_template_info(template_path)

    assert source.map_id is None
    assert source.scenario_name is None


def test_validation_record_to_dict() -> None:
    record = ValidationRecord(
        status=ManifestCategory.VALID,
        errors=("err1",),
        warnings=("warn1",),
        normalized_control_hash="abc123",
    )
    d = record.to_dict()
    assert d["status"] == "valid"
    assert d["errors"] == ["err1"]
    assert d["warnings"] == ["warn1"]
    assert d["normalized_control_hash"] == "abc123"


def test_validation_record_no_hash() -> None:
    record = ValidationRecord()
    d = record.to_dict()
    assert "normalized_control_hash" not in d


def test_source_lineage_to_dict() -> None:
    src = _source()
    d = src.to_dict()
    assert d["map_id"] == "classic_cross_trap"
    assert "scenario_template" in d


def test_source_lineage_to_dict_omits_none() -> None:
    src = SourceLineage()
    d = src.to_dict()
    assert d == {}


def test_generator_info_to_dict() -> None:
    gen = GeneratorInfo(family="test", generator_id="T", seed=7, candidate_index=1)
    d = gen.to_dict()
    assert d["family"] == "test"
    assert d["seed"] == 7
    assert d["candidate_index"] == 1


def test_manifest_from_yaml_round_trips(tmp_path: Path) -> None:
    c = _valid_candidate()
    original = build_manifest(
        c, source=_source(), generator=GeneratorInfo(seed=5, candidate_index=2)
    )
    text = original.to_yaml()
    restored = AdversarialScenarioManifest.from_yaml(text)
    assert restored.generator is not None
    assert original.generator is not None
    assert restored.generator.seed == original.generator.seed
    assert restored.generator.candidate_index == original.generator.candidate_index
    assert restored.source is not None
    assert original.source is not None
    assert restored.source.map_id == original.source.map_id


def _read_llm_fixture(name: str) -> dict[str, object]:
    """Load an LLM manifest fixture payload."""
    path = _LLM_FIXTURE_DIR / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_llm_manifest_fixture_is_validated_ok() -> None:
    search_space = _search_space()
    accepted = _read_llm_fixture("accepted_manifest.yaml")

    record = validate_manifest_payload(accepted, search_space=search_space)

    assert record.status == ManifestCategory.VALID
    assert record.errors == ()


def test_llm_manifest_fixture_is_rejected_fail_closed() -> None:
    search_space = _search_space()
    rejected = _read_llm_fixture("rejected_manifest.yaml")

    record = validate_manifest_payload(rejected, search_space=search_space)

    assert record.status == ManifestCategory.INVALID
    assert "candidate_controls.goal must define x and y" in record.errors


def test_validate_manifest_payload_rejects_missing_required_fields() -> None:
    payload = _valid_manifest_payload()
    payload.pop("execution_status")
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "execution_status must be a string" in record.errors

    payload = _valid_manifest_payload()
    payload.pop("evidence_boundary")
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "evidence_boundary must be a string" in record.errors

    payload = _valid_manifest_payload()
    payload.pop("source")
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "source must be a mapping" in record.errors

    payload = _valid_manifest_payload()
    payload.pop("generator")
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "generator must be a mapping" in record.errors


def test_validate_manifest_payload_rejects_wrong_required_field_types() -> None:
    payload = _valid_manifest_payload()
    payload["execution_status"] = 12
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "execution_status must be a string" in record.errors

    payload = _valid_manifest_payload()
    payload["evidence_boundary"] = 12
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "evidence_boundary must be a string" in record.errors

    payload = _valid_manifest_payload()
    payload["source"] = 12
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "source must be a mapping" in record.errors

    payload = _valid_manifest_payload()
    source = _source().to_dict()
    source["search_space"] = 12
    payload["source"] = source
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "source.search_space must be a string" in record.errors

    payload = _valid_manifest_payload()
    payload["generator"] = 12
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "generator must be a mapping" in record.errors

    payload = _valid_manifest_payload()
    generator = _generator().to_dict()
    generator["candidate_index"] = "3"
    payload["generator"] = generator
    record = validate_manifest_payload(payload)
    assert record.status == ManifestCategory.INVALID
    assert "generator.candidate_index must be an integer" in record.errors
