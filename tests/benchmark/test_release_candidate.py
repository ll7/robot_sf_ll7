"""DOI-free candidate stays exact and cannot enter publication paths (#9863)."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark import release_candidate, spawn_preflight
from robot_sf.benchmark.release_candidate import (
    _APPROVED_008_HYBRID_CONFIGS,
    CANDIDATE_SCHEMA,
    _expected_input_paths,
    create_prepublication_candidate,
    load_preflight_input,
    load_prepublication_candidate,
    verify_prepublication_candidate_after_preflight,
)
from robot_sf.benchmark.release_protocol import load_release_manifest
from robot_sf.training.scenario_loader import load_scenarios_for_validation

SOURCE_ROOT = Path(__file__).parents[2]
CONFIG = (
    SOURCE_ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
MATRIX = SOURCE_ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
SUITE_POLICY = (
    "configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1_suite_policy.yaml"
)
ROUTE_CERTIFICATION = "configs/benchmarks/route_clearance_certifications_v1.yaml"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


@pytest.fixture
def candidate_repo(tmp_path: Path) -> tuple[Path, Path, dict]:
    """Build a small committed checkout with a real 48-scenario source closure."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scenarios = load_scenarios_for_validation(MATRIX, base_dir=SOURCE_ROOT)
    assert scenarios.load_error is None and not scenarios.load_issues and not scenarios.entry_issues
    rows = [dict(row) for row in scenarios.scenarios]
    seed_policy = {
        "mode": "seed-set",
        "seed_set": "paper_eval_s30",
        "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
        "resolved_seeds": list(range(111, 141)),
    }
    inputs = {
        "suite_policy_path": SUITE_POLICY,
        "route_certification_path": ROUTE_CERTIFICATION,
    }
    paths = _expected_input_paths(
        SOURCE_ROOT,
        CONFIG,
        config,
        MATRIX,
        rows,
        {**seed_policy, **inputs},
        config["planners"],
    )
    for source in paths:
        target = tmp_path / source.relative_to(SOURCE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    pins = {
        source.relative_to(SOURCE_ROOT).as_posix(): hashlib.sha256(
            (tmp_path / source.relative_to(SOURCE_ROOT)).read_bytes()
        ).hexdigest()
        for source in paths
    }
    (tmp_path / ".gitignore").write_text("output/\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "candidate_id": "0.0.8-diagnostic-fixture",
        "release_kind": "benchmark-data-prepublication-candidate",
        "source_commit": _git(tmp_path, "rev-parse", "HEAD"),
        "canonical_campaign_config": CONFIG.relative_to(SOURCE_ROOT).as_posix(),
        "scenario": {
            "matrix_path": MATRIX.relative_to(SOURCE_ROOT).as_posix(),
            "identities": [row["name"] for row in rows],
        },
        "planners": {"keys": [row["key"] for row in config["planners"]]},
        "seed_policy": seed_policy,
        "matrix": {"expected_episode_cells": 20160, "horizon_steps": 600},
        "inputs": inputs,
        "sha256_files": pins,
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return tmp_path, candidate_path, payload


def test_doi_free_candidate_loads_but_publication_loader_rejects(candidate_repo) -> None:
    root, path, payload = candidate_repo
    candidate = load_prepublication_candidate(path, repository_root=root)
    assert load_preflight_input(path, repository_root=root) == candidate
    assert candidate.release_id == payload["candidate_id"]
    assert len(candidate.planner_keys) == 14
    assert len(candidate.scenario_identities) == 48
    assert candidate.resolved_seeds == tuple(range(111, 141))
    assert candidate.expected_episode_cells == 20160
    with pytest.raises(ValueError, match="schema_version"):
        load_release_manifest(path, repository_root=root)


def _replace_campaign_algorithm(candidate_repo, arm: str) -> dict:
    """Change only one algo value in real template bytes, then re-pin the checkout."""
    root, path, payload = candidate_repo
    config_path = root / payload["canonical_campaign_config"]
    original = CONFIG.read_bytes()
    assert config_path.read_bytes() == original
    config = yaml.safe_load(original)
    row = next(row for row in config["planners"] if row["key"] == arm)
    replacement = "risk_dwa" if row["algo"] == "goal" else "goal"
    old = f"  - key: {arm}\n    algo: {row['algo']}\n".encode()
    new = f"  - key: {arm}\n    algo: {replacement}\n".encode()
    assert original.count(old) == 1
    changed = original.replace(old, new, 1)
    config_path.write_bytes(changed)
    changed_config = yaml.safe_load(changed)
    row["algo"] = replacement
    assert changed_config == config  # No matrix, config path, or other row changes.
    payload["sha256_files"][payload["canonical_campaign_config"]] = hashlib.sha256(
        changed
    ).hexdigest()
    _git(root, "add", payload["canonical_campaign_config"])
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "substituted campaign algorithm",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return changed_config


@pytest.mark.parametrize(
    "arm", [row["key"] for row in yaml.safe_load(CONFIG.read_bytes())["planners"]]
)
def test_candidate_rejects_algo_only_substitution(candidate_repo, arm: str) -> None:
    """Every real-template arm must retain its actual approved algorithm."""
    root, path, _payload = candidate_repo
    _replace_campaign_algorithm(candidate_repo, arm)
    with pytest.raises(ValueError, match=f"{arm} must bind its approved algorithm"):
        load_prepublication_candidate(path, repository_root=root)


@pytest.mark.parametrize("arm", list(_APPROVED_008_HYBRID_CONFIGS))
def test_hybrid_input_admission_uses_actual_campaign_algorithm(candidate_repo, arm: str) -> None:
    """Runtime admission must use the row algo even without the roster identity gate."""
    root, _path, payload = candidate_repo
    config = _replace_campaign_algorithm(candidate_repo, arm)
    matrix_path = root / payload["scenario"]["matrix_path"]
    scenarios = load_scenarios_for_validation(matrix_path, base_dir=root)
    assert scenarios.load_error is None and not scenarios.load_issues and not scenarios.entry_issues
    with pytest.raises(
        ValueError, match=f"v4 hybrid slot {arm} resolves to a non-v4 planner variant"
    ):
        _expected_input_paths(
            root,
            root / payload["canonical_campaign_config"],
            config,
            matrix_path,
            [dict(row) for row in scenarios.scenarios],
            {**payload["seed_policy"], **payload["inputs"]},
            config["planners"],
        )


def test_candidate_rejects_changed_map_bytes(candidate_repo) -> None:
    root, path, payload = candidate_repo
    map_path = next(name for name in payload["sha256_files"] if name.endswith(".svg"))
    with (root / map_path).open("a", encoding="utf-8") as stream:
        stream.write("\n<!-- changed -->\n")
    _git(root, "add", map_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "changed map",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256_files hash mismatch"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_missing_input_pin(candidate_repo) -> None:
    root, path, payload = candidate_repo
    changed = copy.deepcopy(payload)
    map_path = next(name for name in changed["sha256_files"] if name.endswith(".svg"))
    del changed["sha256_files"][map_path]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="pin exactly the input closure"):
        load_prepublication_candidate(path, repository_root=root)


@pytest.mark.parametrize(
    "nested_path",
    [
        "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
        "configs/algos/issue707_orca_tuned.yaml",
    ],
)
def test_candidate_pins_nested_policy_search_configs(candidate_repo, nested_path: str) -> None:
    root, path, payload = candidate_repo
    assert nested_path in payload["sha256_files"]
    del payload["sha256_files"][nested_path]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pin exactly the input closure"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_changed_nested_policy_search_config(candidate_repo) -> None:
    root, path, payload = candidate_repo
    nested_path = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
    with (root / nested_path).open("a", encoding="utf-8") as stream:
        stream.write("\n# changed after candidate creation\n")
    _git(root, "add", nested_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "changed nested planner config",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256_files hash mismatch"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_roster_and_publication_coordinates(candidate_repo) -> None:
    root, path, payload = candidate_repo
    changed = copy.deepcopy(payload)
    changed["planners"]["keys"] = changed["planners"]["keys"][:-1]
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="14 ordered enabled campaign arms"):
        load_prepublication_candidate(path, repository_root=root)
    changed = copy.deepcopy(payload)
    changed["publication"] = {"version_doi": "10.5281/zenodo.123"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="must omit publication"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_jointly_rewritten_planner_key(candidate_repo) -> None:
    root, path, payload = candidate_repo
    config_path = root / payload["canonical_campaign_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["planners"][0]["key"] = "arbitrary_extra_planner"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    payload["planners"]["keys"][0] = "arbitrary_extra_planner"
    payload["sha256_files"][payload["canonical_campaign_config"]] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    _git(root, "add", payload["canonical_campaign_config"])
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "changed roster",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="approved 0.0.8 14-slot roster"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_v4_key_bound_to_historical_v3_config(candidate_repo) -> None:
    root, path, payload = candidate_repo
    config_path = root / payload["canonical_campaign_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["planners"][7]["algo_config"] = (
        "configs/policy_search/candidates/"
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_s30_h600_release.yaml"
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    payload["sha256_files"][payload["canonical_campaign_config"]] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    _git(root, "add", payload["canonical_campaign_config"])
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "misbound v4 planner key",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must bind its v4 config path"):
        load_prepublication_candidate(path, repository_root=root)


@pytest.mark.parametrize(
    ("arm", "historical_path"),
    [
        ("risk_dwa", "configs/algos/risk_dwa_camera_ready.yaml"),
        ("predictive_mppi", "configs/algos/predictive_mppi_camera_ready.yaml"),
        ("guarded_ppo", "configs/algos/guarded_ppo_camera_ready_cpu.yaml"),
    ],
)
def test_candidate_rejects_historical_waypoint_binding(
    candidate_repo, arm: str, historical_path: str
) -> None:
    """A self-consistent candidate cannot restore any historical waypoint arm."""
    root, path, payload = candidate_repo
    config_path = root / payload["canonical_campaign_config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    next(row for row in config["planners"] if row["key"] == arm)["algo_config"] = historical_path
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    historical_copy = root / historical_path
    historical_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_ROOT / historical_path, historical_copy)
    payload["sha256_files"][payload["canonical_campaign_config"]] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    payload["sha256_files"][historical_path] = hashlib.sha256(
        historical_copy.read_bytes()
    ).hexdigest()
    _git(root, "add", payload["canonical_campaign_config"], historical_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "historical waypoint binding",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{arm} must bind its active-waypoint v2 config path"):
        load_prepublication_candidate(path, repository_root=root)


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi", "guarded_ppo"])
def test_candidate_rejects_mutated_effective_waypoint_selector(candidate_repo, arm: str) -> None:
    root, path, payload = candidate_repo
    campaign = yaml.safe_load((root / payload["canonical_campaign_config"]).read_text())
    algo_path = next(row["algo_config"] for row in campaign["planners"] if row["key"] == arm)
    config_path = root / algo_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selector_config = config["fallback_risk_dwa"] if arm == "guarded_ppo" else config
    selector_config["goal_target_version"] = "legacy_next_goal_v1"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    payload["sha256_files"][algo_path] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    _git(root, "add", algo_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "mutated waypoint selector",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{arm} must select active_waypoint_v2"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_v4_config_bound_to_historical_v3_base(candidate_repo) -> None:
    root, path, payload = candidate_repo
    config_path = _APPROVED_008_HYBRID_CONFIGS[
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4"
    ]
    manifest_path = root / config_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["base_config_path"] = "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    payload["sha256_files"][config_path] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _git(root, "add", config_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "misbound v4 planner base",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must bind the v4 base config"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_jointly_pinned_v3_scenario_algorithm_override(candidate_repo) -> None:
    root, path, payload = candidate_repo
    config_path = _APPROVED_008_HYBRID_CONFIGS[
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4"
    ]
    manifest_path = root / config_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    override = manifest["scenario_algo_overrides"]["francis2023_leave_group"]
    override["algo"] = "hybrid_rule_local_planner"
    nested_path = "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml"
    override["base_config_path"] = nested_path
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    nested_target = root / nested_path
    nested_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_ROOT / nested_path, nested_target)
    payload["sha256_files"][config_path] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload["sha256_files"][nested_path] = hashlib.sha256(nested_target.read_bytes()).hexdigest()
    _git(root, "add", config_path, nested_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "jointly pinned historical scenario override",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved scenario algorithm override"):
        load_prepublication_candidate(path, repository_root=root)


@pytest.mark.parametrize("section", ["params", "family_overrides", "scenario_overrides"])
def test_candidate_rejects_jointly_pinned_v3_hybrid_variant_override(
    candidate_repo, section: str
) -> None:
    root, path, payload = candidate_repo
    config_path = _APPROVED_008_HYBRID_CONFIGS[
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4"
    ]
    manifest_path = root / config_path
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if section == "params":
        manifest[section]["planner_variant"] = "hybrid_rule_v3_teb_like_rollout"
    else:
        scenario_key = "classic_bottleneck_high" if section == "scenario_overrides" else "classic"
        manifest.setdefault(section, {}).setdefault(scenario_key, {})["planner_variant"] = (
            "hybrid_rule_v3_teb_like_rollout"
        )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    payload["sha256_files"][config_path] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _git(root, "add", config_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "jointly pinned historical hybrid variant",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    reason = (
        "frozen v4 slot resolves planner_variant=.*hybrid_rule_v3_teb_like_rollout"
        if section == "params"
        else "resolves to a non-v4 planner variant"
    )
    with pytest.raises(ValueError, match=reason):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_jointly_pinned_v3_variant_in_v4_base(candidate_repo) -> None:
    root, path, payload = candidate_repo
    base_path = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
    base_file = root / base_path
    base = yaml.safe_load(base_file.read_text(encoding="utf-8"))
    base["planner_variant"] = "hybrid_rule_v3_teb_like_rollout"
    base_file.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    payload["sha256_files"][base_path] = hashlib.sha256(base_file.read_bytes()).hexdigest()
    _git(root, "add", base_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "jointly pinned historical hybrid base",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="resolves to a non-v4 planner variant"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_jointly_rewritten_scenario_identity(candidate_repo) -> None:
    root, path, payload = candidate_repo
    matrix_path = root / payload["scenario"]["matrix_path"]
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    matrix["scenario_overrides_by_name"]["classic_bottleneck_low"]["name"] = (
        "classic_bottleneck_low_alternate"
    )
    matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    changed = load_scenarios_for_validation(matrix_path, base_dir=root)
    assert changed.load_error is None and not changed.load_issues and not changed.entry_issues
    payload["scenario"]["identities"] = [row["name"] for row in changed.scenarios]
    payload["sha256_files"][payload["scenario"]["matrix_path"]] = hashlib.sha256(
        matrix_path.read_bytes()
    ).hexdigest()
    _git(root, "add", payload["scenario"]["matrix_path"])
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "changed identity",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="48 frozen 0.0.7 identities"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_source_commit_drift(candidate_repo) -> None:
    root, path, payload = candidate_repo
    changed = copy.deepcopy(payload)
    changed["source_commit"] = "f" * 40
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="exact checkout HEAD"):
        load_prepublication_candidate(path, repository_root=root)


def test_candidate_rejects_pinned_but_untracked_map(candidate_repo) -> None:
    root, path, payload = candidate_repo
    map_path = next(name for name in payload["sha256_files"] if name.endswith(".svg"))
    _git(root, "rm", "--cached", "-q", map_path)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "drop map from source",
    )
    payload["source_commit"] = _git(root, "rev-parse", "HEAD")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not in source_commit"):
        load_prepublication_candidate(path, repository_root=root)


def test_builder_creates_ignored_valid_candidate_without_doi(candidate_repo, monkeypatch) -> None:
    root, _path, _payload = candidate_repo
    output = root / "output" / "candidate.json"
    candidate = create_prepublication_candidate(
        campaign_config=CONFIG.relative_to(SOURCE_ROOT),
        suite_policy=Path(SUITE_POLICY),
        route_certification=Path(ROUTE_CERTIFICATION),
        candidate_id="0.0.8-diagnostic-fixture",
        output=output,
        repository_root=root,
    )
    assert candidate.path == output
    assert "doi" not in json.loads(output.read_text(encoding="utf-8"))
    with pytest.raises(FileExistsError):
        create_prepublication_candidate(
            campaign_config=CONFIG.relative_to(SOURCE_ROOT),
            suite_policy=Path(SUITE_POLICY),
            route_certification=Path(ROUTE_CERTIFICATION),
            candidate_id="0.0.8-diagnostic-fixture",
            output=output,
            repository_root=root,
        )

    def reject_candidate(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid candidate")

    monkeypatch.setattr(release_candidate, "load_prepublication_candidate", reject_candidate)
    rejected_output = root / "output" / "rejected-candidate.json"
    with pytest.raises(ValueError, match="invalid candidate"):
        create_prepublication_candidate(
            campaign_config=CONFIG.relative_to(SOURCE_ROOT),
            suite_policy=Path(SUITE_POLICY),
            route_certification=Path(ROUTE_CERTIFICATION),
            candidate_id="0.0.8-diagnostic-fixture",
            output=rejected_output,
            repository_root=root,
        )
    assert not rejected_output.exists()


def test_post_preflight_readback_rejects_candidate_digest_drift(candidate_repo) -> None:
    root, path, payload = candidate_repo
    candidate = load_prepublication_candidate(path, repository_root=root)
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_prepublication_candidate_after_preflight(
        candidate, manifest_sha256=original_digest, repository_root=root
    )
    payload["candidate_id"] = "changed-after-run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed while preflight"):
        verify_prepublication_candidate_after_preflight(
            candidate, manifest_sha256=original_digest, repository_root=root
        )


@pytest.mark.parametrize("change_during_run", [False, True])
def test_preflight_cli_accepts_candidate_and_rejects_mid_run_drift(
    candidate_repo,
    monkeypatch: pytest.MonkeyPatch,
    change_during_run: bool,
) -> None:
    root, path, payload = candidate_repo
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(release_candidate, "get_repository_root", lambda: root)
    monkeypatch.setattr(spawn_preflight, "get_repository_root", lambda: root)
    observed: dict[str, object] = {"changed": False}

    def diagnostic_scenario(job):
        scenario, _matrix, seeds, *_checks = job
        if change_during_run and not observed["changed"]:
            payload["candidate_id"] = "changed-during-preflight"
            path.write_text(json.dumps(payload), encoding="utf-8")
            observed["changed"] = True
        name = scenario["name"]
        return {
            "scenario": name,
            "rows": [
                {
                    "scenario": name,
                    "seed": seed,
                    "overall_status": "valid",
                }
                for seed in seeds
            ],
            "map_warnings": [],
        }

    monkeypatch.setattr(spawn_preflight, "_check_release_scenario", diagnostic_scenario)
    json_output = root / "report.json"
    result = spawn_preflight.main(
        [
            "--manifest",
            str(path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(root / "report.md"),
        ]
    )
    report = json.loads(json_output.read_text(encoding="utf-8"))
    assert result == (2 if change_during_run else 0)
    assert report["source_commit"] == payload["source_commit"]
    assert report["release_inputs"]["manifest_sha256"] == original_digest
    assert report["expected_cell_count"] == report["cell_count"] == 1440
    if change_during_run:
        assert report["status"] == "invalid"
        assert "candidate_input_drift" in report["input_error"]
        assert "release manifest changed" in report["input_error"]
    else:
        assert report["status"] == "valid"
        assert report["input_error"] is None
