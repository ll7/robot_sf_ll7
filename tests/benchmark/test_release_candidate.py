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
MATRIX = SOURCE_ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v3.yaml"
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
    config["scenario_matrix"] = MATRIX.relative_to(SOURCE_ROOT).as_posix()
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
    config_path = tmp_path / CONFIG.relative_to(SOURCE_ROOT)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
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


def test_builder_creates_ignored_valid_candidate_without_doi(candidate_repo) -> None:
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


@pytest.mark.parametrize(
    ("change_during_run", "prior_error"),
    [(False, None), (True, None), (True, "matrix_worker_failed")],
)
def test_preflight_cli_accepts_candidate_and_rejects_mid_run_drift(
    candidate_repo,
    monkeypatch: pytest.MonkeyPatch,
    change_during_run: bool,
    prior_error: str | None,
) -> None:
    root, path, payload = candidate_repo
    original_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(release_candidate, "get_repository_root", lambda: root)
    observed: dict[str, object] = {}

    def diagnostic_preflight(manifest, *, source_commit, **_kwargs):
        assert isinstance(manifest, release_candidate.PrepublicationCandidate)
        identity, scenarios, seeds = spawn_preflight._release_manifest_inputs(manifest)
        observed["source_commit"] = source_commit
        assert len(scenarios) == 48 and seeds == tuple(range(111, 141))
        if change_during_run:
            payload["candidate_id"] = "changed-during-preflight"
            path.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "schema_version": "spawn_matrix_preflight.v1",
            "status": "blocked",
            "evidence_class": "preflight_diagnostic_only",
            "benchmark_success": None,
            "source_commit": source_commit,
            "release_inputs": identity,
            "rows": [],
            "cell_count": 0,
            "blocked_cell_count": 0,
            "input_error": prior_error,
        }

    monkeypatch.setattr(spawn_preflight, "run_manifest_preflight", diagnostic_preflight)
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
    assert result == 2
    assert observed["source_commit"] == payload["source_commit"]
    assert report["source_commit"] == payload["source_commit"]
    assert report["release_inputs"]["manifest_sha256"] == original_digest
    if change_during_run:
        assert report["status"] == "invalid"
        assert "candidate_input_drift" in report["input_error"]
        if prior_error:
            assert prior_error in report["input_error"]
    else:
        assert report["status"] == "blocked"
        assert report["input_error"] is None
