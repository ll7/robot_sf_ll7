"""D-049 execution and reverse-policy witnesses; all simulation is stubbed."""
# seed-holdout: synthetic-fixture begin

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.benchmark import spawn_preflight as spawn
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from scripts.tools import run_benchmark_release as runner
from tests.benchmark.test_sealed_source_pins import (
    git,
    materialize,
    worker_stub,
)
from tests.benchmark.test_sealed_source_pins import (
    sealed_repository as _sealed_repository,
)

sealed_repository = _sealed_repository

ROOT = Path(__file__).resolve().parents[2]
RELEASES = ROOT / "configs/benchmarks/releases"
HISTORICAL = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
SMOKE = "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
SLICE = "three_width_doorway_release_0_0_8_v1.yaml"


@pytest.fixture(autouse=True)
def forbid_environments(monkeypatch):
    def abort(*args, **kwargs):
        pytest.fail("environment creation attempted in a static witness")

    monkeypatch.setattr(spawn, "make_robot_env", abort)
    monkeypatch.setattr("robot_sf.gym_env.environment_factory.make_robot_env", abort)
    monkeypatch.setattr(runner, "run_campaign", abort)
    monkeypatch.setattr(runner, "prepare_campaign_preflight", abort)
    monkeypatch.setattr(runner, "check_orca_rvo2_preflight", lambda *_a, **_kw: None)


@pytest.mark.parametrize("case", ["invalid-dev", "invalid-retired", "historical"])
def test_runner_refuses_before_spawn(monkeypatch, tmp_path, capsys, case):
    manifest = protocol.load_release_manifest(
        RELEASES / (SMOKE if case == "invalid-dev" else HISTORICAL)
    )
    if case == "invalid-retired":
        manifest = replace(manifest, release_id="anonymous", release_tag="anonymous")
    monkeypatch.setattr(runner, "load_release_manifest", lambda _path: manifest)
    if case == "invalid-dev":
        monkeypatch.setattr(
            runner,
            "validate_release_manifest",
            lambda *_a, **_kw: {"status": "invalid", "problems": ["fixture invalid"]},
        )
    reached = []

    def record(**kwargs):
        reached.append(list(kwargs["manifest"].resolved_seeds))
        return {"blocked_cell_count": 0}, {"status": "invalid", "input_error": "stub"}

    monkeypatch.setattr(runner, "_run_spawn_matrix_preflight", record)
    assert (
        runner.main(
            ["--manifest", "unused.yaml", "--mode", "preflight", "--output-root", str(tmp_path)]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert reached == [], f"spawn reached with {reached}"
    assert payload["campaign_execution_status"] == "not_started"


@pytest.mark.parametrize("entry", ["api", "standalone"])
def test_historical_execution_refused_at_shared_preflight(monkeypatch, tmp_path, entry):
    reached = []

    def record(job):
        reached.append(job[2])
        return {"scenario": job[0]["name"], "rows": [], "map_warnings": []}

    monkeypatch.setattr(spawn, "_check_release_scenario", record)
    if entry == "api":
        report = spawn.run_manifest_preflight(protocol.load_release_manifest(RELEASES / HISTORICAL))
        assert report["status"] == "invalid"
        assert "retired" in report["input_error"]
    else:
        assert (
            spawn.main(
                [
                    "--manifest",
                    str(RELEASES / HISTORICAL),
                    "--workers",
                    "1",
                    "--json-output",
                    str(tmp_path / "report.json"),
                    "--markdown-output",
                    str(tmp_path / "report.md"),
                ]
            )
            == 2
        )
    assert reached == [], f"scenario workers reached with {reached}"


@pytest.mark.parametrize("source", [None, "another-commit", "HEAD"])
def test_slice_requires_freeze_source(sealed_repository, source):
    repo = sealed_repository
    if source is None:
        manifest = protocol.load_release_manifest(repo / "configs/benchmarks/releases" / SLICE)
        # The historical concrete manifest stays immutable; exercise the successor's guard path.
        manifest = replace(
            manifest,
            canonical_campaign_config_path=(
                repo
                / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v2.yaml"
            ),
        )
        result = protocol.validate_release_manifest(manifest, repository_root=repo)
        assert any("sealed" in p and "source_sha" in p for p in result["problems"]), result
        return
    manifest = materialize(repo, "slice")
    if source == "another-commit":
        git(repo, "commit", "--allow-empty", "-qm", "alternate source")
        with pytest.raises(ValueError, match="does not match source_commit"):
            protocol.load_release_manifest(manifest.path, repository_root=repo)
    else:
        result = protocol.validate_release_manifest(manifest, repository_root=repo)
        assert result["status"] == "valid", result["problems"]


def test_anonymous_mixed_sealed_seed_release_refused(tmp_path):
    import yaml

    manifest = protocol.load_release_manifest(RELEASES / SLICE)
    cfg = load_campaign_config(manifest.canonical_campaign_config_path)
    seeds = (1001, 50036)
    cfg = replace(
        cfg, seed_policy=replace(cfg.seed_policy, mode="fixed-list", seed_set=None, seeds=seeds)
    )
    manifest = replace(
        manifest,
        release_id="anonymous",
        release_tag="anonymous",
        release_kind="benchmark-data",
        seed_policy={
            "mode": "fixed-list",
            "seed_set": None,
            "seeds": list(seeds),
            "seed_sets_path": str(cfg.seed_policy.seed_sets_path),
        },
        canonical_campaign_config_path=tmp_path / "campaign.yaml",
        scenario_matrix_path=tmp_path / "matrix.yaml",
    )
    problems = []
    protocol._validate_release_seed_policy(manifest, cfg, problems)
    assert any("sealed" in p for p in problems), problems
    # Repeat through the real config loader, as the runner does.
    original = (
        yaml.safe_load(cfg.source_path.read_text())
        if hasattr(cfg, "source_path")
        else yaml.safe_load(
            (
                ROOT
                / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
            ).read_text()
        )
    )
    original.update(seed_policy=manifest.seed_policy, scenario_matrix=str(cfg.scenario_matrix_path))
    manifest.canonical_campaign_config_path.write_text(yaml.safe_dump(original))
    loaded = load_campaign_config(manifest.canonical_campaign_config_path)
    problems = []
    protocol._validate_release_seed_policy(manifest, loaded, problems)
    assert any("sealed" in p for p in problems), problems


@pytest.mark.parametrize("source", ["HEAD", "another-commit"])
def test_freeze_bound_slice_preflight_uses_runtime_source(sealed_repository, monkeypatch, source):
    manifest = materialize(sealed_repository, "slice")
    reached = worker_stub(monkeypatch)
    report = spawn.run_manifest_preflight(
        manifest,
        workers=1,
        source_commit=manifest.source_sha if source == "HEAD" else "a" * 40,
    )
    if source == "HEAD":
        assert report["status"] == "valid", report
        assert len(reached) == 3, report
        assert all(len(seeds) == 30 for seeds in reached)
    else:
        assert reached == []
        assert "source" in report["input_error"]


def test_smoke_static_validation_stays_valid():
    result = protocol.validate_release_manifest(protocol.load_release_manifest(RELEASES / SMOKE))
    assert result["status"] == "valid", result["problems"]


def test_resolved_slice_matrix_carries_real_derived_dimensions(sealed_repository):
    # Exercise the public v0.2 freeze resolver; v0.1 has no independent schedule pin.
    manifest = materialize(sealed_repository, "slice")
    payload = protocol.build_resolved_release_manifest(manifest, repository_root=sealed_repository)
    assert payload["matrix"] == {
        "planner_arms": 14,
        "scenarios": 3,
        "seeds": 30,
        "expected_episode_cells": 1260,
        "horizon_steps": None,
        "scenario_horizons": "configs/benchmarks/horizon_schedules/three_width_doorway_release_0_0_8_authored_v1.yaml",
        "scenario_horizons_sha256": "e420f41636e59dd1169bf2da048c482ec78ba9ccf8ec5dc2721534c61c8191c9",
        "dt": 0.1,
    }
    cfg = protocol.load_release_campaign_config(manifest, repository_root=sealed_repository)
    assert protocol.resolve_release_horizon_budgets(manifest, cfg) == {
        "francis2023_narrow_doorway_width_2p20": 400,
        "francis2023_narrow_doorway_width_2p80": 400,
        "francis2023_narrow_doorway_width_3p60": 400,
    }
    assert "require source_sha equal to HEAD" in protocol.sealed_seed_execution_problem(
        replace(manifest, source_sha=None),
        manifest.resolved_seeds,
        repository_root=sealed_repository,
    )


# seed-holdout: synthetic-fixture end
