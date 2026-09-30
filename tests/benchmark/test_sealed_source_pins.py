"""Source-byte forgeries at real release entry points; no simulation is executed."""
# seed-holdout: synthetic-fixture begin

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.benchmark import spawn_preflight as spawn
from robot_sf.benchmark.camera_ready import _config as campaign_config
from robot_sf.benchmark.camera_ready import _run_state as campaign_paths
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from scripts.benchmark import preflight_spawn_clearance as standalone
from scripts.tools import resolve_benchmark_release_identity as identity_cli
from scripts.tools import run_benchmark_release as runner

ROOT = Path(__file__).resolve().parents[2]
RELEASES = "configs/benchmarks/releases"
MAIN_TEMPLATE = "benchmark_data_release_s30_h600.template.yaml"
SLICE_TEMPLATE = "three_width_doorway_release_0_0_8_v1.template.yaml"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture
def sealed_repository(tmp_path, monkeypatch):
    """Commit real repository inputs, including templates, in an isolated freeze."""
    repo = tmp_path / "source"
    repo.mkdir()
    shutil.copytree(ROOT / "configs", repo / "configs")
    shutil.copytree(ROOT / "maps", repo / "maps")
    (repo / "docs").mkdir()
    shutil.copy2(ROOT / "docs/RELEASE.md", repo / "docs/RELEASE.md")
    shutil.copy2(ROOT / "CITATION.cff", repo / "CITATION.cff")
    (repo / ".gitignore").write_text("output/\n")
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Sealed Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-qm", "initialize")
    git(repo, "add", "configs", "maps", "docs/RELEASE.md", "CITATION.cff")
    git(repo, "commit", "-qm", "freeze real inputs")
    for module in (protocol, spawn, runner, campaign_config, campaign_paths):
        monkeypatch.setattr(module, "get_repository_root", lambda: repo)
    monkeypatch.setattr(runner, "_current_source_commit", lambda: git(repo, "rev-parse", "HEAD"))
    return repo


def materialize(repo, kind):
    """Use the same public generate command and verified loader for both identities."""
    head = git(repo, "rev-parse", "HEAD")
    output = repo / f"output/{kind}/release_identity.resolved.json"
    template = MAIN_TEMPLATE if kind == "main" else SLICE_TEMPLATE
    assert (
        identity_cli.main(
            [
                "generate",
                "--template",
                str(repo / RELEASES / template),
                "--output",
                str(output),
                "--source-commit",
                head,
                "--release-tag",
                f"{kind}-0.0.8-{head}",
                "--concept-doi",
                "10.5281/zenodo.99000001",
                "--version-doi",
                "10.5281/zenodo.99000002",
                "--repository-root",
                str(repo),
            ]
        )
        == 0
    )
    return protocol.load_release_manifest(output, repository_root=repo)


class EnvironmentAttempt(BaseException):
    """An environment attempt must escape generic exception handling."""


class CampaignReached(BaseException):
    """Stop the base witness before a real campaign can start."""


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def abort(*_args, **_kwargs):
        raise EnvironmentAttempt("environment construction attempted")

    import robot_sf.gym_env.environment_factory as factory
    from robot_sf.gym_env.robot_env import RobotEnv

    for name in vars(factory):
        if name.startswith("make_") and callable(getattr(factory, name)):
            monkeypatch.setattr(factory, name, abort)
    monkeypatch.setattr(RobotEnv, "__init__", abort)
    monkeypatch.setattr(spawn, "make_robot_env", abort)
    monkeypatch.setattr(runner, "check_orca_rvo2_preflight", lambda *_a, **_kw: None)


def worker_stub(monkeypatch):
    reached = []

    def record(job):
        scenario, _matrix, seeds, *_rest = job
        reached.append(list(seeds))
        checks = ("reset_clearance", "footprint_reachability", "passage_width", "respawn_safety")
        return {
            "scenario": scenario["name"],
            "map_warnings": [],
            "rows": [
                {
                    "scenario": scenario["name"],
                    "seed": seed,
                    "overall_status": "valid",
                    **{key: {"status": "pass", "reason": "stub"} for key in checks},
                }
                for seed in seeds
            ],
        }

    class InlinePool:
        def __init__(self, *_a, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def map(self, fn, jobs):
            return [fn(job) for job in jobs]

    monkeypatch.setattr(spawn, "_check_release_scenario", record)
    monkeypatch.setattr(spawn, "ProcessPoolExecutor", InlinePool)
    return reached


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain_payload(repo, kind):
    """Materialize main; make the reviewer's handwritten v0.2 slice on the base too."""
    main = materialize(repo, "main")
    envelope = json.loads(main.path.read_text())
    template = repo / RELEASES / MAIN_TEMPLATE
    payload, *_ = protocol._identity_template_payload(template, repository_root=repo)
    values = protocol._materialize_release_template_payload(
        payload,
        template_path=template,
        metadata_path=main.metadata_path,
        metadata_sha256=main.metadata_sha256,
        source_commit=main.source_sha,
        latest_main_base_commit=main.latest_main_base_commit,
        release_tag=main.release_tag,
        concept_doi=envelope["publication"]["concept_doi"],
        version_doi=envelope["publication"]["version_doi"],
        repository_root=repo,
    )
    if kind == "main":
        return values
    historical = repo / RELEASES / "three_width_doorway_release_0_0_8_v1.yaml"
    result = yaml.safe_load(historical.read_text())
    for key in ("canonical_campaign_config", "citation_path", "release_checklist_path"):
        result[key] = str((historical.parent / result[key]).resolve())
    result["scenario"]["matrix_path"] = str(
        (historical.parent / result["scenario"]["matrix_path"]).resolve()
    )
    result["seed_policy"]["seed_sets_path"] = str(repo / "configs/benchmarks/seed_sets_0_0_8.yaml")
    for key in ("snqi_weights_path", "snqi_baseline_path"):
        result["metrics"][key] = str((historical.parent / result["metrics"][key]).resolve())
    for key in ("schema_version", "source_sha", "latest_main_base_commit", "publication"):
        result[key] = values[key]
    for key in (
        "suite_policy_path",
        "suite_policy_sha256",
        "route_certification_path",
        "route_certification_sha256",
    ):
        result["scenario"][key] = values["scenario"][key]
    result["matrix"] = {"expected_episode_cells": 1260, "horizon_steps": 600}
    result["metrics"]["snqi_claim_policy"] = "advisory_no_ranking"
    # v0.2 requires DOI agreement; the config's DOI is filled in the forged copy below.
    result["provenance"]["doi"] = values["provenance"]["doi"]
    return result


def forge(repo, case):
    payload = plain_payload(repo, "slice" if case == "F2b" else "main")
    real_config = Path(payload["canonical_campaign_config"])
    real_matrix = Path(payload["scenario"]["matrix_path"])
    directory = repo / ("configs/benchmarks/untracked" if case == "F1f" else f"output/forge/{case}")
    directory.mkdir(parents=True)
    config = yaml.safe_load(real_config.read_text())
    config["release_tag"] = payload["release_tag"]
    config["doi"] = payload["provenance"]["doi"]
    if case == "F1c":
        config["bootstrap_samples"] = 7
        social = next(p for p in config["planners"] if p["key"] == "social_force")
        social["algo_config"] = (
            "configs/algos/social_force_resolution_independent_v2_kernel_wrapped_v2.yaml"
        )
    else:
        matrix = directory / real_matrix.name
        matrix.write_text(
            yaml.safe_dump(
                {
                    "includes": [str(real_matrix)],
                    "scenario_overrides": {"simulation_config": {"ped_density": 0.0}},
                }
            )
        )
        config["scenario_matrix"] = str(matrix)
        payload["scenario"].update(matrix_path=str(matrix), matrix_sha256=sha(matrix))
    config_path = directory / real_config.name
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if case != "F1c":
        cfg = protocol.load_campaign_config(config_path, repository_root=repo)
        scenarios = _load_campaign_scenarios(cfg)
        assert {s["simulation_config"]["ped_density"] for s in scenarios} == {0.0}
    payload.update(
        canonical_campaign_config=str(config_path), campaign_config_sha256=sha(config_path)
    )
    manifest = repo / f"output/{case}.json"
    manifest.write_text(json.dumps(payload))
    return manifest


@pytest.mark.parametrize("case", ["F1c", "F1d", "F1f", "F2b"])
@pytest.mark.parametrize("entry", ["preflight", "run", "standalone"])
def test_forged_sealed_inputs_refused_before_workers(
    sealed_repository, monkeypatch, capsys, case, entry
):
    repo = sealed_repository
    path = forge(repo, case)
    capsys.readouterr()
    reached = worker_stub(monkeypatch)
    # The copied matrix changes relative map origins. Keep this unrelated static
    # map oracle out of the source-pin witness, as all clearance workers are stubs.
    monkeypatch.setattr(spawn, "_verified_main_grid_probe", lambda *_a: True)

    def campaign(*_a, **_kw):
        raise CampaignReached("base admitted a forged campaign")

    monkeypatch.setattr(runner, "run_campaign", campaign)
    monkeypatch.setattr(
        runner,
        "prepare_campaign_preflight",
        lambda *_a, **_kw: {
            "campaign_id": "stub",
            "campaign_root": "stub",
            "validate_config_path": "stub",
            "preview_scenarios_path": "stub",
            "matrix_summary_json_path": "stub",
            "matrix_summary_csv_path": "stub",
            "checkpoint_preflight_summary": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "validate_checkpoint_staging_receipt",
        lambda *_a, **_kw: {"generated_at_utc": "stub"},
    )
    monkeypatch.setattr(runner, "validate_runtime_smoke_result", lambda *_a, **_kw: {})
    monkeypatch.setattr(runner, "_admit_release_resume", lambda **_kw: None)
    receipt = repo / "output/receipt.json"
    receipt.write_text("{}\n")
    try:
        if entry == "standalone":
            rc = standalone.main(
                [
                    "--manifest",
                    str(path),
                    "--workers",
                    "1",
                    "--json-output",
                    str(repo / "output/spawn.json"),
                    "--markdown-output",
                    str(repo / "output/spawn.md"),
                ]
            )
        else:
            args = [
                "--manifest",
                str(path),
                "--mode",
                entry,
                "--output-root",
                str(repo / "output/run"),
            ]
            if entry == "run":
                args += [
                    "--checkpoint-receipt",
                    str(receipt),
                    "--runtime-smoke-receipt",
                    str(receipt),
                ]
            rc = runner.main(args)
    except CampaignReached:
        rc = 0
    output = capsys.readouterr().out
    assert reached == [], f"forged {case} reached {len(reached)} spawn workers"
    assert rc == 2, output
    assert "canonical repository path" in output, output


@pytest.mark.parametrize("kind,cells,workers", [("main", 20160, 48), ("slice", 1260, 3)])
def test_real_materialized_identity_passes_frozen_guard(
    sealed_repository, monkeypatch, kind, cells, workers
):
    repo = sealed_repository
    manifest = materialize(repo, kind)
    validation = protocol.validate_release_manifest(manifest, repository_root=repo)
    assert validation["status"] == "valid", validation
    assert manifest.expected_episode_cells == cells
    spawn.guard_manifest_execution(
        manifest, source_commit=git(repo, "rev-parse", "HEAD"), repository_root=repo
    )
    reached = worker_stub(monkeypatch)
    report = spawn.run_manifest_preflight(manifest, workers=1, source_commit=manifest.source_sha)
    assert report["status"] == "valid", report
    assert len(reached) == workers
    assert all(len(seeds) == 30 for seeds in reached)


@pytest.mark.parametrize(
    "input_kind",
    [
        "campaign",
        "matrix",
        "matrix-include",
        "seeds",
        "planner",
        "planner-base",
        "untracked-planner",
    ],
)
def test_canonical_inputs_must_equal_source_blobs(sealed_repository, input_kind):
    repo = sealed_repository
    manifest = materialize(repo, "main")
    cfg = protocol.load_release_campaign_config(manifest, repository_root=repo)
    planner_path = next(p.algo_config_path for p in cfg.planners if p.algo_config_path is not None)
    path = {
        "campaign": manifest.canonical_campaign_config_path,
        "matrix": manifest.scenario_matrix_path,
        "matrix-include": repo
        / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_route_width_v4.yaml",
        "seeds": cfg.seed_policy.seed_sets_path,
        "planner": planner_path,
        "planner-base": repo / "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
        "untracked-planner": planner_path,
    }[input_kind]
    if input_kind == "untracked-planner":
        git(repo, "rm", "--cached", str(path.relative_to(repo)))
    else:
        path.write_bytes(path.read_bytes() + b"\n# changed after freeze\n")
    problem = protocol.sealed_seed_execution_problem(
        manifest, tuple(manifest.resolved_seeds), repository_root=repo
    )
    assert problem is not None, f"{input_kind} mutation accepted at frozen source"
    assert ("not tracked" if input_kind == "untracked-planner" else "bytes differ") in problem


def test_planner_symlink_cannot_substitute_another_tracked_blob(sealed_repository):
    repo = sealed_repository
    manifest = materialize(repo, "main")
    cfg = protocol.load_release_campaign_config(manifest, repository_root=repo)
    config = next(p.algo_config_path for p in cfg.planners if p.key == "prediction_planner")
    replacement = repo / "configs/algos/risk_dwa_camera_ready.yaml"
    config.unlink()
    config.symlink_to(replacement)
    problem = protocol.sealed_seed_execution_problem(
        manifest, tuple(manifest.resolved_seeds), repository_root=repo
    )
    assert problem is not None, "planner symlink substituted another tracked source blob"
    assert "symlink" in problem


# seed-holdout: synthetic-fixture end
