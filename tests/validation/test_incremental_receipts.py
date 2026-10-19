"""Production-path regression controls for incremental receipt acquisition (#10446)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.validation import run_empty_world_sweep as sweep

REPO_ROOT = sweep.REPO_ROOT


def test_duplicate_development_seeds_refused_before_collection():
    """Duplicate seeds must not collapse into an apparently complete census."""
    with pytest.raises(sweep.SeedGuardError, match="duplicate"):
        sweep.assert_dev_seeds([1001, 1001])


def test_contract_preflight_refuses_legacy_metric_anchor(tmp_path, monkeypatch):
    """The pre-trajectory production preflight must check the anchor's meanings."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg_path, scenarios = sweep.build_derived_inputs(
        "main",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({"_metadata": {"metric_schema_version": "robot-sf-metrics.v1"}}))
    cfg = replace(load_campaign_config(cfg_path), snqi_baseline_path=anchor)
    monkeypatch.setattr(sweep, "assert_forecast_support", lambda cfg: None)
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        sweep._preflight_receipt_inputs([("main", cfg_path, scenarios, cfg)])


def test_contract_preflight_refuses_missing_scenario_seed_combination(tmp_path, monkeypatch):
    """Every selected scenario must carry the whole declared seed list."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg_path, scenarios = sweep.build_derived_inputs(
        "main",
        seeds=[1001, 1002],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    scenarios[0]["seeds"] = [1001]
    cfg = load_campaign_config(cfg_path)
    monkeypatch.setattr(sweep, "assert_forecast_support", lambda cfg: None)
    with pytest.raises(ValueError, match="missing.*combination"):
        sweep._preflight_receipt_inputs([("main", cfg_path, scenarios, cfg)])


def test_validated_writer_publishes_completion_before_jsonl_append(tmp_path):
    """A killed append must leave a validated atomically published episode."""
    from robot_sf.benchmark.map_runner.map_runner import _write_validated_to_handle

    descriptor = {
        "schema_version": "receipt-execution.v1",
        "source_sha": "a" * 40,
        "execution_key": "b" * 64,
        "run_id": "fixture",
    }
    (tmp_path / "receipt_execution.json").write_text(json.dumps(descriptor))
    row = {
        "episode_id": "episode-1",
        "scenario_id": "slot",
        "seed": 1001,
        "metrics": {"success": False, "collisions": 1},
        "outcome": {"collision_event": True},
    }

    class KilledAppend:
        name = str(tmp_path / "episodes.jsonl")

        def write(self, text):
            raise OSError("interrupted append")

    with pytest.raises(OSError, match="interrupted append"):
        _write_validated_to_handle(KilledAppend(), {}, row)
    records = list((tmp_path / "completed").glob("*.json"))
    assert len(records) == 1, "completed observations must survive interrupted JSONL writes"
    completion = json.loads(records[0].read_text())
    assert completion["row"] == row
    assert completion["source_sha"] == "a" * 40


def test_scheduler_pins_working_directory_and_checks_helpers(tmp_path, monkeypatch):
    """Batch spooling must not change helper resolution or ignore check-only."""
    from scripts.ci import make_behaviour_receipt as producer

    checkout = tmp_path / "source with space"
    checkout.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(producer, "_ensure_worktree", lambda *args: checkout)
    submitted = []
    monkeypatch.setattr(
        producer.subprocess,
        "check_output",
        lambda command, **kwargs: submitted.append(command) or "123\n",
    )
    producer._submit_sbatch_sweep(
        tmp_path,
        work,
        "head",
        "a" * 40,
        [1001],
        arms=["goal"],
        scenarios=None,
        workers=1,
    )
    assert any(str(arg).startswith("--chdir=") for arg in submitted[0]), (
        "sbatch must pin the immutable source even when its script is spooled"
    )
    script = (work / "behaviour_receipt_head.sbatch").read_text()
    assert "test -f" in script, "missing production helper must fail before a trajectory"
    assert 'export PYTHONPATH="$PWD:$PWD/fast-pysf' in script


def test_generated_campaign_enables_missing_only_resume(tmp_path):
    """The first full seed is retained by the same campaign, with no source substitution."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    path, _ = sweep.build_derived_inputs(
        "main",
        seeds=[1001, 1002],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    assert load_campaign_config(path).resume is True


def test_comparator_refuses_incompatible_row_meanings():
    """Comparison-only recovery must not silently pool old and new metric meanings."""
    from scripts.ci import make_behaviour_receipt as producer

    def rows(version):
        return {
            ("goal", "slot", seed): {
                "success": True,
                "collisions": 0,
                "metrics": {"metric_schema_version": version, "success": True, "collisions": 0},
            }
            for seed in producer.DEFAULT_SEEDS
        }

    with pytest.raises(ValueError, match="incompatible metric definitions"):
        producer.build_rows_and_classifications(
            rows("robot-sf-metrics.v2"),
            rows("robot-sf-metrics.v1"),
            {"arms": ["goal"], "maps": ["slot"]},
            classification_class="known_limitation",
            evidence_base_uri="https://example.org/evidence",
        )


def test_runtime_input_digest_mismatch_refused_before_collection(tmp_path):
    """Read-back assets must match the digest captured by the real resolver."""
    from robot_sf.benchmark.receipt_contract import input_digests

    asset = tmp_path / "map.svg"
    asset.write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="digest"):
        input_digests(tmp_path, [{"path": str(asset), "sha256": "0" * 64}])


def test_atomic_json_accepts_canonical_sequences(tmp_path):
    """The runtime package inventory must survive canonical JSON publication."""
    from robot_sf.benchmark.receipt_stages import atomic_json

    atomic_json(tmp_path / "record.json", {"packages": [("numpy", "1")]})
    assert json.loads((tmp_path / "record.json").read_bytes()) == {"packages": [["numpy", "1"]]}


def test_comparator_refuses_extra_episode_rows():
    """Unexpected rows cannot be silently discarded when constructing a full comparison."""
    from scripts.ci import make_behaviour_receipt as producer

    expected = {
        ("goal", "slot", seed): {"success": True, "collisions": 0}
        for seed in producer.DEFAULT_SEEDS
    }
    head = {**expected, ("goal", "unexpected", 1001): {"success": True, "collisions": 0}}
    with pytest.raises(ValueError, match="unexpected.*episode"):
        producer.build_rows_and_classifications(
            head,
            expected,
            {"arms": ["goal"], "maps": ["slot"]},
            classification_class="known_limitation",
            evidence_base_uri="https://example.org/evidence",
        )


def test_completed_collisions_and_timeouts_resume_without_retry(tmp_path, monkeypatch):
    """The atomic journal rehydrates completed failures, not just successful navigation."""
    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.benchmark.receipt_stages import atomic_json, publish_episode, recover_episodes

    descriptor = {
        "schema_version": "receipt-execution.v1",
        "source_sha": "a" * 40,
        "execution_key": "b" * 64,
        "run_id": "fixture",
    }
    path = tmp_path / "episodes.jsonl"
    atomic_json(tmp_path / "receipt_execution.json", descriptor, immutable=True)
    collision = {
        "episode_id": "collision",
        "scenario_id": "slot",
        "seed": 1001,
        "metrics": {"success": False, "collisions": 1},
        "outcome": {"collision_event": True},
    }
    timeout = {
        "episode_id": "timeout",
        "scenario_id": "slot",
        "seed": 1002,
        "metrics": {"success": False, "collisions": 0},
        "outcome": {"timeout_event": True},
    }
    assert publish_episode(path, {}, collision)
    assert publish_episode(path, {}, timeout)
    path.write_text('{"torn":')
    rows = recover_episodes(path, {}, descriptor)
    assert {row["episode_id"] for row in rows} == {"collision", "timeout"}
    ctx = SimpleNamespace(out_path=path, jobs=[({}, 1001), ({}, 1002), ({}, 1003)])
    monkeypatch.setattr(runner, "_compute_resume_identity_payload", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_compute_map_episode_id",
        lambda payload, seed: {1001: "collision", 1002: "timeout", 1003: "missing"}[seed],
    )
    runner._filter_resumed_jobs(ctx)
    assert ctx.jobs == [({}, 1003)]
    with pytest.raises(ValueError, match="same-source"):
        recover_episodes(path, {}, {**descriptor, "source_sha": "c" * 40})


def test_corrupt_completion_is_preserved_and_requeued(tmp_path):
    """A checksum failure preserves diagnosis and reruns only the invalid observation."""
    from robot_sf.benchmark.receipt_stages import atomic_json, publish_episode, recover_episodes

    descriptor = {
        "schema_version": "receipt-execution.v1",
        "source_sha": "a" * 40,
        "execution_key": "b" * 64,
        "run_id": "fixture",
    }
    path = tmp_path / "episodes.jsonl"
    atomic_json(tmp_path / "receipt_execution.json", descriptor, immutable=True)
    row = {"episode_id": "one", "scenario_id": "slot", "seed": 1001, "metrics": {"success": False}}
    publish_episode(path, {}, row)
    completion = next((tmp_path / "completed").glob("*.json"))
    record = json.loads(completion.read_bytes())
    record["row"]["seed"] = 1002
    completion.write_text(json.dumps(record))
    assert recover_episodes(path, {}, descriptor) == []
    assert len(list((tmp_path / "invalid").glob("*.json"))) == 1
    assert publish_episode(path, {}, row)
    assert recover_episodes(path, {}, descriptor) == [row]


def test_first_seed_dispatch_keeps_full_campaign_contract(tmp_path, monkeypatch):
    """A first-seed stage schedules one seed without changing the execution identity."""
    from robot_sf.benchmark.map_runner import map_runner as runner

    path = tmp_path / "episodes.jsonl"
    path.write_text("")
    descriptor = {
        "schema_version": "receipt-execution.v1",
        "source_sha": "a" * 40,
        "execution_key": "b" * 64,
        "run_id": "fixture",
    }
    (tmp_path / "receipt_execution.json").write_text(json.dumps(descriptor))
    (tmp_path / "receipt_dispatch.json").write_text(
        json.dumps({"execution_key": "b" * 64, "seeds": [1001]})
    )
    ctx = SimpleNamespace(out_path=path, jobs=[({}, 1001), ({}, 1002)])
    monkeypatch.setattr(runner, "_compute_resume_identity_payload", lambda *args: {})
    monkeypatch.setattr(runner, "_compute_map_episode_id", lambda payload, seed: str(seed))
    runner._filter_resumed_jobs(ctx)
    assert ctx.jobs == [({}, 1001)]


def test_paired_submission_requires_compatible_preflight_contracts(tmp_path, monkeypatch):
    """The long paired submission cannot race ahead of its schema/contract preflight."""
    from scripts.ci import make_behaviour_receipt as producer

    args = producer._parse_args(
        [
            "--head-sha",
            "a" * 40,
            "--receipt-id",
            "fixture",
            "--mode",
            "sbatch",
            "--work-dir",
            str(tmp_path),
        ]
    )
    submitted = []
    monkeypatch.setattr(
        producer, "_submit_sbatch_sweep", lambda *args, **kwargs: submitted.append(True)
    )
    with pytest.raises(ValueError, match="preflight contract"):
        producer._prepare_sweeps(args, tmp_path, "b" * 40)
    assert submitted == []


def test_banked_arm_without_summary_rebuilds_reporting(tmp_path, monkeypatch):
    """A kill after the last row must not strand the arm behind skip-complete."""
    from robot_sf.benchmark.camera_ready import _resume_plan as plan

    directory = tmp_path / "goal__differential_drive"
    directory.mkdir()
    (directory / "episodes.jsonl").write_text(
        json.dumps({"scenario_id": "slot", "seed": 1001}) + "\n"
    )
    (directory / "receipt_execution.json").write_text("{}")
    monkeypatch.setattr(plan, "_validate_resume_runtime_identities", lambda *args: None)
    verdict = plan.build_resume_plan(
        tmp_path,
        planners=[{"key": "goal", "algo": "goal"}],
        kinematics_matrix=["differential_drive"],
        scenarios=[
            {"name": "slot", "seeds": [1001], "simulation_config": {"max_episode_steps": 2}}
        ],
    )[0]
    assert verdict.episodes_remaining == 0
    assert verdict.verdict != "skip-complete", (
        "atomic rows require summary reconstruction after a killed writer"
    )


def test_receipt_resume_summary_includes_banked_observations(tmp_path, monkeypatch):
    """Resume must aggregate retained rows together with newly executed rows."""
    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.benchmark.map_runner.map_runner_batch_runner import BatchExecutionResult

    path = tmp_path / "episodes.jsonl"
    rows = [
        {
            "episode_id": str(seed),
            "scenario_id": "slot",
            "seed": seed,
            "metrics": {"success": False},
        }
        for seed in [1001, 1002]
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / "receipt_execution.json").write_text("{}")
    ctx = SimpleNamespace(
        out_path=path,
        schema={},
        jobs=[],
        filtered=[{"name": "slot", "seeds": [1001, 1002]}],
        suite_seeds={},
        suite_key="fixture",
    )
    monkeypatch.setattr(
        runner,
        "_dispatch_batch_execution",
        lambda ctx: BatchExecutionResult(0, [], [], 0, 0, False, None, {}, 0.0, None),
    )
    monkeypatch.setattr(
        runner,
        "_build_final_summary",
        lambda ctx, result: {
            "ids": {row["episode_id"] for row in result.episode_records},
            "written": result.wrote,
            "expected": len(ctx.jobs),
        },
    )
    result = runner._execute_and_summarize(ctx)
    assert result == {"ids": {"1001", "1002"}, "written": 2, "expected": 2}


def test_canary_passes_admitted_short_scenario_to_worker(tmp_path, monkeypatch):
    """Canary shortening must use supported scenario fields before any reset or step."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.benchmark.map_runner import map_runner as runner
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario
    from scripts.validation.receipt_canary import run_canary

    path, scenarios = sweep.build_derived_inputs(
        "main",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=["classic_bottleneck_low"],
        workers=1,
        out_dir=tmp_path / "inputs",
        step_trace=True,
    )

    class ReachedConstructor(Exception):
        pass

    def worker(scenario, seed, **kwargs):
        assert scenario["name"].startswith("canary/")
        build_robot_config_from_scenario(scenario, scenario_path=kwargs["scenario_path"])
        raise ReachedConstructor

    monkeypatch.setattr(runner, "_run_map_episode", worker)
    with pytest.raises(ReachedConstructor):
        run_canary(
            [("main", path, scenarios, load_campaign_config(path))],
            tmp_path / "canary",
            "a" * 40,
            2,
        )


def test_fresh_receipt_does_not_masquerade_as_resumed_campaign(tmp_path):
    """Execution descriptors must not create prior arms before a real campaign manifest."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.benchmark.receipt_contract import prepare_completion_views

    path, scenarios = sweep.build_derived_inputs(
        "main",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=["classic_bottleneck_low"],
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    prepare_completion_views(
        [("main", path, scenarios, load_campaign_config(path))],
        tmp_path,
        {"source_sha": "a" * 40, "execution_key": "b" * 64},
    )
    assert not (tmp_path / "campaigns/empty_world_main/runs").exists(), (
        "fresh preflight must not require a missing campaign_manifest.json"
    )


def test_contract_uses_production_batch_preflight_gates(tmp_path, monkeypatch):
    """Construction checks cannot bypass readiness and per-scenario compatibility."""
    from robot_sf.benchmark import receipt_contract
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.benchmark.map_runner import map_runner as runner

    path, scenarios = sweep.build_derived_inputs(
        "main",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=["classic_bottleneck_low"],
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    monkeypatch.setattr(receipt_contract, "inspect_construction", lambda *args, **kwargs: {})

    def reject(ctx):
        raise RuntimeError("production profile rejection")

    monkeypatch.setattr(runner, "_run_preflight_pipeline", reject)
    with pytest.raises(RuntimeError, match="production profile rejection"):
        receipt_contract.compile_contract(
            [("main", path, scenarios, load_campaign_config(path))],
            source_sha="a" * 40,
            root=REPO_ROOT,
            output_dir=tmp_path,
            minimum_free_bytes=0,
            full_roster=False,
        )


def _comparison_fixture(tmp_path, monkeypatch, *, mismatched_definitions=False):
    from robot_sf.benchmark.receipt_stages import stage_key
    from scripts.validation import compare_behaviour_rows as comparator

    head_dir, baseline_dir = tmp_path / "head", tmp_path / "baseline"
    data = {}
    for directory, source in ((head_dir, "a" * 40), (baseline_dir, "b" * 40)):
        directory.mkdir()
        definitions = ("2" if mismatched_definitions and directory == baseline_dir else "1") * 64
        execution = {"source_sha": source, "metric_definitions_sha256": definitions}
        (directory / "contract.json").write_text(
            json.dumps(
                {
                    "schema_version": "receipt-contract.v1",
                    "source_sha": source,
                    "execution_key": stage_key("execution", execution),
                    "execution": execution,
                    "metric_schema_version": "robot-sf-metrics.v2",
                }
            )
        )
        data[directory] = {
            ("goal", "slot", seed): {
                "git_hash": source,
                "metrics": {
                    "metric_schema_version": "robot-sf-metrics.v2",
                    "success": True,
                    "collisions": 0,
                },
            }
            for seed in range(1001, 1031)
        }
    monkeypatch.setattr(comparator.producer, "_load_sweep_rows", lambda directory: data[directory])
    return comparator, head_dir, baseline_dir


def test_comparison_refuses_same_schema_different_metric_definitions(tmp_path, monkeypatch):
    """An unchanged version label cannot conceal changed scoring meanings."""
    comparator, head, baseline = _comparison_fixture(
        tmp_path, monkeypatch, mismatched_definitions=True
    )
    with pytest.raises(ValueError, match="metric definitions"):
        comparator.compare(
            head, baseline, {"arms": ["goal"], "maps": ["slot"]}, tmp_path / "comparison.json"
        )


def test_comparison_preserves_original_scoring_source(tmp_path, monkeypatch):
    """A newer comparison checkout must not relabel the original scoring source."""
    comparator, head, baseline = _comparison_fixture(tmp_path, monkeypatch)
    result = comparator.compare(
        head, baseline, {"arms": ["goal"], "maps": ["slot"]}, tmp_path / "comparison.json"
    )
    assert result["scoring_keys"]["head"] != result["scoring_keys"]["baseline"], (
        "different producing sources must retain separate execution-time scoring identities"
    )
    assert result.get("scoring_contracts", {}).get("head", {}).get("source_sha") == "a" * 40


def test_nested_forecast_adapter_is_observable():
    """A foresight encoder's nested real adapter cannot disappear from capability checks."""
    from robot_sf.benchmark.receipt_contract import _loaded_predictors
    from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter

    predictor = object.__new__(PredictionPlannerAdapter)
    policy = SimpleNamespace(_predictive_foresight=SimpleNamespace(_adapter=predictor))
    assert _loaded_predictors(policy) == [predictor]


def test_effective_readback_observes_closure_planner_defaults():
    """Closure-only learned planners expose constructed defaults rather than authored metadata."""
    from dataclasses import dataclass

    from robot_sf.benchmark.receipt_contract import _policy_readback

    @dataclass
    class Config:
        max_speed: float = 7.0

    planner = SimpleNamespace(config=Config())

    def policy(obs):
        return planner.config.max_speed

    observed = _policy_readback(policy)
    assert any(
        item["parameters"].get("max_speed") == 7.0 for item in observed["owned_configurations"]
    )


def test_native_goal_readback_includes_instantiated_kinematics():
    """The real native policy must expose constructed speed/command limits."""
    from robot_sf.benchmark.map_runner.map_runner import _build_policy
    from robot_sf.benchmark.receipt_contract import _policy_readback

    policy, _ = _build_policy("goal", {}, robot_kinematics="differential_drive")
    observed = _policy_readback(policy)
    assert any(
        item["parameters"].get("max_linear_speed") is not None
        for item in observed["owned_configurations"]
    )


def test_completion_preserves_unbounded_empty_world_metric(tmp_path):
    """Infinite empty-world clearance is losslessly preserved without invalid JSON records."""
    import math

    from robot_sf.benchmark.receipt_stages import atomic_json, publish_episode, recover_episodes

    descriptor = {
        "schema_version": "receipt-execution.v1",
        "source_sha": "a" * 40,
        "execution_key": "b" * 64,
        "run_id": "fixture",
    }
    path = tmp_path / "episodes.jsonl"
    atomic_json(tmp_path / "receipt_execution.json", descriptor, immutable=True)
    row = {
        "episode_id": "one",
        "scenario_id": "slot",
        "seed": 1001,
        "metrics": {"success": False, "clearing_distance_min": float("inf")},
    }
    assert publish_episode(path, {}, row)
    restored = recover_episodes(path, {}, descriptor)
    assert len(restored) == 1
    assert math.isinf(restored[0]["metrics"]["clearing_distance_min"])
    completion_bytes = next((tmp_path / "completed").glob("*.json")).read_text()
    assert "Infinity" not in completion_bytes


def test_guarded_child_planner_defaults_are_read_back():
    """Guard fallback/prior planners have independent effective configurations."""
    from dataclasses import dataclass

    from robot_sf.benchmark.receipt_contract import _policy_readback

    @dataclass
    class Config:
        limit: int

    guard = SimpleNamespace(
        config=Config(99),
        fallback_adapter=SimpleNamespace(config=Config(11)),
        prior_adapter=SimpleNamespace(config=Config(22)),
    )
    observed = _policy_readback(SimpleNamespace(_planner_adapter=guard))
    limits = {item["parameters"].get("limit") for item in observed["owned_configurations"]}
    assert {11, 22, 99} <= limits


def test_pinned_harness_installs_bounded_acquisition_bundle(tmp_path):
    """Historical runtime policy stays byte-identical while acquisition APIs are overlaid."""
    import subprocess

    from scripts.ci import make_behaviour_receipt as producer

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    runtime_paths = [
        "robot_sf/benchmark/map_runner/map_runner.py",
        "robot_sf/benchmark/map_runner/map_runner_batch_runner.py",
        "robot_sf/benchmark/camera_ready/_resume_plan.py",
    ]
    script = repo / "scripts/validation/run_empty_world_sweep.py"
    script.parent.mkdir(parents=True)
    script.write_text("# old harness\n")
    for relative in runtime_paths:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ORIGINAL_POLICY = 1\n")
    git("add", "scripts", "robot_sf")
    git("commit", "-m", "runtime")
    baseline = git("rev-parse", "HEAD")
    script.write_text("# new harness\n")
    patched = repo / runtime_paths[0]
    patched.write_text("ORIGINAL_POLICY = 1\nPREFLIGHT_ONLY = True\n")
    patch = subprocess.check_output(["git", "diff", "--", runtime_paths[0]], cwd=repo)
    (repo / "scripts/validation/receipt_acquisition.patch").write_bytes(patch)
    modules = [
        "robot_sf/benchmark/receipt_stages.py",
        "robot_sf/benchmark/receipt_contract.py",
        "scripts/validation/receipt_canary.py",
        "scripts/validation/receipt_harness.py",
        "scripts/validation/compare_behaviour_rows.py",
    ]
    for relative in modules:
        (repo / relative).write_text("# acquisition module\n")
    git("add", "scripts", "robot_sf")
    git("commit", "-m", "acquisition bundle")
    harness = git("rev-parse", "HEAD")
    work = tmp_path / "work"
    checkout = producer._ensure_worktree(repo, work, "baseline", baseline)
    producer._install_sweep_harness(repo, checkout, tmp_path / "out", baseline, harness)
    assert (checkout / modules[0]).is_file(), "single-script pins omit required acquisition modules"
    assert (
        checkout / runtime_paths[0]
    ).read_text() == "ORIGINAL_POLICY = 1\nPREFLIGHT_ONLY = True\n"
    manifest = json.loads((tmp_path / "out/harness.json").read_bytes())
    assert manifest.get("bundle_files"), "every installed acquisition byte must be source-bound"
    from scripts.validation.receipt_harness import verify_bundle

    assert runtime_paths[0] in verify_bundle(checkout, tmp_path / "out", baseline)
    (checkout / runtime_paths[0]).write_text("ORIGINAL_POLICY = 2\nPREFLIGHT_ONLY = True\n")
    with pytest.raises(ValueError, match="differs from committed overlay"):
        verify_bundle(checkout, tmp_path / "out", baseline)
