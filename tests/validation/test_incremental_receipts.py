"""Production-path regression controls for incremental receipt acquisition (#10446)."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.validation import run_empty_world_sweep as sweep


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
