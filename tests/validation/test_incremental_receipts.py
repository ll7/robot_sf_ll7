"""Production-path regression controls for incremental receipt acquisition (#10446)."""

import json
from dataclasses import replace

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
