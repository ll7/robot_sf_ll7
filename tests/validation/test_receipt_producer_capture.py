"""Capture a fresh native primary seal before receipt allocation binding."""

import importlib
import json

import pytest

from scripts.ci.receipt_outcomes import raw_outcomes
from scripts.ci.receipt_producer_capture import ProducerCaptureError, install_capture
from scripts.ci.receipt_provenance import verify_producer_seals


@pytest.fixture
def native_campaign(tmp_path):
    """Use the actual MapRunner provenance writer, without running a simulation."""
    runner = importlib.import_module("robot_sf.benchmark.map_runner.map_runner")
    raw = tmp_path / "campaigns" / "open" / "episodes.jsonl"
    raw.parent.mkdir(parents=True)
    row = {
        "episode_id": "open-1001",
        "scenario_id": "open",
        "seed": 1001,
        "algo": "dummy",
        "config_hash": "fixture-config",
        "steps": 2,
        "metrics": {"success": False, "collisions": 0, "total_collision_count": 0},
    }
    raw.write_text(json.dumps(row) + "\n")
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}\n')
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("- name: open\n")
    arguments = {
        "summary": {},
        "out_path": raw,
        "episode_records": [row],
        "schema_path": schema,
        "scenario_path": scenario,
        "scenarios": [{"name": "open"}],
        "algo": "dummy",
        "algo_config_path": None,
        "benchmark_profile": "baseline-safe",
        "suite_key": "main",
        "total_jobs": 1,
        "written": 1,
        "horizon": 2,
        "dt": 0.1,
        "record_forces": False,
        "active_observation_mode": None,
        "active_observation_level": None,
    }
    return runner, raw, arguments


def test_fresh_native_primary_admits_both_required_seals(native_campaign):
    """Capture makes native fresh output satisfy unchanged two-seal admission."""
    runner, raw, arguments = native_campaign
    restore = install_capture(validate_raw_outcomes=raw_outcomes)
    try:
        runner._record_result_provenance_manifest(**arguments)
    finally:
        restore()
    assert raw.with_suffix(".jsonl.provenance.json").is_file()
    verify_producer_seals(raw.parents[2])


def _invoke(native_campaign):
    runner, _raw, arguments = native_campaign
    restore = install_capture(validate_raw_outcomes=raw_outcomes)
    try:
        runner._record_result_provenance_manifest(**arguments)
    finally:
        restore()


def test_changed_raw_byte_after_primary_refuses(native_campaign, monkeypatch):
    runner, raw, _arguments = native_campaign
    original = runner._record_result_provenance_manifest

    def changed(**kwargs):
        original(**kwargs)
        raw.write_bytes(raw.read_bytes() + b" ")

    monkeypatch.setattr(runner, "_record_result_provenance_manifest", changed)
    with pytest.raises(ProducerCaptureError, match="primary raw digest differs"):
        _invoke(native_campaign)
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


@pytest.mark.parametrize("field", ["success", "collisions", "total_collision_count", "steps"])
def test_coherently_sealed_missing_canonical_outcome_refuses(native_campaign, field):
    _runner, raw, arguments = native_campaign
    row = arguments["episode_records"][0]
    del (row if field == "steps" else row["metrics"])[field]
    raw.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=f"missing raw .*{field}"):
        _invoke(native_campaign)
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


@pytest.mark.parametrize("value", [False, "0", None])
def test_coherently_sealed_invalid_collision_type_refuses(native_campaign, value):
    _runner, raw, arguments = native_campaign
    row = arguments["episode_records"][0]
    row["metrics"]["collisions"] = value
    raw.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid raw metrics.collisions"):
        _invoke(native_campaign)
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


def test_existing_secondary_conflict_never_overwrites(native_campaign):
    _runner, raw, _arguments = native_campaign
    secondary = raw.with_suffix(".jsonl.manifest.json")
    secondary.write_bytes(b"existing conflict\n")
    with pytest.raises(ProducerCaptureError, match="secondary manifest exists"):
        _invoke(native_campaign)
    assert secondary.read_bytes() == b"existing conflict\n"


def test_native_swallowed_write_error_never_creates_secondary(native_campaign, monkeypatch):
    runner, raw, arguments = native_campaign

    def failed(**_kwargs):
        raise OSError("native write failed")

    monkeypatch.setattr(runner, "_emit_provenance_manifest", failed)
    with pytest.raises(ProducerCaptureError, match="native primary writer did not succeed"):
        _invoke(native_campaign)
    assert arguments["summary"]["provenance"]["result_manifest_status"] == "error"
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


def test_native_uncaught_exception_never_creates_secondary(native_campaign, monkeypatch):
    runner, raw, _arguments = native_campaign

    def failed(**_kwargs):
        raise RuntimeError("collector failed")

    monkeypatch.setattr(runner, "_record_result_provenance_manifest", failed)
    with pytest.raises(RuntimeError, match="collector failed"):
        _invoke(native_campaign)
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "partial", "count", "digest", "duplicate"])
def test_untrustworthy_primary_never_creates_secondary(native_campaign, monkeypatch, corruption):
    runner, raw, _arguments = native_campaign
    original = runner._record_result_provenance_manifest

    def corrupt(**kwargs):
        original(**kwargs)
        primary = raw.with_suffix(".jsonl.provenance.json")
        if corruption == "missing":
            primary.unlink()
            return
        payload = json.loads(primary.read_bytes())
        if corruption == "partial":
            payload["completeness"]["status"] = "partial"
        elif corruption == "count":
            payload["campaign_identity"]["written"] = True
        elif corruption == "digest":
            payload["raw_artifacts"][0]["sha256"] = "0" * 64
        else:
            primary.write_text('{"schema_version":"duplicate",' + primary.read_text()[1:])
            return
        primary.write_text(json.dumps(payload))

    monkeypatch.setattr(runner, "_record_result_provenance_manifest", corrupt)
    with pytest.raises(ValueError):
        _invoke(native_campaign)
    assert not raw.with_suffix(".jsonl.manifest.json").exists()


def test_capture_preserves_native_raw_and_primary_bytes(native_campaign, monkeypatch):
    runner, raw, _arguments = native_campaign
    original = runner._record_result_provenance_manifest
    recorded = {}

    def observe(**kwargs):
        original(**kwargs)
        recorded["raw"] = raw.read_bytes()
        recorded["primary"] = raw.with_suffix(".jsonl.provenance.json").read_bytes()

    monkeypatch.setattr(runner, "_record_result_provenance_manifest", observe)
    _invoke(native_campaign)
    assert raw.read_bytes() == recorded["raw"]
    assert raw.with_suffix(".jsonl.provenance.json").read_bytes() == recorded["primary"]
    verify_producer_seals(raw.parents[2])


@pytest.mark.parametrize("legacy_metric", [float("nan"), float("inf"), float("-inf")])
def test_legacy_nonfinite_metric_remains_sealed_and_unchanged(native_campaign, legacy_metric):
    _runner, raw, arguments = native_campaign
    row = arguments["episode_records"][0]
    row["metrics"]["legacy_diagnostic"] = legacy_metric
    raw.write_text(json.dumps(row) + "\n")
    before = raw.read_bytes()
    _invoke(native_campaign)
    assert raw.read_bytes() == before
    verify_producer_seals(raw.parents[2])


def test_capture_preserves_runtime_orca_and_declared_hybrid_labels(native_campaign):
    """Capture only: profile/source dispatch attestation belongs to admission."""
    _runner, raw, arguments = native_campaign
    row = arguments["episode_records"][0]
    row["algo"] = "orca"
    arguments["algo"] = "hybrid_rule_local_planner"
    raw.write_text(json.dumps(row) + "\n")
    before = raw.read_bytes()
    _invoke(native_campaign)
    primary = json.loads(raw.with_suffix(".jsonl.provenance.json").read_bytes())
    assert primary["campaign_identity"]["algorithm"] == "hybrid_rule_local_planner"
    assert raw.read_bytes() == before
    assert json.loads(raw.read_bytes())["algo"] == "orca"
    assert raw.with_suffix(".jsonl.manifest.json").is_file()


@pytest.mark.parametrize(
    ("arguments", "error", "reason"),
    [
        ({}, TypeError, "validate_raw_outcomes"),
        (
            {"validate_raw_outcomes": None},
            ProducerCaptureError,
            "canonical outcome validator is required",
        ),
    ],
)
def test_install_requires_explicit_validator(arguments, error, reason):
    with pytest.raises(error, match=reason):
        install_capture(**arguments)
