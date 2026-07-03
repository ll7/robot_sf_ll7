"""Tests for the issue #4205 static-constriction pre-registration checker."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
SCRIPT = REPO_ROOT / "scripts/validation/check_issue_4205_static_constriction_codesign_loop.py"

_SPEC = importlib.util.spec_from_file_location("_issue_4205_check", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_checked_in_preregistration_config_passes() -> None:
    """The tracked contract pre-registers the exact three-arm static-deadlock slice."""
    report = _MODULE.validate_config(_load_config())

    assert report["ok"] is True
    assert report["issue"] == 4205
    assert report["suite_id"] == "static_deadlock_recovery"
    assert report["scenario_ids"] == [
        "classic_bottleneck_low",
        "classic_head_on_corridor_low",
        "narrow_passage",
    ]
    assert report["seeds"] == [111, 112, 113]
    assert report["arm_keys"] == [
        "ppo_frozen",
        "ppo_frozen_wrapper_on",
        "ppo_frozen_cbf_on",
    ]
    assert report["compute_submit_authorized"] is False
    assert report["benchmark_evidence"] is False
    assert (
        report["benchmark_contract"]
        == "configs/research/issue_4205_static_constriction_codesign_loop_v1.yaml"
    )


def test_ppo_checkpoint_or_config_mismatch_fails_closed() -> None:
    """Every arm must share the same frozen PPO config identity."""
    config = _load_config()
    config["arms"][1]["algo_config"] = "configs/baselines/ppo.yaml"

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "algo_config must match frozen_ppo_lineage" in str(exc)
    else:
        raise AssertionError("mismatched PPO config must fail closed")


def test_wrapper_threshold_drift_fails_closed() -> None:
    """Wrapper-on threshold drift is rejected before campaign execution."""
    config = _load_config()
    config["arms"][1]["safety_wrapper"]["capped_speed_m_s"] = 0.25

    try:
        _MODULE.validate_config(config)
    except ValueError as exc:
        assert "predeclared ablation config" in str(exc)
    else:
        raise AssertionError("wrapper threshold drift must fail closed")


def test_unknown_cbf_arm_or_threshold_drift_fails_closed() -> None:
    """Unknown CBF arms and CBF threshold drift are rejected."""
    config = _load_config()
    config["arms"][2]["cbf_safety_filter"]["arm_key"] = "cbf_custom_on"

    try:
        _MODULE.validate_config(config)
    except ValueError as exc:
        assert "arm_key" in str(exc)
    else:
        raise AssertionError("unknown CBF arm must fail closed")

    config = _load_config()
    config["arms"][2]["cbf_safety_filter"]["alpha"] = 2.0
    try:
        _MODULE.validate_config(config)
    except ValueError as exc:
        assert "predeclared ablation config" in str(exc)
    else:
        raise AssertionError("CBF threshold drift must fail closed")


def test_missing_static_deadlock_suite_metadata_fails_before_campaign() -> None:
    """The contract must bind to the static-deadlock mechanism suite."""
    config = _load_config()
    config["mechanism_suite"]["suite_id"] = "guard_domination"

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "static_deadlock_recovery" in str(exc)
    else:
        raise AssertionError("wrong mechanism suite must fail closed")


def test_submit_authorization_cannot_be_enabled_in_public_config() -> None:
    """Tracked pre-registration must not authorize Slurm or GPU submission."""
    config = _load_config()
    config["campaign_authorization"]["compute_submit_authorized"] = True

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "compute_submit_authorized must stay false" in str(exc)
    else:
        raise AssertionError("public config must not authorize compute submit")


def test_benchmark_contract_drift_fails_closed(tmp_path: Path) -> None:
    """The paired benchmark contract must stay aligned with the research contract."""
    config = _load_config()
    benchmark = yaml.safe_load(
        (REPO_ROOT / config["benchmark_contract"]).read_text(encoding="utf-8")
    )
    benchmark["arms"] = ["ppo_frozen"]
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(benchmark), encoding="utf-8")
    config["benchmark_contract"] = str(benchmark_path)

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "benchmark_contract arms drifted" in str(exc)
    else:
        raise AssertionError("benchmark contract drift must fail closed")


def test_transient_routing_state_is_rejected() -> None:
    """Target-host or queue-route state belongs outside tracked pre-registration files."""
    config = _load_config()
    config["target_host"] = "do-not-track"

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "transient routing keys" in str(exc)
    else:
        raise AssertionError("tracked config must reject transient routing state")


def test_cli_writes_compact_cpu_smoke_manifest_without_raw_artifacts(tmp_path: Path) -> None:
    """The CLI emits compact CPU smoke metadata without raw artifacts."""
    smoke_path = tmp_path / "cpu_smoke.json"
    exit_code = _MODULE.main(
        [
            "--config",
            str(CONFIG),
            "--json",
            "--smoke-out",
            str(smoke_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot_sf.issue_4205.cpu_smoke_manifest.v1"
    assert payload["benchmark_evidence"] is False
    assert [row["arm_key"] for row in payload["rows"]] == [
        "ppo_frozen",
        "ppo_frozen_wrapper_on",
        "ppo_frozen_cbf_on",
    ]
    serialized = json.dumps(payload, sort_keys=True)
    forbidden_tokens = ["episodes.jsonl", ".mp4", "checkpoint", "slurm"]
    assert all(token not in serialized.lower() for token in forbidden_tokens)


def test_arms_remain_mutually_exclusive() -> None:
    """The first slice rejects wrapper and CBF composition in the same arm."""
    config = deepcopy(_load_config())
    config["arms"][2]["safety_wrapper"] = {"enabled": True, "arm_key": "wrapper_on"}

    try:
        _MODULE.validate_config(config)
    except _MODULE.ContractError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("wrapper and CBF composition is outside this slice")


def test_cli_verifies_cpu_smoke_evidence_and_writes_packet(tmp_path: Path) -> None:
    """Fixture-backed arm smoke writes only compact pre-run evidence files."""
    smoke_path = REPO_ROOT / "tests/fixtures/issue_4205_cpu_smoke_evidence.json"
    manifest_path = tmp_path / "cpu_smoke_verified.json"
    evidence_dir = tmp_path / "evidence"

    exit_code = _MODULE.main(
        [
            "--config",
            str(CONFIG),
            "--json",
            "--smoke-input",
            str(smoke_path),
            "--smoke-out",
            str(manifest_path),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["evidence_status"] == "cpu_arm_plumbing_smoke"
    assert payload["rows"][1]["wrapper_intervention_rate"] == 0.125
    assert payload["rows"][2]["cbf_status_counts"] == {"active": 1, "nominal": 0}

    expected_files = {
        "README.md",
        "metadata.json",
        "pre_registration.json",
        "intervention_summary.csv",
        "failure_mode_counts.csv",
        "claim_boundary.md",
        "SHA256SUMS",
    }
    assert {path.name for path in evidence_dir.iterdir()} == expected_files
    serialized = "\n".join(path.read_text(encoding="utf-8") for path in evidence_dir.iterdir())
    forbidden_tokens = ["episodes.jsonl", ".mp4", "checkpoint.zip", "slurm.log"]
    assert all(token not in serialized.lower() for token in forbidden_tokens)


def test_cpu_smoke_evidence_missing_cbf_status_fails_closed() -> None:
    """The CBF arm must emit CBF status metadata before evidence packet writing."""
    report = _MODULE.validate_config(_load_config())
    smoke_path = REPO_ROOT / "tests/fixtures/issue_4205_cpu_smoke_evidence.json"
    payload = json.loads(smoke_path.read_text(encoding="utf-8"))
    payload["rows"][2]["cbf_status_counts"] = {}

    try:
        _MODULE._validate_cpu_smoke_evidence(report, payload)
    except _MODULE.ContractError as exc:
        assert "CBF status counts" in str(exc)
    else:
        raise AssertionError("CBF smoke evidence without status counts must fail closed")
