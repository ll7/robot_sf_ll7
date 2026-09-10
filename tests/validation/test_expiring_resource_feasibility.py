"""Focused contract tests for the expiring-resource feasibility check (#8905)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.validation import check_expiring_resource_feasibility as tool

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "expiring_resource_feasibility" / "cases.json"
TOOL = ROOT / "scripts" / "validation" / "check_expiring_resource_feasibility.py"
CASES = tool.load_case_manifests(FIXTURE)
CASE_CONTRACT = [
    ("known_fit", "fits_conservative", "", 0),
    ("conservative_failure", "fits_expected", "", 0),
    ("unknown_start", "unknown", "queue_start_unknown", 2),
    ("stale_deadline_source", "unknown", "expired_deadline_evidence,stale_deadline_source", 2),
    ("zero_reserve", "unknown", "zero_reserve", 2),
    ("timezone_boundaries", "too_late", "", 1),
]
PLANNERS = [{"key": "orca", "algo": "orca"}]


def _case(name: str = "known_fit") -> dict[str, Any]:
    manifest = json.loads(json.dumps(CASES[name]))
    manifest["manifest_id"] = "synthetic"
    return manifest


MISSING_BASIS = _case()
del MISSING_BASIS["expiring_resource"]["output"]["basis"]


def _with(path: str, value: Any) -> dict[str, Any]:
    """Return the known-fit manifest with one dotted contract path overridden."""
    manifest = _case()
    keys = path.split(".")
    node = manifest["expiring_resource"]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    return manifest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOL), "--manifest", str(FIXTURE), "--check", *args]
    return subprocess.run(command, capture_output=True, text=True, check=False, cwd=ROOT)


@pytest.mark.parametrize(("name", "verdict", "codes", "exit_code"), CASE_CONTRACT)
def test_fixture_cases_match_contract_end_to_end(
    name: str, verdict: str, codes: str, exit_code: int
) -> None:
    report = tool.evaluate_manifest(CASES[name])
    assert report.verdict == verdict
    assert set(report.reason_codes) == set(filter(None, codes.split(",")))
    assert report.blocking is (verdict not in tool.POSITIVE_VERDICTS)
    result = _run_cli("--case", name, "--json")
    assert (result.returncode, json.loads(result.stdout)["verdict"]) == (exit_code, verdict)


def test_cli_repeat_runs_are_byte_stable() -> None:
    first = _run_cli("--case", "known_fit", "--json")
    assert first.stdout == _run_cli("--case", "known_fit", "--json").stdout


def test_historical_and_no_expiry_manifests_are_not_applicable() -> None:
    legacy = tool.evaluate_manifest(
        {"manifest_id": "legacy", "generated_at": "2026-09-10T08:00:00Z"}
    )
    assert (legacy.verdict, legacy.applicable, legacy.blocking) == ("unknown", False, False)
    assert legacy.reason_codes == ("contract_absent",)
    report = tool.evaluate_manifest(_with("deadline.kind", "none"))
    assert report.applicable is False and report.reason_codes == ("no_declared_expiry",)


@pytest.mark.parametrize(
    ("manifest", "code"),
    [
        (_with("deadline.timestamp", "2026-09-15T06:00:00"), "naive_timestamp"),
        (_with("deadline.timestamp", "2026-09-15T06:00:00+15:00"), "incompatible_timezone"),
        (_with("reserves.retrieval_seconds", -1), "negative_reserve"),
        (_with("runtime.conservative_seconds", 3600), "runtime_estimate_conflict"),
        (_with("reserves.retrieval_seconds", 10), "retrieval_reserve_insufficient"),
        (
            _with("latest_safe_submission", "2026-09-15T00:00:00Z"),
            "latest_safe_submission_conflict",
        ),
        (MISSING_BASIS, "missing_output_size_basis"),
        (
            _with("queue_start", {"status": "running", "start_timestamp": "2026-09-11T00:00:00Z"}),
            "queue_start_contradiction",
        ),
    ],
)
def test_fail_closed_evidence_and_contradiction_codes(manifest: dict[str, Any], code: str) -> None:
    report = tool.evaluate_manifest(manifest)
    assert code in report.reason_codes and report.verdict == "unknown" and report.blocking


def test_private_source_values_are_rejected_and_never_echoed() -> None:
    secret = "user@private-host.example/srv/mount"
    report = tool.evaluate_manifest(_with("deadline.source", secret))
    assert "missing_deadline_source" in report.reason_codes
    assert secret not in json.dumps(report.to_dict())


def test_timezone_boundary_equals_same_instant_in_utc() -> None:
    offset = tool.evaluate_manifest(CASES["timezone_boundaries"])
    utc = _case("timezone_boundaries")
    utc["expiring_resource"]["deadline"]["timestamp"] = "2026-09-14T12:00:00Z"
    equivalent = tool.evaluate_manifest(utc)
    assert offset.verdict == equivalent.verdict == "too_late"
    assert offset.deadline_utc == equivalent.deadline_utc == "2026-09-14T12:00:00Z"


def _write_campaign(
    tmp_path: Path, name: str, *, policy: str | None = None, with_contract: bool = True
) -> Path:
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text("scenarios: []\n", encoding="utf-8")
    config: dict[str, Any] = {
        "name": f"deadline_gate_{name}",
        "scenario_matrix": str(scenarios),
        "planners": PLANNERS,
        "seed_policy": {"mode": "explicit", "seed_set": "paper_eval_s20"},
    }
    if with_contract:
        case = _case(name)
        contract = case["expiring_resource"] | ({"admission_policy": policy} if policy else {})
        config |= {"as_of": case["as_of"], "expiring_resource": contract}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_preflight_gate_blocks_only_present_contracts_under_block_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts.benchmark.preflight_campaign_checkpoints import main as preflight_main

    assert (
        preflight_main(
            ["--config", str(_write_campaign(tmp_path, "historical", with_contract=False))]
        )
        == 0
    )
    assert preflight_main(["--config", str(_write_campaign(tmp_path, "known_fit"))]) == 0
    assert (
        preflight_main(
            ["--config", str(_write_campaign(tmp_path, "zero_reserve", policy="report"))]
        )
        == 0
    )
    blocked = _write_campaign(tmp_path, "timezone_boundaries")
    capsys.readouterr()
    assert preflight_main(["--config", str(blocked), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "expiring_resource_deadline_gate"
    assert payload["expiring_resource"]["verdict"] == "too_late"


def test_preflight_uses_current_time_not_stale_generated_at(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale manifest generated_at must not make an expired window look feasible."""
    from scripts.benchmark.preflight_campaign_checkpoints import main as preflight_main

    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text("scenarios: []\n", encoding="utf-8")
    contract: dict[str, Any] = {
        "schema": "expiring_resource_contract.v1",
        "resource_class": "slurm_gpu",
        "admission_policy": "block",
        "retention_class": "durable_required",
        "deadline": {
            "kind": "known",
            "timestamp": "2026-09-05T00:00:00Z",
            "source": "ops_window_notice",
            "freshness": "fresh",
            "evidence_as_of": "2026-09-01T00:00:00Z",
        },
        "runtime": {
            "expected_seconds": 14400,
            "conservative_seconds": 21600,
            "basis": "campaign_history",
        },
        "queue_start": {"status": "not_queued"},
        "output": {"estimate_bytes": 1073741824, "basis": "episode_rate_measurement"},
        "reserves": {
            "retrieval_seconds": 3600,
            "verification_seconds": 1800,
            "preservation_seconds": 1800,
        },
    }
    config: dict[str, Any] = {
        "name": "deadline_gate_stale_clock",
        "scenario_matrix": str(scenarios),
        "planners": PLANNERS,
        "seed_policy": {"mode": "explicit", "seed_set": "paper_eval_s20"},
        "generated_at": "2026-08-01T00:00:00Z",
        "expiring_resource": contract,
    }
    path = tmp_path / "stale-clock.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    capsys.readouterr()
    assert preflight_main(["--config", str(path), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "expiring_resource_deadline_gate"
    # The deadline is evaluated against the current time, not the stale generated_at, so the
    # past deadline is detected even though the manifest was generated before it.
    assert "deadline_expired" in payload["expiring_resource"]["reason_codes"]


def test_standalone_checker_uses_current_time_not_stale_generated_at(tmp_path: Path) -> None:
    """The direct checker must not treat historical generated_at as its evaluation clock."""
    now = datetime.now(UTC)
    manifest = _case()
    manifest["generated_at"] = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")
    manifest.pop("as_of", None)
    manifest["expiring_resource"]["deadline"]["timestamp"] = (
        (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    )
    path = tmp_path / "stale-generated-at.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(path), "--check", "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "too_late"
    assert "deadline_expired" in payload["reason_codes"]
