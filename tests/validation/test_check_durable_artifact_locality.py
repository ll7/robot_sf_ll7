"""Focused contract tests for the durable artifact locality audit."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from scripts.validation import check_durable_artifact_locality as tool

FIXTURES = Path(__file__).parent / "fixtures" / "durable_artifact_locality"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _packet(
    tmp_path: Path,
    references: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    **overrides: Any,
) -> Path:
    path = tmp_path / "packet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema": tool.SCHEMA,
        "generated_at": "2026-09-10",
        "verification_max_age_days": 30,
        "minimum_release_copies": 2,
        "references": references,
        "artifacts": artifacts,
    } | overrides
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path


def _ref(reference_id: str, artifact_id: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "reference_id": reference_id,
        "artifact_id": artifact_id,
        "version": "1.0.0",
        "digest": DIGEST_A,
        "consumer_path": f"configs/benchmarks/{artifact_id}.yaml",
        "consumer_status": "active",
        "retention_class": "durable_required",
    }
    return entry | overrides


def _locator(**overrides: Any) -> dict[str, Any]:
    locator: dict[str, Any] = {
        "locator_class": "public_release",
        "verification": "verified",
        "verified_at": "2026-09-05",
        "failure_domain_id": "release-public",
        "mutable_alias": False,
    }
    return locator | overrides


def _artifact(artifact_id: str, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "artifact_id": artifact_id,
        "version": "1.0.0",
        "digest": DIGEST_A,
        "locators": [_locator()],
    }
    return entry | overrides


def _codes(report: tool.LocalityReport) -> set[str]:
    return {finding.code for finding in report.findings}


LOCATOR_CASES = {
    "stale_verification": _locator(verified_at="2026-01-01"),
    "mutable_alias": _locator(mutable_alias=True),
    "institutional_only": _locator(locator_class="institutional_durable"),
    "cache_only": _locator(locator_class="institutional_cache"),
    "non_durable_custody": _locator(locator_class="local_scratch"),
    "no_verified_locator": _locator(verification="unverified", verified_at=None),
    "unknown_locator_class": _locator(locator_class="mystery"),
}


def test_committed_compliant_packet_passes_and_is_deterministic() -> None:
    report = tool.audit_locality(FIXTURES / "compliant.json")
    assert report.status == "pass", [finding.message for finding in report.findings]
    assert report.render_json() == tool.audit_locality(FIXTURES / "compliant.json").render_json()
    outcomes = {outcome["reference_id"]: outcome for outcome in report.outcomes}
    assert outcomes["release.camera_ready_table_1"]["verified_non_institutional_copies"] == 2
    assert outcomes["model.learned_risk_v2"]["outcome"] == "pass"
    assert outcomes["evidence.retired_campaign"]["outcome"] == "inactive"


def test_committed_failing_packet_reports_required_codes() -> None:
    report = tool.audit_locality(FIXTURES / "failing.json")
    expected = {
        "institutional_only",
        "cache_only",
        "stale_verification",
        "digest_mismatch",
        "same_failure_domain",
        "insufficient_redundancy",
        "missing_projection_row",
    }
    assert report.status == "fail"
    assert expected <= _codes(report)


def test_cli_check_exit_codes_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    compliant, failing = FIXTURES / "compliant.json", FIXTURES / "failing.json"
    assert tool.main(["--projection", str(compliant), "--check"]) == 0
    assert '"status": "pass"' in capsys.readouterr().out
    assert tool.main(["--projection", str(failing), "--check"]) == 1
    assert '"status": "fail"' in capsys.readouterr().out
    assert tool.main(["--projection", str(compliant), "--format", "markdown"]) == 0
    assert "# Durable Artifact Locality Audit" in capsys.readouterr().out
    missing = tmp_path / "missing.json"
    assert tool.audit_locality(missing).status == "unknown"
    assert tool.main(["--projection", str(missing), "--check"]) == 2


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        *(({"locators": [locator]}, code) for code, locator in LOCATOR_CASES.items()),
        ({"version": "9.9.9"}, "version_mismatch"),
        ({"digest": DIGEST_B}, "digest_mismatch"),
        ({"locators": []}, "empty_locator_set"),
    ],
)
def test_failure_modes(tmp_path: Path, overrides: dict[str, Any], expected: str) -> None:
    report = tool.audit_locality(_packet(tmp_path, [_ref("r", "x")], [_artifact("x", **overrides)]))
    assert expected in _codes(report)
    assert report.status == "fail"


def test_inactive_history_never_satisfies_active_requirement(tmp_path: Path) -> None:
    references = [
        _ref("active", "x"),
        _ref("history", "x", consumer_status="inactive", retention_class="historical"),
    ]
    artifact = _artifact("x", locators=[_locator(locator_class="institutional_durable")])
    outcomes = {
        outcome["reference_id"]: outcome
        for outcome in tool.audit_locality(_packet(tmp_path, references, [artifact])).outcomes
    }
    assert outcomes["active"]["outcome"] == "fail"
    assert outcomes["history"]["outcome"] == "inactive"


def test_private_locator_values_never_reach_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "s3://private-host.invalid/secret-bucket/model.bin"
    packet = _packet(
        tmp_path, [_ref("r", "x")], [_artifact("x", locators=[_locator(locator=secret)])]
    )
    report = tool.audit_locality(packet)
    assert "private_locator_value_rejected" in _codes(report)
    assert secret not in report.render_json() + report.render_markdown()
    assert tool.main(["--projection", str(packet), "--check"]) == 1
    assert secret not in capsys.readouterr().out


def test_as_of_override_controls_staleness(tmp_path: Path) -> None:
    packet = _packet(tmp_path, [_ref("r", "x")], [_artifact("x")])
    assert tool.audit_locality(packet).status == "pass"
    assert "stale_verification" in _codes(tool.audit_locality(packet, as_of=date(2027, 1, 1)))
    assert tool.main(["--projection", str(packet), "--as-of", "2027-01-01", "--check"]) == 1
