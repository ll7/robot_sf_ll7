"""Tests for the structural benchmark release suite contract checker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.release_suite_contract import (
    CLAIM_BOUNDARY,
    RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
    REQUIRED_SUITE_METADATA_FIELDS,
    ReleaseSuiteContractError,
    evaluate_release_suite_contract,
    load_release_suite_contract,
)
from scripts.validation import check_release_suite_contract

if TYPE_CHECKING:
    from pathlib import Path


def _suite(suite_id: str = "nominal") -> dict[str, str]:
    """Build one structurally complete suite declaration."""

    return {
        "suite_id": suite_id,
        "odd_contract": f"contracts/{suite_id}/odd.yaml",
        "scenario_contract": f"contracts/{suite_id}/scenarios.yaml",
        "scenario_certification": f"certification/{suite_id}.json",
        "planner_row_status": f"rows/{suite_id}.json",
        "seed_schedule": f"seeds/{suite_id}.yaml",
        "artifact_manifest": f"artifacts/{suite_id}.json",
    }


def _manifest(suites: list[object]) -> dict[str, object]:
    """Build a minimal suite-contract manifest payload."""

    return {
        "schema_version": RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
        "release_id": "benchmark-v0.1-candidate",
        "suites": suites,
    }


def _write_manifest(path: Path, payload: object) -> None:
    """Write a YAML or JSON manifest fixture."""

    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_complete_multi_suite_manifest_passes_structural_check(tmp_path: Path) -> None:
    """Every declared suite must carry all six required metadata references."""

    path = tmp_path / "release_suites.yaml"
    _write_manifest(path, _manifest([_suite("nominal"), _suite("stress")]))

    report = evaluate_release_suite_contract(load_release_suite_contract(path))

    assert report["status"] == "pass"
    assert report["suite_count"] == 2
    assert report["complete_suite_count"] == 2
    assert report["blocked_suite_count"] == 0
    assert report["required_suite_metadata_fields"] == list(REQUIRED_SUITE_METADATA_FIELDS)
    assert report["claim_boundary"] == CLAIM_BOUNDARY


@pytest.mark.parametrize("field", REQUIRED_SUITE_METADATA_FIELDS)
@pytest.mark.parametrize("missing_value", [None, "", "   ", []])
def test_each_missing_or_invalid_required_field_blocks(
    tmp_path: Path,
    field: str,
    missing_value: object,
) -> None:
    """Missing, blank, and non-string references fail closed for every field."""

    suite: dict[str, object] = _suite()
    suite[field] = missing_value
    path = tmp_path / "release_suites.yaml"
    _write_manifest(path, _manifest([suite]))

    report = evaluate_release_suite_contract(load_release_suite_contract(path))

    assert report["status"] == "blocked"
    assert report["complete_suite_count"] == 0
    assert report["blocked_suite_count"] == 1
    assert report["suites"][0]["missing_fields"] == [field]
    assert report["blockers"] == [f"nominal.{field} is missing or is not a non-empty string"]


def test_incomplete_later_suite_blocks_whole_manifest(tmp_path: Path) -> None:
    """The checker must evaluate every suite instead of passing on the first complete one."""

    incomplete = _suite("adversarial")
    del incomplete["artifact_manifest"]
    path = tmp_path / "release_suites.json"
    _write_manifest(path, _manifest([_suite("nominal"), incomplete]))

    report = evaluate_release_suite_contract(load_release_suite_contract(path))

    assert report["status"] == "blocked"
    assert report["complete_suite_count"] == 1
    assert report["blocked_suite_count"] == 1
    assert report["suites"][1] == {
        "suite_id": "adversarial",
        "status": "blocked",
        "missing_fields": ["artifact_manifest"],
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "must be a mapping"),
        ({"schema_version": "wrong"}, "schema_version"),
        (
            {
                "schema_version": RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
                "release_id": 2910,
                "suites": [_suite()],
            },
            "release_id",
        ),
        (
            {
                "schema_version": RELEASE_SUITE_CONTRACT_SCHEMA_VERSION,
                "release_id": "candidate",
                "suites": [],
            },
            "non-empty list",
        ),
        (_manifest(["not-a-mapping"]), r"suites\[0\] must be a mapping"),
        (_manifest([{"suite_id": 1}]), r"suites\[0\]\.suite_id"),
        (_manifest([_suite("same"), _suite("same")]), "duplicate suite_id"),
    ],
)
def test_ambiguous_manifest_structure_is_rejected(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    """Malformed containers cannot be interpreted as an empty successful check."""

    path = tmp_path / "release_suites.yaml"
    _write_manifest(path, payload)

    with pytest.raises(ReleaseSuiteContractError, match=match):
        load_release_suite_contract(path)


def test_cli_exit_codes_distinguish_pass_blocked_and_malformed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The command exposes shell-friendly pass, blocked, and malformed outcomes."""

    complete_path = tmp_path / "complete.yaml"
    blocked_path = tmp_path / "blocked.yaml"
    malformed_path = tmp_path / "malformed.yaml"
    _write_manifest(complete_path, _manifest([_suite()]))
    incomplete = _suite()
    del incomplete["seed_schedule"]
    _write_manifest(blocked_path, _manifest([incomplete]))
    _write_manifest(malformed_path, {"schema_version": "wrong"})

    assert check_release_suite_contract.main(["--manifest", str(complete_path), "--json"]) == 0
    assert '"status": "pass"' in capsys.readouterr().out
    assert check_release_suite_contract.main(["--manifest", str(blocked_path)]) == 1
    assert "blocker: nominal.seed_schedule" in capsys.readouterr().out
    assert check_release_suite_contract.main(["--manifest", str(malformed_path)]) == 2
    assert "error:" in capsys.readouterr().err
