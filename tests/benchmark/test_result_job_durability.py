"""Tests for the ``result_job_durability.v1`` gate contract.

The gate is the producer-side counterpart to the orchestrator's admission
discipline: a result job must publish checksum + schema + rerun command +
durable pointer before any ``analysis:`` successor issue is created.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.result_job_durability import (
    RESULT_JOB_DURABILITY_SCHEMA_VERSION,
    DurabilityVerdict,
    ResultJobDurabilityError,
    load_result_job_durability,
    main,
    result_job_durability_from_dict,
    sha256_file,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "result_job_durability" / "v1"
VALID_MANIFEST = FIXTURE_DIR / "valid_manifest.yaml"
PRIVATE_MANIFEST = FIXTURE_DIR / "valid_private_manifest.yaml"


def _payload(path: Path = VALID_MANIFEST) -> dict[str, object]:
    """Return a mutable valid fixture manifest payload."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_load_valid_manifest_passes_all_four_gates() -> None:
    """A complete tracked-path manifest should be durable with all gates passing."""

    verdict = load_result_job_durability(VALID_MANIFEST)

    assert verdict.ok
    assert verdict.gate_id == "job_13512_snqi_rank_identifiability"
    assert verdict.gate_results == {
        "checksum": True,
        "schema": True,
        "rerun_command": True,
        "durable_pointer": True,
    }
    assert verdict.issues == ()


def test_valid_manifest_has_correct_schema_version_and_checksum() -> None:
    """The fixture should carry the current schema version and a matching checksum."""

    payload = _payload()
    assert payload["schema_version"] == RESULT_JOB_DURABILITY_SCHEMA_VERSION
    rows = FIXTURE_DIR / "job_13512_snqi_rows.csv"
    assert payload["sha256"] == sha256_file(rows)


def test_private_safe_registry_pointer_is_durable_without_local_file() -> None:
    """A private-safe registry pointer need not resolve to a tracked file.

    This is the #5912 case: raw rows stay private behind a checksummed
    sufficient-derived input reached through a registry/release pointer.
    """

    verdict = load_result_job_durability(PRIVATE_MANIFEST)

    assert verdict.ok, verdict.issues
    assert verdict.gate_results["durable_pointer"]


def test_schema_path_directory_fails_closed() -> None:
    """A schema_path directory must not pass the schema gate."""

    payload = _payload()
    payload["input_schema"]["schema_path"] = "tests"

    verdict = result_job_durability_from_dict(payload, manifest_path=VALID_MANIFEST)

    assert not verdict.ok
    assert any(
        issue.gate == "schema" and "does not resolve to a file on clean checkout" in issue.message
        for issue in verdict.issues
    )


@pytest.mark.parametrize("pointer_kind", ["registry_entry", "release_artifact"])
def test_hydration_pointer_kind_requires_hydration_command(pointer_kind: str) -> None:
    """A registry/release pointer must carry a hydration_command to be durable.

    The schema description and gate doc state hydration_command is required for
    registry_entry/release_artifact pointers (the artifact is reproduced by
    hydrating from the durable pointer on a clean checkout). A manifest that
    omits it must fail closed on the durable_pointer gate.
    """

    payload = _payload(PRIVATE_MANIFEST)
    payload["analysis_input"]["pointer_kind"] = pointer_kind
    del payload["analysis_input"]["hydration_command"]
    verdict = result_job_durability_from_dict(payload, manifest_path=PRIVATE_MANIFEST)

    assert not verdict.ok
    assert verdict.gate_results["durable_pointer"] is False
    matching = [
        issue
        for issue in verdict.issues
        if issue.gate == "durable_pointer" and "hydration_command" in issue.path
    ]
    assert matching, f"expected a hydration_command issue, got {verdict.issues}"
    assert all(issue.remedy for issue in matching)


@pytest.mark.parametrize(
    ("mutate", "failed_gate", "fragment"),
    [
        # durable_pointer: local-only pointer is not durable (root cause of #5890/#5891)
        (
            lambda p: p["analysis_input"].__setitem__("pointer", "output/job_13512/snqi_rows.csv"),
            "durable_pointer",
            "local-only analysis input pointer is not durable",
        ),
        # durable_pointer: tracked path does not resolve on clean checkout
        (
            lambda p: p["analysis_input"].__setitem__("pointer", "missing_input.csv"),
            "durable_pointer",
            "tracked analysis input does not resolve on clean checkout",
        ),
        # checksum: declared digest does not match the resolved artifact
        (
            lambda p: p.__setitem__("sha256", "0" * 64),
            "checksum",
            "checksum mismatch",
        ),
        # schema: declared schema path does not resolve on clean checkout
        (
            lambda p: p["input_schema"].__setitem__(
                "schema_path", "robot_sf/benchmark/schemas/missing.v9.json"
            ),
            "schema",
            "input schema does not resolve to a file on clean checkout",
        ),
        # rerun_command: a bare host-only stub cannot reproduce the analysis
        (
            lambda p: (
                p["analysis_input"].__setitem__("pointer", "output/job_13512/snqi_rows.csv")
                or p.__setitem__(
                    "rerun_command",
                    "cat output/job_13512/snqi_rows.csv",
                )
            ),
            "rerun_command",
            "cannot reproduce the analysis from a local-only input pointer",
        ),
    ],
)
def test_each_gate_failure_is_reported_with_a_remedy(
    mutate,
    failed_gate: str,
    fragment: str,
) -> None:
    """Each of the four gate properties should fail closed with a clear remedy."""

    payload = _payload()
    mutate(payload)
    verdict = result_job_durability_from_dict(payload, manifest_path=VALID_MANIFEST)

    assert not verdict.ok
    assert verdict.gate_results[failed_gate] is False
    matching = [issue for issue in verdict.issues if issue.gate == failed_gate]
    assert matching, f"expected a {failed_gate} issue, got {verdict.issues}"
    assert any(fragment in issue.message for issue in matching)
    assert all(issue.remedy for issue in matching), "every gate issue must name a remedy"


def test_missing_required_field_fails_closed() -> None:
    """Dropping a required gate field must surface a manifest-level issue."""

    payload = _payload()
    del payload["rerun_command"]
    verdict = result_job_durability_from_dict(payload, manifest_path=VALID_MANIFEST)

    assert not verdict.ok
    assert any(issue.gate == "rerun_command" for issue in verdict.issues)


def test_error_exception_aggregates_issues() -> None:
    """The error type should aggregate every gate issue with its remedy."""

    payload = _payload()
    payload["analysis_input"]["pointer"] = "output/x.csv"
    payload["sha256"] = "0" * 64
    issues = list(result_job_durability_from_dict(payload, manifest_path=VALID_MANIFEST).issues)
    error = ResultJobDurabilityError(issues)
    text = str(error)
    assert "remedy" in text
    assert len(error.issues) == len(issues)


def test_unloadable_manifest_returns_verdict_not_raises(tmp_path: Path) -> None:
    """A malformed manifest file yields a not-ok verdict, not an exception."""

    bad = tmp_path / "broken.yaml"
    bad.write_text(":\n  - [unbalanced", encoding="utf-8")
    verdict = load_result_job_durability(bad)

    assert isinstance(verdict, DurabilityVerdict)
    assert not verdict.ok
    assert verdict.gate_id is None
    assert any("failed to load manifest" in issue.message for issue in verdict.issues)


def test_cli_returns_zero_for_durable_manifest_and_two_for_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI must return shell-friendly exit codes for admission automation."""

    assert main([str(VALID_MANIFEST)]) == 0
    out = capsys.readouterr().out
    assert "DURABLE" in out

    bad_payload = _payload()
    bad_payload["analysis_input"]["pointer"] = "output/x.csv"
    bad_manifest = FIXTURE_DIR / "_tmp_bad.yaml"
    bad_manifest.write_text(yaml.safe_dump(bad_payload), encoding="utf-8")
    try:
        assert main([str(bad_manifest), "--json"]) == 2
        import json

        report = json.loads(capsys.readouterr().out)
        assert report["ok"] is False
        assert any(issue["gate"] == "durable_pointer" for issue in report["issues"])
    finally:
        bad_manifest.unlink(missing_ok=True)


def test_probe_does_not_mutate_input_payload() -> None:
    """Validation must not mutate the caller's payload."""

    payload = _payload()
    snapshot = copy.deepcopy(payload)
    result_job_durability_from_dict(payload, manifest_path=VALID_MANIFEST)
    assert payload == snapshot
