"""Focused contract tests for the scheduler/launcher failure classifier (#8843)."""

from __future__ import annotations

import contextlib
import json
from io import StringIO
from typing import Any

import pytest

from scripts.tools import classify_scheduler_failure as tool

PRIVATE_MESSAGE = (
    "failed on gpu-node-17.cluster.invalid for user=researcher42 at "
    "/scratch/researcher42/out token=abc123 account=proj-rm"
)
EXPECTED_KEYS = set(
    "schema vocabulary_version check_only classification causes source confidence evidence retryability owner required_remediation outputs_may_require_harvest job_alias campaign_id notes claim_boundary".split()
)


def _loads(text: str) -> dict[str, Any]:
    return json.loads(text) if text else {}


# id; scheduler-json; launcher-result; artifacts-json; manifest-json; log; class; source; harvest
CASES = tuple(
    (p[0], _loads(p[1]), p[2], _loads(p[3]), _loads(p[4]), p[5], p[6], p[7], p[8])
    for row in r"""
pending-delay;{"state":"PENDING","exit_code":null,"queue_health":"constrained"};not_observed;{};{};-;pending_or_queue_delay;structured;not_applicable
queue-timeout;{"state":"PENDING","exit_code":null,"queue_timeout":true};not_observed;{};{};-;queue_timeout;structured;no
preemption;{"state":"PREEMPTED","termination_class":"preempted"};not_observed;{};{};-;preemption;structured;possible
node-failure;{"state":"NODE_FAIL"};not_observed;{};{};-;node_failure;structured;possible
out-of-memory;{"state":"OUT_OF_MEMORY"};not_observed;{};{};-;out_of_memory;structured;possible
walltime;{"state":"TIMEOUT","termination_class":"time_limit"};not_observed;{};{};-;walltime_timeout;structured;possible
cancellation;{"state":"CANCELLED","termination_class":"cancelled"};not_observed;{};{};-;cancellation;structured;possible
missing-command-module;{};missing_command;{};{};-;missing_command_or_module;structured;no
import-failure;{};application_failure;{};{};Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch';import_failure;structured+log_fingerprint;no
invalid-config;{};invalid_config;{};{};-;invalid_config_or_input;structured;no
output-capacity;{};application_failure;{"output_capacity_exceeded":true};{};-;output_capacity;structured;possible
duplicate-submission;{};application_failure;{};{"duplicate_of":"job-0000"};-;duplicate_submission;structured;no
application-nonzero;{};application_failure;{};{};-;application_failure;structured;possible
harvest-failure;{"state":"COMPLETED","exit_code":0};completed;{"harvest_status":"failed","harvest_required":true};{};-;harvest_failure;structured;required
""".strip().splitlines()
    if (p := row.split(";"))
)


def _receipt(
    scheduler: dict[str, Any] | None = None,
    launcher: str = "application_failure",
    artifacts: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": tool.RECEIPT_SCHEMA,
        "job_alias": "job-0001",
        "campaign_id": "campaign-01",
        "scheduler": {"state": "FAILED", "exit_code": 1, **(scheduler or {})},
        "launcher": {"result": launcher},
        "artifacts": artifacts or {},
        "job_manifest": manifest or {},
    }


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_class_fixtures(case: tuple[Any, ...]) -> None:
    """Each required operational fixture classifies with full policy metadata."""
    _, scheduler, launcher, artifacts, manifest, log, expected, source, harvest = case
    report = tool.classify_failure(
        _receipt(scheduler, launcher, artifacts, manifest), "" if log == "-" else log
    )
    assert report["classification"] == expected
    assert (report["source"], report["outputs_may_require_harvest"]) == (source, harvest)
    assert report["retryability"] == {
        "disposition": tool.CLASS_POLICY[expected].retryability,
        "automatic_retry": False,
        "policy_changed": False,
    }
    assert report["check_only"] is True and report["owner"] and report["evidence"]


def test_structured_evidence_outranks_log_prose() -> None:
    """A structured OOM state wins over a conflicting import-failure fingerprint."""
    report = tool.classify_failure(
        _receipt({"state": "OUT_OF_MEMORY"}, "not_observed"),
        log_excerpt="ModuleNotFoundError: No module named 'torch'",
    )
    assert (report["classification"], report["source"]) == ("out_of_memory", "structured")
    suppressed = [entry for entry in report["evidence"] if entry["suppressed"]]
    assert suppressed and all(entry["source"] == "log_fingerprint" for entry in suppressed)


def test_insufficient_evidence_and_internal_conflict_stay_explicit() -> None:
    """Missing evidence yields unknown; contradictory scheduler fields also stay unknown."""
    unknown = tool.classify_failure(
        _receipt({"state": "FAILED", "exit_code": None}, "not_observed")
    )
    assert unknown["classification"] == "unknown"
    assert (unknown["source"], unknown["confidence"], unknown["causes"]) == ("none", "none", [])
    assert unknown["retryability"]["disposition"] == "unknown"
    conflict = tool.classify_failure(_receipt({"state": "FAILED", "exit_code": 0}, "not_observed"))
    assert (conflict["classification"], conflict["source"], conflict["confidence"]) == (
        "unknown",
        "structured",
        "low",
    )
    assert any("conflicts" in note for note in conflict["notes"])


def test_conflicting_sources_and_post_run_failures_preserve_multiple_causes() -> None:
    """Distinct structured sources and post-run failures both stay multiple_causes."""
    conflicting = tool.classify_failure(_receipt({"state": "OUT_OF_MEMORY"}, "invalid_config"))
    assert conflicting["classification"] == "multiple_causes"
    assert conflicting["causes"] == ["invalid_config_or_input", "out_of_memory"]
    post_run = tool.classify_failure(_receipt(artifacts={"harvest_status": "failed"}))
    assert post_run["classification"] == "multiple_causes"
    assert set(post_run["causes"]) == {"application_failure", "harvest_failure"}
    assert post_run["outputs_may_require_harvest"] == "required"


def test_private_values_and_topology_are_redacted_before_output() -> None:
    """Paths, hosts, identities, accounts, tokens, and signed URLs never reach output."""
    report = tool.classify_failure(
        {
            **_receipt(launcher="launcher_error"),
            "job_alias": "/scratch/researcher42/job",
            "launcher": {
                "result": "launcher_error",
                "message": PRIVATE_MESSAGE,
                "outer_timeout": False,
            },
            "log_excerpt": "sbatch error: see https://cluster.invalid/log?sig=SECRETVALUE",
        }
    )
    blob = tool.render_report_json(report) + tool.render_report_text(report)
    for private in ("gpu-node-17", "researcher42", "/scratch", "abc123", "proj-rm", "SECRETVALUE"):
        assert private not in blob
    assert report["job_alias"] == "unavailable" and "<redacted-" in blob


def test_bare_host_labels_are_redacted() -> None:
    """Short scheduler host labels without a domain are redacted."""
    for label in ("node-42", "login01", "worker-7", "compute12"):
        assert label not in tool.sanitize_text(f"failed on {label} during launch")


def test_completed_run_and_deterministic_output_are_stable() -> None:
    """A completed run without a failure signature stays unknown and renders identically."""
    success = _receipt({"state": "COMPLETED", "exit_code": 0}, "completed")
    first = tool.classify_failure(success)
    assert first["classification"] == "unknown"
    assert first["source"] == "structured"
    assert any("nothing to classify" in note for note in first["notes"])
    second = tool.classify_failure(success)
    assert first == second
    assert tool.render_report_json(first) == tool.render_report_json(second)
    assert tool.render_report_json(first).endswith("\n") and set(first) == EXPECTED_KEYS
    assert frozenset(tool.CLASS_POLICY) == frozenset(tool.VOCABULARY)


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = StringIO(), StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tool.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_cli_check_json_text_bounds_log_excerpt_and_rejects_invalid(tmp_path: Any) -> None:
    """The check-only CLI prints deterministic JSON/text and fails closed on bad input."""
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_receipt({"state": "OUT_OF_MEMORY"})), encoding="utf-8")
    code, stdout, _ = _run(["--check", "--receipt", str(receipt_path), "--format", "json"])
    assert code == 0 and json.loads(stdout)["classification"] == "out_of_memory"
    code, stdout, _ = _run(["--check", "--receipt", str(receipt_path), "--format", "text"])
    assert code == 0 and stdout.startswith("classification: out_of_memory\n")
    small_log = tmp_path / "small.log"
    small_log.write_text("ModuleNotFoundError: No module named 'torch'", encoding="utf-8")
    app_receipt = tmp_path / "app.json"
    app_receipt.write_text(json.dumps(_receipt()), encoding="utf-8")
    code, stdout, _ = _run(
        ["--check", "--receipt", str(app_receipt), "--log-excerpt", str(small_log)]
    )
    assert code == 0 and json.loads(stdout)["classification"] == "import_failure"
    huge_log = tmp_path / "huge.log"
    huge_log.write_text("x" * tool.MAX_LOG_BYTES + "ModuleNotFoundError", encoding="utf-8")
    code, stdout, _ = _run(
        ["--check", "--receipt", str(app_receipt), "--log-excerpt", str(huge_log)]
    )
    assert code == 0 and json.loads(stdout)["classification"] == "application_failure"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code, _, stderr = _run(["--check", "--receipt", str(bad)])
    assert code == 2 and "FAIL invalid_input" in stderr
