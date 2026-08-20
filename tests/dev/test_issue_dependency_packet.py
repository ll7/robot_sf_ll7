"""Deterministic contract tests for typed issue dependency packets."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from scripts.dev import issue_dependency_packet as packet

if TYPE_CHECKING:
    from pathlib import Path

REPOSITORY = "ll7/robot_sf_ll7"
ISSUE = 7613
SHA = "a" * 40
TARGET_SHA = "b" * 40
DIGEST = "c" * 64


def _row(
    identifier: str,
    kind: str,
    requirement: dict[str, Any],
    *,
    mandatory: bool = True,
    observed: dict[str, Any] | None = None,
    verdict: str = "unavailable",
) -> dict[str, Any]:
    """Build one compact fixture row."""
    freshness_key = {
        "issue_state": "issue_state",
        "pull_request_state": "pull_request_state",
        "commit_present": "commit",
        "path_present": "path",
        "artifact_digest": "artifact",
        "external_input": "external_input",
        "environment_capability": "environment",
        "human_ruling": "human_ruling",
    }[kind]
    return {
        "id": identifier,
        "repository": REPOSITORY,
        "kind": kind,
        "requirement": requirement,
        "mandatory": mandatory,
        "source": {"kind": "fixture", "ref": f"tests/dev/{identifier}.json"},
        "observed": observed or {},
        "verdict": verdict,
        "unblock_condition": f"verify dependency {identifier}",
        "freshness": [freshness_key],
    }


def _packet(*rows: dict[str, Any]) -> dict[str, Any]:
    """Build a packet with a stable contract digest."""
    return packet.build_packet(
        {
            "repository": REPOSITORY,
            "issue": ISSUE,
            "contract": {"source": "issue-body.md", "digest": DIGEST},
            "dependencies": list(rows),
        }
    )


def test_issue_dependency_packet_build_and_validate_self_digest() -> None:
    """A canonical packet validates and its digest changes with its predicate."""
    built = _packet(
        _row("issue-open", "issue_state", {"number": 12, "state": "OPEN"}),
    )

    report = packet.validate_packet(built, expected_repository=REPOSITORY, expected_issue=ISSUE)

    assert report["ok"] is True
    assert report["packet_digest"] == packet.compute_packet_digest(built)
    changed = dict(built)
    changed["dependencies"] = [dict(built["dependencies"][0], mandatory=False)]
    assert packet.validate_packet(changed)["ok"] is False


def test_issue_dependency_packet_link_without_predicate_is_invalid() -> None:
    """An informational link cannot be represented as a satisfied predicate."""
    with pytest.raises(ValueError, match="predicate"):
        _packet(
            _row(
                "external-input",
                "external_input",
                {"identifier": "dataset-1"},
            )
        )


def test_issue_dependency_packet_satisfied_and_advisory_rows_are_visible() -> None:
    """Mandatory rows admit while an unavailable advisory row remains visible."""
    built = _packet(
        _row("issue-open", "issue_state", {"number": 12, "state": "OPEN"}),
        _row(
            "optional-artifact",
            "artifact_digest",
            {"path": "output/report.json", "schema": "report.v1", "digest": DIGEST},
            mandatory=False,
        ),
    )
    context = {
        "observations": {
            "issue-open": {"available": True, "state": "OPEN"},
            "optional-artifact": {"available": False, "reason": "not staged"},
        }
    }

    result = packet.evaluate_packet(built, context)

    assert result["ok"] is True
    assert result["verdict"] == "satisfied"
    assert result["mandatory_failures"] == []
    assert result["advisory_failures"][0]["id"] == "optional-artifact"
    assert result["advisory_failures"][0]["verdict"] == "unavailable"


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        ({"available": True, "state": "CLOSED"}, "unsatisfied"),
        ({"available": False, "reason": "not fetched"}, "unavailable"),
        ({"available": True, "state": "OPEN", "head_sha": TARGET_SHA}, "conflict"),
    ],
)
def test_issue_dependency_packet_mandatory_failure_verdicts_block(
    observation: dict[str, Any], expected: str
) -> None:
    """Every mandatory non-satisfied verdict blocks admission and names its condition."""
    built = _packet(
        _row(
            "pr-exact",
            "pull_request_state",
            {"number": 42, "state": "OPEN", "head_sha": SHA},
        )
    )
    result = packet.evaluate_packet(built, {"observations": {"pr-exact": observation}})

    assert result["ok"] is False
    assert result["verdict"] == "blocked"
    assert result["mandatory_failures"] == [
        {
            "id": "pr-exact",
            "kind": "pull_request_state",
            "verdict": expected,
            "reason": result["mandatory_failures"][0]["reason"],
            "unblock_condition": "verify dependency pr-exact",
        }
    ]
    assert result["mandatory_failures"][0]["unblock_condition"]


def test_issue_dependency_packet_exact_artifact_digest_and_closed_state_are_distinct() -> None:
    """A closed prerequisite does not prove an artifact, and a digest mismatch conflicts."""
    built = _packet(
        _row("closed-issue", "issue_state", {"number": 12, "state": "CLOSED"}),
        _row(
            "artifact",
            "artifact_digest",
            {"path": "report.json", "schema": "report.v1", "digest": DIGEST},
        ),
    )
    context = {
        "observations": {
            "closed-issue": {"available": True, "state": "CLOSED"},
            "artifact": {
                "available": True,
                "verified": True,
                "schema": "report.v1",
                "digest": "d" * 64,
            },
        }
    }

    result = packet.evaluate_packet(built, context)

    assert result["rows"][0]["verdict"] == "satisfied"
    assert result["rows"][1]["verdict"] == "conflict"
    assert result["mandatory_failures"][0]["id"] == "artifact"


def test_issue_dependency_packet_artifact_and_external_need_verified_sources() -> None:
    """Matching metadata without an explicit source verification remains unavailable."""
    built = _packet(
        _row(
            "artifact",
            "artifact_digest",
            {"path": "report.json", "schema": "report.v1", "digest": DIGEST},
        ),
        _row(
            "external",
            "external_input",
            {"identifier": "dataset-1", "predicate": "licensed-for-test"},
        ),
    )
    context = {
        "observations": {
            "artifact": {"available": True, "schema": "report.v1", "digest": DIGEST},
            "external": {
                "available": True,
                "predicate": "licensed-for-test",
            },
        }
    }

    result = packet.evaluate_packet(built, context)

    assert {row["verdict"] for row in result["rows"]} == {"unavailable"}
    assert {failure["id"] for failure in result["mandatory_failures"]} == {
        "artifact",
        "external",
    }


def test_issue_dependency_packet_rejects_malformed_freshness_and_issue_merge_state() -> None:
    """Malformed freshness values and impossible issue states fail closed."""
    row = _row("bad", "issue_state", {"number": 12, "state": "MERGED"})
    row["freshness"] = [{}]
    unsigned = {
        "repository": REPOSITORY,
        "issue": ISSUE,
        "contract": {"source": "issue-body.md", "digest": DIGEST},
        "dependencies": [row],
    }
    packet_payload = dict(unsigned, schema=packet.SCHEMA)
    built = dict(packet_payload, packet_digest=packet.compute_packet_digest(packet_payload))

    report = packet.validate_packet(built)

    assert report["ok"] is False
    assert any("freshness" in error for error in report["errors"])
    assert any("OPEN or CLOSED" in error for error in report["errors"])


def test_issue_dependency_packet_evaluation_is_byte_stable() -> None:
    """Unchanged packet/context inputs produce byte-identical JSON output."""
    built = _packet(
        _row("path", "path_present", {"path": "scripts/dev", "path_type": "directory"}),
    )
    context = {"observations": {"path": {"available": True, "exists": True, "type": "directory"}}}

    first = packet.evaluate_packet(built, context)
    second = packet.evaluate_packet(built, context)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_issue_dependency_packet_gate_never_allows_write_on_block() -> None:
    """The #7611 adapter attaches the aggregate result and removes write permission."""
    report = packet.apply_dependency_gate(
        {"classification": "ready", "ready": True, "write_allowed": True, "reasons": []},
        {
            "schema": packet.EVALUATION_SCHEMA,
            "ok": False,
            "verdict": "blocked",
            "packet_digest": DIGEST,
            "mandatory_failures": [
                {
                    "id": "missing",
                    "reason": "not found",
                    "unblock_condition": "stage it",
                }
            ],
            "advisory_failures": [],
        },
    )

    assert report["ready"] is False
    assert report["write_allowed"] is False
    assert report["classification"] == "needs_dependency"
    assert report["dependency_gate"]["mandatory_failures"][0]["id"] == "missing"


def test_issue_dependency_packet_resolver_uses_rest_and_local_readers(tmp_path: Path) -> None:
    """Resolver reads public state and local predicates through injectable readers."""
    local_dir = tmp_path / "scripts" / "dev"
    local_dir.mkdir(parents=True)
    (local_dir / "ready.txt").write_text("fixture\n", encoding="utf-8")
    calls: list[list[str]] = []
    gh_payloads = {
        "repos/ll7/robot_sf_ll7/issues/12": {"number": 12, "state": "open", "body": ""},
        "repos/ll7/robot_sf_ll7/pulls/42": {
            "number": 42,
            "state": "open",
            "merged_at": None,
            "head": {"sha": SHA, "ref": "feature"},
            "base": {"sha": TARGET_SHA, "ref": "main"},
        },
    }

    def gh_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        endpoint = command[2]
        payload = gh_payloads[endpoint]
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    def git_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    built = _packet(
        _row("issue", "issue_state", {"number": 12, "state": "OPEN"}),
        _row("pr", "pull_request_state", {"number": 42, "state": "OPEN", "head_sha": SHA}),
        _row("commit", "commit_present", {"sha": SHA, "ancestor_of": TARGET_SHA}),
        _row("path", "path_present", {"path": "scripts/dev", "path_type": "directory"}),
    )

    result = packet.resolve_packet(
        built,
        repo_root=tmp_path,
        gh_runner=gh_runner,
        git_runner=git_runner,
    )

    assert result["ok"] is True
    assert {row["id"] for row in result["rows"] if row["verdict"] == "satisfied"} == {
        "issue",
        "pr",
        "commit",
        "path",
    }
    assert any(
        command[:3] == ["gh", "api", "repos/ll7/robot_sf_ll7/issues/12"] for command in calls
    )
    assert any(command[:2] == ["git", "cat-file"] for command in calls)


def test_issue_dependency_packet_cli_build_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI build and offline verification expose the same contract."""
    declaration = tmp_path / "declaration.json"
    packet_path = tmp_path / "packet.json"
    context_path = tmp_path / "context.json"
    declaration.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "issue": ISSUE,
                "contract": {"source": "issue-body.md", "digest": DIGEST},
                "dependencies": [
                    _row("issue", "issue_state", {"number": 12, "state": "OPEN"}),
                ],
            }
        ),
        encoding="utf-8",
    )
    context_path.write_text(
        json.dumps({"observations": {"issue": {"available": True, "state": "OPEN"}}}),
        encoding="utf-8",
    )

    assert packet.main(["build", "--input", str(declaration), "--output", str(packet_path)]) == 0
    assert (
        packet.main(["verify", "--packet", str(packet_path), "--context", str(context_path)]) == 0
    )
    output = capsys.readouterr().out
    assert '"verdict": "satisfied"' in output
