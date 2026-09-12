"""Focused contract tests for the check-only log-retention helper (#8856)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from scripts.tools.check_log_retention import EXIT_OK, build_report, main

if TYPE_CHECKING:
    from pathlib import Path


def _metadata(data: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(data),
        "line_count": data.count(b"\n") + int(bool(data) and not data.endswith(b"\n")),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _case(
    tmp_path: Path,
    *,
    data: bytes = b"start\nready\nERROR: useful detail\nend\n",
) -> tuple[dict[str, object], Path]:
    (root := tmp_path / "logs").mkdir()
    (root / "task-0.log").write_bytes(data)
    entry = json.loads(
        '{"path":"task-0.log","role":"task_stderr","job_id":"job-1","task_id":"0","encoding":"utf-8","completion":"complete","retention_class":"disposable"}'
    )
    entry.update(_metadata(data))
    manifest = json.loads(
        '{"schema":"log_retention_manifest.v1","job":{"job_id":"job-1"},"excerpt_policy":{"version":"log_excerpt.v1","max_bytes":180,"first_lines":1,"last_lines":1,"context_lines":1,"max_log_bytes":4096,"loss_declared":true},"logs":[]}'
    )
    manifest["job"].update(state="completed", diagnosis="not_required")
    manifest["logs"].append(entry)
    manifest["custody"] = json.loads(
        '{"status":"verified","durability_class":"cloud_durable","independent_verification":true,"transfer_verification":true,"consumer_review":true,"files":[{"relative_path":"task-0.log","state":"verified"}]}'
    )
    manifest["custody"]["files"][0].update(sha256=entry["sha256"], byte_size=entry["size_bytes"])
    return manifest, root


def _assert_blocked(report: dict, reason: str | None = None) -> None:
    assert report["status"] == "blocked" and report["logs"][0]["prune_eligible"] is False
    assert report["storage_estimate"]["prune_eligible_bytes"] == 0
    if reason:
        assert reason in report["logs"][0]["reasons"]


def test_valid_log_has_provenance_excerpt_and_storage_estimate(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    report = build_report(manifest, root)
    entry = report["logs"][0]

    assert report["status"] == "eligible"
    assert entry["identity"] == {"job_id": "job-1", "task_id": "0"}
    assert entry["line_count"] == 4
    assert entry["excerpt"]["source_location"] == "root-relative:task-0.log"
    assert len(entry["excerpt"]["text"].encode()) <= 180
    assert report["storage_estimate"]["prune_eligible_bytes"] > 0
    assert report["deletion_plan"] == []


def test_excerpt_is_deterministic_and_sanitized(tmp_path: Path) -> None:
    data = b"first\nkeep\nERROR host=node42 user=alice /scratch/alice/x https://internal.local/a\nkeep\nlast\n"
    manifest, root = _case(tmp_path, data=data)
    first = build_report(manifest, root)
    excerpt = first["logs"][0]["excerpt"]

    assert first == build_report(deepcopy(manifest), root)
    assert {item["kind"] for item in excerpt["ranges"]} == {"first", "error_context", "last"}
    assert all(value not in excerpt["text"] for value in ("node42", "alice", "internal.local"))


@pytest.mark.parametrize("kind", ("root", "parent", "child", "dangling"))
def test_symlinked_or_escaped_source_paths_block_pruning(tmp_path: Path, kind: str) -> None:
    manifest, root = _case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task-0.log").write_bytes(root.joinpath("task-0.log").read_bytes())
    if kind == "root":
        linked = tmp_path / "logs-link"
        linked.symlink_to(root, target_is_directory=True)
        root = linked
    elif kind == "parent":
        linked = tmp_path / "parent-link"
        linked.symlink_to(tmp_path, target_is_directory=True)
        root = linked / "logs"
    else:
        link = root / "nested"
        if kind == "child":
            link.symlink_to(outside, target_is_directory=True)
        else:
            link.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
        manifest["logs"][0]["path"] = "nested/task-0.log"
    report = build_report(manifest, root)

    _assert_blocked(report)
    codes = {problem["code"] for problem in report["problems"]}
    assert "source_root_symlink" in codes or "source_path_symlink" in codes
    assert "source_outside_root" in codes or kind not in ("child", "dangling")


@pytest.mark.parametrize(
    ("path", "code"),
    [("..", "path_escape"), ("../outside/task-0.log", "path_escape")],
)
def test_parent_paths_block_pruning(tmp_path: Path, path: str, code: str) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"][0]["path"] = path
    report = build_report(manifest, root)

    _assert_blocked(report)
    assert any(problem["code"] == code for problem in report["problems"])


def test_unavailable_source_root_blocks_pruning(tmp_path: Path) -> None:
    manifest, _root = _case(tmp_path)
    report = build_report(manifest, tmp_path / "missing-root")

    _assert_blocked(report)
    assert any(problem["code"] == "source_root_unavailable" for problem in report["problems"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("independent_verification", False),
        ("transfer_verification", None),
        ("consumer_review", "yes"),
    ],
)
def test_pruning_requires_independent_custody_transfer_and_consumer_proof(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest, root = _case(tmp_path)
    if value is None:
        manifest["custody"].pop(field)
    else:
        manifest["custody"][field] = value

    report = build_report(manifest, root)
    _assert_blocked(report)
    assert any(problem["location"] == f"custody.{field}" for problem in report["problems"])


@pytest.mark.parametrize(
    ("alias", "value"),
    [("transfer_verified", True), ("consumer_reviewed", True)],
)
def test_custody_proof_alias_spellings_are_reconciled(
    tmp_path: Path, alias: str, value: bool
) -> None:
    manifest, root = _case(tmp_path)
    canonical = "transfer_verification" if alias == "transfer_verified" else "consumer_review"
    manifest["custody"].pop(canonical)
    manifest["custody"][alias] = value

    report = build_report(manifest, root)

    assert report["status"] == "eligible"
    assert report["custody"]["verified"] is True


def test_array_tasks_have_distinct_prunable_identities(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    data = b"array task one\n"
    (root / "task-1.log").write_bytes(data)
    entry = deepcopy(manifest["logs"][0])
    entry.update(path="task-1.log", task_id="1", **_metadata(data))
    manifest["logs"].append(entry)
    custody_entry = deepcopy(manifest["custody"]["files"][0])
    custody_entry.update(
        relative_path="task-1.log", sha256=entry["sha256"], byte_size=entry["size_bytes"]
    )
    manifest["custody"]["files"].append(custody_entry)

    report = build_report(manifest, root)

    assert report["status"] == "eligible"
    assert {entry["identity"]["task_id"] for entry in report["logs"]} == {"0", "1"}
    assert report["storage_estimate"]["prune_eligible_bytes"] == sum(
        item["size_bytes"] for item in report["logs"]
    )


@pytest.mark.parametrize(
    ("data", "change", "reason"),
    [
        (b"safe\n", {"completion": "active"}, "active_log"),
        (b"safe\n", {"completion": "truncated"}, "truncated_log"),
        (b"safe\n", {"sha256": "0" * 64}, "digest_mismatch"),
        (b"safe\n", {"task_id": None}, "task_identity_missing"),
        (b"safe\n\x00binary\n", {}, "binary_log"),
        (b"repeat" * 1000, {}, "unbounded_verbosity"),
        (b"\xff\n", {}, "encoding_error"),
    ],
)
def test_invalid_logs_fail_closed(tmp_path: Path, data: bytes, change, reason: str) -> None:
    manifest, root = _case(tmp_path, data=data)
    manifest["logs"][0].update(change)
    _assert_blocked(build_report(manifest, root), reason)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("completion", "unknown", "unknown_completion"),
        ("active_writer", "true", "active_writer_invalid"),
        ("truncated", "false", "truncated_invalid"),
    ],
)
def test_unestablished_completion_or_flags_block_pruning(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"][0][field] = value
    _assert_blocked(build_report(manifest, root), reason)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"completion": "complete", "completion_status": "truncated"}, "completion_contradictory"),
        ({"completion": "complete", "completion_status": "pending"}, "completion_unknown"),
        ({"completion": None}, "completion_malformed"),
        ({"completion": "complete", "completion_status": None}, "completion_malformed"),
    ],
)
def test_completion_aliases_are_reconciled_fail_closed(
    tmp_path: Path, updates: dict[str, object], reason: str
) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"][0].update(updates)
    report = build_report(manifest, root)

    _assert_blocked(report)
    assert any(problem["code"] == reason for problem in report["problems"])


def test_missing_completion_aliases_block_pruning(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"][0].pop("completion")
    manifest["logs"][0].pop("completion_status", None)

    _assert_blocked(build_report(manifest, root), "completion_missing")


def test_completion_status_alias_can_establish_completion(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"][0].pop("completion")
    manifest["logs"][0]["completion_status"] = "complete"

    report = build_report(manifest, root)

    assert report["status"] == "eligible"


def test_public_excerpt_redacts_structured_identity_and_private_values(tmp_path: Path) -> None:
    data = rb'ERROR {"username":"alice"} /root/alice /var/lib/private C:\Users\alice\x.txt \\server\share\alice\x'
    manifest, root = _case(tmp_path, data=data)
    report = build_report(manifest, root)

    assert all(
        value not in json.dumps(report) for value in ("alice", "/root/alice", "/var/lib/private")
    )
    assert report["logs"][0]["excerpt"] is not None


@pytest.mark.parametrize(
    "data",
    (
        b'ERROR {"username":"alice\\" /root/alice}\n',
        b'ERROR {"username":"alice}\n',
        rb"ERROR {\"username\":\"alice\"}\n",
        b'ERROR {"username: "alice"}\n',
    ),
)
def test_malformed_structured_identity_suppresses_excerpt(tmp_path: Path, data: bytes) -> None:
    manifest, root = _case(tmp_path, data=data)
    report = build_report(manifest, root)

    _assert_blocked(report, "redaction_failed")
    assert report["logs"][0]["excerpt"] is None
    assert "alice" not in json.dumps(report)


@pytest.mark.parametrize(
    "change", ({"max_bytes": None}, {"max_log_bytes": 0}, {"loss_declared": "true"})
)
def test_malformed_excerpt_policy_blocks_without_excerpt(
    tmp_path: Path, change: dict[str, object]
) -> None:
    manifest, root = _case(tmp_path)
    manifest["excerpt_policy"].update(change)

    report = build_report(manifest, root)

    _assert_blocked(report)
    assert report["logs"][0]["excerpt"] is None


@pytest.mark.parametrize(
    "data",
    (
        b'ERROR {"token":"supersecret"}\n',
        b'ERROR {"token":"supersecret}\n',
        rb"ERROR {\"token\":\"supersecret\"}\n",
    ),
)
def test_structured_credentials_are_not_exposed(tmp_path: Path, data: bytes) -> None:
    manifest, root = _case(tmp_path, data=data)
    report = build_report(manifest, root)

    assert (report["logs"][0]["excerpt"], "supersecret" in json.dumps(report)) == (None, False)


@pytest.mark.parametrize("case", ("missing_source", "missing_custody", "invalid_member"))
def test_custody_requires_source_and_valid_member_metadata(tmp_path: Path, case: str) -> None:
    manifest, root = _case(tmp_path)
    if case == "missing_source":
        (root / "task-0.log").unlink()
    elif case == "missing_custody":
        manifest.pop("custody")
    else:
        manifest["custody"]["files"][0].update(sha256=None, byte_size=None)
    _assert_blocked(report := build_report(manifest, root))
    assert report["custody"]["verified"] is False


def test_duplicate_and_malformed_logs_block_all_pruning(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    manifest["logs"].extend((deepcopy(manifest["logs"][0]), ["malformed"]))
    report = build_report(manifest, root)

    _assert_blocked(report)
    assert {"duplicate_log", "log_entry_invalid"} <= {p["code"] for p in report["problems"]}


def test_failed_job_keeps_full_log_until_diagnosis_and_transfer(tmp_path: Path) -> None:
    manifest, root = _case(tmp_path)
    manifest["job"].update(state="failed", diagnosis="pending")
    entry = build_report(manifest, root)["logs"][0]

    assert entry["retention_action"] == "retain_full"
    assert "failed_job_full_retention" in entry["reasons"]
    assert entry["prune_eligible"] is False


def test_cli_json_smoke_and_check_only_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, root = _case(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        main(["--check", "--manifest", str(path), "--root", str(root), "--format", "json"])
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["check_only"] is True
