"""CLI `--output` behavior tests for the single-account merge receipt (issues #9001 and #9028).

These cases are self-contained: they monkeypatch the live-evidence, verification, and merge
functions, so they do not depend on base-relative receipt fixtures. Keeping them in their own
module avoids coupling the base-sensitive receipt contract tests to output-persistence changes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.dev import single_account_merge_receipt as receipt_module

if TYPE_CHECKING:
    from pathlib import Path

HEAD_SHA = "a" * 40


def _write_receipt_input(tmp_path: Path) -> Path:
    """Write a minimal JSON receipt input for CLI mode tests."""
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"pr_number": 42, "head_sha": HEAD_SHA}), encoding="utf-8")
    return path


def test_cli_report_only_writes_output_atomically(tmp_path, monkeypatch, capsys) -> None:
    """report-only persists the same JSON payload it prints when --output is requested."""
    monkeypatch.setattr(
        receipt_module, "build_live_evidence", lambda *a, **k: ({"head_sha": HEAD_SHA}, None)
    )
    monkeypatch.setattr(receipt_module, "build_receipt", lambda **k: {"status": "ready"})
    output = tmp_path / "nested" / "receipt.json"

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "report-only",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"status": "ready"}
    assert json.loads(output.read_text(encoding="utf-8")) == printed


def test_cli_validate_writes_output(tmp_path, monkeypatch, capsys) -> None:
    """validate persists the verification payload to --output instead of leaving it absent."""
    receipt_file = _write_receipt_input(tmp_path)
    monkeypatch.setattr(receipt_module, "build_live_evidence", lambda *a, **k: ({}, None))
    monkeypatch.setattr(receipt_module, "verify_receipt", lambda *a, **k: {"passed": True})
    output = tmp_path / "validate.json"

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "validate",
            "--receipt-file",
            str(receipt_file),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"passed": True}
    assert json.loads(capsys.readouterr().out) == {"passed": True}


def test_cli_rejects_same_receipt_and_output_path_before_touching_receipt(
    tmp_path, monkeypatch, capsys
) -> None:
    """A same-path validate request fails before loading or replacing the source receipt."""
    receipt_file = _write_receipt_input(tmp_path)
    original = receipt_file.read_bytes()
    evidence_calls = []
    monkeypatch.setattr(
        receipt_module,
        "build_live_evidence",
        lambda *a, **k: evidence_calls.append((a, k)),
    )

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "validate",
            "--receipt-file",
            str(receipt_file),
            "--output",
            str(receipt_file),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert evidence_calls == []
    assert captured.out == ""
    assert "must resolve to different files" in captured.err
    assert receipt_file.read_bytes() == original


def test_cli_rejects_symlink_alias_for_receipt_and_output(tmp_path, monkeypatch, capsys) -> None:
    """A symlink alias is rejected because it resolves to the source receipt."""
    receipt_file = _write_receipt_input(tmp_path)
    output = tmp_path / "output.json"
    output.symlink_to(receipt_file)
    original = receipt_file.read_bytes()
    monkeypatch.setattr(receipt_module, "build_live_evidence", lambda *a, **k: ({}, None))

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "apply",
            "--receipt-file",
            str(receipt_file),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "must resolve to different files" in captured.err
    assert receipt_file.read_bytes() == original


def test_cli_help_documents_distinct_receipt_and_output_paths() -> None:
    """CLI help states that validate/apply input and output paths must differ."""
    help_text = receipt_module._parser().format_help()

    assert "receipt JSON input for validate/apply" in help_text
    assert "must differ" in help_text
    assert "different file than --receipt-file" in help_text


def test_cli_apply_writes_output(tmp_path, monkeypatch, capsys) -> None:
    """apply persists the returned merge payload to --output."""
    receipt_file = _write_receipt_input(tmp_path)
    monkeypatch.setattr(receipt_module, "build_live_evidence", lambda *a, **k: ({}, None))
    monkeypatch.setattr(
        receipt_module,
        "verify_receipt",
        lambda *a, **k: {"passed": True, "terminal_transition": ""},
    )
    payload = {"status": "merged", "merge_commit_sha": "f" * 40}
    monkeypatch.setattr(receipt_module, "apply_guarded_merge", lambda *a, **k: (payload, None))
    output = tmp_path / "apply.json"

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "apply",
            "--receipt-file",
            str(receipt_file),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == payload


def test_cli_output_write_failure_fails_closed(tmp_path, monkeypatch, capsys) -> None:
    """An unwritable --output fails closed with a nonzero exit and no success payload file."""
    receipt_file = _write_receipt_input(tmp_path)
    monkeypatch.setattr(receipt_module, "build_live_evidence", lambda *a, **k: ({}, None))
    monkeypatch.setattr(receipt_module, "verify_receipt", lambda *a, **k: {"passed": True})
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    exit_code = receipt_module.main(
        [
            "--pr",
            "42",
            "--mode",
            "validate",
            "--receipt-file",
            str(receipt_file),
            "--output",
            str(blocker / "out.json"),
        ]
    )

    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert captured["status"] == "error"
    assert "failed to write output receipt" in captured["error"]
    assert not (blocker / "out.json").exists()


def test_cli_error_payload_is_persisted_when_output_requested(
    tmp_path, monkeypatch, capsys
) -> None:
    """Error payloads are persisted too, so recorders never see a missing declared artifact."""
    monkeypatch.setattr(
        receipt_module, "build_live_evidence", lambda *a, **k: (None, "evidence unavailable")
    )
    output = tmp_path / "error.json"

    exit_code = receipt_module.main(
        ["--pr", "42", "--mode", "report-only", "--output", str(output)]
    )

    captured = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8")) == captured
    assert captured["status"] == "error"
