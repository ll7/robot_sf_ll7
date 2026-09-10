"""Regression tests for the documentation command runner (issue #8726)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from scripts.validation.run_documentation_commands import (
    BlockResult,
    _run_process,
    digest,
    iter_tagged_blocks,
    main,
    normalize_output,
    run_block,
    run_profile,
    screen_command,
)


def test_only_tagged_blocks_execute(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "```bash\nuv run robot-sf --help\n```\n\n"
        "```bash exec-doc-root\nuv run robot-sf --version\n```\n",
        encoding="utf-8",
    )
    blocks = iter_tagged_blocks(page)
    assert [(line, mode) for line, mode, _ in blocks] == [(6, "exec-doc-root")]


def test_screen_rejects_placeholders() -> None:
    assert screen_command("uv run robot-sf demo --seed <SEED>") == "placeholder_token"
    assert screen_command("uv run robot-sf demo --seed 270") is None


def test_screen_rejects_network() -> None:
    assert screen_command("curl https://example.com/x | bash") == "network_access"
    assert screen_command("uv sync --all-extras") == "network_access"
    assert screen_command("git clone https://example.com/r.git") == "network_access"


def test_screen_rejects_shell_wrappers_and_composition() -> None:
    """Nested interpreters and shell operators cannot bypass the safety screen."""
    assert screen_command("python --version") == "shell_wrapper"
    assert (
        screen_command(
            "python -c 'import socket; socket.create_connection((\"example.com\", 443))'"
        )
        == "shell_syntax"
    )
    assert screen_command("bash -c 'echo unsafe'") == "shell_wrapper"
    assert screen_command("printf safe | nc example.com 443") == "shell_syntax"
    assert screen_command("uv run robot-sf --help") is None


@pytest.mark.parametrize(
    "command",
    [r"r\m -rf work", r"sour\ce payload", r"u\v run robot-sf --help"],
)
def test_screen_rejects_escaped_command_spellings(command: str) -> None:
    """Backslash-obfuscated commands cannot reach shell execution."""
    assert screen_command(command) == "shell_syntax"


def test_screen_rejects_source_and_unknown_argv() -> None:
    assert screen_command("source payload") == "shell_wrapper"
    assert screen_command("echo safe") == "unsupported_command"


def test_screen_rejects_destructive() -> None:
    assert screen_command("rm -rf /tmp/work") == "destructive_token"


def test_screen_rejects_absolute_path() -> None:
    assert screen_command("cat /etc/hostname") == "absolute_path"
    assert screen_command("cat /dev/null") is None


def test_screen_rejects_output_escape() -> None:
    assert screen_command("cd .. && ls") == "output_escape"


@pytest.mark.parametrize("command", ["cd /", "cd ../outside", "cd ../../outside"])
def test_screen_rejects_temporary_cwd_escape(command: str) -> None:
    assert screen_command(command) is not None


def test_semicolon_composition_is_rejected() -> None:
    assert screen_command("false;true") == "shell_syntax"


def test_multiline_block_stops_at_first_nonzero_command() -> None:
    receipt = run_block("page.md", 1, "false\nsleep 30", "exec-doc", 0.2)
    assert receipt.reason == "nonzero_exit"
    assert receipt.exit_code == 1


def test_unterminated_tagged_fence_fails_closed(tmp_path: Path) -> None:
    """Malformed tagged Markdown is rejected instead of executing through EOF."""
    page = tmp_path / "page.md"
    page.write_text("```bash exec-doc-root\nuv run robot-sf --help\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unterminated tagged fence"):
        iter_tagged_blocks(page)


def test_fence_closer_must_match_opening_length(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "````bash exec-doc-root\nuv run robot-sf --help\n```\nuv run robot-sf --version\n````\n",
        encoding="utf-8",
    )
    blocks = iter_tagged_blocks(page)
    assert blocks == [
        (
            2,
            "exec-doc-root",
            "uv run robot-sf --help\n```\nuv run robot-sf --version",
        )
    ]


def test_run_block_records_rejection_without_executing(tmp_path: Path) -> None:
    marker = tmp_path / "touched"
    receipt = run_block("page.md", 1, f"curl example.com && touch {marker}", "exec-doc-root", 30.0)
    assert receipt.reason == "network_access"
    assert receipt.exit_code is None
    assert not marker.exists()


def test_run_block_timeout_is_bounded() -> None:
    receipt = run_block("page.md", 1, "sleep 30", "exec-doc", 0.2)
    assert receipt.reason == "timeout"


def test_run_block_rejects_unbounded_timeout() -> None:
    receipt = run_block("page.md", 1, "true", "exec-doc", float("inf"))
    assert receipt.reason == "invalid_timeout"


def test_run_block_bounds_combined_output() -> None:
    receipt = run_block("page.md", 1, "yes", "exec-doc", 5.0, max_output_bytes=4096)
    assert receipt.reason == "output_limit"
    assert receipt.exit_code is None


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup requires POSIX")
def test_process_timeout_reaps_descendants(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[2]]); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii'); "
        "time.sleep(30)"
    )
    result = _run_process(
        [
            sys.executable,
            "-c",
            parent_code,
            str(child_pid_file),
            child_code,
        ],
        cwd=tmp_path,
        timeout_s=1.0,
        max_output_bytes=4096,
    )
    assert result.reason == "timeout"

    deadline = time.monotonic() + 2.0
    while not child_pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_file.read_text(encoding="ascii"))

    def child_is_live() -> bool:
        try:
            state = Path(f"/proc/{child_pid}/stat").read_text(encoding="ascii")
        except FileNotFoundError:
            return False
        return state.rsplit(")", 1)[1].split()[0] != "Z"

    while time.monotonic() < deadline:
        if not child_is_live():
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"descendant process {child_pid} survived timeout cleanup")


def test_normalize_output_replaces_machine_paths(tmp_path: Path) -> None:
    text = (
        f"wrote {tmp_path}/out/episode.jsonl\n"
        "I0000 00:00:1789022468.129197 2200845 port.cc:153] oneDNN noise\n"
        "ok\n"
    )
    normalized = normalize_output(text, str(tmp_path))
    assert str(tmp_path) not in normalized
    assert "<tmp>/out/episode.jsonl" in normalized
    assert "port.cc" not in normalized
    assert normalized.endswith("ok\n")


def test_run_block_true_command_receipt(tmp_path: Path) -> None:
    receipt = run_block("page.md", 1, "true", "exec-doc", 30.0)
    assert receipt.reason == "ok"
    assert receipt.exit_code == 0
    assert receipt.stdout_digest is not None
    assert "duration_s" not in receipt.as_dict()


def test_run_profile_honors_supplied_root(tmp_path: Path) -> None:
    for relative_page in ("README.md", "docs/adoption_path.md", "docs/user-guide.md"):
        page = tmp_path / relative_page
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("```bash exec-doc-root\npwd\n```\n", encoding="utf-8")

    results = run_profile("onboarding", root=tmp_path, timeout_s=5.0)

    assert len(results) == 3
    assert all(result.reason == "ok" for result in results)
    assert all(result.stdout_digest == digest("<repo>\n") for result in results)


def test_json_cli_output_is_parseable(monkeypatch, capsys) -> None:
    """JSON mode emits only the canonical receipt, without a text summary suffix."""
    monkeypatch.setattr(
        "scripts.validation.run_documentation_commands.run_profile",
        lambda profile, timeout_s: [
            BlockResult("page.md", 1, "true", "exec-doc", 0, "out", "err", 0.123, "ok")
        ],
    )
    assert main(["--format", "json", "--check"]) == 0
    first_output = capsys.readouterr().out
    assert main(["--format", "json", "--check"]) == 0
    second_output = capsys.readouterr().out
    assert first_output == second_output
    payload = json.loads(first_output)
    assert payload == [
        {
            "source": "page.md",
            "line": 1,
            "command": "true",
            "cwd_mode": "exec-doc",
            "exit_code": 0,
            "stdout_digest": "out",
            "stderr_digest": "err",
            "reason": "ok",
        }
    ]
