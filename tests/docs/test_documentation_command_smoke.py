"""Regression tests for the documentation command runner (issue #8726)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validation.run_documentation_commands import (
    BlockResult,
    iter_tagged_blocks,
    main,
    normalize_output,
    run_block,
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


def test_screen_rejects_destructive() -> None:
    assert screen_command("rm -rf /tmp/work") == "destructive_token"


def test_screen_rejects_absolute_path() -> None:
    assert screen_command("cat /etc/hostname") == "absolute_path"
    assert screen_command("cat /dev/null") is None


def test_screen_rejects_output_escape() -> None:
    assert screen_command("cd .. && ls") == "output_escape"


def test_unterminated_tagged_fence_fails_closed(tmp_path: Path) -> None:
    """Malformed tagged Markdown is rejected instead of executing through EOF."""
    page = tmp_path / "page.md"
    page.write_text("```bash exec-doc-root\nuv run robot-sf --help\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unterminated tagged fence"):
        iter_tagged_blocks(page)


def test_run_block_records_rejection_without_executing(tmp_path: Path) -> None:
    marker = tmp_path / "touched"
    receipt = run_block("page.md", 1, f"curl example.com && touch {marker}", "exec-doc-root", 30.0)
    assert receipt.reason == "network_access"
    assert receipt.exit_code is None
    assert not marker.exists()


def test_run_block_timeout_is_bounded() -> None:
    receipt = run_block("page.md", 1, "sleep 30", "exec-doc", 0.2)
    assert receipt.reason == "timeout"


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


def test_json_cli_output_is_parseable(monkeypatch, capsys) -> None:
    """JSON mode emits only the canonical receipt, without a text summary suffix."""
    monkeypatch.setattr(
        "scripts.validation.run_documentation_commands.run_profile",
        lambda profile, timeout_s: [
            BlockResult("page.md", 1, "true", "exec-doc", 0, "out", "err", 0.123, "ok")
        ],
    )
    assert main(["--format", "json", "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
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
