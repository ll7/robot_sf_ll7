"""Contract tests for token-saving and shared-routing compatibility entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(
    script: str, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env.pop("CODEX_ROUTING_REPO", None)
    if env:
        child_env.update(env)
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *arguments],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_checkpoint_is_machine_readable_and_conservative() -> None:
    """The checkpoint stays callable and records unavailable route state safely."""
    result = _run(
        "save-codex-token-checkpoint.py",
        "--task-class",
        "issue_implementation",
        "--prompt",
        "bounded test task",
        "--format",
        "json",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "token_saving_checkpoint.v1"
    assert payload["route"]["status"] == "unavailable"
    assert "route-unavailable" in payload["next_action"]
    assert any("read-active-ledger.py" in command for command in payload["recommended_commands"])


def test_missing_canonical_route_entrypoints_fail_closed_with_json() -> None:
    """Missing shared routing capability is explicit, not silently replaced."""
    for script in ("advise-provider-routing.py", "resolve-route.py"):
        result = _run(script, "--json")
        assert result.returncode == 2
        payload = json.loads(result.stdout)
        assert payload["status"] == "unavailable"
        assert payload["route_evidence_only"] is True


def test_route_help_is_available_without_shared_checkout() -> None:
    """The documented help command remains usable when shared routing is absent."""
    result = _run("resolve-route.py", "--help")

    assert result.returncode == 0
    assert "CODEX_ROUTING_REPO" in result.stdout


def test_route_wrapper_forwards_to_configured_shared_checkout(tmp_path: Path) -> None:
    """Configured routing delegates instead of recreating provider policy locally."""
    shared = tmp_path / "shared"
    scripts = shared / "scripts"
    scripts.mkdir(parents=True)
    resolver = scripts / "resolve-route.py"
    resolver.write_text(
        "import json\n"
        "print(json.dumps({'schema_version': 'route_resolution.v1', 'status': 'available'}))\n",
        encoding="utf-8",
    )

    result = _run("resolve-route.py", "--json", env={"CODEX_ROUTING_REPO": str(shared)})

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "schema_version": "route_resolution.v1",
        "status": "available",
    }


def test_active_ledger_reader_returns_compact_latest_entry(tmp_path: Path) -> None:
    """The ledger reader handles an explicit directory without broad log output."""
    ledger = tmp_path / "active"
    ledger.mkdir()
    (ledger / "latest.md").write_text("line one\nline two\n", encoding="utf-8")

    result = _run(
        "read-active-ledger.py",
        "--ledger-dir",
        str(ledger),
        "--json",
        "--tail-lines",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["entries"][0]["tail"] == ["line two"]
