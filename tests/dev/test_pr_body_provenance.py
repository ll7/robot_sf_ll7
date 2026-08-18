"""Regression coverage for truthful PR exact-head metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.dev.pr_body_provenance import (
    extract_sha_carriers,
    main,
    validate_sha_carriers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_SCRIPT = REPO_ROOT / "scripts" / "dev" / "pr_body_provenance.py"
LIVE_SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def test_extracts_supported_full_sha_carriers() -> None:
    body = (
        f"gate-verdict: accepted @ {LIVE_SHA}\n"
        f"base-policy: current-base @ {OTHER_SHA}\n"
        f"Exact head: `{LIVE_SHA}`\n"
    )

    assert [(carrier.label, carrier.sha) for carrier in extract_sha_carriers(body)] == [
        ("gate-verdict", LIVE_SHA),
        ("base-policy", OTHER_SHA),
        ("Exact head", LIVE_SHA),
    ]


def test_live_head_carrier_passes_without_local_object_lookup(tmp_path: Path) -> None:
    with patch("scripts.dev.pr_body_provenance.git_object_type") as object_type:
        errors = validate_sha_carriers(
            f"gate-verdict: accepted @ {LIVE_SHA}",
            live_head_sha=LIVE_SHA,
            repo_root=tmp_path,
        )

    assert errors == []
    object_type.assert_not_called()


def test_existing_local_object_preserves_historical_reference(tmp_path: Path) -> None:
    with patch("scripts.dev.pr_body_provenance.git_object_type", return_value="commit"):
        errors = validate_sha_carriers(
            f"base-policy: ordinary-cas @ {OTHER_SHA}",
            live_head_sha=LIVE_SHA,
            repo_root=tmp_path,
        )

    assert errors == []


def test_fabricated_sha_is_rejected_when_not_live_or_local(tmp_path: Path) -> None:
    with patch("scripts.dev.pr_body_provenance.git_object_type", return_value=None):
        errors = validate_sha_carriers(
            f"Exact head: {OTHER_SHA}",
            live_head_sha=LIVE_SHA,
            repo_root=tmp_path,
        )

    assert len(errors) == 1
    assert OTHER_SHA in errors[0]
    assert "neither the live PR head" in errors[0]


def test_cli_accepts_live_head_from_github_event(tmp_path: Path, capsys) -> None:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {"pull_request": {"body": f"Exact head: {LIVE_SHA}", "head": {"sha": LIVE_SHA}}}
        ),
        encoding="utf-8",
    )

    assert main(["--github-event-path", str(event), "--repo-root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["carriers"][0]["sha"] == LIVE_SHA


def test_cli_rejects_fabricated_body_sha(tmp_path: Path, capsys) -> None:
    body = tmp_path / "body.md"
    body.write_text(f"gate-verdict: accepted @ {OTHER_SHA}\n", encoding="utf-8")
    with patch("scripts.dev.pr_body_provenance.git_object_type", return_value=None):
        rc = main(
            [
                "--body-file",
                str(body),
                "--head-sha",
                LIVE_SHA,
                "--repo-root",
            ]
            + [str(tmp_path)]
        )

    assert rc == 1
    assert "validation failed" in capsys.readouterr().out


def test_script_is_executable_and_helpful() -> None:
    result = subprocess.run(
        [sys.executable, str(PROVENANCE_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--github-event-path" in result.stdout
