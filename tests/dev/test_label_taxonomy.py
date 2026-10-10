"""Keep the shared decision-cockpit label reference tied to local authorities."""

from __future__ import annotations

from pathlib import Path

from scripts.dev import issue_audit_core

REPO_ROOT = Path(__file__).parents[2]
TAXONOMY = REPO_ROOT / "docs/ai/label-taxonomy.md"


def test_decision_cockpit_taxonomy_exists_and_links_authorities() -> None:
    """The shared skill's repository-local taxonomy must remain discoverable."""
    text = TAXONOMY.read_text(encoding="utf-8")

    assert "../../CONTRIBUTING.md#issue-state-labels-and-dispatch" in text
    assert "../../scripts/dev/issue_audit_core.py" in text
    for heading in (
        "## Decision flow",
        "## Lifecycle and origin",
        "## Execution state",
        "## Resource",
        "## Type",
        "## Evidence",
    ):
        assert heading in text


def test_taxonomy_covers_classifier_execution_states() -> None:
    """The prose reference must not drift behind the fail-closed classifier."""
    text = TAXONOMY.read_text(encoding="utf-8")

    for label in issue_audit_core.EXECUTION_STATE_LABELS:
        assert f"`{label}`" in text


def test_taxonomy_covers_every_classifier_qualifier() -> None:
    """The prose qualifier list must not drift behind STATE_QUALIFIER_LABELS."""
    from scripts.dev import issue_state_taxonomy

    text = TAXONOMY.read_text(encoding="utf-8")

    missing = [
        label
        for label in sorted(issue_state_taxonomy.STATE_QUALIFIER_LABELS)
        if f"`{label}`" not in text
    ]
    assert missing == [], f"undocumented qualifiers: {missing}"


def test_state_label_inventory_check_fails_closed_on_unclassified_label() -> None:
    """An unknown state:* label must be reported, never silently tolerated."""
    from scripts.dev import check_state_label_inventory as check

    result = check.classify_labels(["state:ready", "state:typo-invented"])

    assert result["ok"] is False
    assert result["unclassified"] == ["state:typo-invented"]
    assert result["deliberately_unclassified"] == []


def test_state_label_inventory_check_allows_only_named_exclusions() -> None:
    """state:done is a named exclusion; a near-miss typo must still fail."""
    from scripts.dev import check_state_label_inventory as check
    from scripts.dev import issue_state_taxonomy

    assert issue_state_taxonomy.DELIBERATELY_UNCLASSIFIED_STATE_LABELS == {"state:done"}

    ok = check.classify_labels(["state:ready", "state:done"])
    assert ok["ok"] is True
    assert ok["deliberately_unclassified"] == ["state:done"]

    near_miss = check.classify_labels(["state:done ", "state:done-ish"])
    assert near_miss["ok"] is False
    assert "state:done-ish" in near_miss["unclassified"]


def test_state_label_inventory_check_accepts_current_qualifiers() -> None:
    from scripts.dev import check_state_label_inventory as check
    from scripts.dev import issue_state_taxonomy

    labels = sorted(issue_state_taxonomy.KNOWN_STATE_LABELS) + ["state:done"]
    result = check.classify_labels(labels)

    assert result["ok"] is True
    assert result["unclassified"] == []


def test_state_label_inventory_check_reads_a_labels_file(tmp_path) -> None:
    """The offline input keeps the unit test hermetic (no network)."""
    from scripts.dev import check_state_label_inventory as check

    path = tmp_path / "labels.txt"
    path.write_text("state:ready\n# a comment\nstate:reviewing\n", encoding="utf-8")

    result = check.classify_labels(check._read_labels_file(str(path)))
    assert result["ok"] is True


def test_live_inventory_uses_shared_rest_transport(monkeypatch, capsys) -> None:
    """The live CLI reads paginated label names using the real REST adapter."""
    import json
    import subprocess

    from scripts.dev import _gh_rest
    from scripts.dev import check_state_label_inventory as check

    calls = []

    def capture_request(args, payload=None, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, stdout="state:ready\nstate:reviewing\nstate:done\n", stderr=""
        )

    monkeypatch.setattr(_gh_rest, "run_gh_command", capture_request)
    assert check.main(["--repo", "ll7/robot_sf_ll7", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["source"] == "github_api"
    assert result["state_labels"] == ["state:done", "state:ready", "state:reviewing"]
    assert calls == [["api", "repos/ll7/robot_sf_ll7/labels", "--paginate", "--jq", ".[].name"]]


def test_failed_live_inventory_cannot_report_empty_success(monkeypatch, capsys) -> None:
    """A failed inventory read must exit with an explicit unavailable receipt."""
    import json
    import subprocess

    from scripts.dev import _gh_rest
    from scripts.dev import check_state_label_inventory as check

    monkeypatch.setattr(
        _gh_rest,
        "run_gh_command",
        lambda args, *a, **kw: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="unavailable"
        ),
    )
    assert check.main(["--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["source"] == "github_api"
    assert result["error"] == "label inventory read failed (exit 1)"


def test_inventory_direct_cli_prefers_its_checkout(tmp_path) -> None:
    """Direct execution must not import taxonomy from a competing checkout."""
    import json
    import os
    import subprocess
    import sys

    competitor = tmp_path / "other"
    package = competitor / "scripts" / "dev"
    package.mkdir(parents=True)
    (competitor / "scripts" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "issue_state_taxonomy.py").write_text("raise RuntimeError('wrong checkout')\n")
    labels = tmp_path / "labels.txt"
    labels.write_text("state:reviewing\n")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/dev/check_state_label_inventory.py"),
            "--labels-file",
            str(labels),
            "--json",
        ],
        cwd=competitor,
        env={**os.environ, "PYTHONPATH": str(competitor)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
