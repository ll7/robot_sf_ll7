"""Tests for the evidence promotion gate validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "validation" / "validate_evidence_promotion_gate.py"


def run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the validator CLI with captured output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_allows_only_adjacent_promotion_steps() -> None:
    """Promotion checks allow only adjacent evidence-tier transitions."""
    valid = run_validator(
        "--check-promotion",
        "--current-tier",
        "proposal",
        "--claimed-tier",
        "preflight",
    )
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["promotion_check"]["transition_valid"] is True

    invalid = run_validator(
        "--check-promotion",
        "--current-tier",
        "proposal",
        "--claimed-tier",
        "smoke",
    )
    assert invalid.returncode == 1
    payload = json.loads(invalid.stdout)
    assert payload["promotion_check"]["transition_valid"] is False
    assert payload["promotion_check"]["errors"] == ["Invalid transition: proposal -> smoke"]


def test_evidence_bundle_fails_closed_when_smoke_artifacts_are_missing(tmp_path: Path) -> None:
    """Smoke evidence fails closed when reproducibility artifacts are missing."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "summary.md").write_text(
        "Smoke evidence\n\ncommand: uv run python demo.py\n",
        encoding="utf-8",
    )

    result = run_validator("--evidence-bundle", str(bundle), "--claimed-tier", "smoke")

    assert result.returncode == 1
    payload = json.loads(result.stdout)["validation_result"]
    assert payload["claimed_tier"] == "smoke"
    assert payload["transition_valid"] is False
    assert "commit_or_checksum" in payload["missing_artifacts"]
    assert "metric_or_summary" in payload["missing_artifacts"]


def test_evidence_bundle_passes_when_smoke_artifacts_are_present(tmp_path: Path) -> None:
    """Smoke evidence passes when all required artifact signals are present."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "summary.json").write_text(
        json.dumps(
            {
                "command": "uv run python scripts/validation/example.py",
                "commit": "2072e083a6554cbc03638f0941e5c5c74317ef6c",
                "summary": {"episodes_total": 2, "success_rate": 1.0},
            }
        ),
        encoding="utf-8",
    )

    result = run_validator("--evidence-bundle", str(bundle), "--claimed-tier", "smoke")

    assert result.returncode == 0
    payload = json.loads(result.stdout)["validation_result"]
    assert payload["transition_valid"] is True
    assert payload["missing_artifacts"] == []


def test_context_note_matching_is_case_insensitive(tmp_path: Path) -> None:
    """Context-note evidence matching does not depend on author casing."""
    note = tmp_path / "note.md"
    note.write_text(
        "\n".join(
            [
                "# Smoke Evidence",
                "",
                "COMMAND: uv run python scripts/validation/example.py",
                "COMMIT: 2072e083a6554cbc03638f0941e5c5c74317ef6c",
                "SUMMARY JSON: output/summary.json",
            ]
        ),
        encoding="utf-8",
    )

    result = run_validator("--context-note", str(note))

    assert result.returncode == 0
    payload = json.loads(result.stdout)["validation_result"]
    assert payload["claimed_tier"] == "smoke"
    assert payload["transition_valid"] is True
    assert payload["missing_artifacts"] == []


def test_validate_all_emits_diagnostic_only_stuck_results(tmp_path: Path) -> None:
    """Repository-wide validation reports diagnostic-only notes as stuck."""
    note_dir = tmp_path / "docs" / "context"
    note_dir.mkdir(parents=True)
    (note_dir / "diagnostic_note.md").write_text(
        "# Diagnostic Note\n\nThis is diagnostic-only evidence and not a promotion.",
        encoding="utf-8",
    )

    result = run_validator("--root", str(tmp_path))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["validation_summary"]["diagnostic_only"] == 1
    assert payload["diagnostic_only_stuck"] == [
        {
            "context_note": str(note_dir / "diagnostic_note.md"),
            "reason": "diagnostic_only_no_promotion",
        }
    ]


def test_validate_all_ignores_non_directory_scan_roots(tmp_path: Path) -> None:
    """Repository-wide scanning skips placeholder files at expected directory paths."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "context").write_text("not a directory", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "benchmarks").write_text("not a directory", encoding="utf-8")

    result = run_validator("--root", str(tmp_path))

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["validation_summary"]["total_validated"] == 0
