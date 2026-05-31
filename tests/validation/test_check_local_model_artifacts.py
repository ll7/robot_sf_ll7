"""Tests for the local model artifact preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.check_local_model_artifacts import (
    check_local_model_artifacts,
    main,
)


def test_checked_in_baseline_local_paths_are_explicitly_blocked() -> None:
    """Tracked baseline local model paths should be visible blockers, not silent dependencies."""
    rows = check_local_model_artifacts([Path("configs/baselines")])

    assert rows
    assert {row.status for row in rows} == {"blocked"}
    assert {
        "configs/baselines/ppo_15m_grid_socnav_holonomic.yaml",
        "configs/baselines/ppo_issue_576_br06_v2_15m.yaml",
    }.isdisjoint({row.path for row in rows})


def test_checked_in_promoted_surfaces_have_no_local_output_model_paths() -> None:
    """Promoted benchmark config surfaces should resolve through durable ids or pointers."""
    rows = check_local_model_artifacts([])

    assert rows == []


def test_unblocked_local_model_path_fails_preflight(tmp_path: Path) -> None:
    """A new output model path should fail unless it is explicitly blocklisted."""
    config = tmp_path / "candidate.yaml"
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")

    rows = check_local_model_artifacts([config], blocklist_path=tmp_path / "missing.yaml")

    assert len(rows) == 1
    assert rows[0].status == "unblocked"
    assert rows[0].field == "model_path"


def test_promoted_local_model_path_fails_even_when_blocklisted(tmp_path: Path) -> None:
    """Promoted benchmark config surfaces cannot be softened by the local blocker list."""
    config = tmp_path / "promoted.yaml"
    blocklist = tmp_path / "blocklist.yaml"
    promoted_surfaces = tmp_path / "promoted_surfaces.yaml"
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")
    blocklist.write_text(
        f"""
version: 1
blocked_references:
  - path: {config.as_posix()}
    field: model_path
    value: output/model_cache/demo/model.zip
    reason: known local-only blocker
""".strip()
        + "\n",
        encoding="utf-8",
    )
    promoted_surfaces.write_text(
        f"""
version: 1
promoted_configs:
  - path: {config.as_posix()}
    reason: Synthetic promoted config.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = check_local_model_artifacts(
        [config],
        blocklist_path=blocklist,
        promoted_surfaces_path=promoted_surfaces,
    )

    assert len(rows) == 1
    assert rows[0].status == "promoted_blocked"
    assert rows[0].surface == "benchmark_promoted"
    assert "durable model_id" in rows[0].reason


def test_promoted_model_id_config_passes_preflight(tmp_path: Path) -> None:
    """Promoted configs may use durable model ids instead of local output paths."""
    config = tmp_path / "promoted.yaml"
    promoted_surfaces = tmp_path / "promoted_surfaces.yaml"
    config.write_text("model_id: durable_registry_id\n", encoding="utf-8")
    promoted_surfaces.write_text(
        f"""
version: 1
promoted_configs:
  - path: {config.as_posix()}
    reason: Synthetic promoted config.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = check_local_model_artifacts(
        [config],
        blocklist_path=tmp_path / "missing.yaml",
        promoted_surfaces_path=promoted_surfaces,
    )

    assert rows == []


def test_blocklist_must_name_exact_field_and_value(tmp_path: Path) -> None:
    """Blockers are exact so stale allowlist rows do not hide new paths."""
    config = tmp_path / "candidate.yaml"
    blocklist = tmp_path / "blocklist.yaml"
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")
    blocklist.write_text(
        """
version: 1
blocked_references:
  - path: candidate.yaml
    field: model_path
    value: output/model_cache/other/model.zip
    reason: wrong artifact
""".strip()
        + "\n",
        encoding="utf-8",
    )

    rows = check_local_model_artifacts([config], blocklist_path=blocklist)

    assert rows[0].status == "unblocked"


def test_cli_json_reports_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should emit machine-readable rows and fail on unblocked references."""
    config = tmp_path / "candidate.yaml"
    config.write_text("resume_from: output/slurm/job/checkpoint.zip\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_local_model_artifacts.py",
            "--json",
            "--blocklist",
            str(tmp_path / "missing.yaml"),
            str(config),
        ],
    )

    assert main() == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["field"] == "resume_from"
    assert rows[0]["status"] == "unblocked"


def test_fail_on_blocked_returns_nonzero_for_explicit_blockers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strict mode should let CI or audits fail until blocked artifacts are promoted."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_local_model_artifacts.py",
            "--fail-on-blocked",
            "configs/baselines",
        ],
    )

    assert main() == 1
    assert "BLOCKED:" in capsys.readouterr().out
