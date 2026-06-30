"""Tests for the local model artifact preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.local_model_artifacts import (
    BLOCKLIST_ACTIVE,
    BLOCKLIST_ORPHANED_CONFIG_MISSING,
    BLOCKLIST_ORPHANED_REFERENCE_GONE,
    audit_blocklist_coverage,
    validate_no_local_model_artifacts,
)
from scripts.validation.check_local_model_artifacts import (
    DEFAULT_BLOCKLIST,
    REPO_ROOT,
    audit_blocklist,
    check_local_model_artifacts,
    main,
)


def _write_blocklist(path: Path, entries: str) -> None:
    """Write a minimal v1 blocklist with the given ``blocked_references`` block."""
    path.write_text(
        "version: 1\nblocked_references:\n" + entries,
        encoding="utf-8",
    )


def test_checked_in_baseline_configs_have_no_local_output_model_paths() -> None:
    """Tracked baseline local model paths should be visible blockers, not silent dependencies."""
    rows = check_local_model_artifacts([Path("configs/baselines")])

    assert rows == []
    assert {row.status for row in rows} == set()
    assert {row.availability for row in rows} == set()
    assert {row.decision for row in rows} == set()
    assert {
        "configs/baselines/ppo_15m_grid_socnav_holonomic.yaml",
        "configs/baselines/ppo_issue_576_br06_v2_15m.yaml",
    }.isdisjoint({row.path for row in rows})


def test_checked_in_promoted_surfaces_have_no_local_output_model_paths() -> None:
    """Promoted benchmark config surfaces should resolve through durable ids or pointers."""
    rows = check_local_model_artifacts([])

    assert rows == []


def test_default_cli_paths_are_stable_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default repo config surfaces should resolve even when invoked below the repo root."""
    monkeypatch.chdir(REPO_ROOT / "scripts")
    monkeypatch.setattr("sys.argv", ["check_local_model_artifacts.py"])

    assert main() == 0
    stdout = capsys.readouterr().out
    assert stdout == "OK: no local output model_path/resume_from references found.\n"
    assert "../configs" not in stdout


def test_promoted_surface_paths_expand_from_repo_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Repo-relative promoted surfaces should not depend on the caller's current directory."""
    config = tmp_path / "configs" / "promoted.yaml"
    promoted_surfaces = tmp_path / "configs" / "promoted_config_surfaces.yaml"
    caller_cwd = tmp_path / "scripts"
    config.parent.mkdir()
    caller_cwd.mkdir()
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")
    promoted_surfaces.write_text(
        """
version: 1
promoted_configs:
  - path: configs/promoted.yaml
    reason: Synthetic promoted config.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(caller_cwd)
    monkeypatch.setattr(
        "scripts.validation.check_local_model_artifacts.REPO_ROOT",
        tmp_path,
    )

    rows = check_local_model_artifacts(
        [],
        blocklist_path=tmp_path / "missing.yaml",
        promoted_surfaces_path=promoted_surfaces,
    )

    assert len(rows) == 1
    assert rows[0].path == "configs/promoted.yaml"
    assert rows[0].status == "promoted_blocked"


def test_unblocked_local_model_path_fails_preflight(tmp_path: Path) -> None:
    """A new output model path should fail unless it is explicitly blocklisted."""
    config = tmp_path / "candidate.yaml"
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")

    rows = check_local_model_artifacts([config], blocklist_path=tmp_path / "missing.yaml")

    assert len(rows) == 1
    assert rows[0].status == "unblocked"
    assert rows[0].field == "model_path"


def test_scanner_and_runtime_validator_share_blocklist_contract(tmp_path: Path) -> None:
    """Scanner rows and benchmark runtime errors should agree on local-artifact blockers."""
    config = tmp_path / "candidate.yaml"
    blocklist = tmp_path / "blocklist.yaml"
    promoted_surfaces = tmp_path / "promoted_surfaces.yaml"
    config.write_text("model_path: output/model_cache/demo/model.zip\n", encoding="utf-8")
    blocklist.write_text(
        f"""
version: 1
follow_up_issue: https://github.com/ll7/robot_sf_ll7/issues/1764
blocked_references:
  - path: {config.as_posix()}
    field: model_path
    value: output/model_cache/demo/model.zip
    reason: Synthetic local artifact is not durable.
""".strip()
        + "\n",
        encoding="utf-8",
    )
    promoted_surfaces.write_text("version: 1\npromoted_configs: []\n", encoding="utf-8")

    rows = check_local_model_artifacts(
        [config],
        blocklist_path=blocklist,
        promoted_surfaces_path=promoted_surfaces,
    )

    assert rows[0].status == "blocked"
    assert rows[0].availability == ""
    assert rows[0].reason == "Synthetic local artifact is not durable."
    with pytest.raises(ValueError, match="Synthetic local artifact is not durable") as exc_info:
        validate_no_local_model_artifacts(
            yaml.safe_load(config.read_text(encoding="utf-8")),
            config_path=config,
            blocklist_path=blocklist,
        )
    assert "https://github.com/ll7/robot_sf_ll7/issues/1764" in str(exc_info.value)


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


def test_fail_on_blocked_is_zero_when_shipped_baselines_are_retired(
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

    assert main() == 0
    assert "OK: no local output" in capsys.readouterr().out


def test_checked_in_blocklist_is_empty_after_retirement() -> None:
    """Every shipped blocklist entry should still cover a present local reference."""
    entries = audit_blocklist_coverage(DEFAULT_BLOCKLIST, repo_root=REPO_ROOT)

    assert entries == []
    assert {entry.status for entry in entries} == set()


def test_audit_flags_retired_config_as_orphan(tmp_path: Path) -> None:
    """A blocklist entry whose config was removed should be reported as orphaned."""
    blocklist = tmp_path / "blocklist.yaml"
    _write_blocklist(
        blocklist,
        """  - path: configs/baselines/retired.yaml
    field: model_path
    value: output/model_cache/retired/model.zip
    reason: Retired checkpoint.
""",
    )

    entries = audit_blocklist_coverage(blocklist, repo_root=tmp_path)

    assert len(entries) == 1
    assert entries[0].status == BLOCKLIST_ORPHANED_CONFIG_MISSING
    assert entries[0].path == "configs/baselines/retired.yaml"


def test_audit_flags_migrated_config_as_orphan(tmp_path: Path) -> None:
    """A config migrated to a durable model_id should orphan its old blocklist entry."""
    config = tmp_path / "configs" / "baselines" / "migrated.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model_id: durable_registry_id\n", encoding="utf-8")
    blocklist = tmp_path / "blocklist.yaml"
    _write_blocklist(
        blocklist,
        """  - path: configs/baselines/migrated.yaml
    field: model_path
    value: output/model_cache/migrated/model.zip
    reason: Pre-migration local artifact.
""",
    )

    entries = audit_blocklist_coverage(blocklist, repo_root=tmp_path)

    assert len(entries) == 1
    assert entries[0].status == BLOCKLIST_ORPHANED_REFERENCE_GONE


def test_audit_marks_present_reference_active(tmp_path: Path) -> None:
    """A blocklist entry that still matches a live local reference stays active."""
    config = tmp_path / "configs" / "baselines" / "live.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model_path: output/model_cache/live/model.zip\n", encoding="utf-8")
    blocklist = tmp_path / "blocklist.yaml"
    _write_blocklist(
        blocklist,
        """  - path: configs/baselines/live.yaml
    field: model_path
    value: output/model_cache/live/model.zip
    reason: Still-local checkpoint.
""",
    )

    entries = audit_blocklist_coverage(blocklist, repo_root=tmp_path)

    assert len(entries) == 1
    assert entries[0].status == BLOCKLIST_ACTIVE


def test_audit_cli_fails_on_orphan_and_reports_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``--audit-blocklist`` CLI mode should fail and emit rows when orphans exist."""
    blocklist = tmp_path / "blocklist.yaml"
    _write_blocklist(
        blocklist,
        """  - path: configs/baselines/retired.yaml
    field: model_path
    value: output/model_cache/retired/model.zip
    reason: Retired checkpoint.
""",
    )
    monkeypatch.setattr("scripts.validation.check_local_model_artifacts.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_local_model_artifacts.py",
            "--audit-blocklist",
            "--json",
            "--blocklist",
            str(blocklist),
        ],
    )

    assert main() == 1
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["status"] == BLOCKLIST_ORPHANED_CONFIG_MISSING


def test_audit_helper_uses_default_blocklist() -> None:
    """The script-level helper should audit the shipped blocklist by default."""
    entries = audit_blocklist()

    assert entries == []
    assert all(entry.status == BLOCKLIST_ACTIVE for entry in entries)


def test_audit_fails_closed_on_missing_blocklist(tmp_path: Path) -> None:
    """A missing blocklist path must raise rather than report a vacuously green audit."""
    with pytest.raises(FileNotFoundError):
        audit_blocklist_coverage(tmp_path / "absent.yaml", repo_root=tmp_path)
