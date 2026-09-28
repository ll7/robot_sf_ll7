"""Tests for the /tmp validation-copy doctor and task temp registry (#9720)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from scripts.dev import task_temp_registry as registry
from scripts.dev import temp_copy_doctor as doctor

OLD_EPOCH = time.time() - 72 * 3600
FRESH_EPOCH = time.time() - 1 * 3600


def _entry(root: Path, name: str, *, old: bool, git: bool = False) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "payload.bin").write_bytes(b"x" * 1024)
    if git:
        (path / ".git").mkdir()
    stamp = OLD_EPOCH if old else FRESH_EPOCH
    os.utime(path, (stamp, stamp))
    return path


def test_parse_pr_patterns() -> None:
    assert doctor.parse_pr("robot-sf-pr9709-validation-1") == 9709
    assert doctor.parse_pr("robot-sf-9105-audit.AbC123") == 9105
    assert doctor.parse_pr("robot-sf-unrelated") is None


def test_report_classifies_entries(tmp_path: Path) -> None:
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    _entry(tmp_root, "robot-sf-pr-ready-locks", old=True)
    _entry(tmp_root, "robot-sf-pr1111-fresh", old=False)
    _entry(tmp_root, "robot-sf-pr2222-old", old=True, git=True)
    _entry(tmp_root, "robot-sf-stray-copy", old=True)
    done = _entry(tmp_root, "robot-sf-pr3333-done", old=True)
    (registry_dir / "task.json").write_text(
        json.dumps(
            {
                "schema": registry.SCHEMA,
                "task": "task",
                "status": "completed",
                "temporary_paths": [{"path": str(done), "registered_at": "x", "removed_at": None}],
            }
        ),
        encoding="utf-8",
    )
    held = _entry(tmp_root, "robot-sf-pr4444-held", old=True)
    fd = os.open(held, os.O_RDONLY)
    try:
        report = doctor.build_report(tmp_root, registry_dir=registry_dir, min_age_hours=24.0)
    finally:
        os.close(fd)
    classes = {row["path"]: row["classification"] for row in report["entries"]}
    assert classes[str(tmp_root / "robot-sf-pr-ready-locks")] == "protected"
    assert classes[str(tmp_root / "robot-sf-pr1111-fresh")] == "active"
    assert classes[str(tmp_root / "robot-sf-pr2222-old")] == "orphan_pr"
    assert classes[str(tmp_root / "robot-sf-stray-copy")] == "unclassified"
    assert classes[str(done)] == "registered_done"
    if Path("/proc").is_dir():
        assert classes[str(held)] == "active"
    else:
        assert classes[str(held)] in {"active", "orphan_pr"}
    assert report["schema"] == doctor.SCHEMA
    assert report["by_pr"]["2222"]["count"] == 1
    assert report["by_pr"]["no-pr"]["count"] == 2  # lock root + stray copy


def test_apply_requires_explicit_selection(tmp_path: Path) -> None:
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    _entry(tmp_root, "robot-sf-pr5555-orphan", old=True)
    _entry(tmp_root, "robot-sf-stray", old=True)
    report = doctor.build_report(tmp_root, registry_dir=tmp_path / "missing", min_age_hours=24)
    outcome = doctor.apply_removals(report, assume_merged_prs=set())
    assert outcome == []
    assert (tmp_root / "robot-sf-pr5555-orphan").exists()

    outcome = doctor.apply_removals(report, assume_merged_prs={5555})
    assert len(outcome) == 1 and outcome[0]["ok"] is True
    assert not (tmp_root / "robot-sf-pr5555-orphan").exists()
    assert (tmp_root / "robot-sf-stray").exists()


def test_cli_dry_run_and_guarded_apply(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tmp_root = tmp_path / "tmp"
    tmp_root.mkdir()
    _entry(tmp_root, "robot-sf-pr6666-old", old=True)
    argv = ["--tmp-root", str(tmp_root), "--registry-dir", str(tmp_path / "reg")]
    assert doctor.main(argv) == 0
    assert (tmp_root / "robot-sf-pr6666-old").exists()
    assert "dry-run" in capsys.readouterr().out

    assert doctor.main([*argv, "--apply"]) == 0
    assert (tmp_root / "robot-sf-pr6666-old").exists(), "unselected orphan must survive"

    assert doctor.main([*argv, "--apply", "--pr", "6666", "--assume-merged"]) == 0
    assert not (tmp_root / "robot-sf-pr6666-old").exists()


def test_assume_merged_requires_pr(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        doctor.main(
            [
                "--tmp-root",
                str(tmp_path),
                "--registry-dir",
                str(tmp_path / "reg"),
                "--apply",
                "--assume-merged",
            ]
        )


def test_registry_register_complete_remove(tmp_path: Path) -> None:
    tmp_root = tmp_path / "tmp"
    copy = tmp_root / "robot-sf-pr7777-validation"
    copy.mkdir(parents=True)
    registry_dir = str(tmp_path / "registry")
    assert (
        registry.main(
            ["--registry-dir", registry_dir, "register", "--task", "t1", "--path", str(copy)]
        )
        == 0
    )
    receipt = json.loads((tmp_path / "registry" / "t1.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == registry.SCHEMA
    assert receipt["temporary_paths"][0]["path"] == str(copy)

    assert (
        registry.main(["--registry-dir", registry_dir, "complete", "--task", "t1", "--remove"]) == 0
    )
    assert not copy.exists()
    receipt = json.loads((tmp_path / "registry" / "t1.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["temporary_paths"][0]["removed_at"] is not None

    report = doctor.build_report(tmp_root, registry_dir=tmp_path / "registry", min_age_hours=24.0)
    assert report["entries"] == [], "removed registered path must leave no entry"


def test_registry_refuses_non_temp_and_repo_paths(tmp_path: Path) -> None:
    registry_dir = str(tmp_path / "registry")
    repo_root = Path(registry.__file__).resolve().parents[2]
    rc = registry.main(
        [
            "--registry-dir",
            registry_dir,
            "register",
            "--task",
            "t",
            "--path",
            str(repo_root / "scripts"),
        ]
    )
    assert rc == 1
    assert not (tmp_path / "registry" / "t.json").exists()
    rc = registry.main(
        ["--registry-dir", registry_dir, "register", "--task", "t", "--path", "relative/path"]
    )
    assert rc == 1
