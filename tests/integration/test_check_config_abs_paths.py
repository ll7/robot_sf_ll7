"""
Tests for the configs/** absolute-path lint hook (issue #3605).

Covers the reject path (an unannotated `/home/...` leak), the allowlist path
(an intentional, annotated absolute path), the clean path, and that files
outside `configs/` are ignored.
"""

import subprocess
from pathlib import Path

import pytest

from hooks.check_config_abs_paths import find_abs_path_violations, main


def _write(base: Path, rel: str, content: str) -> str:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestCheckConfigAbsPaths:
    """Behavioural tests for ``find_abs_path_violations``."""

    def test_rejects_unannotated_home_dir_path(self, tmp_path, monkeypatch):
        """An unannotated /home/ path under configs/ is a violation."""
        monkeypatch.chdir(tmp_path)
        f = _write(
            tmp_path,
            "configs/training/leaky.yaml",
            "script: /home/someone/git/thing/run.sl\n",
        )
        result = find_abs_path_violations([f])
        assert result["status"] == "fail"
        assert len(result["violations"]) == 1
        assert result["violations"][0]["line"] == 1

    def test_allows_annotated_path(self, tmp_path, monkeypatch):
        """A line marked with `allow-abs-path` is permitted."""
        monkeypatch.chdir(tmp_path)
        f = _write(
            tmp_path,
            "configs/training/routed.yaml",
            "script: /home/u/private-ops/run.sl  # allow-abs-path: private-ops routing\n",
        )
        result = find_abs_path_violations([f])
        assert result["status"] == "pass"
        assert result["violations"] == []

    def test_passes_clean_relative_config(self, tmp_path, monkeypatch):
        """A config using only repo-relative paths passes."""
        monkeypatch.chdir(tmp_path)
        f = _write(
            tmp_path,
            "configs/training/clean.yaml",
            "config: configs/training/ppo/recipe.yaml\nnum_envs: 22\n",
        )
        result = find_abs_path_violations([f])
        assert result["status"] == "pass"

    def test_detects_users_and_root_prefixes(self, tmp_path, monkeypatch):
        """/Users/ and /root/ prefixes are flagged alongside /home/."""
        monkeypatch.chdir(tmp_path)
        f = _write(
            tmp_path,
            "configs/a.yaml",
            "a: /Users/x/y\nb: /root/z\n",
        )
        result = find_abs_path_violations([f])
        assert result["status"] == "fail"
        assert len(result["violations"]) == 2

    def test_ignores_files_outside_configs(self, tmp_path, monkeypatch):
        """Files outside a configs/ directory are not scanned."""
        monkeypatch.chdir(tmp_path)
        f = _write(tmp_path, "scripts/run.sh", "echo /home/someone/x\n")
        result = find_abs_path_violations([f])
        assert result["status"] == "pass"
        assert "nothing to check" in result["message"].lower()

    def test_repo_baseline_is_clean(self, monkeypatch):
        """The real tracked configs/ tree must already pass ``--all``."""
        monkeypatch.setattr("sys.argv", ["check_config_abs_paths.py", "--all"])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_all_ignores_untracked_config_files(self, tmp_path, monkeypatch):
        """``--all`` scans Git-tracked configs, not local untracked outputs."""
        monkeypatch.chdir(tmp_path)
        _write(tmp_path, "configs/training/clean.yaml", "script: scripts/run.sh\n")
        _write(tmp_path, "configs/local/leaky.yaml", "script: /home/dev/cache/run.sh\n")

        subprocess.run(["git", "init"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "add", "configs/training/clean.yaml"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test User",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "track clean config",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        monkeypatch.setattr("sys.argv", ["check_config_abs_paths.py", "--all"])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0

    def test_all_checks_tracked_path_with_spaces(self, tmp_path, monkeypatch):
        """``--all`` still inspects tracked configs whose path contains spaces.

        ``git ls-files`` C-quotes such paths in its default output, so the
        scan must use the null-delimited (``-z``) listing to see them.
        """
        monkeypatch.chdir(tmp_path)
        rel = "configs/my run/leaky.yaml"
        _write(tmp_path, rel, "script: /home/dev/cache/run.sh\n")

        subprocess.run(["git", "init"], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "add", rel],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test User",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "track spaced config",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        monkeypatch.setattr("sys.argv", ["check_config_abs_paths.py", "--all"])
        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
