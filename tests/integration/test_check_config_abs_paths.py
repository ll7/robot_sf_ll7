"""
Tests for the configs/** absolute-path lint hook (issue #3605).

Covers the reject path (an unannotated `/home/...` leak), the allowlist path
(an intentional, annotated absolute path), the clean path, and that files
outside `configs/` are ignored.
"""

from pathlib import Path

from hooks.check_config_abs_paths import find_abs_path_violations


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

    def test_repo_baseline_is_clean(self):
        """The real configs/ tree must already pass (only annotated paths)."""
        configs = Path("configs")
        files = [str(p) for p in configs.rglob("*") if p.is_file()]
        result = find_abs_path_violations(files)
        assert result["status"] == "pass", result["message"]
