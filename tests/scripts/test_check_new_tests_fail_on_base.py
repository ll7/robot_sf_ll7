"""Tests for the advisory new-tests-fail-on-base check."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validation"
    / "check_new_tests_fail_on_base.py"
)
_spec = importlib.util.spec_from_file_location("check_new_tests_fail_on_base", _SCRIPT)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules["check_new_tests_fail_on_base"] = mod
_spec.loader.exec_module(mod)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Tiny repo whose base commit has `calc.value() == 1` and one old test."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _write(r, "calc.py", "def value():\n    return 1\n")
    _write(
        r,
        "tests/test_old.py",
        "import calc\n\n\ndef test_unchanged():\n    assert calc.value() == 1\n\n\n"
        "def test_will_change():\n    assert calc.value() == 1\n",
    )
    _commit(r, "base")
    _git(r, "branch", "base")
    _git(r, "checkout", "-q", "-b", "feature")
    return r


def test_collect_tests_detects_class_methods_and_marker() -> None:
    src = (
        "import pytest\n\n"
        "@pytest.mark.parametrize('x', [1, 2])\n"
        "def test_a(x):\n    pass\n\n"
        "class TestK:\n"
        "    @pytest.mark.new_feature_no_base(reason='new api')\n"
        "    def test_m(self):\n        pass\n"
        "    def helper(self):\n        pass\n"
    )
    got = mod.collect_tests(src)
    assert set(got) == {"test_a", "TestK.test_m"}
    assert got["TestK.test_m"][1] == "new api"
    assert got["test_a"][1] is None


def test_param_change_counts_as_changed() -> None:
    a = mod.collect_tests(
        "import pytest\n@pytest.mark.parametrize('x',[1])\ndef test_a(x):\n    pass\n"
    )
    b = mod.collect_tests(
        "import pytest\n@pytest.mark.parametrize('x',[1,2])\ndef test_a(x):\n    pass\n"
    )
    assert a["test_a"][0] != b["test_a"][0]


# seed-holdout: synthetic-fixture begin
def test_seed_guard_needs_benchmark_context() -> None:
    assert mod.has_seed_literals("def test_x():\n    run_episode(seed=120)\n")
    assert not mod.has_seed_literals("def test_x():\n    assert 120 == 120\n")
    assert not mod.has_seed_literals("def test_x():\n    run_episode(seed=42)\n")
    assert mod.has_seed_literals("SEEDS = [1, 120]\ndef test_x():\n    benchmark(SEEDS)\n")
    assert mod.has_seed_literals("def test_x():\n    benchmark(range(111, 141))\n# seed\n")
    # a 111-140 literal unrelated to seeds does not trigger the guard
    # seed-holdout: synthetic-fixture end
    assert not mod.has_seed_literals(
        "# episode seed docs\ndef test_x():\n    assert width == 120\n"
    )


def test_classify_error_separates_third_party_imports() -> None:
    pk = {"robot_sf", "calc"}
    assert (
        mod.classify_error("ModuleNotFoundError", "No module named 'torch'", "call", pk)[0]
        == mod.UNRELATED
    )
    assert (
        mod.classify_error("ModuleNotFoundError", "No module named 'robot_sf.new'", "call", pk)[0]
        == mod.FAILS
    )
    assert mod.classify_error("AttributeError", "no attr", "call", pk)[0] == mod.FAILS
    assert mod.classify_error("RuntimeError", "boom", "setup", pk)[0] == mod.UNRELATED


def test_end_to_end_classification(repo: Path, tmp_path: Path) -> None:
    _write(repo, "calc.py", "def value():\n    return 2\n\n\ndef newfn():\n    return 5\n")
    _write(
        repo,
        "tests/test_old.py",
        "import calc\n\n\ndef test_unchanged():\n    assert calc.value() == 1\n\n\n"
        "def test_will_change():\n    assert calc.value() == 2\n",
    )
    _write(
        repo,
        "tests/test_new.py",
        "import pytest\nimport calc\n\n\n"
        "def test_weak():\n    assert True\n\n\n"
        "def test_detects_fix():\n    assert calc.value() == 2\n\n\n"
        "def test_new_api():\n    assert calc.newfn() == 5\n\n\n"
        "def test_needs_third_party():\n    import no_such_third_party_pkg  # noqa: F401\n\n\n"
        "@pytest.mark.new_feature_no_base(reason='feature absent on base')\n"
        "def test_exempt():\n    assert calc.newfn() == 5\n\n\n"
        "class TestGroup:\n"
        "    @pytest.mark.parametrize('n', [1, 2])\n"
        "    def test_param(self, n):\n        assert calc.value() == 2\n",
    )
    _write(
        # seed-holdout: synthetic-fixture begin
        repo,
        "tests/test_seeds.py",
        "def test_bench():\n    run_episode(seed=120)  # benchmark episode\n",
    )
    # seed-holdout: synthetic-fixture end
    _commit(repo, "feature")

    report = mod.check(repo, "base", "HEAD", timeout=300)
    by_name = {f"{r.path}::{r.qualname}": r for r in report.tests}
    assert "tests/test_old.py::test_unchanged" not in by_name
    assert by_name["tests/test_old.py::test_will_change"].classification == mod.FAILS
    assert by_name["tests/test_old.py::test_will_change"].reason == "changed"
    assert by_name["tests/test_new.py::test_weak"].classification == mod.PASSES
    assert by_name["tests/test_new.py::test_detects_fix"].classification == mod.FAILS
    assert by_name["tests/test_new.py::test_new_api"].classification == mod.FAILS
    assert by_name["tests/test_new.py::test_needs_third_party"].classification == mod.UNRELATED
    assert by_name["tests/test_new.py::test_exempt"].classification == mod.EXEMPT
    assert by_name["tests/test_new.py::TestGroup.test_param"].classification == mod.FAILS
    assert by_name["tests/test_seeds.py::test_bench"].classification == mod.SEED_GUARD
    # the temporary worktree is gone
    assert "newtests-base" not in _git(repo, "worktree", "list")


def test_collection_error_from_missing_api_fails_on_base(repo: Path) -> None:
    _write(
        repo,
        "tests/test_imp.py",
        "from calc import missing_name\n\n\ndef test_it():\n    assert missing_name\n",
    )
    _commit(repo, "feature")
    report = mod.check(repo, "base", "HEAD", timeout=300, allow_test_only=True)
    (rec,) = report.tests
    assert rec.classification == mod.FAILS


def test_main_exit_codes_and_outputs(repo: Path, tmp_path: Path) -> None:
    _write(repo, "calc.py", "def value():\n    return 4\n")
    _write(repo, "tests/test_weak.py", "def test_weak():\n    assert True\n")
    _commit(repo, "feature")
    out = tmp_path / "r.json"
    md = tmp_path / "r.md"
    args = ["--repo", str(repo), "--base", "base", "--json-out", str(out), "--summary-out", str(md)]
    assert mod.main(args) == 0
    data = json.loads(out.read_text())
    assert data["counts"] == {mod.PASSES: 1}
    assert "tests/test_weak.py::test_weak" in md.read_text()
    assert mod.main([*args, "--strict"]) == 1


def test_no_test_changes_is_clean(repo: Path) -> None:
    _write(repo, "calc.py", "def value():\n    return 3\n")
    _commit(repo, "prod only")
    report = mod.check(repo, "base", "HEAD")
    assert report.tests == []


def test_test_only_change_is_skipped_unless_allowed(repo: Path) -> None:
    _write(repo, "tests/test_weak.py", "def test_weak():\n    assert True\n")
    _commit(repo, "tests only")
    report = mod.check(repo, "base", "HEAD", timeout=300)
    assert [r.classification for r in report.tests] == [mod.SKIPPED_TEST_ONLY]
    forced = mod.check(repo, "base", "HEAD", timeout=300, allow_test_only=True)
    assert [r.classification for r in forced.tests] == [mod.PASSES]


def _mini_pkg(repo: Path) -> None:
    _write(repo, "calcpkg/__init__.py", "VALUE = 1\n")


def test_import_guard_passes_for_worktree_package(repo: Path) -> None:
    _mini_pkg(repo)
    _commit(repo, "pkg")
    _git(repo, "branch", "-f", "base")
    _write(repo, "calcpkg/__init__.py", "VALUE = 2\n")
    _write(
        repo,
        "tests/test_pkg.py",
        "import calcpkg\n\n\ndef test_v():\n    assert calcpkg.VALUE == 2\n",
    )
    _commit(repo, "feature")
    report = mod.check(repo, "base", "HEAD", timeout=300, guard_packages=["calcpkg"])
    assert [r.classification for r in report.tests] == [mod.FAILS]


def test_import_guard_aborts_when_stale_package_shadows_worktree(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mini_pkg(repo)
    _commit(repo, "pkg")
    _git(repo, "branch", "-f", "base")
    _write(repo, "calcpkg/__init__.py", "VALUE = 2\n")
    _write(
        repo,
        "tests/test_pkg.py",
        "import calcpkg\n\n\ndef test_v():\n    assert calcpkg.VALUE == 2\n",
    )
    _commit(repo, "feature")
    # a stale head copy earlier on sys.path (safe-path drops the cwd entry)
    stale = tmp_path / "stale"
    _write(stale, "calcpkg/__init__.py", "VALUE = 2\n")
    monkeypatch.setenv("PYTHONPATH", str(stale))
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    report = mod.check(repo, "base", "HEAD", timeout=300, guard_packages=["calcpkg"])
    (rec,) = report.tests
    assert rec.classification == mod.UNRELATED
    assert "import guard" in rec.detail and "outside base worktree" in rec.detail


def _commit_index(repo: Path, msg: str) -> None:
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", msg)


def test_submodule_pointer_change_is_reported_not_symlinked(repo: Path) -> None:
    sha_a, sha_b = "a" * 40, "b" * 40
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha_a},fast-pysf")
    _commit_index(repo, "gitlink a")
    _git(repo, "branch", "-f", "base")
    _git(repo, "update-index", "--cacheinfo", f"160000,{sha_b},fast-pysf")
    _write(repo, "calc.py", "def value():\n    return 9\n")
    _write(
        repo, "tests/test_sub.py", "import calc\n\n\ndef test_v():\n    assert calc.value() == 9\n"
    )
    _git(repo, "add", "calc.py", "tests")
    _commit_index(repo, "feature")
    assert mod._submodule_changed(repo, _git(repo, "rev-parse", "base"), "HEAD")
    report = mod.check(repo, "base", "HEAD", timeout=300, guard_packages=[])
    (rec,) = report.tests
    assert rec.classification == mod.UNRELATED
    assert "fast-pysf submodule pointer changed" in rec.detail
