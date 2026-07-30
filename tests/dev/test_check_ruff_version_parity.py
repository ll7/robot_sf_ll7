"""Tests for the Ruff version parity guard.

Proves the guard passes at HEAD and fails when any one axis is mutated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.dev.check_ruff_version_parity import (
    evaluate,
    normalize_version,
    read_pre_commit_version,
    read_pyproject_dev_dep_pin,
    read_pyproject_required_version,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v0.16.0", "0.16.0"),
        ("==0.16.0", "0.16.0"),
        ("ruff==0.16.0", "0.16.0"),
        ("ruff == 0.16.0", "0.16.0"),
        ("0.16.0", "0.16.0"),
        ("  v0.16.0  ", "0.16.0"),
        ('"==0.16.0"', "0.16.0"),
        ("ruff==v0.16.0", "0.16.0"),
    ],
)
def test_normalize_version(raw: str, expected: str) -> None:
    """Various prefix forms are stripped to the bare X.Y.Z."""
    assert normalize_version(raw) == expected


def test_read_pre_commit_version_from_repo() -> None:
    """The real pre-commit config yields a plausible Ruff version string."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PRE_COMMIT

    version = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    assert version.startswith("v") or version[0].isdigit()


def test_read_pyproject_required_version_from_repo() -> None:
    """The real pyproject.toml has a required-version field."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT

    version = read_pyproject_required_version(DEFAULT_PYPROJECT)
    assert "==" in version or version[0].isdigit()


def test_read_pyproject_dev_dep_pin_from_repo() -> None:
    """The real pyproject.toml has a ruff== pin in a dependency group."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT

    pin = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)
    assert "ruff==" in pin


def test_read_pyproject_dev_dep_pin_uses_only_dev_group(tmp_path: Path) -> None:
    """A Ruff dependency in another group must not override the development pin."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[dependency-groups]\ndocs = ["ruff==99.99.99"]\ndev = ["ruff==0.16.0"]\n',
        encoding="utf-8",
    )

    assert read_pyproject_dev_dep_pin(pyproject) == "ruff==0.16.0"


def test_read_pyproject_dev_dep_pin_rejects_lookalike_dependency(tmp_path: Path) -> None:
    """A package whose name merely contains ``ruff`` is not the Ruff tool pin."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[dependency-groups]\ndev = ["some-ruff==0.16.0"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[dependency-groups\]\.dev"):
        read_pyproject_dev_dep_pin(pyproject)


def test_read_pre_commit_version_rejects_lookalike_repo(tmp_path: Path) -> None:
    """A repository URL that merely prefixes the canonical URL must not pass."""
    pre_commit = tmp_path / ".pre-commit-config.yaml"
    pre_commit.write_text(
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit-lookalike\n"
        "    rev: v0.16.0\n"
        "    hooks:\n"
        "      - id: ruff\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no astral-sh/ruff-pre-commit repo"):
        read_pre_commit_version(pre_commit)


def test_evaluate_passes_at_head() -> None:
    """All three version axes agree at the current HEAD."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)

    problems = evaluate(pre, req, dev)
    assert problems == [], f"Ruff version drift detected: {problems}"


def test_evaluate_fails_when_pre_commit_version_mutated(tmp_path: Path) -> None:
    """Mutating the pre-commit rev produces a mismatch."""
    yaml_content = (
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    rev: v99.99.99\n"
        "    hooks:\n"
        "      - id: ruff\n"
    )
    yaml_file = tmp_path / ".pre-commit-config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT

    pre = read_pre_commit_version(yaml_file)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)

    problems = evaluate(pre, req, dev)
    assert len(problems) >= 1


def test_evaluate_fails_when_required_version_mutated(tmp_path: Path) -> None:
    """Mutating the required-version produces a mismatch."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PRE_COMMIT, DEFAULT_PYPROJECT

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)

    problems = evaluate(pre, "==99.99.99", dev)
    assert len(problems) >= 1


def test_evaluate_fails_when_dev_pin_mutated(tmp_path: Path) -> None:
    """Mutating the dev dependency pin produces a mismatch."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PRE_COMMIT, DEFAULT_PYPROJECT

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)

    problems = evaluate(pre, req, "ruff==99.99.99")
    assert len(problems) >= 1


def test_main_passes_at_head() -> None:
    """Running main() at HEAD exits 0."""
    from scripts.dev.check_ruff_version_parity import main

    assert main([]) == 0


def test_main_exits_nonzero_on_mismatch(tmp_path: Path) -> None:
    """Running main() with a mutated pre-commit config exits non-zero."""
    from scripts.dev.check_ruff_version_parity import main

    yaml_content = (
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    rev: v99.99.99\n"
        "    hooks:\n"
        "      - id: ruff\n"
    )
    yaml_file = tmp_path / ".pre-commit-config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT

    exit_code = main(["--pre-commit-config", str(yaml_file), "--pyproject", str(DEFAULT_PYPROJECT)])
    assert exit_code != 0


def test_main_advisory_exits_zero_on_mismatch(tmp_path: Path) -> None:
    """Even with a mismatch, --advisory makes main() exit 0."""
    from scripts.dev.check_ruff_version_parity import main

    yaml_content = (
        "repos:\n"
        "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "    rev: v99.99.99\n"
        "    hooks:\n"
        "      - id: ruff\n"
    )
    yaml_file = tmp_path / ".pre-commit-config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT

    exit_code = main(
        [
            "--pre-commit-config",
            str(yaml_file),
            "--pyproject",
            str(DEFAULT_PYPROJECT),
            "--advisory",
        ]
    )
    assert exit_code == 0


def test_ci_lint_phase_enforces_ruff_version_parity() -> None:
    """CI must fail on version drift instead of hiding it behind advisory mode."""
    from scripts.dev.check_ruff_version_parity import REPO_ROOT

    ci_driver = (REPO_ROOT / "scripts" / "dev" / "ci_driver.sh").read_text(encoding="utf-8")
    invocation = next(
        line.strip() for line in ci_driver.splitlines() if "check_ruff_version_parity.py" in line
    )

    assert invocation == "_run_lint_check uv run python scripts/dev/check_ruff_version_parity.py"


def test_pre_commit_hook_runs_when_guard_changes() -> None:
    """Editing the parity guard itself must trigger its local pre-commit hook."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PRE_COMMIT

    config = DEFAULT_PRE_COMMIT.read_text(encoding="utf-8")
    assert r"scripts/dev/check_ruff_version_parity\.py" in config
