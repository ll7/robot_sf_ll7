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
    read_workflow_ruff_pins,
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
    """All four version axes agree at the current HEAD."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
        DEFAULT_WORKFLOWS_DIR,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)
    workflow_pins = tuple(read_workflow_ruff_pins(DEFAULT_WORKFLOWS_DIR))

    assert workflow_pins, "expected at least one uvx ruff launcher pin at HEAD"
    problems = evaluate(pre, req, dev, workflow_pins)
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

    from scripts.dev.check_ruff_version_parity import DEFAULT_PYPROJECT, DEFAULT_WORKFLOWS_DIR

    pre = read_pre_commit_version(yaml_file)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)
    workflow_pins = tuple(read_workflow_ruff_pins(DEFAULT_WORKFLOWS_DIR))

    problems = evaluate(pre, req, dev, workflow_pins)
    assert len(problems) >= 1


def test_evaluate_fails_when_required_version_mutated() -> None:
    """Mutating the required-version produces a mismatch."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
        DEFAULT_WORKFLOWS_DIR,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)
    workflow_pins = tuple(read_workflow_ruff_pins(DEFAULT_WORKFLOWS_DIR))

    problems = evaluate(pre, "==99.99.99", dev, workflow_pins)
    assert len(problems) >= 1


def test_evaluate_fails_when_dev_pin_mutated() -> None:
    """Mutating the dev dependency pin produces a mismatch."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
        DEFAULT_WORKFLOWS_DIR,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    workflow_pins = tuple(read_workflow_ruff_pins(DEFAULT_WORKFLOWS_DIR))

    problems = evaluate(pre, req, "ruff==99.99.99", workflow_pins)
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


def test_pre_commit_hook_runs_on_workflow_changes() -> None:
    """Editing a CI workflow must re-run the parity guard (issue #10395)."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_PRE_COMMIT

    config = DEFAULT_PRE_COMMIT.read_text(encoding="utf-8")
    assert r"\.github/workflows/" in config


def test_read_workflow_ruff_pins_enumerates_pins(tmp_path: Path) -> None:
    """Every uvx ruff launcher pin is enumerated with file:line locations."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "ci.yml").write_text(
        "jobs:\n"
        "  lint:\n"
        "    steps:\n"
        "      - run: uvx --from ruff==0.16.10 ruff check .\n"
        "      - run: uvx --from ruff==0.16.10 ruff format --check .\n",
        encoding="utf-8",
    )
    (workflows / "other.yaml").write_text(
        "jobs:\n  x:\n    steps:\n      - run: uvx --from ruff==0.16.10 ruff check .\n",
        encoding="utf-8",
    )
    (workflows / "notes.txt").write_text("uvx --from ruff==9.9.9\n", encoding="utf-8")
    (workflows / "plain.yml").write_text(
        "# a comment mentioning ruff==9.9.9 without any launcher keyword\n"
        "jobs:\n  x:\n    steps:\n      - run: echo ruff==9.9.9\n",
        encoding="utf-8",
    )

    pins = read_workflow_ruff_pins(workflows)

    assert pins == [
        ("ci.yml:4", "0.16.10"),
        ("ci.yml:5", "0.16.10"),
        ("other.yaml:4", "0.16.10"),
    ]


def test_read_workflow_ruff_pins_missing_dir_returns_empty(tmp_path: Path) -> None:
    """A missing workflows directory yields no pins (fail-closed downstream)."""
    assert read_workflow_ruff_pins(tmp_path / "does-not-exist") == []


def test_read_workflow_ruff_pins_at_head() -> None:
    """The real workflows directory exposes the CI lint launcher pins."""
    from scripts.dev.check_ruff_version_parity import DEFAULT_WORKFLOWS_DIR

    pins = read_workflow_ruff_pins(DEFAULT_WORKFLOWS_DIR)

    assert pins, "expected at least one uvx ruff launcher pin in .github/workflows"
    assert all(location.startswith("ci.yml:") for location, _ in pins)


def test_evaluate_fails_when_workflow_pin_stale() -> None:
    """A stale workflow launcher pin is reported with its file:line location."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)

    problems = evaluate(pre, req, dev, (("ci.yml:199", "0.0.0"),))

    assert len(problems) >= 1
    assert any("ci.yml:199" in problem for problem in problems)


def test_evaluate_fails_when_no_workflow_pins_found() -> None:
    """An empty workflow scan fails closed instead of passing silently."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
    )

    pre = read_pre_commit_version(DEFAULT_PRE_COMMIT)
    req = read_pyproject_required_version(DEFAULT_PYPROJECT)
    dev = read_pyproject_dev_dep_pin(DEFAULT_PYPROJECT)

    problems = evaluate(pre, req, dev, ())

    assert len(problems) >= 1


def test_main_detects_stale_workflow_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mutated workflow pin makes main() exit non-zero and names the file."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PRE_COMMIT,
        DEFAULT_PYPROJECT,
        main,
    )

    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n      - run: uvx --from ruff==9.9.9 ruff check .\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--pre-commit-config",
            str(DEFAULT_PRE_COMMIT),
            "--pyproject",
            str(DEFAULT_PYPROJECT),
            "--workflows-dir",
            str(workflows),
        ]
    )

    assert exit_code != 0
    assert "ci.yml:4" in capsys.readouterr().out


def test_main_detects_dev_only_bump(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Bumping only the dev ruff pin names the stale workflow pins (issue #10395)."""
    from scripts.dev.check_ruff_version_parity import (
        DEFAULT_PYPROJECT,
        DEFAULT_WORKFLOWS_DIR,
        main,
        read_pyproject_required_version,
    )

    required = read_pyproject_required_version(DEFAULT_PYPROJECT)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f'[tool.ruff]\nrequired-version = "{required}"\n'
        '\n[dependency-groups]\ndev = ["ruff==9.9.9"]\n',
        encoding="utf-8",
    )

    exit_code = main(["--pyproject", str(pyproject), "--workflows-dir", str(DEFAULT_WORKFLOWS_DIR)])

    assert exit_code != 0
    out = capsys.readouterr().out
    assert "ci.yml" in out
