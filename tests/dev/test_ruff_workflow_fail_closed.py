"""Public-entry-point regressions for all four reviews and fail-closed scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.dev import check_ruff_version_parity as guard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parity_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """Supply matching axes through inputs available on unchanged main too."""
    pre_commit = tmp_path / "pre-commit.yaml"
    pre_commit.write_text(
        "repos:\n  - repo: https://github.com/astral-sh/ruff-pre-commit\n    rev: v0.17.0\n"
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.ruff]\nrequired-version = "==0.17.0"\n[dependency-groups]\ndev = ["ruff==0.17.0"]\n'
    )
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    monkeypatch.setattr(guard, "DEFAULT_WORKFLOWS_DIR", workflows, raising=False)
    return workflows, ["--pre-commit-config", str(pre_commit), "--pyproject", str(pyproject)]


@pytest.mark.parametrize(
    ("defaults", "step"),
    [
        pytest.param(
            "",
            "      - run: |\n"
            "          uvx --from ruff==0.17.0 ruff check .\n"
            "          uvx --from \\\n            ruff==9.9.9 ruff format --check .\n",
            id="round1-bash",
        ),
        pytest.param(
            "",
            "      - run: >-\n          uvx --from\n          ruff==9.9.9 ruff format --check .\n",
            id="round2-folded",
        ),
        pytest.param(
            "",
            "      - shell: pwsh\n        run: |-\n          uvx --from `\n"
            "            ruff==9.9.9 ruff format --check .\n",
            id="round3-pwsh-step",
        ),
        pytest.param(
            "    defaults:\n      run:\n        shell: pwsh\n",
            "      - run: |-\n          uvx --from `\n"
            "            ruff==9.9.9 ruff format --check .\n",
            id="round3-pwsh-job-default",
        ),
        pytest.param(
            "defaults:\n  run:\n    shell: pwsh\n",
            "      - run: |-\n          uvx --from `\n"
            "            ruff==9.9.9 ruff format --check .\n",
            id="round3-pwsh-workflow-default",
        ),
        pytest.param(
            "",
            "      - shell: cmd\n        run: |-\n          uvx --from ^\n"
            "            ruff==9.9.9 ruff format --check .\n",
            id="round3-cmd",
        ),
    ],
)
def test_all_review_reproductions_reject(
    parity_inputs: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    defaults: str,
    step: str,
) -> None:
    """Every review's mixed good/stale workflow fails through the existing CLI API."""
    workflows, args = parity_inputs
    workflow_defaults = defaults if defaults.startswith("defaults:") else ""
    job_defaults = "" if workflow_defaults else defaults
    (workflows / "ci.yml").write_text(
        workflow_defaults + "jobs:\n  lint:\n" + job_defaults + "    steps:\n"
        "      - run: uvx --from ruff==0.17.0 ruff check .\n" + step
    )

    assert guard.main(args) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ci.yml:jobs.lint.steps[1].run" in output
    assert "OK: all four" not in output


@pytest.mark.parametrize(
    "command",
    [
        "ruff check .",
        "uv run ruff check .",
        "python -m ruff check .",
        "pip install ruff==9.9.9",
        "UVX --from RUFF==9.9.9 RUFF check .",
        "echo Ruff",
        "echo unrelated_ruff_filename",
        "# ruff is only a comment",
        "uvx --from ruff ruff check .",
        "uvx --from ruff==0.17.0 ruff check .; ruff format .",
        "uvx --from ruff==0.17.0 ruff check . && uvx --from ruff==9.9.9 ruff check .",
        "echo uvx --from ruff==0.17.0 ruff check .",
        "uvx --from ruff==0.17.0 ruff check .\nruff format .",
        "uvx --from `\nruff==0.17.0 ruff check .",
        "uvx --from ^\nruff==0.17.0 ruff check .",
        "r\\\nuff check .",
        "uvx --from ruff==0.17.0 ruff check . | tee log",
        'uvx --from ruff==0.17.0 ruff check "$(ruff format .)"',
        "uvx --from ruff==0.17.0 ruff check ruff==9.9.9",
    ],
)
@pytest.mark.parametrize("shell", ["bash", "pwsh", "cmd", "python", "custom {0}"])
def test_unknown_mentions_fail_under_any_shell(
    parity_inputs: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    command: str,
    shell: str,
) -> None:
    """A known good pin cannot conceal any unrecognised case-insensitive mention."""
    workflows, args = parity_inputs
    indented = "\n".join("          " + line for line in command.splitlines())
    (workflows / "ci.yaml").write_text(
        "jobs:\n  lint:\n    steps:\n"
        "      - run: uvx --from ruff==0.17.0 ruff check .\n"
        f"      - shell: {shell}\n        run: |\n{indented}\n"
    )

    assert guard.main(args) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ci.yaml:jobs.lint.steps[1].run" in output
    assert "unrecognised" in output.lower()
    assert "OK: all four" not in output


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param('ru""ff', id="round4-reviewer-double-empty"),
        pytest.param("r''uff", id="single-empty"),
        pytest.param('"ru"ff', id="quoted-prefix"),
        pytest.param(r"ru\ff", id="backslash-escape"),
        pytest.param("r$''uff", id="ansi-empty"),
        pytest.param('r$""uff', id="locale-empty"),
    ],
)
@pytest.mark.parametrize("advisory", [False, True], ids=["strict", "advisory"])
def test_literal_shell_spellings_cannot_hide_stale_launcher(
    parity_inputs: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    tool: str,
    advisory: bool,
) -> None:
    """Literal concatenation and escapes cannot conceal a stale pin beside a good one."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "name: parity-probe\non: push\njobs:\n  lint:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - shell: bash\n        run: |\n"
        "          uvx --from ruff==0.17.0 ruff check .\n"
        f"          uvx --from {tool}==0.16.10 {tool} check .\n"
    )

    assert guard.main(args + (["--advisory"] if advisory else [])) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "ci.yml:jobs.lint.steps[0].run" in output
    assert "unrecognised" in output.lower()
    assert "0.16.10" in output
    assert "OK: all four" not in output


def test_unknown_mentions_also_fail_in_advisory_mode(
    parity_inputs: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Advisory drift reporting must not turn an incomplete scan into success."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n"
        "      - run: uvx --from ruff==0.17.0 ruff check .\n"
        "      - run: RUFF check .\n"
    )
    assert guard.main(args + ["--advisory"]) == 1
    assert "unrecognised" in capsys.readouterr().err.lower()


def test_known_launchers_after_backslash_joining_pass(
    parity_inputs: tuple[Path, list[str]],
) -> None:
    """Joining continuations precedes mention detection, including a split tool name."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n      - run: |\n"
        "          uvx --from ru\\\n          ff==0.17.0 ru\\\n          ff check .\n"
        "          uvx --from ruff==0.17.0 ruff format --check .\n"
    )
    assert guard.main(args) == 0


@pytest.mark.parametrize("config", ["'lint.ignore=[]'", '"lint.ignore=[]"'])
@pytest.mark.parametrize("version", ["0.17.0", "9.9.9"])
def test_literal_quoted_config_keeps_pin_enforcement(
    parity_inputs: tuple[Path, list[str]], config: str, version: str
) -> None:
    """Literal quoted config overrides work without admitting shell expansion or stale pins."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n"
        "      - run: uvx --from ruff==0.17.0 ruff check .\n"
        "      - run: |\n"
        f"          uvx --from ruff=={version} ruff check --config {config} --select S .\n"
    )
    assert guard.main(args) == (0 if version == "0.17.0" else 1)
