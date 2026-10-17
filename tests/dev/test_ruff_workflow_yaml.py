"""Exercise YAML command equivalence through the public Ruff parity entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.dev import check_ruff_version_parity as guard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parity_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """Keep three matching axes and redirect the workflow directory on old and new code."""
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
    "scalar",
    [
        pytest.param(
            ">\n          uvx --from\n          ruff=={version} ruff format --check .\n",
            id="folded",
        ),
        pytest.param(
            ">-\n          uvx --from\n          ruff=={version} ruff format --check .\n",
            id="folded-strip-reviewer",
        ),
        pytest.param(
            ">+\n          uvx --from\n          ruff=={version} ruff format --check .\n",
            id="folded-keep",
        ),
        pytest.param(
            ">2-\n          uvx --from\n          ruff=={version} ruff format --check .\n",
            id="folded-explicit-indent",
        ),
        pytest.param(
            "|\n          uvx --from ruff==0.17.0 ruff check .\n"
            "          uvx --from \\\n            ruff=={version} ruff format --check .\n",
            id="literal-continuation-reviewer",
        ),
        pytest.param(
            "'uvx --from\n          ruff=={version} ruff format --check .'\n",
            id="single-quoted-multiline",
        ),
        pytest.param(
            '"uvx --from\n          ruff=={version} ruff format --check ."\n',
            id="double-quoted-multiline",
        ),
        pytest.param(
            '"uvx --from \\\n          ruff=={version} ruff format --check ."\n',
            id="double-quoted-escaped-break",
        ),
        pytest.param("uvx --from ruff=={version} ruff format --check .\n", id="single-line"),
    ],
)
@pytest.mark.parametrize("version", ["9.9.9", "0.17.0"], ids=["stale", "matching"])
def test_main_checks_yaml_run_scalars(
    parity_inputs: tuple[Path, list[str]],
    capsys: pytest.CaptureFixture[str],
    scalar: str,
    version: str,
) -> None:
    """A matching launcher must not conceal another step's stale resolved command."""
    workflows, args = parity_inputs
    workflow = (
        "jobs:\n  lint:\n    steps:\n"
        "      - run: uvx --from ruff==0.17.0 ruff check .\n"
        f"      - run: {scalar.format(version=version)}"
    )
    resolved = yaml.safe_load(workflow)["jobs"]["lint"]["steps"][1]["run"]
    assert resolved.replace("\\\n", "").split()[-7:] == [
        "uvx",
        "--from",
        f"ruff=={version}",
        "ruff",
        "format",
        "--check",
        ".",
    ]
    (workflows / "ci.yml").write_text(workflow)

    assert guard.main(args) == (1 if version == "9.9.9" else 0)
    output = capsys.readouterr().out
    assert f"ruff=={version}" in output
    if version == "9.9.9":
        assert "ci.yml:" in output
        assert "OK: all four" not in output


@pytest.mark.parametrize("advisory", [False, True])
def test_main_rejects_unparseable_workflow(
    parity_inputs: tuple[Path, list[str]], capsys: pytest.CaptureFixture[str], advisory: bool
) -> None:
    """A good pin in one file cannot turn a YAML parse failure into a passing guard."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "jobs:\n  lint:\n    steps:\n      - run: uvx --from ruff==0.17.0 ruff check .\n"
    )
    (workflows / "broken.yaml").write_text("jobs: [\n")

    assert guard.main(args + (["--advisory"] if advisory else [])) == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "broken.yaml" in output
    assert "parse" in output.lower()
    assert "OK: all four" not in output


def test_main_only_checks_step_run_values(parity_inputs: tuple[Path, list[str]]) -> None:
    """Comments and step metadata mentioning a launcher are not commands."""
    workflows, args = parity_inputs
    (workflows / "ci.yml").write_text(
        "# uvx --from ruff==9.9.9 is obsolete\n"
        "jobs:\n  lint:\n    steps:\n"
        "      - name: uvx --from ruff==9.9.9 is obsolete\n"
        "        env:\n          NOTE: uvx --from ruff==9.9.9\n"
        "        run: uvx --from ruff==0.17.0 ruff check .\n"
    )

    assert guard.main(args) == 0
