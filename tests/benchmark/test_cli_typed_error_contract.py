"""CLI-boundary contract ratchet for the benchmark CLI (issue #4988).

Every input-consuming benchmark subcommand must translate a *missing* or
*malformed* input file into a documented non-zero exit code, never a raw
traceback that escapes ``cli_main``. This is a boundary contract test: it does
not assert on any success-path output or metric value, only that the failure
path fails closed with a typed surface.

Owner note: issue #4988 owns this CLI-boundary contract. Sibling issues #4880
and #4960 own the exact in-handler ``except`` narrowing; this file only ratchets
the observable boundary behavior so a regression re-introducing a bare traceback
is caught.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.cli import cli_main

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Malformed payloads that exist on disk but cannot be parsed as the input the
# subcommand expects. Keyed by the logical input kind each subcommand consumes.
_MALFORMED = {
    "jsonl": "{not valid json at all\nalso } bad\n",
    "yaml": "::: not : valid : yaml : [\n",
    "json": "{ this is : not json\n",
}


@dataclass(frozen=True)
class CliCase:
    """One input-consuming subcommand invocation.

    Attributes:
        cmd: Subcommand name (used only for the parametrize id).
        kind: Logical input kind (``jsonl``/``yaml``/``json``) selecting the
            malformed payload and the input file extension.
        argv: Builds the full argv given the input path and a scratch dir.
    """

    cmd: str
    kind: str
    argv: Callable[[Path, Path], list[str]]


# Representative coverage of the input-file-consuming subcommands. Each reads a
# single primary input artifact whose absence or corruption must fail closed.
CASES: tuple[CliCase, ...] = (
    CliCase(
        "summary", "jsonl", lambda inp, d: ["summary", "--in", str(inp), "--out-dir", str(d / "o")]
    ),
    CliCase(
        "aggregate",
        "jsonl",
        lambda inp, d: ["aggregate", "--in", str(inp), "--out", str(d / "o.json")],
    ),
    CliCase(
        "metric-layers",
        "jsonl",
        lambda inp, d: ["metric-layers", "--episodes", str(inp), "--output", str(d / "o.json")],
    ),
    CliCase(
        "stress-coverage-report",
        "jsonl",
        lambda inp, d: [
            "stress-coverage-report",
            "--episodes-jsonl",
            str(inp),
            "--out",
            str(d / "o.json"),
        ],
    ),
    CliCase(
        "classify-failure-mechanisms",
        "jsonl",
        lambda inp, d: [
            "classify-failure-mechanisms",
            "--episodes-jsonl",
            str(inp),
            "--out-json",
            str(d / "o.json"),
            "--out-csv",
            str(d / "o.csv"),
        ],
    ),
    CliCase(
        "collision-scenario-similarity",
        "jsonl",
        lambda inp, d: [
            "collision-scenario-similarity",
            "--episodes-jsonl",
            str(inp),
            "--out-json",
            str(d / "o.json"),
        ],
    ),
    CliCase(
        "validate-row-claims", "json", lambda inp, d: ["validate-row-claims", "--sidecar", str(inp)]
    ),
    CliCase(
        "export-parquet",
        "jsonl",
        lambda inp, d: ["export-parquet", "--in", str(inp), "--out-dir", str(d / "o")],
    ),
    CliCase(
        "snqi-ablate",
        "jsonl",
        lambda inp, d: ["snqi-ablate", "--in", str(inp), "--out", str(d / "o.json")],
    ),
    CliCase(
        "seed-variance",
        "jsonl",
        lambda inp, d: ["seed-variance", "--in", str(inp), "--out", str(d / "o.json")],
    ),
    CliCase(
        "extract-failures",
        "jsonl",
        lambda inp, d: ["extract-failures", "--in", str(inp), "--out", str(d / "o.jsonl")],
    ),
    CliCase("rank", "jsonl", lambda inp, d: ["rank", "--in", str(inp), "--out", str(d / "o.md")]),
    CliCase(
        "table",
        "jsonl",
        lambda inp, d: [
            "table",
            "--in",
            str(inp),
            "--out",
            str(d / "o.md"),
            "--metrics",
            "collisions",
        ],
    ),
    CliCase(
        "plot-pareto",
        "jsonl",
        lambda inp, d: [
            "plot-pareto",
            "--in",
            str(inp),
            "--out",
            str(d / "o.png"),
            "--x-metric",
            "a",
            "--y-metric",
            "b",
        ],
    ),
    CliCase(
        "plot-distributions",
        "jsonl",
        lambda inp, d: [
            "plot-distributions",
            "--in",
            str(inp),
            "--out-dir",
            str(d / "o"),
            "--metrics",
            "collisions",
        ],
    ),
    CliCase("list-scenarios", "yaml", lambda inp, d: ["list-scenarios", "--matrix", str(inp)]),
    CliCase("validate-config", "yaml", lambda inp, d: ["validate-config", "--matrix", str(inp)]),
    CliCase(
        "preview-scenarios", "yaml", lambda inp, d: ["preview-scenarios", "--matrix", str(inp)]
    ),
    CliCase(
        "baseline",
        "yaml",
        lambda inp, d: [
            "baseline",
            "--matrix",
            str(inp),
            "--out",
            str(d / "o.json"),
            "--jsonl",
            str(d / "ep.jsonl"),
        ],
    ),
)

_CASE_IDS = tuple(case.cmd for case in CASES)


def _invoke(argv: list[str]) -> int:
    """Invoke the CLI, converting an argparse SystemExit into its int code.

    A raw (non-SystemExit) exception escaping ``cli_main`` is a contract
    violation and is left to propagate so the test fails with that traceback.
    """
    try:
        return cli_main(["--quiet", *argv])
    except SystemExit as exc:  # argparse-level typed exit is an acceptable surface
        code = exc.code
        return code if isinstance(code, int) else 1


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_missing_input_fails_closed(case: CliCase, tmp_path: Path) -> None:
    """A missing input path yields a non-zero exit, not a raw traceback."""
    missing = tmp_path / f"does_not_exist.{case.kind}"
    rc = _invoke(case.argv(missing, tmp_path))
    assert rc != 0, f"{case.cmd} returned rc=0 for missing input {missing}"


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_malformed_input_fails_closed(case: CliCase, tmp_path: Path) -> None:
    """A malformed input file yields a non-zero exit, not a raw traceback.

    Ratchet: ``export-parquet`` previously re-raised a bare ``ValueError`` for
    malformed JSONL, which escaped ``cli_main`` as a raw traceback (issue #4988).
    """
    bad = tmp_path / f"malformed.{case.kind}"
    bad.write_text(_MALFORMED[case.kind], encoding="utf-8")
    rc = _invoke(case.argv(bad, tmp_path))
    assert rc != 0, f"{case.cmd} returned rc=0 for malformed input {bad}"
