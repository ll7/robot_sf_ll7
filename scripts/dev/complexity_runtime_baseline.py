"""Report lightweight complexity and pytest runtime baseline indicators."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_DURATION_RE = re.compile(
    r"^\s*(?P<seconds>\d+(?:\.\d+)?)s\s+(?P<phase>\w+)\s+(?P<nodeid>\S.*)$",
)


@dataclass(frozen=True, slots=True)
class ModuleMetric:
    """Simple physical-size metric for one Python source file."""

    path: Path
    code_lines: int
    total_lines: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "path": self.path.as_posix(),
            "code_lines": self.code_lines,
            "total_lines": self.total_lines,
        }


@dataclass(frozen=True, slots=True)
class FunctionMetric:
    """Line-span metric for one function or method."""

    path: Path
    qualified_name: str
    length_lines: int
    lineno: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "path": self.path.as_posix(),
            "qualified_name": self.qualified_name,
            "length_lines": self.length_lines,
            "lineno": self.lineno,
        }


@dataclass(frozen=True, slots=True)
class PytestDurationSample:
    """One row from pytest's ``--durations`` report."""

    nodeid: str
    duration_seconds: float
    phase: str


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """Combined complexity and runtime-pressure baseline."""

    modules: list[ModuleMetric]
    functions: list[FunctionMetric]
    pytest_durations: list[PytestDurationSample]

    def to_json(self) -> str:
        """Serialize the baseline as deterministic JSON."""
        payload = {
            "modules": [metric.to_dict() for metric in self.modules],
            "functions": [metric.to_dict() for metric in self.functions],
            "pytest_durations": [asdict(sample) for sample in self.pytest_durations],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


class _FunctionVisitor(ast.NodeVisitor):
    """Collect function spans while preserving class/function qualification."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stack: list[str] = []
        self.functions: list[FunctionMetric] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = ".".join([*self.stack, node.name])
        length_lines = int((node.end_lineno or node.lineno) - node.lineno + 1)
        self.functions.append(
            FunctionMetric(
                path=self.path,
                qualified_name=qualified_name,
                length_lines=length_lines,
                lineno=int(node.lineno),
            ),
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def _iter_python_files(roots: list[Path]) -> list[Path]:
    """Return Python files under roots, skipping hidden and generated directories."""
    files: list[Path] = []
    skip_parts = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "output"}
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in skip_parts for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def _module_metric(path: Path) -> ModuleMetric:
    """Count total and simple non-comment source lines for a Python file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    code_lines = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return ModuleMetric(path=path, code_lines=code_lines, total_lines=len(lines))


def _function_metrics(path: Path) -> list[FunctionMetric]:
    """Parse function spans from a Python file, returning no metrics on syntax errors."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    except SyntaxError:
        return []
    visitor = _FunctionVisitor(path)
    visitor.visit(tree)
    return visitor.functions


def parse_pytest_durations(text: str) -> list[PytestDurationSample]:
    """Parse pytest ``--durations`` rows from captured output."""
    samples: list[PytestDurationSample] = []
    for line in text.splitlines():
        match = _DURATION_RE.match(line)
        if match is None:
            continue
        samples.append(
            PytestDurationSample(
                nodeid=match.group("nodeid").strip(),
                duration_seconds=float(match.group("seconds")),
                phase=match.group("phase"),
            ),
        )
    return sorted(samples, key=lambda sample: sample.duration_seconds, reverse=True)


def build_baseline(
    roots: list[Path],
    *,
    top: int = 10,
    pytest_log: Path | None = None,
) -> BaselineReport:
    """Build a bounded baseline from source roots and an optional pytest output log."""
    files = _iter_python_files(roots)
    modules = sorted(
        (_module_metric(path) for path in files),
        key=lambda metric: (-metric.code_lines, -metric.total_lines, metric.path.as_posix()),
    )[:top]
    functions = sorted(
        (metric for path in files for metric in _function_metrics(path)),
        key=lambda metric: (-metric.length_lines, metric.path.as_posix(), metric.lineno),
    )[:top]
    pytest_durations: list[PytestDurationSample] = []
    if pytest_log is not None:
        pytest_durations = parse_pytest_durations(pytest_log.read_text(encoding="utf-8"))[:top]
    return BaselineReport(
        modules=modules,
        functions=functions,
        pytest_durations=pytest_durations,
    )


def _format_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    """Format compact Markdown-style table rows for terminal and docs reuse."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def format_text_report(report: BaselineReport) -> str:
    """Format a human-readable baseline report."""
    lines: list[str] = ["# Complexity and Test Runtime Baseline", ""]
    lines.append("## Largest modules")
    lines.extend(
        _format_table(
            ("path", "code_lines", "total_lines"),
            [
                (metric.path.as_posix(), metric.code_lines, metric.total_lines)
                for metric in report.modules
            ],
        ),
    )
    lines.extend(["", "## Longest functions"])
    lines.extend(
        _format_table(
            ("path", "qualified_name", "length_lines", "lineno"),
            [
                (
                    metric.path.as_posix(),
                    metric.qualified_name,
                    metric.length_lines,
                    metric.lineno,
                )
                for metric in report.functions
            ],
        ),
    )
    lines.extend(["", "## Test runtime indicators"])
    if report.pytest_durations:
        lines.extend(
            _format_table(
                ("duration_seconds", "phase", "nodeid"),
                [
                    (f"{sample.duration_seconds:.2f}", sample.phase, sample.nodeid)
                    for sample in report.pytest_durations
                ],
            ),
        )
    else:
        lines.append(
            "No pytest duration log supplied; pass `--pytest-log <path>` with captured pytest "
            "output to include slow-test indicators.",
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments for the baseline reporter."""
    parser = argparse.ArgumentParser(
        description="Report lightweight complexity and pytest runtime baseline indicators.",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["robot_sf", "scripts", "tests"],
        help="Python files or directories to scan.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of rows per section.")
    parser.add_argument("--pytest-log", type=Path, help="Captured pytest output to parse.")
    parser.add_argument("--json", action="store_true", help="Print deterministic JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the baseline reporter."""
    args = _parse_args(argv)
    if args.top <= 0:
        raise SystemExit("--top must be a positive integer")
    roots = [Path(root) for root in args.roots]
    report = build_baseline(roots, top=args.top, pytest_log=args.pytest_log)
    sys.stdout.write(report.to_json() if args.json else format_text_report(report))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
