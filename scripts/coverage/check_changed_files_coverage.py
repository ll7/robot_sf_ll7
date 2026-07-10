#!/usr/bin/env python3
"""Check coverage for files changed relative to a base ref.

This tool is intended for PR readiness checks. It:
- Detects files changed vs a base ref (default: origin/main).
- Filters to included patterns (default: robot_sf/**/*.py).
- Reads coverage.json and enforces per-file coverage thresholds.

Goal: 100% coverage on changed files. Minimum requirement: 80%.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class _CoverageFileData(TypedDict, total=False):
    """Typed subset of a coverage.py JSON file payload."""

    executed_lines: list[int]
    missing_lines: list[int]
    summary: dict[str, object]


class _CoverageResult(TypedDict):
    """Coverage result row for a single changed file."""

    file: str
    coverage: float | None
    resolved: str | None
    scope: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run a command and return stripped stdout.

    Returns:
        Captured standard output with surrounding whitespace removed.
    """
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _repo_root() -> Path:
    """Resolve the current git repository root.

    Returns:
        Absolute repository root path.
    """
    return Path(_run(["git", "rev-parse", "--show-toplevel"]))


def _changed_files(base: str, repo_root: Path) -> list[Path]:
    """List files changed relative to a base ref.

    Returns:
        Repository-relative paths changed in the comparison.
    """
    output = _run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}...HEAD"],
        cwd=repo_root,
    )
    files = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    return files


def _file_at_ref(base: str, path: Path, repo_root: Path) -> str | None:
    """Read a repository file at the merge-base for a comparison ref.

    Returns:
        File contents at the comparison base, or ``None`` when the file did not
        exist there.
    """
    merge_base = _run(["git", "merge-base", base, "HEAD"], cwd=repo_root)
    proc = subprocess.run(
        ["git", "show", f"{merge_base}:{path.as_posix()}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstring expressions while preserving executable syntax."""

    @staticmethod
    def _strip_body(body: list[ast.stmt]) -> list[ast.stmt]:
        """Drop a leading string expression from a Python statement body."""
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Strip module docstring before visiting child statements."""
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Strip class docstring before visiting child statements."""
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Strip function docstring before visiting child statements."""
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Strip async-function docstring before visiting child statements."""
        node.body = self._strip_body(node.body)
        self.generic_visit(node)
        return node


def _normalized_python_ast(source: str) -> str | None:
    """Parse Python source into a docstring-free AST dump.

    Returns:
        Attribute-free AST dump, or ``None`` when the source cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    stripped = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped)
    return ast.dump(stripped, include_attributes=False)


def _is_doc_or_comment_only_python_change(before: str, after: str) -> bool:
    """Return whether a Python change only affects comments, formatting, or docstrings."""
    before_ast = _normalized_python_ast(before)
    after_ast = _normalized_python_ast(after)
    if before_ast is None or after_ast is None:
        return False
    return before_ast == after_ast


def _is_doc_or_comment_only_changed_file(path: Path, base: str, repo_root: Path) -> bool:
    """Check whether a changed Python file has no executable AST changes."""
    if path.suffix != ".py":
        return False
    before = _file_at_ref(base, path, repo_root)
    if before is None:
        return False
    after_path = repo_root / path
    try:
        after = after_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return _is_doc_or_comment_only_python_change(before, after)


def _expression_terminal_name(node: ast.expr) -> str | None:
    """Return the local or attribute name represented by a simple expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _ast_dumps(nodes: Sequence[ast.AST]) -> list[str]:
    """Return attribute-free AST dumps for a node list."""
    return [ast.dump(node, include_attributes=False) for node in nodes]


def _added_leading_bases(before: ast.ClassDef, after: ast.ClassDef) -> list[str] | None:
    """Return newly prepended class bases when every other class detail is unchanged."""
    before_bases = [ast.dump(base, include_attributes=False) for base in before.bases]
    after_bases = [ast.dump(base, include_attributes=False) for base in after.bases]
    existing_bases_match = not before_bases or after_bases[-len(before_bases) :] == before_bases
    if len(after_bases) <= len(before_bases) or not existing_bases_match:
        return None
    if (
        before.name != after.name
        or _ast_dumps(getattr(before, "type_params", []))
        != _ast_dumps(getattr(after, "type_params", []))
        or _ast_dumps(before.keywords) != _ast_dumps(after.keywords)
        or _ast_dumps(before.decorator_list) != _ast_dumps(after.decorator_list)
        or _ast_dumps(before.body) != _ast_dumps(after.body)
    ):
        return None
    added_names = [
        _expression_terminal_name(base)
        for base in after.bases[: len(after_bases) - len(before_bases)]
    ]
    if any(name is None for name in added_names):
        return None
    return [name for name in added_names if name is not None]


def _import_bindings(node: ast.stmt) -> set[str]:
    """Return bindings introduced by one import statement."""
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.partition(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        return {alias.asname or alias.name for alias in node.names}
    return set()


def _module_imports(tree: ast.Module) -> list[ast.stmt]:
    """Return top-level import statements from a module."""
    return [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]


def _imports_only_add_required_bases(
    before_imports: list[ast.stmt],
    after_imports: list[ast.stmt],
    required_bases: set[str],
) -> bool:
    """Return whether imports only add bindings for the new class bases."""
    before_dumps = {ast.dump(node, include_attributes=False) for node in before_imports}
    after_dumps = {ast.dump(node, include_attributes=False) for node in after_imports}
    if not before_dumps <= after_dumps:
        return False
    added_bindings = set().union(
        *(
            _import_bindings(node)
            for node in after_imports
            if ast.dump(node, include_attributes=False) not in before_dumps
        )
    )
    return added_bindings <= required_bases


def _class_base_requirements(
    before_nodes: list[ast.stmt],
    after_nodes: list[ast.stmt],
) -> set[tuple[str, str]] | None:
    """Return new class-base proof requirements when all other nodes are unchanged."""
    if len(before_nodes) != len(after_nodes):
        return None
    requirements: set[tuple[str, str]] = set()
    for before_node, after_node in zip(before_nodes, after_nodes, strict=True):
        if isinstance(before_node, ast.ClassDef) and isinstance(after_node, ast.ClassDef):
            added_bases = _added_leading_bases(before_node, after_node)
            if added_bases is not None:
                requirements.update((before_node.name, base) for base in added_bases)
                continue
        if ast.dump(before_node, include_attributes=False) != ast.dump(
            after_node,
            include_attributes=False,
        ):
            return None
    return requirements or None


def _declaration_only_class_base_requirements(
    before: str,
    after: str,
) -> set[tuple[str, str]] | None:
    """Return direct-test requirements for a pure class-base migration.

    The exemption deliberately accepts only imports required for newly prepended
    bases and otherwise byte-for-byte equivalent AST statements.  It therefore
    cannot cover a changed class body, function, decorator, or existing base.
    """
    try:
        before_tree = ast.parse(before)
        after_tree = ast.parse(after)
    except SyntaxError:
        return None

    before_imports = _module_imports(before_tree)
    after_imports = _module_imports(after_tree)
    requirements = _class_base_requirements(
        [node for node in before_tree.body if node not in before_imports],
        [node for node in after_tree.body if node not in after_imports],
    )
    if requirements is None:
        return None
    required_bases = {base for _, base in requirements}
    if not _imports_only_add_required_bases(before_imports, after_imports, required_bases):
        return None
    return requirements


def _names_in_static_collection(node: ast.expr) -> set[str]:
    """Return simple names from a tuple, list, or set literal."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return set()
    return {
        name for element in node.elts if (name := _expression_terminal_name(element)) is not None
    }


def _parametrize_values(
    tree: ast.Module, bindings: dict[str, set[str]]
) -> dict[int, dict[str, set[str]]]:
    """Resolve simple pytest parametrization collections by function node id."""
    values: dict[int, dict[str, set[str]]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
                and len(decorator.args) >= 2
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                continue
            parameter_name = decorator.args[0].value
            if "," in parameter_name:
                continue
            collection = decorator.args[1]
            candidates = (
                bindings.get(collection.id, set())
                if isinstance(collection, ast.Name)
                else _names_in_static_collection(collection)
            )
            if candidates:
                values.setdefault(id(node), {})[parameter_name] = candidates
    return values


def _candidate_names(node: ast.expr, parameters: dict[str, set[str]]) -> set[str]:
    """Resolve a directly named class or a simple parametrized class name."""
    name = _expression_terminal_name(node)
    if name is None:
        return set()
    return parameters.get(name, {name})


def _proofs_from_statements(
    statements: list[ast.stmt],
    parameters: dict[str, set[str]],
) -> set[tuple[str, str]]:
    """Collect direct ``issubclass`` and ``pytest.raises`` relationship proofs."""
    proofs: set[tuple[str, str]] = set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Call):
            call = node.test
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "issubclass"
                and len(call.args) == 2
            ):
                base = _expression_terminal_name(call.args[1])
                if base is not None:
                    proofs.update(
                        (candidate, base)
                        for candidate in _candidate_names(call.args[0], parameters)
                    )
        if not isinstance(node, ast.With):
            continue
        raised_bases = {
            _expression_terminal_name(item.context_expr.args[0])
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pytest"
            and item.context_expr.func.attr == "raises"
            and item.context_expr.args
        }
        for raised_base in raised_bases:
            if raised_base is None:
                continue
            for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(child, ast.Raise) and child.exc is not None:
                    raised = child.exc.func if isinstance(child.exc, ast.Call) else child.exc
                    proofs.update(
                        (candidate, raised_base)
                        for candidate in _candidate_names(raised, parameters)
                    )
    return proofs


def _source_module_name(path: Path) -> str | None:
    """Return the importable module name for a repository-relative Python path."""
    if path.suffix != ".py" or not path.parts:
        return None
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def _direct_imports_from_module(tree: ast.Module, module_name: str) -> set[str]:
    """Return unaliased names directly imported from one absolute module."""
    return {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == module_name
        for alias in node.names
        if alias.asname is None
    }


def _declaration_proofs_from_test_source(
    source: str,
    *,
    source_module: str | None = None,
) -> set[tuple[str, str]]:
    """Collect direct declaration proofs from one test module source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    bindings: dict[str, set[str]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            bindings[node.targets[0].id] = _names_in_static_collection(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bindings[node.target.id] = _names_in_static_collection(node.value)
    parameter_values = _parametrize_values(tree, bindings)
    proofs: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            proofs.update(_proofs_from_statements(node.body, parameter_values.get(id(node), {})))
    if source_module is None:
        return proofs
    source_imports = _direct_imports_from_module(tree, source_module)
    return {(candidate, base) for candidate, base in proofs if candidate in source_imports}


def _changed_tests_prove_declaration_requirements(
    requirements: set[tuple[str, str]],
    changed_files: Iterable[Path],
    repo_root: Path,
    source_path: Path,
) -> bool:
    """Return whether changed tests prove a migration from its source module."""
    source_module = _source_module_name(source_path)
    if source_module is None:
        return False
    proofs: set[tuple[str, str]] = set()
    for path in changed_files:
        if path.suffix != ".py" or not path.parts or path.parts[0] != "tests":
            continue
        try:
            source = (repo_root / path).read_text(encoding="utf-8")
        except OSError:
            continue
        proofs.update(_declaration_proofs_from_test_source(source, source_module=source_module))
    return requirements <= proofs


def _has_declaration_only_test_proof(
    path: Path,
    base: str,
    repo_root: Path,
    changed_files: Iterable[Path],
) -> bool:
    """Return whether a changed source file has bounded, changed-test proof."""
    before = _file_at_ref(base, path, repo_root)
    if before is None:
        return False
    try:
        after = (repo_root / path).read_text(encoding="utf-8")
    except OSError:
        return False
    requirements = _declaration_only_class_base_requirements(before, after)
    return requirements is not None and _changed_tests_prove_declaration_requirements(
        requirements,
        changed_files,
        repo_root,
        path,
    )


def _normalize_path(path: Path, repo_root: Path) -> str:
    """Normalize an absolute or relative path to repository POSIX form.

    Returns:
        Repository-relative path when possible, otherwise POSIX path text.
    """
    if path.is_absolute():
        try:
            path = path.relative_to(repo_root)
        except ValueError:
            pass
    return path.as_posix()


def _matches_any(path_str: str, patterns: Iterable[str]) -> bool:
    """Check whether a path matches any glob pattern.

    Returns:
        True when at least one pattern matches.
    """
    return any(fnmatch(path_str, pattern) for pattern in patterns)


def _resolve_coverage(
    path_str: str,
    coverage_index: dict[str, float],
) -> tuple[float | None, str | None]:
    """Resolve coverage for a changed file from the coverage index.

    Returns:
        Pair of coverage percentage and matched coverage key, or ``(None,
        None)`` when no unambiguous coverage row exists.
    """
    if path_str in coverage_index:
        return coverage_index[path_str], path_str
    matches = [(k, v) for k, v in coverage_index.items() if k.endswith(path_str)]
    if len(matches) == 1:
        key, value = matches[0]
        return value, key
    return None, None


def _load_coverage_index(coverage_path: Path, repo_root: Path) -> dict[str, float]:
    """Load file coverage percentages from coverage.py JSON output.

    Returns:
        Mapping from normalized file path to percent covered.
    """
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    index: dict[str, float] = {}
    for file_path, file_data in data.get("files", {}).items():
        summary = file_data.get("summary", {})
        percent = summary.get("percent_covered", 0.0)
        try:
            percent_value = float(percent)
        except (TypeError, ValueError):
            percent_value = 0.0
        normalized = _normalize_path(Path(file_path), repo_root)
        index[normalized] = percent_value
    return index


def _load_coverage_file_data(coverage_path: Path, repo_root: Path) -> dict[str, _CoverageFileData]:
    """Load normalized coverage.py file data keyed by repository path.

    Returns:
        Mapping from normalized file path to the raw coverage.py file payload.
    """
    data = json.loads(coverage_path.read_text(encoding="utf-8"))
    files: dict[str, _CoverageFileData] = {}
    for file_path, file_data in data.get("files", {}).items():
        if isinstance(file_data, dict):
            files[_normalize_path(Path(file_path), repo_root)] = cast(
                "_CoverageFileData", file_data
            )
    return files


def _changed_line_numbers(base: str, path: Path, repo_root: Path) -> set[int]:
    """Return new-file line numbers touched by the diff against *base*.

    Returns:
        Set of line numbers on ``HEAD`` that were added or modified relative to the merge-base.
    """
    proc = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            "--diff-filter=ACMRT",
            f"{base}...HEAD",
            "--",
            path.as_posix(),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()

    changed: set[int] = set()
    new_line: int | None = None
    for line in proc.stdout.splitlines():
        hunk_match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if hunk_match:
            new_line = int(hunk_match.group(1))
            continue
        if new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            changed.add(new_line)
            new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        elif line.startswith("\\"):
            continue
        else:
            new_line += 1
    return changed


def _coverage_for_changed_lines(
    *,
    file_data: _CoverageFileData | None,
    changed_lines: set[int],
) -> tuple[float | None, str]:
    """Return executable changed-line coverage, falling back when data is insufficient.

    Returns:
        Pair of coverage percent and a compact scope label.
    """
    if file_data is None or not changed_lines:
        return None, "file"

    executed = {int(line) for line in file_data.get("executed_lines", [])}
    missing = {int(line) for line in file_data.get("missing_lines", [])}
    statement_lines = executed | missing
    changed_statements = changed_lines & statement_lines
    if not changed_statements:
        return 100.0, "changed executable lines 0/0"
    covered_changed = changed_statements & executed
    return (
        100.0 * len(covered_changed) / len(changed_statements),
        f"changed executable lines {len(covered_changed)}/{len(changed_statements)}",
    )


def _print_lines(lines: Iterable[str]) -> None:
    """Print lines in order."""
    for line in lines:
        print(line)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the changed-files coverage gate.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Check per-file coverage for changed files relative to a base ref.",
    )
    parser.add_argument("--base", default="origin/main", help="Base ref to diff against")
    parser.add_argument(
        "--coverage",
        default="output/coverage/coverage.json",
        help="Path to coverage.json",
    )
    parser.add_argument("--min", type=float, default=80.0, help="Minimum required coverage")
    parser.add_argument("--goal", type=float, default=100.0, help="Target coverage goal")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern(s) to include (can be repeated)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern(s) to exclude (can be repeated)",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print files skipped by include/exclude filters",
    )
    return parser.parse_args()


def _resolve_coverage_path(coverage_arg: str, repo_root: Path) -> Path:
    """Resolve the coverage JSON path relative to the repository root.

    Returns:
        Absolute coverage JSON path.
    """
    coverage_path = Path(coverage_arg)
    if not coverage_path.is_absolute():
        coverage_path = repo_root / coverage_path
    return coverage_path


def _select_changed_files(
    changed_files: list[Path],
    repo_root: Path,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Apply include and exclude patterns to changed files.

    Returns:
        Tuple of selected normalized paths and skipped normalized paths.
    """
    selected: list[str] = []
    skipped: list[str] = []
    for path in changed_files:
        if not (repo_root / path).exists():
            continue
        path_str = _normalize_path(path, repo_root)
        if _matches_any(path_str, include_patterns) and not _matches_any(
            path_str,
            exclude_patterns,
        ):
            selected.append(path_str)
        else:
            skipped.append(path_str)
    return selected, skipped


def _build_results(
    selected: list[str],
    coverage_index: dict[str, float],
    coverage_file_data: dict[str, _CoverageFileData],
    *,
    base: str,
    repo_root: Path,
    changed_files: Iterable[Path],
) -> list[_CoverageResult]:
    """Attach coverage data to selected changed files.

    Returns:
        Result rows containing file path, coverage value, and resolved coverage
        key.
    """
    results: list[_CoverageResult] = []
    for path_str in selected:
        file_coverage, resolved = _resolve_coverage(path_str, coverage_index)
        coverage = file_coverage
        scope = "file"
        if resolved is not None:
            changed_coverage, changed_scope = _coverage_for_changed_lines(
                file_data=coverage_file_data.get(resolved),
                changed_lines=_changed_line_numbers(base, Path(path_str), repo_root),
            )
            if changed_coverage is not None:
                coverage = changed_coverage
                scope = changed_scope
                if _has_declaration_only_test_proof(
                    Path(path_str),
                    base,
                    repo_root,
                    changed_files,
                ):
                    coverage = 100.0
                    scope = "declaration-only proof"
        results.append(
            {
                "file": path_str,
                "coverage": coverage,
                "resolved": resolved,
                "scope": scope,
            }
        )
    return results


def _summarize_results(
    results: list[_CoverageResult],
    min_required: float,
    goal: float,
) -> tuple[list[_CoverageResult], list[_CoverageResult], list[_CoverageResult]]:
    """Partition coverage results by missing data, minimum failures, and goal warnings.

    Returns:
        Missing rows, rows below the required minimum, and rows below the goal.
    """
    missing = [r for r in results if r["coverage"] is None]
    below_min = [r for r in results if r["coverage"] is not None and r["coverage"] < min_required]
    below_goal = [r for r in results if r["coverage"] is not None and r["coverage"] < goal]
    return missing, below_min, below_goal


def _print_results(
    results: list[_CoverageResult],
    min_required: float,
    goal: float,
) -> None:
    """Print per-file coverage status rows."""
    for r in results:
        cov = r["coverage"]
        file_path = r["file"]
        if cov is None:
            print(f"- {file_path}: coverage missing")
            continue
        scope = r.get("scope")
        scope_suffix = f" ({scope})" if scope and scope != "file" else ""
        status = "OK"
        if cov < min_required:
            status = "FAIL"
        elif cov < goal:
            status = "WARN"
        print(f"- {file_path}: {cov:.1f}% [{status}]{scope_suffix}")


def _report_failures(
    missing: list[_CoverageResult],
    below_min: list[_CoverageResult],
) -> None:
    """Print coverage failures that should fail the gate."""
    print("Test coverage requirement not met:")
    if missing:
        _print_lines(["- Missing coverage data for:"] + [f"  - {r['file']}" for r in missing])
    if below_min:
        _print_lines(
            ["- Below minimum coverage:"]
            + [f"  - {r['file']} ({r['coverage']:.1f}%)" for r in below_min]
        )


def _report_warnings(below_goal: list[_CoverageResult]) -> None:
    """Print coverage rows below the aspirational goal."""
    _print_lines(
        ["Coverage goal not met (warning only):"]
        + [f"- {r['file']} ({r['coverage']:.1f}%)" for r in below_goal]
    )


def _ensure_coverage_path(coverage_path: Path) -> bool:
    """Check whether the requested coverage artifact exists.

    Returns:
        True when the coverage artifact can be read.
    """
    if not coverage_path.exists():
        print(f"coverage.json not found at {coverage_path}", file=sys.stderr)
        return False
    return True


def _log_header(args: argparse.Namespace) -> None:
    """Print the configured coverage check header."""
    print(
        "Changed files test coverage check "
        f"(base={args.base}, min={args.min:.1f}%, goal={args.goal:.1f}%)"
    )


def _report_skipped(skipped: list[str], show_skipped: bool) -> None:
    """Optionally print files skipped by include/exclude filters."""
    if show_skipped and skipped:
        _print_lines(["Skipped:"] + [f"- {p}" for p in skipped])


def _handle_missing_or_below_min(
    missing: list[_CoverageResult],
    below_min: list[_CoverageResult],
) -> int:
    """Return a failing exit code when hard coverage requirements are unmet.

    Returns:
        ``1`` for missing/below-minimum coverage, otherwise ``0``.
    """
    if missing or below_min:
        _report_failures(missing, below_min)
        return 1
    return 0


def _run_check(args: argparse.Namespace) -> int:
    """Execute the changed-files coverage check.

    Returns:
        Process exit code for the configured coverage gate.
    """
    repo_root = _repo_root()
    coverage_path = _resolve_coverage_path(args.coverage, repo_root)
    if not _ensure_coverage_path(coverage_path):
        return 1

    include_patterns = args.include or ["robot_sf/*.py", "robot_sf/**/*.py"]
    exclude_patterns = args.exclude or []

    changed_files = _changed_files(args.base, repo_root)
    if not changed_files:
        print(f"No changed files vs {args.base}.")
        return 0

    selected, skipped = _select_changed_files(
        changed_files,
        repo_root,
        include_patterns,
        exclude_patterns,
    )
    if not selected:
        print("No changed files matched include patterns.")
        _report_skipped(skipped, args.show_skipped)
        return 0

    coverage_index = _load_coverage_index(coverage_path, repo_root)
    coverage_file_data = _load_coverage_file_data(coverage_path, repo_root)
    doc_only = [
        path_str
        for path_str in selected
        if _is_doc_or_comment_only_changed_file(Path(path_str), args.base, repo_root)
    ]
    if doc_only:
        doc_only_set = set(doc_only)
        selected = [path_str for path_str in selected if path_str not in doc_only_set]
        skipped.extend(f"{path_str} (doc/comment-only)" for path_str in doc_only)

    if not selected:
        print("No changed files with executable Python changes matched include patterns.")
        _report_skipped(skipped, args.show_skipped)
        return 0

    results = _build_results(
        selected,
        coverage_index,
        coverage_file_data,
        base=args.base,
        repo_root=repo_root,
        changed_files=changed_files,
    )

    _log_header(args)
    _print_results(results, args.min, args.goal)
    _report_skipped(skipped, args.show_skipped)

    missing, below_min, below_goal = _summarize_results(results, args.min, args.goal)
    failure_code = _handle_missing_or_below_min(missing, below_min)
    if failure_code:
        return failure_code

    if below_goal:
        _report_warnings(below_goal)

    return 0


def main() -> int:
    """Run the changed-files coverage check and return exit code."""
    args = _parse_args()
    return _run_check(args)


if __name__ == "__main__":
    sys.exit(main())
