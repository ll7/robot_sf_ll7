"""Select PR slow witnesses by tracked imports and paths, with explicit fallback."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path


def changed_paths(root: Path, base: str, head: str = "HEAD") -> set[str]:
    """Include deleted paths and both rename endpoints without executing them."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=ACDMRT",
            f"{base}...{head}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    fields = iter(result.stdout.decode().split("\0")[:-1])
    paths = set()
    for status in fields:
        paths.add(next(fields))
        if status.startswith(("R", "C")):
            paths.add(next(fields))
    return paths


def _module(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _joined_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        left, right = _joined_literal(node.left), _joined_literal(node.right)
        if isinstance(node.op, ast.Div) and left is None:
            return right
        if left is not None and right is not None:
            return left + ("/" if isinstance(node.op, ast.Div) else "") + right
    if isinstance(node, ast.Call) and node.args:
        if isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath"}:
            return _joined_literal(node.args[0])
    return None


def _references(path: Path, relative: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    imports, literals = set(), set()
    package = _module(relative).split(".")[:-1]
    if path.name == "__init__.py":
        package = _module(relative).split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                prefix = ".".join(
                    package[: len(package) - node.level + 1] + ([prefix] if prefix else [])
                )
            imports.add(prefix)
            imports.update(f"{prefix}.{alias.name}" for alias in node.names)
        literal = _joined_literal(node)
        if literal:
            literals.add(literal.strip("/"))
    return imports, literals


def _dependency_index(vertices: set[str]) -> tuple[dict, dict, dict]:
    """Index module aliases, directory prefixes and unambiguous basenames."""
    basenames = {}
    for path in vertices:
        basenames.setdefault(Path(path).name, set()).add(path)
    modules = {}
    path_prefixes = {}
    for dependency in sorted(vertices):
        if dependency.endswith(".py"):
            aliases = {_module(dependency)}
            if dependency.startswith("fast-pysf/"):
                aliases.add(_module(dependency.removeprefix("fast-pysf/")))
            for alias in aliases:
                modules.setdefault(alias, set()).add(dependency)
        parts = dependency.split("/")
        for index in range(1, len(parts) + 1):
            path_prefixes.setdefault("/".join(parts[:index]), set()).add(dependency)
    return modules, path_prefixes, basenames


def _source_dependencies(imports, literals, text, modules, path_prefixes, basenames) -> dict:
    """Resolve static imports and path references without running input code."""
    dependencies = {}
    for name in sorted(imports):
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            prefix = ".".join(parts[:index])
            for dependency in modules.get(prefix, ()):
                dependencies[dependency] = "import"
    # Text inputs can themselves point at other inputs (YAML -> SVG, etc.).
    references = literals | {token for token in re.findall(r"[\w./-]+", text) if "/" in token}
    for literal in sorted(references):
        for dependency in path_prefixes.get(literal.strip("/"), ()) if "/" in literal else ():
            dependencies.setdefault(dependency, "path")
        if len(basenames.get(literal, ())) == 1:
            dependency = next(iter(basenames[literal]))
            dependencies.setdefault(dependency, "path")
    return dependencies


def _reverse_graph(vertices: set[str], sources: dict) -> dict:
    """Build edges from each dependency to the files that consume it."""
    modules, path_prefixes, basenames = _dependency_index(vertices)
    reverse = {}
    for source, (imports, literals, text) in sources.items():
        dependencies = _source_dependencies(
            imports, literals, text, modules, path_prefixes, basenames
        )
        for dependency, kind in sorted(dependencies.items()):
            if source != dependency:
                reverse.setdefault(dependency, []).append((source, kind))
    return reverse


def _closure(paths: set[str], reverse: dict) -> dict:
    """Find consumers breadth first, retaining the first sorted explanation chain."""
    reached = {path: "changed: " + path for path in sorted(paths)}
    queue = list(reached)
    for dependency in queue:
        for source, kind in reverse.get(dependency, []):
            if source not in reached:
                reached[source] = reached[dependency] + f" -> {kind}: {source}"
                queue.append(source)
    return reached


def _reaches_test(changed: str, tests: list[str], reverse: dict) -> bool:
    """Check each changed input independently, including overlapping closures."""
    seen = {changed}
    pending = [changed]
    for dependency in pending:
        if dependency in tests:
            return True
        for source, _ in reverse.get(dependency, ()):
            if source not in seen:
                seen.add(source)
                pending.append(source)
    return False


def _always_witness(path: str, text: str) -> bool:
    """Recognize integrity witnesses by purpose and digest use, regardless of marks."""
    stem = Path(path).stem
    return (
        bool(re.search(r"(?:^|_)(?:pins?|pinned|inventory|manifest|registry)(?:_|$)", stem))
        or any(
            word in stem
            for word in ("optimized_assert", "optimized_mode_assert", "docs_evidence", "issue_5303")
        )
        or any(word in text for word in ("sha256", "sha1", "hashlib", "required_pin_inventory"))
    )


def _scan_sources(root: Path, files: set[str], tests: list[str]) -> tuple[dict, dict]:
    """Read tracked text and Python ASTs; no untracked artifacts enter the graph."""
    reasons = {}
    sources = {}
    for path in sorted(files):
        target = root / path
        if target.stat().st_size > 2_000_000:
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        imports, literals = set(), set()
        if path.endswith(".py"):
            try:
                imports, literals = _references(target, path)
            except (SyntaxError, ValueError):
                raise ValueError(f"cannot parse {path}") from None
        sources[path] = (imports, literals, text)
        if path in tests and _always_witness(path, text):
            reasons[path] = "always: pin/inventory/manifest witness"
    return sources, reasons


def selection_reasons(root: Path, paths: set[str]) -> tuple[str, dict[str, str]]:
    """Trace tracked imports and input references; retain full fallback for unknown inputs.

    No dependency code is executed. Include package initializers, vendored module
    aliases, joined path literals, unique basenames, and transitive data references.
    Fixture/build/CI changes and unmapped inputs require complete admission.
    """
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, text=True).split("\0")
    files = {path for path in tracked if path and (root / path).is_file()}
    tests = sorted(
        path
        for path in files
        if path.endswith(".py")
        and path.startswith(("tests/", "fast-pysf/tests/"))
        and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
    )
    if any(
        Path(path).name == "conftest.py"
        or path.startswith(".github/")
        or path in {"pyproject.toml", "uv.lock", "setup.cfg", "pytest.ini"}
        or path.startswith("scripts/dev/")
        for path in paths
    ):
        return "full", dict.fromkeys(tests, "fallback: shared collection/build/CI input")
    try:
        sources, reasons = _scan_sources(root, files, tests)
    except ValueError as error:
        return "full", dict.fromkeys(tests, f"fallback: {error}")
    # Deleted and renamed paths remain vertices even when no longer tracked.
    vertices = files | paths
    reverse = _reverse_graph(vertices, sources)
    reached = _closure(paths, reverse)
    for path in tests:
        if path in reached:
            reasons.setdefault(path, reached[path])
    unmapped = sorted(path for path in paths if not _reaches_test(path, tests, reverse))
    if unmapped:
        return "full", dict.fromkeys(tests, "fallback: unmapped input " + unmapped[0])
    return "affected", dict(sorted(reasons.items()))


def affected_tests(root: Path, paths: set[str]) -> list[str]:
    """Return deterministic affected files plus unconditional integrity witnesses."""
    return sorted(selection_reasons(root, paths)[1])


def _identity(root: Path, ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", ref], cwd=root, text=True
    ).strip()


def selection_report(root: Path, base: str, head: str = "HEAD") -> dict:
    """Prepare one commit-bound diff decision for reuse across all shards."""
    base_sha = _identity(root, f"{base}^{{commit}}")
    head_sha = _identity(root, f"{head}^{{commit}}")
    paths = changed_paths(root, base_sha, head_sha)
    mode, reasons = selection_reasons(root, paths)
    return {
        "schema_version": "affected-test-selection.v1",
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tree_sha": _identity(root, f"{head_sha}^{{tree}}"),
        "mode": mode,
        "reason": "Tracked import/path closure plus always-run integrity witnesses; full fallback for uncertain inputs.",
        "reasons": reasons,
        "changed_paths": sorted(paths),
        "tests": sorted(reasons),
    }


def read_report(root: Path, path: Path, base: str, head: str) -> dict:
    """Reject stale, incomplete or unknown decisions before shard admission."""
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != "affected-test-selection.v1"
        or report.get("head_sha") != _identity(root, f"{head}^{{commit}}")
        or report.get("base_sha") != _identity(root, f"{base}^{{commit}}")
        or report.get("tree_sha") != _identity(root, f"{head}^{{tree}}")
        or report.get("mode") not in {"full", "affected", "unchanged"}
        or (
            report.get("mode") == "unchanged"
            and (
                report.get("changed_paths") != []
                or _identity(root, f"{base}^{{tree}}") != _identity(root, f"{head}^{{tree}}")
            )
        )
    ):
        raise ValueError("Selection decision is stale or invalid")
    return report


def main() -> None:
    """Prepare one decision or read its bound artifact cheaply in each shard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--read-report", type=Path)
    parser.add_argument("--format", choices=("paths", "mode"), default="paths")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = (
        read_report(root, args.read_report, args.base, args.head)
        if args.read_report
        else selection_report(root, args.base, args.head)
    )
    print(f"[test-selection] mode={report['mode']} files={len(report['tests'])}", file=sys.stderr)
    if not args.read_report:
        for test, reason in sorted(report.get("reasons", {}).items()):
            print(f"[test-selection] {test}: {reason}", file=sys.stderr)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["mode"] if args.format == "mode" else "\n".join(report["tests"]))


if __name__ == "__main__":
    main()
