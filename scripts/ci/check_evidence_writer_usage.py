"""Guard changed Python files against unmarked evidence-tree writers."""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path

EVIDENCE_PATH_FRAGMENT = "docs/context/evidence"
EXEMPTION_PATTERN = re.compile(r"#\s*evidence-writer-exempt:\s*(.*)$", re.IGNORECASE)
WRITE_METHODS = frozenset({"write_bytes", "write_text"})
INVENTORY_PATH_PREFIXES = ("hooks/", "scripts/", "robot_sf/")
BENCHMARK_PATH_PREFIXES = ("scripts/benchmark/", "robot_sf/benchmark/", "tests/benchmark/")
# The shared evidence-writer package owns the canonical raw-write wrappers; flagging
# its own definitions would be circular (those wrappers exist so other modules can
# adopt them). The guard enforces adoption everywhere else.
_SHARED_WRITER_MODULE_SUFFIX = ("robot_sf", "evidence", "writers.py")
_ForwardingHelper = tuple[str, int | None]


@dataclass(frozen=True, order=True)
class EvidenceWriterInventoryFinding:
    """One raw evidence-writer bypass found by inventory mode."""

    path: str
    line: int
    operation: str
    kind: str
    exemption_status: str
    exemption_reason: str | None = None


def _string_literal_parts(expr: ast.AST) -> list[str]:
    """Return string literal parts from ``expr`` in AST source order."""
    parts: list[str] = []
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        parts.append(expr.value)
    for child in ast.iter_child_nodes(expr):
        parts.extend(_string_literal_parts(child))
    return parts


def _has_write_mode(call: ast.Call) -> bool:
    """Return whether an ``open`` call requests a write-capable mode."""
    mode: ast.expr | None = None
    if len(call.args) >= 2:
        mode = call.args[1]
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
            break
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(flag in mode.value for flag in ("w", "a", "x", "+"))
    return False


def _expr_mentions_evidence(
    expr: ast.AST, evidence_names: set[str], evidence_arg_dests: set[str]
) -> bool:
    """Return whether an expression resolves to an evidence-tree path.

    Three resolution forms are recognized:

    - a string literal containing the evidence path fragment;
    - a name previously assigned from an evidence-mentioning expression; or
    - an ``args.<dest>`` attribute access whose argparse ``add_argument`` default
      points at the evidence tree (the indirect CLI-arg bypass pattern, e.g.
      ``args.output.write_text(...)``).
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return EVIDENCE_PATH_FRAGMENT in expr.value
    literal_path = "/".join(part.strip("/\\") for part in _string_literal_parts(expr))
    if EVIDENCE_PATH_FRAGMENT in literal_path:
        return True
    if isinstance(expr, ast.Name):
        return expr.id in evidence_names
    if isinstance(expr, ast.Attribute) and expr.attr in evidence_arg_dests:
        return True
    return any(
        _expr_mentions_evidence(child, evidence_names, evidence_arg_dests)
        for child in ast.iter_child_nodes(expr)
    )


def _evidence_argparse_dests(
    tree: ast.AST, evidence_names: set[str], evidence_arg_dests: set[str]
) -> set[str]:
    """Collect argparse destinations whose ``add_argument`` default is evidence.

    Flags the indirect CLI-arg bypass pattern: a writer does
    ``args.output.write_text(...)`` where ``--output`` defaults to
    ``docs/context/evidence/...``. The destination name (``--output-dir`` ->
    ``output_dir``) is returned only when its default literal or a name it resolves
    to mentions the evidence tree, so ordinary CLI outputs are not classified as
    evidence.
    """
    dests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute) and function.attr == "add_argument"):
            continue
        default = next(
            (kw.value for kw in node.keywords if kw.arg == "default" and kw.value is not None),
            None,
        )
        if default is None or not _expr_mentions_evidence(
            default, evidence_names, evidence_arg_dests
        ):
            continue
        dest: str | None = None
        argument_names = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        ]
        option_strings = [name for name in argument_names if name.startswith("-")]
        if option_strings:
            long_options = [option for option in option_strings if option.startswith("--")]
            selected_option = (long_options or option_strings)[0]
            dest = selected_option.lstrip("-").replace("-", "_")
        elif argument_names:
            dest = argument_names[0]
        explicit_dest = next(
            (
                kw.value
                for kw in node.keywords
                if kw.arg == "dest"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ),
            None,
        )
        if explicit_dest is not None:
            dest = explicit_dest.value
        if dest:
            dests.add(dest)
    return dests


def _evidence_path_names(tree: ast.AST, evidence_arg_dests: set[str]) -> set[str]:
    """Collect simple names assigned from expressions containing the evidence path."""
    evidence_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if node.value is None or not _expr_mentions_evidence(
                node.value, evidence_names, evidence_arg_dests
            ):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in evidence_names:
                    evidence_names.add(target.id)
                    changed = True
    return evidence_names


def _evidence_name_sets(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Resolve evidence names and argparse dests jointly to a fixed point.

    A destination whose default is a module constant (``default=DEFAULT_OUTPUT``)
    is only recognizable once that constant has been classified as evidence, and a
    name assigned from ``args.<dest>`` is only recognizable once the dest is known,
    so the two sets are propagated together.
    """
    evidence_names: set[str] = set()
    evidence_arg_dests: set[str] = set()
    changed = True
    while changed:
        changed = False
        names = _evidence_path_names(tree, evidence_arg_dests)
        if names - evidence_names:
            evidence_names |= names
            changed = True
        dests = _evidence_argparse_dests(tree, evidence_names, evidence_arg_dests)
        if dests - evidence_arg_dests:
            evidence_arg_dests |= dests
            changed = True
    return evidence_names, evidence_arg_dests


def _write_path_exprs(node: ast.Call) -> list[ast.AST]:
    """Return the path expressions a write-primitive call targets.

    Used by indirect-helper detection to decide whether a private helper forwards
    one of its parameters to a raw evidence write.
    """
    function = node.func
    if isinstance(function, ast.Attribute):
        if function.attr in WRITE_METHODS:
            return [function.value]
        if function.attr == "open" and _has_write_mode(node):
            return [function.value]
        if function.attr == "dump" and len(node.args) >= 2:
            return [node.args[1]]
        if function.attr == "DictWriter" and node.args:
            return [node.args[0]]
    elif isinstance(function, ast.Name) and function.id == "open":
        if _has_write_mode(node) and node.args:
            return [node.args[0]]
    return []


def _local_path_origins(func_def: ast.FunctionDef, params: list[str]) -> dict[str, set[str]]:
    """Map helper-local path aliases back to the parameters they derive from."""
    path_origins = {parameter: {parameter} for parameter in params}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if node.value is None:
                continue
            origins = {
                origin
                for child in ast.walk(node.value)
                if isinstance(child, ast.Name)
                for origin in path_origins.get(child.id, ())
            }
            for target in targets:
                if not origins or not isinstance(target, ast.Name):
                    continue
                previous = path_origins.setdefault(target.id, set())
                if origins - previous:
                    previous.update(origins)
                    changed = True
    return path_origins


def _forward_path_parameter(func_def: ast.FunctionDef) -> _ForwardingHelper | None:
    """Return the parameter a helper forwards to a write.

    A module-level helper like ``_write(path, text)`` that calls
    ``path.write_text(...)`` forwards its ``path`` parameter to a raw write the
    direct-write visitor cannot see (the parameter is not bound to an evidence
    literal at the definition site). Positional-only, positional-or-keyword, and
    keyword-only parameters are supported. The result carries the parameter name
    and its positional call index (``None`` for keyword-only parameters).
    """
    positional_params = [arg.arg for arg in (*func_def.args.posonlyargs, *func_def.args.args)]
    keyword_only_params = [arg.arg for arg in func_def.args.kwonlyargs]
    params = [*positional_params, *keyword_only_params]
    path_origins = _local_path_origins(func_def, params)
    forwarded: set[str] = set()
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        for expr in _write_path_exprs(node):
            for child in ast.walk(expr):
                if isinstance(child, ast.Name):
                    forwarded.update(path_origins.get(child.id, ()))
    if not forwarded:
        return None
    parameter_name = min(forwarded, key=params.index)
    positional_index = (
        positional_params.index(parameter_name) if parameter_name in positional_params else None
    )
    return parameter_name, positional_index


def _forwarding_helpers(tree: ast.AST) -> dict[str, _ForwardingHelper]:
    """Map module-level helpers to the parameters they forward to raw writes."""
    helpers: dict[str, _ForwardingHelper] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            parameter = _forward_path_parameter(node)
            if parameter is not None:
                helpers[node.name] = parameter
    return helpers


class _DirectWriterVisitor(ast.NodeVisitor):
    """Find writes whose target resolves to the evidence tree.

    Detects direct writes (``path.write_text()`` where ``path`` is an evidence
    literal, derived name, or evidence CLI dest) and indirect bypasses: private
    helpers that forward a parameter to a raw write (``_write(path, ...)``) called
    with an evidence path, and CLI-arg writes (``args.output.write_text()``) where
    the argparse default points at the evidence tree.
    """

    def __init__(
        self,
        evidence_names: set[str],
        evidence_arg_dests: set[str],
        forwarding_helpers: dict[str, _ForwardingHelper],
    ) -> None:
        self.violations: list[tuple[int, str, str]] = []
        self.evidence_names = evidence_names
        self.evidence_arg_dests = evidence_arg_dests
        self.forwarding_helpers = forwarding_helpers

    def _mentions(self, expr: ast.AST) -> bool:
        return _expr_mentions_evidence(expr, self.evidence_names, self.evidence_arg_dests)

    def _record(self, node: ast.Call, operation: str, kind: str) -> None:
        self.violations.append((node.lineno, operation, kind))

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute):
            self._check_attribute_call(node, function)
        elif isinstance(function, ast.Name):
            self._check_name_call(node, function)
        self.generic_visit(node)

    def _check_attribute_call(self, node: ast.Call, function: ast.Attribute) -> None:
        if function.attr in WRITE_METHODS and self._mentions(function.value):
            self._record(node, f".{function.attr}()", "direct")
        elif function.attr == "open" and _has_write_mode(node) and self._mentions(function.value):
            self._record(node, ".open(..., write mode)", "direct")
        elif function.attr == "DictWriter" and node.args and self._mentions(node.args[0]):
            self._record(node, "csv.DictWriter()", "direct")
        elif function.attr == "dump" and len(node.args) >= 2 and self._mentions(node.args[1]):
            self._record(node, "json.dump()", "direct")
        elif function.attr in {"_write_sha256sums", "write_sha256sums"} and any(
            self._mentions(argument) for argument in node.args
        ):
            if not (isinstance(function.value, ast.Name) and function.value.id == "writers"):
                self._record(node, f"{function.attr}()", "direct")

    def _check_name_call(self, node: ast.Call, function: ast.Name) -> None:
        if (
            function.id == "open"
            and _has_write_mode(node)
            and node.args
            and self._mentions(node.args[0])
        ):
            self._record(node, "open(..., write mode)", "direct")
        elif function.id in self.forwarding_helpers:
            parameter_name, positional_index = self.forwarding_helpers[function.id]
            path_arguments = [
                keyword.value for keyword in node.keywords if keyword.arg == parameter_name
            ]
            if positional_index is not None and positional_index < len(node.args):
                path_arguments.append(node.args[positional_index])
            if any(self._mentions(argument) for argument in path_arguments):
                operation = (
                    f"{function.id}() with an evidence-tree path "
                    "(forwards to a raw write that bypasses the shared writer)"
                )
                self._record(node, operation, "call")


def _structural_helper_violations(
    forwarding_helpers: dict[str, _ForwardingHelper],
    tree: ast.AST,
    has_evidence_output: bool,
) -> list[tuple[int, str, str]]:
    """Flag module-level forwarding helpers in modules that produce evidence.

    Some writers route evidence through an interprocedural or keyword-only parameter
    chain (e.g. ``def _write_outputs(*, output_dir, ...): _write(output_dir / ...)``)
    so the evidence path is invisible at every call site. Rather than fail to detect
    that bypass, the guard treats a module that both produces evidence and defines a
    private write-forwarding helper as an adoption gap: the helper exists to wrap a
    raw write, which is exactly what the shared writers replace. Genuine non-evidence
    helpers resolve with a justified ``# evidence-writer-exempt`` comment.
    """
    if not has_evidence_output or not forwarding_helpers:
        return []
    helper_lines = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in forwarding_helpers
    }
    return [
        (
            line,
            f"{name}() (forwards a path parameter to a raw write)",
            "definition",
        )
        for name, line in helper_lines.items()
    ]


def _format_blocker(source_path: Path, line: int, operation: str, kind: str) -> str:
    """Format a guard blocker with an actionable remediation hint."""
    lead = {
        "direct": f"directly uses {operation}",
        "call": f"calls {operation}",
        "definition": f"defines {operation}",
    }[kind]
    return (
        f"BLOCKER: '{source_path}:{line}' {lead} in a file that writes generated evidence. "
        "Use robot_sf.evidence.writers.write_json/write_csv/write_text/write_sha256sums, "
        "or add a justified '# evidence-writer-exempt: <reason>' comment."
    )


def _is_shared_writer_module(source_path: Path) -> bool:
    """Return whether ``source_path`` is the canonical shared writer module."""
    resolved = source_path.resolve()
    suffix_length = len(_SHARED_WRITER_MODULE_SUFFIX)
    return resolved.parts[-suffix_length:] == _SHARED_WRITER_MODULE_SUFFIX


def _exemption_status(source: str) -> tuple[str, str | None, str | None]:
    """Return ``(status, reason, error)`` for a file-level exemption marker."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comment_tokens = (token for token in tokens if token.type == tokenize.COMMENT)
        for token in comment_tokens:
            if token.start[1] != 0:
                continue
            match = EXEMPTION_PATTERN.match(token.string)
            if match is None:
                continue
            reason = match.group(1).strip()
            if not reason:
                return (
                    "invalid",
                    None,
                    f"line {token.start[0]} has an empty evidence-writer exemption reason",
                )
            reason_parts = [reason]
            source_lines = source.splitlines()
            next_line_index = token.start[0]
            while next_line_index < len(source_lines):
                continuation = source_lines[next_line_index]
                if not continuation.startswith("#"):
                    break
                continuation_text = continuation[1:].strip()
                if not continuation_text or EXEMPTION_PATTERN.match(continuation):
                    break
                reason_parts.append(continuation_text)
                next_line_index += 1
            return "valid", " ".join(reason_parts), None
    except (IndentationError, tokenize.TokenError):
        # Let ast.parse below report malformed source as a fail-closed blocker.
        return "none", None, None
    return "none", None, None


def _exemption(source: str) -> tuple[bool, str | None]:
    """Return whether source has a valid file-level exemption."""
    status, _, error = _exemption_status(source)
    return status in {"valid", "invalid"}, error


def _repo_root() -> Path:
    """Return the current Git repository root."""
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _repo_relative_path(path: Path, repo_root: Path) -> str:
    """Return a stable slash-separated repository-relative path when possible."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _violation_tuples(source_path: Path, source: str) -> list[tuple[int, str, str]]:
    """Return raw evidence-writer bypasses in ``source``."""
    tree = ast.parse(source, filename=str(source_path))
    evidence_names, evidence_arg_dests = _evidence_name_sets(tree)
    forwarding_helpers = _forwarding_helpers(tree)
    has_evidence_output = bool(evidence_names) or bool(evidence_arg_dests)

    visitor = _DirectWriterVisitor(evidence_names, evidence_arg_dests, forwarding_helpers)
    visitor.visit(tree)
    violations = visitor.violations
    violations.extend(_structural_helper_violations(forwarding_helpers, tree, has_evidence_output))
    return sorted(violations, key=lambda item: (item[0], item[1], item[2]))


def inventory_file(
    path: str | Path, repo_root: Path | None = None
) -> list[EvidenceWriterInventoryFinding]:
    """Return raw evidence-writer bypasses for one Python file, including exemption status."""
    source_path = Path(path)
    if (
        source_path.suffix != ".py"
        or not source_path.is_file()
        or _is_shared_writer_module(source_path)
    ):
        return []
    if repo_root is None:
        repo_root = _repo_root()
    relative_path = _repo_relative_path(source_path, repo_root)
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [
            EvidenceWriterInventoryFinding(
                path=relative_path,
                line=1,
                operation="read",
                kind="error",
                exemption_status="invalid",
                exemption_reason=f"cannot read source: {exc}",
            )
        ]
    exemption_status, exemption_reason, exemption_error = _exemption_status(source)
    if exemption_error is not None:
        exemption_reason = exemption_error
    try:
        violations = _violation_tuples(source_path, source)
    except SyntaxError as exc:
        return [
            EvidenceWriterInventoryFinding(
                path=relative_path,
                line=exc.lineno or 1,
                operation="parse",
                kind="error",
                exemption_status="invalid",
                exemption_reason=f"cannot parse source: {exc.msg}",
            )
        ]
    return [
        EvidenceWriterInventoryFinding(
            path=relative_path,
            line=line,
            operation=operation,
            kind=kind,
            exemption_status=exemption_status,
            exemption_reason=exemption_reason,
        )
        for line, operation, kind in violations
    ]


def check_file(path: str | Path) -> list[str]:
    """Return fail-closed guard messages for one changed Python file."""
    source_path = Path(path)
    if source_path.suffix != ".py" or not source_path.is_file():
        return []
    if _is_shared_writer_module(source_path):
        # The shared evidence-writer package owns the canonical raw-write wrappers;
        # flagging its own definitions would be circular.
        return []
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"BLOCKER: could not read changed Python file '{source_path}': {exc}"]

    has_exemption, exemption_error = _exemption(source)
    if exemption_error is not None:
        return [f"BLOCKER: evidence-writer exemption in '{source_path}' {exemption_error}"]
    if has_exemption:
        return []
    try:
        violations = _violation_tuples(source_path, source)
    except SyntaxError as exc:
        return [f"BLOCKER: cannot parse changed Python file '{source_path}': {exc}"]
    return [
        _format_blocker(source_path, line, operation, kind) for line, operation, kind in violations
    ]


def _is_changed_from_base(path: str | Path, base_ref: str) -> bool:
    """Return whether a repository file differs from ``base_ref``.

    Historical PR regression checks replay old file lists against the current
    checkout. Skipping files that are no longer changed avoids re-linting those
    old paths; an unavailable base is treated as changed so the guard fails
    closed in CI.
    """
    source_path = Path(path)
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        relative_path = source_path.resolve().relative_to(repo_root.resolve())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return True
    result = subprocess.run(
        ["git", "diff", "--quiet", base_ref, "--", relative_path.as_posix()],
        capture_output=True,
        check=False,
    )
    return result.returncode != 0


def check_changed_files(changed_files: list[str], base_ref: str = "origin/main") -> list[str]:
    """Check only changed Python files, preserving the PR contract boundary."""
    blockers: list[str] = []
    for path in changed_files:
        if not _is_changed_from_base(path, base_ref):
            continue
        blockers.extend(check_file(path))
    return blockers


def _is_inventory_path(path: str) -> bool:
    """Return whether a tracked Python path belongs in the non-benchmark inventory."""
    return (
        path.endswith(".py")
        and any(path.startswith(prefix) for prefix in INVENTORY_PATH_PREFIXES)
        and not any(path.startswith(prefix) for prefix in BENCHMARK_PATH_PREFIXES)
    )


def inventory_tracked_files() -> tuple[list[EvidenceWriterInventoryFinding], int]:
    """Scan tracked non-benchmark Python files for raw evidence-writer bypasses."""
    repo_root = _repo_root()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        check=True,
        capture_output=True,
    )
    paths = sorted(
        path.decode("utf-8")
        for path in result.stdout.split(b"\0")
        if path and _is_inventory_path(path.decode("utf-8"))
    )
    findings: list[EvidenceWriterInventoryFinding] = []
    for path in paths:
        findings.extend(inventory_file(repo_root / path, repo_root))
    return sorted(findings), len(paths)


def _print_json(payload: dict[str, object]) -> None:
    """Print deterministic JSON output."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_inventory(*, json_output: bool) -> int:
    """Run the read-only tracked-file inventory mode."""
    findings, scanned_paths = inventory_tracked_files()
    scan_errors = any(finding.kind == "error" for finding in findings)
    if json_output:
        _print_json(
            {
                "mode": "inventory",
                "scanned_paths": scanned_paths,
                "approved_prefixes": list(INVENTORY_PATH_PREFIXES),
                "excluded_prefixes": list(BENCHMARK_PATH_PREFIXES),
                "count": len(findings),
                "findings": [asdict(finding) for finding in findings],
            }
        )
        return 1 if scan_errors else 0
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}: operation={finding.operation} "
            f"kind={finding.kind} exemption_status={finding.exemption_status}"
        )
    print(f"Inventory findings: {len(findings)} across {scanned_paths} tracked Python files")
    return 1 if scan_errors else 0


def main() -> int:
    """Run the changed-file evidence-writer guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--changed-files-file",
        type=Path,
        help="newline-delimited changed-file paths",
    )
    parser.add_argument("--base-ref", default="origin/main", help="git base ref")
    mode_group.add_argument(
        "--inventory",
        action="store_true",
        help=(
            "read-only inventory of tracked non-benchmark Python files; excludes "
            "scripts/benchmark/, robot_sf/benchmark/, and tests/benchmark/"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit deterministic JSON output")
    args = parser.parse_args()
    if args.inventory:
        return _run_inventory(json_output=args.json)
    if args.changed_files_file is None:
        parser.error("--changed-files-file is required unless --inventory is set")
    changed_files = [
        line.strip()
        for line in args.changed_files_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blockers = check_changed_files(changed_files, args.base_ref)
    if args.json:
        _print_json({"mode": "changed-files", "blockers": blockers, "count": len(blockers)})
    else:
        for blocker in blockers:
            print(blocker)
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
