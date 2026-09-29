#!/usr/bin/env python3
"""Advisory check: tests added or changed by a PR should fail on the base commit.

A new regression test that already passes on the base code does not detect the
regression it claims to cover. This script collects test functions that are new
or whose definition changed between the merge base and head, overlays the head
versions of the changed test files onto a temporary worktree at the base, runs
only those tests, and classifies each one:

* ``FAILS_ON_BASE``    - good: the test fails (assertion, missing API, collection error).
* ``PASSES_ON_BASE``   - flagged: the test does not detect its regression on the base.
* ``ERROR_UNRELATED``  - the run broke for reasons unrelated to the change (missing
  third-party dependency, fixture setup error, timeout, skip); reported separately.
* ``EXEMPT``           - marked ``@pytest.mark.new_feature_no_base(reason=...)``; not run.
* ``SKIPPED_SEED_GUARD`` - file uses benchmark/episode seeds 111-140; never run here.

Limits (by design, advisory): only test functions whose own AST (body, decorators,
arguments) changed are selected; edits that only touch helpers or fixtures used by an
unchanged test are not detected.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

FAILS = "FAILS_ON_BASE"
PASSES = "PASSES_ON_BASE"
UNRELATED = "ERROR_UNRELATED"
EXEMPT = "EXEMPT"
SEED_GUARD = "SKIPPED_SEED_GUARD"

EXEMPT_MARKER = "new_feature_no_base"
SEED_RANGE = range(111, 141)
TEST_DIRS = ("tests/", "fast-pysf/tests/")
DEFAULT_MAX_TESTS = 200
_BENCH_CONTEXT = re.compile(r"benchmark|episode", re.IGNORECASE)
_SEED_WORD = re.compile(r"seed", re.IGNORECASE)
_NO_MODULE = re.compile(r"No module named ['\"]([\w.]+)['\"]")
_CANNOT_IMPORT = re.compile(r"cannot import name .* from ['\"]([\w.]+)['\"]")


@dataclass
class TestRecord:
    """One selected test function and its classification."""

    __test__ = False  # not a pytest class

    path: str
    qualname: str
    reason: str = "new"  # new | changed
    exempt_reason: str | None = None
    classification: str = ""
    detail: str = ""

    @property
    def node_id(self) -> str:
        """Return the pytest node id without parameter suffix."""
        return f"{self.path}::{self.qualname.replace('.', '::')}"


@dataclass
class Report:
    """Full result of one check run."""

    base: str
    head: str
    tests: list[TestRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Return classification counts."""
        out: dict[str, int] = {}
        for rec in self.tests:
            out[rec.classification] = out.get(rec.classification, 0) + 1
        return out


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _show(repo: Path, ref: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def resolve_base(repo: Path, base: str | None, head: str) -> str:
    """Return the merge-base commit of ``base`` (default origin/main) and ``head``."""
    candidates = [base] if base else ["origin/main", "main"]
    last_err = ""
    for cand in candidates:
        proc = subprocess.run(
            ["git", "-C", str(repo), "merge-base", cand, head],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        last_err = proc.stderr.strip()
    raise RuntimeError(f"cannot resolve merge base: {last_err}")


def changed_files(repo: Path, base: str, head: str) -> list[tuple[str, str, str | None]]:
    """Return (status, head_path, base_path) for added/modified/renamed files."""
    out = _git(repo, "diff", "--name-status", "-M", "--diff-filter=AMR", base, head)
    rows: list[tuple[str, str, str | None]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        status = parts[0][0]
        if status == "R":
            rows.append((status, parts[2], parts[1]))
        else:
            rows.append((status, parts[1], parts[1] if status == "M" else None))
    return rows


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------


def is_test_file(path: str) -> bool:
    """Return whether ``path`` is a pytest test module in a known test directory."""
    name = Path(path).name
    return (
        path.endswith(".py")
        and path.startswith(TEST_DIRS)
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _marker_reason(decorators: list[ast.expr]) -> tuple[bool, str | None]:
    """Return (has_marker, reason) for the exemption marker among decorators."""
    for dec in decorators:
        if _dotted(dec).split(".")[-1] != EXEMPT_MARKER:
            continue
        reason = None
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                    reason = str(kw.value.value).strip() or None
            if reason is None and dec.args and isinstance(dec.args[0], ast.Constant):
                reason = str(dec.args[0].value).strip() or None
        return True, reason
    return False, None


def _module_marker(tree: ast.Module) -> tuple[bool, str | None]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            items = (
                node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            )
            found, reason = _marker_reason(list(items))
            if found:
                return found, reason
    return False, None


def collect_tests(source: str) -> dict[str, tuple[str, str | None]]:
    """Map test qualname -> (ast dump, exempt reason or '' when marker lacks a reason).

    The exemption value is ``None`` when the test is not exempt.
    Includes module-level ``test_*`` functions and methods of ``Test*`` classes
    (also nested). Parametrize decorators are part of the dump.
    """
    tree = ast.parse(source)
    mod_found, mod_reason = _module_marker(tree)
    result: dict[str, tuple[str, str | None]] = {}

    def visit(body: list[ast.stmt], prefix: str, inherited: tuple[bool, str | None]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                found, reason = _marker_reason(node.decorator_list)
                cls_mark = (found, reason) if found else inherited
                visit(node.body, f"{prefix}{node.name}.", cls_mark)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test"
            ):
                found, reason = _marker_reason(node.decorator_list)
                mark = (found, reason) if found else inherited
                dump = ast.dump(node, include_attributes=False)
                exempt = (mark[1] or "") if mark[0] else None
                result[f"{prefix}{node.name}"] = (dump, exempt)

    visit(tree.body, "", (mod_found, mod_reason))
    return result


def has_seed_literals(source: str) -> bool:
    """Return whether a file uses seeds 111-140 as literals in benchmark/episode context."""
    if not (_BENCH_CONTEXT.search(source) and _SEED_WORD.search(source)):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, int)
        and not isinstance(n.value, bool)
        and n.value in SEED_RANGE
        for n in ast.walk(tree)
    )


def _records_for_file(
    repo: Path, base: str, head: str, head_path: str, base_path: str | None
) -> tuple[list[TestRecord], str | None]:
    """Return new/changed test records for one file, plus an optional note."""
    head_src = _show(repo, head, head_path)
    if head_src is None:
        return [], None
    try:
        head_tests = collect_tests(head_src)
    except SyntaxError as exc:
        return [], f"{head_path}: syntax error at head ({exc.msg}); skipped"
    base_tests: dict[str, tuple[str, str | None]] = {}
    base_src = _show(repo, base, base_path) if base_path else None
    if base_src is not None:
        try:
            base_tests = collect_tests(base_src)
        except SyntaxError:
            base_tests = {}
    seed_guard = has_seed_literals(head_src)
    records: list[TestRecord] = []
    for qual, (dump, exempt) in head_tests.items():
        if qual in base_tests and base_tests[qual][0] == dump:
            continue
        rec = TestRecord(
            path=head_path,
            qualname=qual,
            reason="changed" if qual in base_tests else "new",
            exempt_reason=exempt or None,
        )
        if seed_guard:
            rec.classification = SEED_GUARD
            rec.detail = "file uses seed literals 111-140 in benchmark/episode context"
        elif exempt:
            rec.classification = EXEMPT
            rec.detail = exempt
        elif exempt == "":
            rec.detail = f"{EXEMPT_MARKER} marker without reason is ignored"
        records.append(rec)
    return records, None


def select_tests(repo: Path, base: str, head: str) -> tuple[list[TestRecord], list[str], list[str]]:
    """Return (records, overlay_files, notes) for tests new/changed in head vs base."""
    records: list[TestRecord] = []
    overlay: list[str] = []
    notes: list[str] = []
    for _status, head_path, base_path in changed_files(repo, base, head):
        if not head_path.startswith(TEST_DIRS):
            continue
        overlay.append(head_path)
        if not is_test_file(head_path):
            continue
        found, note = _records_for_file(repo, base, head, head_path, base_path)
        records.extend(found)
        if note:
            notes.append(note)
    return records, overlay, notes


# ---------------------------------------------------------------------------
# execution and classification
# ---------------------------------------------------------------------------


def _repo_packages(worktree: Path) -> set[str]:
    names = {"robot_sf", "scripts", "tests", "pysocialforce", "fast_pysf"}
    for child in worktree.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            names.add(child.name)
        elif child.suffix == ".py":
            names.add(child.stem)
    return names


def classify_error(err_type: str, message: str, phase: str, repo_pkgs: set[str]) -> tuple[str, str]:
    """Classify a failing/erroring test result."""
    text = f"{err_type} {message}"
    if "Timeout" in err_type or "Timeout" in message[:200]:
        return UNRELATED, "timeout"
    if "ModuleNotFoundError" in text or "ImportError" in text:
        match = _NO_MODULE.search(message) or _CANNOT_IMPORT.search(message)
        if match:
            root = match.group(1).split(".")[0]
            if root not in repo_pkgs:
                return UNRELATED, f"missing third-party module {root}"
        return FAILS, "import error on base (API missing)"
    if phase == "setup":
        return UNRELATED, f"setup/fixture error: {err_type}"
    return FAILS, err_type or "failure"


def _case_key(classname: str, name: str) -> tuple[str, str]:
    """Return (module path guess, qualname without params) from junit attributes."""
    base_name = name.split("[", 1)[0]
    parts = classname.split(".") if classname else []
    return ".".join(parts), base_name


def _case_outcome(case: ET.Element, collect_err: bool, repo_pkgs: set[str]) -> tuple[str, str]:
    """Classify one junit testcase element."""
    problem = case.find("failure")
    phase = "call"
    if problem is None:
        problem = case.find("error")
        phase = "setup"
    if problem is not None:
        msg = (problem.get("message") or "") + "\n" + (problem.text or "")[:2000]
        err_type = problem.get("type") or ""
        if not err_type:
            m = re.search(r"^E\s+(\w+(?:\.\w+)*(?:Error|Exception))", msg, re.MULTILINE)
            err_type = m.group(1) if m else ""
        return classify_error(err_type, msg, "collect" if collect_err else phase, repo_pkgs)
    skipped = case.find("skipped")
    if skipped is not None:
        return UNRELATED, f"skipped on base: {(skipped.get('message') or '')[:120]}"
    return PASSES, "passed on base"


def _merge_outcomes(results: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    """Aggregate parameter cases; any failing case means the test detects a difference."""
    merged: dict[str, tuple[str, str]] = {}
    for node, outs in results.items():
        for wanted in (FAILS, PASSES, UNRELATED):
            hit = [o for o in outs if o[0] == wanted]
            if hit:
                detail = hit[0][1] if len(outs) == 1 else f"{hit[0][1]} ({len(outs)} cases)"
                merged[node] = (wanted, detail)
                break
    return merged


def parse_junit(
    xml_path: Path, records: list[TestRecord], repo_pkgs: set[str]
) -> dict[str, tuple[str, str]]:
    """Return {record.node_id: (classification, detail)} aggregated over parameter cases."""
    by_key: dict[tuple[str, str], TestRecord] = {}
    by_module: dict[str, list[TestRecord]] = {}
    for rec in records:
        module = rec.path[:-3].replace("/", ".")
        cls_parts = rec.qualname.split(".")
        by_key[(".".join([module, *cls_parts[:-1]]), cls_parts[-1])] = rec
        by_module.setdefault(module, []).append(rec)
    if not xml_path.exists():
        return {}
    results: dict[str, list[tuple[str, str]]] = {}
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        rec = by_key.get(_case_key(classname, name))
        collect_err = rec is None
        if rec is not None:
            matched = [rec]
        else:  # collection error: reported against the whole module
            cands = (classname, name.replace("/", ".").removesuffix(".py"))
            matched = next((by_module[c] for c in cands if c in by_module), [])
        outcome = _case_outcome(case, collect_err, repo_pkgs) if matched else None
        for target in matched:
            results.setdefault(target.node_id, []).append(outcome)  # type: ignore[arg-type]
    return _merge_outcomes(results)


def run_on_base(
    repo: Path,
    base: str,
    head: str,
    records: list[TestRecord],
    overlay: list[str],
    *,
    pytest_args: list[str],
    timeout: int,
) -> list[str]:
    """Run selected records on a base worktree and set classifications. Return notes."""
    notes: list[str] = []
    runnable = [r for r in records if not r.classification]
    if not runnable:
        return notes
    tmp = Path(tempfile.mkdtemp(prefix="newtests-base-"))
    wt = tmp / "base"
    try:
        _git(repo, "worktree", "add", "--detach", str(wt), base)
        # submodule content is not checked out in the new worktree; reuse the current one
        fast = repo / "fast-pysf"
        if fast.is_dir() and any(fast.iterdir()) and not any((wt / "fast-pysf").iterdir()):
            shutil.rmtree(wt / "fast-pysf")
            (wt / "fast-pysf").symlink_to(fast, target_is_directory=True)
        for rel in overlay:
            content = subprocess.run(
                ["git", "-C", str(repo), "show", f"{head}:{rel}"], capture_output=True, check=True
            ).stdout
            dest = wt / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        junit = tmp / "junit.xml"
        node_ids = sorted({r.node_id for r in runnable})
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            "--timeout=120",
            "-q",
            f"--junitxml={junit}",
            *pytest_args,
            *node_ids,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=wt, capture_output=True, text=True, timeout=timeout, check=False
            )
            tail = "\n".join(proc.stdout.splitlines()[-5:])
            notes.append(f"pytest exit={proc.returncode}: {tail}")
        except subprocess.TimeoutExpired:
            notes.append(f"pytest run timed out after {timeout}s")
        results = parse_junit(junit, runnable, _repo_packages(wt))
        for rec in runnable:
            cls, detail = results.get(rec.node_id, (UNRELATED, "no result reported by pytest"))
            rec.classification, rec.detail = cls, detail
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)],
            capture_output=True,
            check=False,
        )
        shutil.rmtree(tmp, ignore_errors=True)
    return notes


def check(
    repo: Path,
    base: str | None,
    head: str,
    *,
    max_tests: int = DEFAULT_MAX_TESTS,
    pytest_args: list[str] | None = None,
    timeout: int = 1800,
) -> Report:
    """Run the full check and return the report."""
    base_sha = resolve_base(repo, base, head)
    head_sha = _git(repo, "rev-parse", head).strip()
    records, overlay, notes = select_tests(repo, base_sha, head_sha)
    report = Report(base=base_sha, head=head_sha, tests=records, notes=notes)
    runnable = [r for r in records if not r.classification]
    if len(runnable) > max_tests:
        for rec in runnable[max_tests:]:
            rec.classification = UNRELATED
            rec.detail = f"not run: more than {max_tests} selected tests"
        report.notes.append(f"truncated to first {max_tests} selected tests")
    report.notes.extend(
        run_on_base(
            repo,
            base_sha,
            head_sha,
            records,
            overlay,
            pytest_args=pytest_args or [],
            timeout=timeout,
        )
    )
    return report


def render_human(report: Report) -> str:
    """Render a Markdown summary."""
    counts = report.counts()
    lines = [
        "## New tests must fail on base (advisory)",
        "",
        f"base `{report.base[:12]}` head `{report.head[:12]}`; "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if counts
        else f"base `{report.base[:12]}` head `{report.head[:12]}`: no new or changed tests.",
        "",
    ]
    flagged = [r for r in report.tests if r.classification == PASSES]
    if flagged:
        lines.append("Tests that PASS on base (do not detect their regression):")
        lines += [f"- `{r.node_id}` ({r.reason})" for r in flagged]
        lines.append("")
    for label in (UNRELATED, SEED_GUARD, EXEMPT):
        rows = [r for r in report.tests if r.classification == label]
        if rows:
            lines.append(f"{label}:")
            lines += [f"- `{r.node_id}`: {r.detail}" for r in rows]
            lines.append("")
    lines += [f"note: {n}" for n in report.notes]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", help="base ref; merge-base with HEAD is used (default origin/main)")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--json-out", type=Path, help="write JSON report here")
    ap.add_argument("--summary-out", type=Path, help="write Markdown summary here")
    ap.add_argument("--max-tests", type=int, default=DEFAULT_MAX_TESTS)
    ap.add_argument("--timeout", type=int, default=1800, help="overall pytest timeout in seconds")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any test PASSES on base")
    ap.add_argument("--pytest-arg", action="append", default=[], help="extra pytest argument")
    args = ap.parse_args(argv)

    report = check(
        args.repo.resolve(),
        args.base,
        args.head,
        max_tests=args.max_tests,
        pytest_args=args.pytest_arg,
        timeout=args.timeout,
    )
    payload = {
        "base": report.base,
        "head": report.head,
        "counts": report.counts(),
        "notes": report.notes,
        "tests": [asdict(r) | {"node_id": r.node_id} for r in report.tests],
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = render_human(report)
    if args.summary_out:
        args.summary_out.write_text(summary, encoding="utf-8")
    print(summary)
    print(json.dumps(payload["counts"]))
    if args.strict and report.counts().get(PASSES):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
