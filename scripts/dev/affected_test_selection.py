"""Conservative PR test admission, including every slow test after any change.

Import graphs cannot prove completeness for dynamic imports, fixture hooks or
literal/data dependencies. Until that proof exists, every nonempty diff runs the
full set. This intentionally trades extra execution for zero mapped omissions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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


def affected_tests(root: Path, paths: set[str]) -> list[str]:
    """Return the complete tracked test inventory for every changed input.

    Enumerate every tracked Python file, including vendored packages and roots
    omitted by the old import scan. No package-name inference or path-literal
    matching is used to exclude a test. Full pytest roots remain authoritative
    for execution, including custom collection rules.
    """
    if not paths:
        return []
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.py"], cwd=root, check=True, capture_output=True, text=True
    )
    return sorted(
        path
        for path in result.stdout.split("\0")
        if path
        and (root / path).is_file()
        and (path.startswith("tests/") or path.startswith("fast-pysf/tests/"))
        and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
    )


def _identity(root: Path, ref: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", ref], cwd=root, text=True
    ).strip()


def selection_report(root: Path, base: str, head: str = "HEAD") -> dict:
    """Prepare a reusable immutable diff decision without scanning test bodies."""
    base_sha = _identity(root, f"{base}^{{commit}}")
    head_sha = _identity(root, f"{head}^{{commit}}")
    paths = changed_paths(root, base_sha, head_sha)
    return {
        "schema_version": "affected-test-selection.v1",
        "base_sha": base_sha,
        "head_sha": head_sha,
        "tree_sha": _identity(root, f"{head_sha}^{{tree}}"),
        "mode": "full" if paths else "unchanged",
        "reason": "Any changed input requires complete admission; dependency inference is not exclusion proof.",
        "changed_paths": sorted(paths),
        "tests": affected_tests(root, paths),
    }


def read_report(root: Path, path: Path, base: str, head: str) -> dict:
    """Reject stale, incomplete or unknown decisions before shard admission."""
    report = json.loads(path.read_text())
    if (
        report.get("schema_version") != "affected-test-selection.v1"
        or report.get("head_sha") != _identity(root, f"{head}^{{commit}}")
        or report.get("base_sha") != _identity(root, f"{base}^{{commit}}")
        or report.get("tree_sha") != _identity(root, f"{head}^{{tree}}")
        or report.get("mode") not in {"full", "unchanged"}
        or (report.get("mode") == "unchanged" and report.get("changed_paths") != [])
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
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["mode"] if args.format == "mode" else "\n".join(report["tests"]))


if __name__ == "__main__":
    main()
