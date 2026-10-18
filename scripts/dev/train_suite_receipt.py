#!/usr/bin/env python3
"""Bind a complete pytest run to a clean, fully materialized Git commit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def git(*args: str) -> str:
    """Read Git metadata without exposing environment values in the receipt."""
    return subprocess.check_output(["git", *args], text=True).strip()


def binding(expected_head: str, expected_tree: str | None = None) -> tuple[str, str]:
    """Refuse hidden, omitted, modified or stale sources before accepting evidence."""
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if head != expected_head:
        raise ValueError("expected head differs from checkout HEAD")
    if expected_tree is not None and tree != expected_tree:
        raise ValueError("expected tree differs from checkout tree")
    sparse = subprocess.run(
        ["git", "config", "--bool", "core.sparseCheckout"],
        capture_output=True,
        text=True,
        check=False,
    )
    if sparse.returncode not in (0, 1):
        raise ValueError("cannot verify complete checkout configuration")
    flags = git("ls-files", "-v", "-z").split("\0")
    if sparse.stdout.strip() == "true" or any(
        entry and (entry[0] == "S" or entry[0].islower()) for entry in flags
    ):
        raise ValueError("train suite requires a complete checkout without sparse or hidden files")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("train suite requires a clean tree including untracked files")
    return head, tree


def validate_destination(receipt: Path) -> None:
    """Keep evidence outside tracked inputs, including when a caller chooses its location."""
    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    for path in (receipt, receipt.with_suffix(".log")):
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        tracked = git("ls-files", "--", str(relative))
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", "--", str(relative)], check=False
            ).returncode
            == 0
        )
        if tracked or not ignored:
            raise ValueError("receipt and log must be outside the checkout or in ignored output")


def summary(log: Path) -> dict[str, int]:
    """Use the final pytest terminal summary, rejecting missing or deselected results."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
    summaries = [line for line in text.splitlines() if re.search(r"\bin \d+(?:\.\d+)?s\b", line)]
    for line in reversed(summaries):
        counts = {
            label: int(count)
            for count, label in re.findall(
                r"\b(\d+) (passed|failed|skipped|xfailed|xpassed|deselected|errors?|warnings?)\b",
                line,
            )
            if label not in ("warning", "warnings")
        }
        if counts:
            return counts
    return {}


def main() -> int:
    """Preflight the binding or finalize a durable receipt with the effective exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "finalize"))
    parser.add_argument("--expect-head", required=True)
    parser.add_argument("--expect-tree")
    parser.add_argument("--receipt-file", type=Path, required=True)
    parser.add_argument("--pytest-exit-code", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        try:
            _, tree = binding(args.expect_head)
            validate_destination(args.receipt_file)
        except ValueError as error:
            parser.exit(2, f"{error}\n")
        print(tree)
        return 0
    if args.expect_tree is None or args.pytest_exit_code is None:
        parser.error("finalize requires --expect-tree and --pytest-exit-code")
    error = ""
    try:
        binding(args.expect_head, args.expect_tree)
    except ValueError as failure:
        error = str(failure)
    counts = summary(args.receipt_file.with_suffix(".log"))
    if (
        not counts
        or counts.get("deselected", 0)
        or not sum(
            counts.get(label, 0) for label in ("passed", "failed", "skipped", "xfailed", "xpassed")
        )
    ):
        error = error or "missing complete pytest summary or deselected tests"
    exit_code = 2 if error else args.pytest_exit_code
    receipt = {
        "schema": "train-suite.v1",
        "head_sha": args.expect_head,
        "tree_sha": args.expect_tree,
        "current_head_sha": git("rev-parse", "HEAD"),
        "current_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "counts": counts,
        "pytest_exit_code": args.pytest_exit_code,
        "exit_code": exit_code,
        "complete": exit_code == 0,
        "binding_error": error,
        "log_file": args.receipt_file.with_suffix(".log").name,
    }
    temporary = args.receipt_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(args.receipt_file)
    if error:
        print(error)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
