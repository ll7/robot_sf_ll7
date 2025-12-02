"""Emit a grouped Ruff docstring report for contributor triage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

DOCSTRING_RULES = [
    "D100",
    "D101",
    "D102",
    "D103",
    "D104",
    "D105",
    "D106",
    "D107",
    "D201",
    "D417",
    "D419",
]

OUTPUT_PATH = Path("output/issues/docstrings_summary.json")


def run_ruff(paths: list[str]) -> list[dict]:
    """Run Ruff with JSON output and return parsed diagnostics."""
    cmd = [
        "ruff",
        "check",
        f"--select={','.join(DOCSTRING_RULES)}",
        "--output-format",
        "json",
        *paths,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    if not proc.stdout.strip():
        return []
    return json.loads(proc.stdout)


def build_summary(diagnostics: list[dict]) -> dict:
    """Group diagnostics by file path."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for diag in diagnostics:
        grouped[diag["filename"]].append(
            {
                "rule": diag["code"],
                "message": diag["message"],
                "line": diag["location"]["row"],
                "column": diag["location"]["column"],
            },
        )
    files = []
    total = 0
    for path, entries in sorted(grouped.items()):
        files.append({"path": path, "count": len(entries), "violations": entries})
        total += len(entries)
    return {"files": files, "total_files": len(files), "total_violations": total}


def main() -> None:
    """Parse CLI arguments, run Ruff, and persist the grouped docstring summary."""
    parser = argparse.ArgumentParser(description="Group Ruff docstring diagnostics by file.")
    parser.add_argument("paths", nargs="*", default=["."], help="Paths to inspect.")
    args = parser.parse_args()
    diagnostics = run_ruff(args.paths)
    summary = build_summary(diagnostics)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Docstring report written to {OUTPUT_PATH} ({summary['total_violations']} violations).")


if __name__ == "__main__":
    main()
