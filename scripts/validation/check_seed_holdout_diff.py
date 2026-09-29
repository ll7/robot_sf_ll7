"""Reject newly introduced evaluation seeds in executable pilot contexts.

This is a diff gate, not an audit of historical releases or archived evidence.
See issue #9668 for the evaluation split and anchor barrier.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

RELEASE_CONFIGS = frozenset(
    {
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml",
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml",
    }
)
MARKER = re.compile(r"#\s*seed-holdout:\s*(?:setup-only|synthetic-fixture)\b")
FILE_MARKER = re.compile(r"\s*#\s*seed-holdout:\s*(?:setup-only|synthetic-fixture)\s*")
SEED = re.compile(
    r"(?<![\w.])(?:11[1-9]|12\d|13\d|140)(?![\w.])"  # seed-holdout: synthetic-fixture
)
SEED_FIELD = re.compile(
    r"(?i)(?:^|[\s,({])['\"]?(?:seed|seeds|seed_list|seed_set|resolved_seeds|"
    r"eval_seeds|evaluation_seeds|pilot_seeds|diagnostic_seeds|base_seed|"
    r"master_seed|episode_seed|scenario_seed)['\"]?\s*[:=]"
)
SCENARIO_SEEDS = re.compile(r"(?i)\bscenario\s*\[\s*['\"]seeds?['\"]\s*\]\s*=")
EPISODE_SEED_LOOP = re.compile(
    r"\bfor\s+seed\s+in\s+range\s*\([^)]*\)\s*:\s*.*\brun_episode\s*\(\s*seed\b"
)
SAMPLER_CLI_DEFAULT = re.compile(r"\bseeds?\s*=\s*\(?\s*args\.seed\s+or\b")
CLI_SEED = re.compile(r"(?<![\w-])--seeds?(?:\s+|=)")
SEED_PARAM = re.compile(r"(?i)\bparametrize\s*\(\s*['\"][^'\"]*\bseed\b")
SEED_CONSTANT = re.compile(r"(?i)\b(?:DEFAULT|TEST|PILOT|DIAGNOSTIC|EVAL)_SEEDS?\s*=")
HUNK = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class Finding:
    """An added line that may step a planner on a held-out seed."""

    path: str
    line: int
    text: str


def _eligible(path: str) -> bool:
    if path.startswith("tests/analysis/fixtures/") and path.endswith(".jsonl"):
        return False
    return path.startswith(
        ("configs/", "tests/", "scripts/", "docs/plan/", "docs/context/evidence/")
    )


def _rejection_test_seed(path: str, text: str, before: list[str], after: list[str]) -> bool:
    """Recognize negative tests that pass holdout values only to validation."""
    if not path.startswith("tests/"):
        return False
    if SEED_PARAM.search(text):
        next_lines = "\n".join(after[:10])
        return (
            bool(re.search(r"def test_\w*reject\w*\(", next_lines))
            and "pytest.raises" in next_lines
        )
    if text.lstrip().startswith("payload["):
        declarations = [line for line in before if line.startswith("def test_")]
        return bool(
            declarations
            and re.search(r"def test_\w*reject\w*\(", declarations[-1])
            and "pytest.raises" in "\n".join(after[:20])
        )
    return False


def _non_episode_seed(path: str, text: str, before: list[str], after: list[str]) -> bool:
    """Recognize narrow setup metadata and sampler RNG defaults."""
    if _rejection_test_seed(path, text, before, after):
        return True
    if path == "tests/benchmark/test_release_candidate.py" and re.search(
        r"['\"]resolved_seeds['\"]\s*:", text
    ):
        # The candidate_repo fixture assembles preflight manifest metadata;
        # no episode runner consumes this seed list in the test.
        declarations = [line for line in before if line.lstrip().startswith("def ")]
        if declarations and declarations[-1].lstrip().startswith("def candidate_repo("):
            return True
    return path == "scripts/tools/compare_adversarial_samplers.py" and bool(
        SAMPLER_CLI_DEFAULT.search(text)
    )


def _seed_context(path: str, text: str, before: list[str], after: list[str]) -> bool:
    if CLI_SEED.search(text):
        return True
    if path.startswith("docs/") or _non_episode_seed(path, text, before, after):
        return False
    if (
        SEED_FIELD.search(text)
        or SCENARIO_SEEDS.search(text)
        or EPISODE_SEED_LOOP.search(text)
        or SEED_CONSTANT.search(text)
        or SEED_PARAM.search(text)
    ):
        return True
    # YAML block lists and multiline pytest parametrizations put the value on
    # a separate line from the seed key. Use the closest enclosing declaration.
    if re.match(
        r"^\s*(?:-\s*)?\[?\s*(?:11[1-9]|12\d|13\d|140)(?:\s*,\s*\d+)*\s*\]?,?\s*(?:#.*)?$",
        text,
    ):
        if path == "configs/benchmarks/seed_list_v1.yaml":
            return True
        if any("parametrize" in line for line in before[-12:]) and any(
            re.search(r"['\"]seed['\"]", line) for line in before[-12:]
        ):
            return True
        for previous in reversed(before):
            if SEED_FIELD.search(previous) or SEED_PARAM.search(previous):
                return True
            if previous.strip() and not re.match(
                r"^\s*(?:-\s*)?\[?\s*\d+(?:\s*,\s*\d+)*\s*,?\]?\s*(?:#.*)?$|^\s*#",
                previous,
            ):
                break
    return False


def _removed_lines(diff: str) -> dict[str, Counter[str]]:
    """Count deleted lines by file, ignoring indentation changed by a move."""
    removed: dict[str, Counter[str]] = defaultdict(Counter)
    path = ""
    for row in diff.splitlines():
        if row.startswith("diff --git "):
            path = row.rsplit(" b/", 1)[-1]
        elif row.startswith("+++ b/"):
            path = row[6:]
        elif row.startswith("-") and not row.startswith("--- "):
            removed[path][row[1:].strip()] += 1
    return removed


def _consume_moved(removed: dict[str, Counter[str]], path: str, content: str) -> bool:
    """Consume one matching deletion for an added line, if present."""
    normalized = content.strip()
    if removed[path][normalized] == 0:
        return False
    removed[path][normalized] -= 1
    return True


def check_diff(diff: str, root: Path) -> list[Finding]:
    """Return violations on added lines; root supplies complete file context."""
    findings: list[Finding] = []
    removed = _removed_lines(diff)
    path = ""
    line_number = 0
    before: list[str] = []
    file_marked = False
    file_lines: list[str] = []
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
            file_path = root / path
            file_lines = (
                file_path.read_text(errors="replace").splitlines() if file_path.is_file() else []
            )
            file_marked = any(FILE_MARKER.fullmatch(line) for line in file_lines[:30])
            before = []
        elif row.startswith("@@ "):
            match = HUNK.match(row)
            if match:
                line_number = int(match.group(1))
                if file_lines:
                    before = file_lines[: line_number - 1]
        elif row.startswith("+") and not row.startswith("+++ "):
            content = row[1:]
            if (
                not _consume_moved(removed, path, content)
                and _eligible(path)
                and path not in RELEASE_CONFIGS
                and not file_marked
                and not MARKER.search(content)
                and SEED.search(content)
                and _seed_context(path, content, before, file_lines[line_number:])
            ):
                findings.append(Finding(path, line_number, content.strip()))
            before.append(content)
            line_number += 1
        elif row.startswith(" "):
            before.append(row[1:])
            line_number += 1
    return findings


def main() -> int:
    """Check the PR's added lines against the evaluation seed holdout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", default="HEAD")
    args = parser.parse_args()
    command = [
        "git",
        "diff",
        "--no-ext-diff",
        "--no-renames",
        "--unified=3",
        args.base_ref,
        args.head_ref,
        "--",
        "configs/",
        "tests/",
        "scripts/",
        "docs/plan/",
        "docs/context/evidence/",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    findings = check_diff(result.stdout, Path.cwd())
    for finding in findings:
        print(
            f"{finding.path}:{finding.line}: evaluation seed in executable context: {finding.text}"
        )
    if findings:
        print(
            "Use dev seeds 1001-1030 for tests/tuning; calibration 101-102 and diagnostics 103-105 are reserved for their registered roles. Mark setup-only or synthetic fixtures explicitly.",
            file=sys.stderr,
        )
        return 1
    print("Seed holdout diff check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
