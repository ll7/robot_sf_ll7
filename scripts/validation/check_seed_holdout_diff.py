"""Reject newly introduced evaluation seeds in executable pilot contexts.

This is a diff gate, not an audit of historical releases or archived evidence.
See issue #9668 for the evaluation split and anchor barrier.

Accepted syntactic limits: quoted seed values on continuation lines of multiline
JSON lists and seeds passed through environment variables need exact-head review.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

RELEASE_CONFIGS = frozenset(
    {
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml",
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml",
    }
)
MARKER = re.compile(r"#\s*seed-holdout:\s*(?:setup-only|synthetic-fixture)\b")
BLOCK_MARKER = re.compile(r"\s*#\s*seed-holdout:\s*(setup-only|synthetic-fixture)\s+(begin|end)\s*")
SEED = re.compile(
    r"(?<![\w.])(?:11[1-9]|12\d|13\d|140)(?![\w.])"  # seed-holdout: synthetic-fixture
)
SEED_FIELD = re.compile(
    r"(?i)(?:^|[\s,({])['\"]?(?:seed|seeds|seed_list|seed_set|resolved_seeds|"
    r"eval_seeds|evaluation_seeds|pilot_seeds|diagnostic_seeds|base_seed|"
    r"master_seed|episode_seed|scenario_seed|simulator_seed|simulation_seed|"
    r"environment_seed|env_seed|world_seed|run_seed|rollout_seed|task_seed|"
    r"map_seed|spawn_seed)['\"]?\s*[:=]"
)
SCENARIO_SEEDS = re.compile(r"(?i)\bscenario\s*\[\s*['\"]seeds?['\"]\s*\]\s*=")
EPISODE_SEED_LOOP = re.compile(
    r"\bfor\s+seed\s+in\s+range\s*\([^)]*\)\s*:\s*.*\brun_episode\s*\(\s*seed\b"
)
RANGE = re.compile(r"\brange\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
YAML_RANGE_BOUND = re.compile(r"^\s*(?:-\s*)?(?:min|max|low|high)\s*:", re.I)
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


def _enclosed_by_search_config(before: list[str]) -> bool:
    """Require the seed argument's innermost open call to be SearchConfig.from_files."""
    opened: list[tuple[str, bool]] = []
    recent: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO("\n".join(before) + "\n").readline)
        for token in tokens:
            if token.type != tokenize.OP:
                if token.type == tokenize.NAME:
                    recent.append(token.string)
                    recent = recent[-3:]
                continue
            symbol = token.string
            if symbol in "([{":
                opened.append(
                    (symbol, symbol == "(" and recent[-3:] == ["SearchConfig", ".", "from_files"])
                )
            elif symbol in ")]}":
                if opened:
                    opened.pop()
            recent.append(symbol)
            recent = recent[-3:]
    except tokenize.TokenError:
        # The line under inspection is inside an unfinished call.
        pass
    except IndentationError:
        return False
    return bool(opened and opened[-1] == ("(", True))


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
    if path != "scripts/tools/compare_adversarial_samplers.py":
        return False
    if text.strip() == "seeds = args.seed or [123]":  # seed-holdout: synthetic-fixture
        return True
    if text.strip() != "seed=(args.seed or [123])[0],":  # seed-holdout: synthetic-fixture
        return False
    # Only the SearchConfig keyword is a sampler RNG default. An earlier
    # SearchConfig call cannot exempt a later environment or episode call.
    return _enclosed_by_search_config(before)


def _range_overlaps_holdout(text: str) -> bool:
    """Recognize literal two-bound ranges, whose stop bound is exclusive."""
    return any(
        int(match.group(1)) <= 140 and int(match.group(2)) > 111 for match in RANGE.finditer(text)
    )


def _marked_block_lines(lines: list[str]) -> set[int]:
    """Only complete, matching marker pairs exempt their enclosed lines."""
    marked: set[int] = set()
    opened: tuple[str, int] | None = None
    for number, line in enumerate(lines, 1):
        marker = BLOCK_MARKER.fullmatch(line)
        if marker is None:
            continue
        kind, boundary = marker.groups()
        if boundary == "begin":
            opened = (kind, number) if opened is None else None
        elif opened is not None and opened[0] == kind:
            marked.update(range(opened[1] + 1, number))
            opened = None
    return marked


def _seed_range_context(text: str, before: list[str]) -> bool:
    """Recognize a range inside a nearby seed list or parametrization."""
    if not _range_overlaps_holdout(text):
        return False
    if SEED_PARAM.search("\n".join(before[-12:] + [text])):
        return True
    for previous in reversed(before[-12:]):
        if SEED_FIELD.search(previous):
            return True
        if previous.strip().startswith((")", "]", "}")) or re.match(r"^\s*[\w.]+\s*=", previous):
            break
    return False


def _yaml_seed_range_bound(path: str, text: str, before: list[str]) -> bool:
    """Match a range bound only under its nearest YAML seed key."""
    if not path.startswith("configs/") or not path.endswith((".yaml", ".yml")):
        return False
    if not YAML_RANGE_BOUND.match(text):
        return False
    indentation = len(text) - len(text.lstrip())
    for previous in reversed(before):
        stripped = previous.strip()
        if not stripped or stripped.startswith("#"):
            continue
        previous_indent = len(previous) - len(previous.lstrip())
        if previous_indent < indentation:
            return bool(SEED_FIELD.search(stripped)) and stripped.endswith(":")
    return False


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
        or _yaml_seed_range_bound(path, text, before)
        or _seed_range_context(text, before)
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


def check_diff(diff: str, root: Path) -> list[Finding]:
    """Return violations on added lines; root supplies complete file context."""
    findings: list[Finding] = []
    path = ""
    line_number = 0
    before: list[str] = []
    marked_block_lines: set[int] = set()
    file_lines: list[str] = []
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
            file_path = root / path
            file_lines = (
                file_path.read_text(errors="replace").splitlines() if file_path.is_file() else []
            )
            marked_block_lines = _marked_block_lines(file_lines)
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
                _eligible(path)
                and path not in RELEASE_CONFIGS
                and line_number not in marked_block_lines
                and not MARKER.search(content)
                and (SEED.search(content) or _range_overlaps_holdout(content))
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
