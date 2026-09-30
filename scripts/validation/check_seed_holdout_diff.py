"""Reject newly introduced evaluation seeds in executable pilot contexts.

This is a diff gate, not an audit of historical releases or archived evidence.
See issue #9668 for the evaluation split and anchor barrier.

Accepted syntactic limits: quoted seed values on continuation lines of multiline
JSON lists, non-literal or aliased seed generation (including dynamic ``range``
bounds), and seeds passed through environment variables need exact-head review.
Literal Python ``range`` calls and nearby episode loops are checked across common
wrappers and line breaks; unknown syntax remains an exact-head review obligation.

Markers: ``# seed-holdout: setup-only`` and ``synthetic-fixture`` (line or
``begin``/``end`` block) exempt fixtures anywhere. ``release-evaluation`` (line
or block) exempts the evaluation seed list of a release manifest and is valid
only under ``configs/benchmarks/releases/`` and
``configs/benchmarks/paper_experiment_matrix_*``; elsewhere it is reported as an
error. Unpaired or mismatched blocks exempt nothing.
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
MARKER_KINDS = "setup-only|synthetic-fixture|release-evaluation"
MARKER = re.compile(rf"#\s*seed-holdout:\s*(?P<kind>{MARKER_KINDS})\b")
BLOCK_MARKER = re.compile(rf"\s*#\s*seed-holdout:\s*({MARKER_KINDS})\s+(begin|end)\s*")
# The release-evaluation marker is valid only in release evaluation manifests,
# where the held-out seeds 111-140 are the intended evaluation set. Anywhere
# else it is an error, so it cannot become a blanket escape.
RELEASE_EVALUATION_KIND = "release-evaluation"
RELEASE_EVALUATION_PREFIXES = (
    "configs/benchmarks/releases/",
    "configs/benchmarks/paper_experiment_matrix_",
)
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
PYTHON_INT_LITERAL = (
    r"[+-]?\s*(?:"
    r"0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*|"
    r"0[bB][01](?:_?[01])*|"
    r"0[oO][0-7](?:_?[0-7])*|"
    r"(?:0|[1-9](?:_?\d)*)"
    r")"
)
RANGE = re.compile(
    rf"\brange\s*\(\s*(?P<first>{PYTHON_INT_LITERAL})\s*"
    rf"(?:,\s*(?P<second>{PYTHON_INT_LITERAL})\s*"
    rf"(?:,\s*(?P<third>{PYTHON_INT_LITERAL})\s*)?)?,?\s*\)"
)
EPISODE_RANGE_LOOP = re.compile(
    r"\bfor\s+[A-Za-z_]\w*\s+in\b(?P<body>.{0,1200}?)\brun_episode\s*\(",
    re.DOTALL,
)
EPISODE_RANGE_MAP = re.compile(r"\bmap\s*\(\s*run_episode\b", re.DOTALL)
EPISODE_CALL = re.compile(
    r"\b(?:run_episode|run_map_episode|execute_episode)\s*\(|\.\s*(?:step|reset)\s*\(",
)
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


def _release_evaluation_path(path: str) -> bool:
    return path.startswith(RELEASE_EVALUATION_PREFIXES)


def _marker_allowed(path: str, kind: str) -> bool:
    return kind != RELEASE_EVALUATION_KIND or _release_evaluation_path(path)


def _line_marker_exempts(path: str, content: str) -> bool:
    return any(_marker_allowed(path, match.group("kind")) for match in MARKER.finditer(content))


def _misplaced_release_marker(path: str, content: str) -> bool:
    return not _release_evaluation_path(path) and any(
        match.group("kind") == RELEASE_EVALUATION_KIND for match in MARKER.finditer(content)
    )


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
        next_lines = "\n".join(after[:40])
        return (
            bool(re.search(r"def test_\w*reject\w*\(", next_lines))
            and "pytest.raises" in next_lines
            and not EPISODE_CALL.search(next_lines)
        )
    if text.lstrip().startswith("payload["):
        declarations = [line for line in before if line.startswith("def test_")]
        next_lines = "\n".join(after[:40])
        return bool(
            declarations
            and re.search(r"def test_\w*reject\w*\(", declarations[-1])
            and "pytest.raises" in next_lines
            and not EPISODE_CALL.search(next_lines)
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
    """Recognize literal Python ranges whose values include a held-out seed."""
    for match in RANGE.finditer(text):
        first = _parse_python_int_literal(match.group("first"))
        second = match.group("second")
        if first is None:
            continue
        if second is None:
            start, stop, step = 0, first, 1
        else:
            stop = _parse_python_int_literal(second)
            if stop is None:
                continue
            third = match.group("third")
            step = _parse_python_int_literal(third) if third is not None else 1
            if step is None:
                continue
            start = first
        if step == 0:
            continue
        if step > 0:
            candidate = start + max(0, (111 - start + step - 1) // step) * step
            if 111 <= candidate <= 140 and candidate < stop:
                return True
        else:
            magnitude = -step
            candidate = start - max(0, (start - 140 + magnitude - 1) // magnitude) * magnitude
            if 111 <= candidate <= 140 and candidate > stop:
                return True
    return False


def _parse_python_int_literal(value: str) -> int | None:
    """Parse the integer literal spellings accepted by Python's ``range`` call."""
    normalized = value.replace("_", "").replace(" ", "")
    sign = ""
    if normalized[:1] in {"+", "-"}:
        sign, normalized = normalized[0], normalized[1:]
    try:
        base = 0 if normalized.lower().startswith(("0x", "0o", "0b")) else 10
        return int(f"{sign}{normalized}", base)
    except ValueError:
        return None


def _episode_range_context(text: str, before: list[str], after: list[str]) -> bool:
    """Recognize an overlapping literal range used by a nearby episode call."""
    window = "\n".join(before[-20:] + [text] + after[:20])
    if not _range_overlaps_holdout(window):
        return False
    if EPISODE_RANGE_MAP.search(window):
        return True
    return any(
        _range_overlaps_holdout(match.group("body"))
        for match in EPISODE_RANGE_LOOP.finditer(window)
    )


def _marked_block_lines(lines: list[str], path: str = "") -> set[int]:
    """Only complete, matching marker pairs exempt their enclosed lines.

    A release-evaluation pair exempts lines only inside release manifests.
    """
    marked: set[int] = set()
    opened: tuple[str, int] | None = None
    for number, line in enumerate(lines, 1):
        marker = BLOCK_MARKER.fullmatch(line)
        if marker is None:
            continue
        kind, boundary = marker.groups()
        if not _marker_allowed(path, kind):
            continue
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
        or _episode_range_context(text, before, after)
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
            marked_block_lines = _marked_block_lines(file_lines, path)
            before = []
        elif row.startswith("@@ "):
            match = HUNK.match(row)
            if match:
                line_number = int(match.group(1))
                if file_lines:
                    before = file_lines[: line_number - 1]
        elif row.startswith("+") and not row.startswith("+++ "):
            content = row[1:]
            if _eligible(path) and _misplaced_release_marker(path, content):
                findings.append(
                    Finding(
                        path,
                        line_number,
                        f"{content.strip()}  [release-evaluation marker outside release manifests]",
                    )
                )
            elif (
                _eligible(path)
                and path not in RELEASE_CONFIGS
                and line_number not in marked_block_lines
                and not _line_marker_exempts(path, content)
                and (
                    SEED.search(RANGE.sub("", content))
                    or _range_overlaps_holdout(content)
                    or _episode_range_context(content, before, file_lines[line_number:])
                )
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
            "Use dev seeds 1001-1030 for tests/tuning; calibration 101-102 and diagnostics 103-105 are reserved for their registered roles. Mark setup-only or synthetic fixtures explicitly "
            "(# seed-holdout: setup-only|synthetic-fixture, line or begin/end block). "
            "Release evaluation manifests under configs/benchmarks/releases/ and "
            "configs/benchmarks/paper_experiment_matrix_* may mark their evaluation seed "
            "lists in a seed-holdout release-evaluation begin/end block (see the module "
            "docstring); that marker kind is an error in any other path.",
            file=sys.stderr,
        )
        return 1
    print("Seed holdout diff check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
