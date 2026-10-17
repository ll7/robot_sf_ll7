"""Reject newly introduced evaluation seeds in executable pilot contexts.

This is a diff gate, not an audit of historical releases or archived evidence.
See issue #9668 for the evaluation split and anchor barrier.

Accepted syntactic limits: non-literal or aliased seed generation (including
dynamic ``range`` bounds), dynamically named environment keys, and non-literal
environment seed values need exact-head review. Literal environment assignments
and quoted numeric seed-list continuations are checked.
Literal Python ``range`` loops use complete-file AST context when available.
Positional episode seeds require a resolvable local or directly imported source
signature with a parameter named ``seed``; unresolved callees use keywords only.
Bounded diff-only fragments, dynamic rebinding, re-exports and unknown syntax
remain exact-head review obligations.

Markers: ``# seed-holdout: setup-only`` and ``synthetic-fixture`` (line or
``begin``/``end`` block) exempt fixtures anywhere. ``release-evaluation`` (line
or block) exempts the evaluation seed list of a release manifest and is valid
only under ``configs/benchmarks/releases/`` and
``configs/benchmarks/paper_experiment_matrix_*``; elsewhere it is reported as an
error. Unpaired or mismatched blocks exempt nothing.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import textwrap
import tokenize
from dataclasses import dataclass
from pathlib import Path

import yaml

from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS

RELEASE_CONFIGS = frozenset(
    {
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml",
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml",
    }
)
# Explicit consumers of release identities. New executable consumers require review.
SEALED_REFERENCE_ALLOWLIST = frozenset(
    [
        "robot_sf/benchmark/seed_bands.py",
        "robot_sf/benchmark/release_protocol.py",
        "robot_sf/benchmark/spawn_preflight.py",
        "robot_sf/benchmark/release_acceptance.py",
        "robot_sf/benchmark/release_candidate.py",
        "scripts/validation/check_seed_holdout_diff.py",
        "scripts/validation/check_issue_9748_dev_split.py",
        "scripts/benchmark/run_issue_9748_v4_tuning.py",
        "configs/benchmarks/seed_sets_0_0_8.yaml",
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml",
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml",
        # #10112 successor retains the sealed guard; only its static source inventory is allowed.
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v2.yaml",
        "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml",
        "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.yaml",
        "configs/benchmarks/releases/three_width_doorway_release_0_0_8_v1.template.yaml",
        "tests/analysis/test_pinned_successor_lineage.py",
        "tests/analysis/test_compare_issue_9431_release.py",
        "tests/validation/test_check_seed_holdout_diff.py",
        "tests/benchmark/test_newseeds.py",
        "tests/benchmark/test_release_campaign_authority.py",
        "tests/benchmark/test_release_candidate.py",
        "tests/benchmark/test_release_resolved_identity.py",
        "tests/benchmark/test_sealed_source_pins.py",
        "tests/benchmark/test_sealed_runtime_sources.py",
        "tests/benchmark/test_s30_h600_runtime_smoke_contract.py",
        # Reviewed guard policy and refusal witnesses (#10053; 2026-10-01 ruling).
    ]
)
SEALED_REFERENCE = re.compile(
    r"\b(?:release_eval_0_0_8|seed_sets_0_0_8\.yaml|EVAL_SEEDS_0_0_8|"
    r"EVAL_SEED_SET_0_0_8|HELD_OUT_SEEDS|RETIRED_EVAL_SEEDS_0_0_7|paper_eval_s30)\b"
)

MARKER_KINDS = "setup-only|synthetic-fixture|release-evaluation"
MARKER = re.compile(rf"#\s*seed-holdout:\s*(?P<kind>{MARKER_KINDS})\b")
BLOCK_MARKER = re.compile(rf"\s*#\s*seed-holdout:\s*({MARKER_KINDS})\s+(begin|end)\s*")
# The release-evaluation marker is valid only in release evaluation manifests,
# where sealed evaluation seeds are the intended release set. Anywhere
# else it is an error, so it cannot become a blanket escape.
RELEASE_EVALUATION_KIND = "release-evaluation"
RELEASE_EVALUATION_PREFIXES = (
    "configs/benchmarks/releases/",
    "configs/benchmarks/paper_experiment_matrix_",
)
SEALED_SEED_PATTERN = "(?:" + "|".join(map(str, sorted(HELD_OUT_SEEDS))) + ")"
SEED = re.compile(rf"(?<![\w.]){SEALED_SEED_PATTERN}(?![\w.])")
SEED_FIELD = re.compile(
    r"(?i)(?:^|[\s,.({])['\"]?(?:seed|seeds|seed_list|seed_set|resolved_seeds|"
    r"eval_seeds|evaluation_seeds|pilot_seeds|diagnostic_seeds|base_seed|"
    r"master_seed|episode_seed|scenario_seed|simulator_seed|simulation_seed|"
    r"environment_seed|env_seed|world_seed|run_seed|rollout_seed|task_seed|"
    r"map_seed|spawn_seed|pedestrian_seed|desired_speed_seed|route_spawn_seed|archetype_seed|response_law_seed)['\"]?\s*[:=]"
)
SCENARIO_SEEDS = re.compile(r"(?i)\bscenario\s*\[\s*['\"]seeds?['\"]\s*\]\s*=")
ENVIRONMENT_SEEDS = re.compile(
    r"(?i)\bos\.environ\s*\[\s*['\"](?:[A-Z_][A-Z0-9_]*_)?SEEDS?['\"]\s*\]\s*="
)
SEED_LIST_VALUE = r"(?:\d+|['\"]\d+['\"])"
SEED_LIST_LINE = re.compile(
    rf"^\s*(?:-\s*)?\[?\s*{SEED_LIST_VALUE}(?:\s*,\s*{SEED_LIST_VALUE})*\s*,?\s*\]?,?\s*(?:#.*)?$"
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
        ("robot_sf/", "configs/", "tests/", "scripts/", "docs/plan/", "docs/context/evidence/")
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
        if any(seed in range(start, stop, step) for seed in HELD_OUT_SEEDS):
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


EPISODE_FUNCTIONS = frozenset({"run_episode", "run_map_episode", "execute_episode"})


def _call_name(node: ast.AST) -> str:
    """Return a dotted static callee name, or an empty unresolved name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


class _SeedSignatures:
    """Resolve local/imported function signatures from source, never imports."""

    def __init__(self, tree: ast.AST, root: Path):
        self.root = root
        self.parents = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        self.modules: dict[str, ast.Module | None] = {}
        self.scopes: dict[ast.AST, dict[str, list[ast.AST]]] = {}
        for scope in ast.walk(tree):
            if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                self.scopes[scope] = self._bindings(scope)

    @staticmethod
    def _bindings(
        scope: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, list[ast.AST]]:
        """Keep lexical bindings separate, treating ambiguous rebinding as unknown."""
        bindings: dict[str, list[ast.AST]] = {}

        def collect(node: ast.AST) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.setdefault(node.name, []).append(node)
                return
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    bindings.setdefault(name, []).append(node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bindings.setdefault(node.id, []).append(node)
            for child in ast.iter_child_nodes(node):
                collect(child)

        for statement in scope.body:
            collect(statement)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs:
                bindings.setdefault(arg.arg, []).append(arg)
        return bindings

    def _imported(self, module: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        if module not in self.modules:
            source = self.root.joinpath(*module.split(".")).with_suffix(".py")
            try:
                self.modules[module] = ast.parse(source.read_text())
            except (OSError, SyntaxError, UnicodeError):
                self.modules[module] = None
        tree = self.modules[module]
        if tree is None:
            return None
        matches = [node for node in tree.body if getattr(node, "name", None) == name]
        if len(matches) == 1 and isinstance(matches[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return matches[0]
        return None

    def seed_position(self, call: ast.Call, callee: ast.AST | None = None) -> int | None:
        """Only an unambiguous callee's parameter named seed can bind positionally."""
        name = _call_name(callee or call.func)
        parts = name.split(".")
        parent = self.parents.get(call)
        binding = None
        while parent is not None:
            candidates = self.scopes.get(parent, {}).get(parts[0], [])
            if candidates:
                if len(candidates) != 1:
                    return None
                binding = candidates[0]
                break
            parent = self.parents.get(parent)
        function = None
        if isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(parts) == 1:
            function = binding
        elif isinstance(binding, ast.ImportFrom) and binding.module and not binding.level:
            alias = next(a for a in binding.names if (a.asname or a.name) == parts[0])
            if len(parts) == 1:
                function = self._imported(binding.module, alias.name)
        elif isinstance(binding, ast.Import):
            alias = next(a for a in binding.names if (a.asname or a.name.split(".")[0]) == parts[0])
            module = alias.name if alias.asname else ".".join(parts[:-1])
            function = self._imported(module, parts[-1])
        if function is None or function.name not in EPISODE_FUNCTIONS:
            return None
        parameters = function.args.posonlyargs + function.args.args
        return next((i for i, arg in enumerate(parameters) if arg.arg == "seed"), None)


def _loop_seed_calls(loop: ast.For, signatures: _SeedSignatures) -> list[ast.Call]:
    """Bind calls inside this loop, excluding nested scopes and shadowed loops."""
    if not isinstance(loop.target, ast.Name):
        return []
    variable = loop.target.id
    calls: list[ast.Call] = []

    def is_variable(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == variable

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.For) and is_variable(node.target):
            return
        if isinstance(node, ast.Call):
            name = _call_name(node.func).split(".")[-1]
            position = signatures.seed_position(node)
            episode = name in EPISODE_FUNCTIONS or position is not None
            reset = isinstance(node.func, ast.Attribute) and name == "reset"
            positional = (
                position is not None
                and not any(isinstance(arg, ast.Starred) for arg in node.args[: position + 1])
                and len(node.args) > position
                and is_variable(node.args[position])
            )
            keyword = any(k.arg == "seed" and is_variable(k.value) for k in node.keywords)
            if (episode and positional) or ((episode or reset) and keyword):
                calls.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in loop.body:
        visit(statement)
    return calls


def _episode_seed_lines(source: str, root: Path) -> set[int]:
    """Report loop headers and their consumers using the complete AST context."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    signatures = _SeedSignatures(tree, root)
    flagged: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and _range_overlaps_holdout(
            ast.get_source_segment(source, node.iter) or ""
        ):
            calls = _loop_seed_calls(node, signatures)
            if calls:
                flagged.update(range(node.lineno, (node.iter.end_lineno or node.lineno) + 1))
                for call in calls:
                    flagged.update(range(call.lineno, (call.end_lineno or call.lineno) + 1))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "map"
            and len(node.args) == 2
            and signatures.seed_position(node, node.args[0]) == 0
            and _range_overlaps_holdout(ast.get_source_segment(source, node.args[1]) or "")
        ):
            flagged.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return flagged


def _episode_range_context(text: str, before: list[str], after: list[str], root: Path) -> bool:
    """Parse bounded diff-only loops, dropping partial preceding suites."""
    lines = before[-20:] + [text] + after[:20]
    if not _range_overlaps_holdout("\n".join(lines)):
        return False
    # Prefer the whole fragment so local function signatures remain available.
    starts = [0] + [i for i, line in enumerate(lines) if line.lstrip().startswith("for ")]
    for start in starts:
        fragment = textwrap.dedent("\n".join(lines[start:]))
        # Closing an unfinished call preserves its original indentation context.
        for source in (
            fragment,
            fragment + "\n" + " " * (len(lines[start]) - len(lines[start].lstrip())) + ")",
        ):
            if len(before[-20:]) + 1 - start in _episode_seed_lines(source, root):
                return True
    return False


def _literal_environment_seed_lines(lines: list[str]) -> set[int]:
    """Inspect literal Python environment assignments without executing source."""
    try:
        tree = ast.parse(textwrap.dedent("\n".join(lines)))
    except SyntaxError:
        return set()
    flagged: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        if not SEED.search(value.value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            environment = target.value
            key = target.slice
            if (
                isinstance(environment, ast.Attribute)
                and isinstance(environment.value, ast.Name)
                and environment.value.id == "os"
                and environment.attr == "environ"
                and isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and re.fullmatch(r"(?:[A-Z_][A-Z0-9_]*_)?SEEDS?", key.value, re.I)
            ):
                flagged.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return flagged


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


def _environment_seed_continuation(text: str, before: list[str], after: list[str]) -> bool:
    """Check a bounded literal assignment when the full Python file is absent."""
    for offset, previous in enumerate(reversed(before[-12:]), 1):
        if ENVIRONMENT_SEEDS.search(previous):
            fragment = before[-offset:] + [text] + after[:12]
            if _literal_environment_seed_lines(fragment):
                return True
            # A diff-only window can end before the closing parenthesis.
            return bool(
                _literal_environment_seed_lines(
                    before[-offset:] + [text, " " * (len(previous) - len(previous.lstrip())) + ")"]
                )
            )
        if previous.strip() and not previous.lstrip().startswith(("#", '"', "'", "(")):
            break
    return False


def _seed_context(path: str, text: str, before: list[str], after: list[str]) -> bool:
    if CLI_SEED.search(text):
        return True
    # Every value in a seed inventory is a seed, irrespective of the set name
    # or whether YAML uses a flow list or a block list.
    if path.startswith("configs/") and re.fullmatch(r"seed_(?:set|list).*\.ya?ml", Path(path).name):
        return True
    if path.startswith("docs/") or _non_episode_seed(path, text, before, after):
        return False
    if (
        SEED_FIELD.search(text)
        or SCENARIO_SEEDS.search(text)
        or ENVIRONMENT_SEEDS.search(text)
        or SEED_CONSTANT.search(text)
        or SEED_PARAM.search(text)
        or _yaml_seed_range_bound(path, text, before)
        or _seed_range_context(text, before)
        or (path.endswith(".py") and _environment_seed_continuation(text, before, after))
    ):
        return True
    # YAML block lists and multiline pytest parametrizations put the value on
    # a separate line from the seed key. Use the closest enclosing declaration.
    if SEED_LIST_LINE.fullmatch(text):
        if any("parametrize" in line for line in before[-12:]) and any(
            re.search(r"['\"]seed['\"]", line) for line in before[-12:]
        ):
            return True
        for previous in reversed(before):
            if (
                SEED_FIELD.search(previous)
                or SEED_PARAM.search(previous)
                or ENVIRONMENT_SEEDS.search(previous)
            ):
                return True
            if previous.strip() and not (
                SEED_LIST_LINE.fullmatch(previous) or previous.lstrip().startswith("#")
            ):
                break
    return False


def _normalize_integer_spellings(text: str) -> str:
    """Normalize whole integer tokens, leaving floats and identifiers alone."""
    pattern = rf"(?<![\w.])(?:{PYTHON_INT_LITERAL})(?![\w.])"
    return re.sub(pattern, lambda m: str(_parse_python_int_literal(m[0])), text)


def _yaml_bounds_overlap(path: str, text: str, before: list[str], after: list[str]) -> bool:
    """Join bounds inside one YAML seed mapping, including unchanged diff context."""
    if not path.startswith("configs/") or not path.endswith((".yaml", ".yml")):
        return False
    if SEED_FIELD.search(text) and "{" in text:
        block = text
    elif _yaml_seed_range_bound(path, text, before):
        indentation = len(text) - len(text.lstrip())
        siblings = [text]
        for lines in (reversed(before), iter(after)):
            for line in lines:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if len(line) - len(line.lstrip()) < indentation:
                    break
                if len(line) - len(line.lstrip()) == indentation:
                    siblings.append(line)
        block = "\n".join(siblings)
    else:
        return False
    bounds = dict(re.findall(rf"\b(min|max|low|high)\s*:\s*({PYTHON_INT_LITERAL})(?![\w.])", block))
    lower = bounds.get("min", bounds.get("low"))
    upper = bounds.get("max", bounds.get("high"))
    if lower is None or upper is None:
        return False
    low, high = _parse_python_int_literal(lower), _parse_python_int_literal(upper)
    return (
        low is not None and high is not None and any(low <= seed <= high for seed in HELD_OUT_SEEDS)
    )


RUNTIME_POLICY_LINES = {
    "robot_sf/benchmark/runtime_seed_guard.py": frozenset(
        [
            "if value not in seed_bands.HELD_OUT_SEEDS:",
            "if value in seed_bands.EVAL_SEEDS_0_0_8 and value in identity.resolved_seeds:",
        ]
    ),
    "tests/test_runtime_seed_guard.py": frozenset(
        [
            'monkeypatch.setattr(seed_bands, "HELD_OUT_SEEDS", frozenset({SENTINEL}))',
            "seed_bands.HELD_OUT_SEEDS = frozenset({1030})",
            # These reviewed policy doubles use development seeds and never execute episodes.
            'monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (SENTINEL,))',
            'monkeypatch.setattr(protocol, "EVAL_SEEDS_0_0_8", (SENTINEL,))',
            'monkeypatch.setattr(seed_bands, "EVAL_SEEDS_0_0_8", (1001,))',
        ]
    ),
    "tests/support/seedguard_boundaries.py": frozenset(
        [
            "FALLBACK_HELD_OUT_SEEDS = frozenset(range(111, 141)) | frozenset(",
            "HELD_OUT_SEEDS = FALLBACK_HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",
            "global HELD_OUT_SEEDS, _POLICY_RESOLVED  # seed-holdout: setup-only (guard policy metadata)",
            "HELD_OUT_SEEDS as canonical,  # seed-holdout: setup-only (guard policy metadata)",
            "HELD_OUT_SEEDS = frozenset(",
            "return HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",
            "if value in FALLBACK_HELD_OUT_SEEDS and module.__name__ not in _LEGACY_RESTORE_STATES:",
            "if _LEGACY_SEEDS.get(name) in FALLBACK_HELD_OUT_SEEDS:",
            "from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",
        ]
    ),
    "tests/test_heldout_seed_guard.py": frozenset(
        [
            '"guard.HELD_OUT_SEEDS = frozenset({1030})\\n"',
            '"guard.FALLBACK_HELD_OUT_SEEDS = frozenset({1030})\\n"',
            'monkeypatch.setattr(guard, "HELD_OUT_SEEDS", guard.resolve_held_out_seeds() | {SENTINEL})',
            'guard, "FALLBACK_HELD_OUT_SEEDS", guard.FALLBACK_HELD_OUT_SEEDS | {SENTINEL}',
            "guard.HELD_OUT_SEEDS = frozenset({SENTINEL})",
            "guard.FALLBACK_HELD_OUT_SEEDS = frozenset({SENTINEL})",
            "assert seedguard_boundaries.resolve_held_out_seeds() == frozenset(policy.HELD_OUT_SEEDS)",
            "assert seedguard_boundaries.HELD_OUT_SEEDS == frozenset(policy.HELD_OUT_SEEDS)",
            "from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",
        ]
    ),
    "tests/validation/test_seed_hardening.py": frozenset(
        [
            'text = "for seed in EVAL_SEEDS_0_0_8:\\n    env.reset(seed=seed)\\n"',
            '"from robot_sf.benchmark.seed_bands import HELD_OUT_SEEDS  # seed-holdout: setup-only (guard policy metadata)",',
            "assert all(str(seed) not in source for seed in seed_bands.EVAL_SEEDS_0_0_8)",
        ]
    ),
}


def _unallowlisted_sealed_reference(path: str, content: str) -> bool:
    """Named release identities need explicit review outside their static consumers."""
    return bool(
        _eligible(path)
        and SEALED_REFERENCE.search(content)
        and path not in SEALED_REFERENCE_ALLOWLIST
        and content.strip() not in RUNTIME_POLICY_LINES.get(path, ())
        and not path.startswith("docs/")
    )


def _yaml_alias_holdout(  # noqa: C901 - traverse YAML merge/alias graphs with cycle guards
    path: str, number: int, content: str, lines: list[str]
) -> bool:
    """Inspect the resolved value of an added YAML merge or seed alias."""
    if not path.endswith((".yaml", ".yml")) or "*" not in content:
        return False
    try:
        root = yaml.compose("\n".join(lines), Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return bool("<<:" in content or SEED_FIELD.search(content))

    def contains_seed(node, seen):
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if SEED_FIELD.search(key.value + ":"):
                    resolved = yaml.safe_load(yaml.serialize(value))
                    values = resolved if isinstance(resolved, list) else [resolved]
                    if any(type(seed) is int and seed in HELD_OUT_SEEDS for seed in values):
                        return True
                if contains_seed(value, seen):
                    return True
        elif isinstance(node, yaml.SequenceNode):
            return any(contains_seed(value, seen) for value in node.value)
        return False

    def visit(node, seen):
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if key.start_mark.line == number - 1:
                    if contains_seed(value, set()):
                        return True
                    if SEED_FIELD.search(key.value + ":"):
                        resolved = yaml.safe_load(yaml.serialize(value))
                        values = resolved if isinstance(resolved, list) else [resolved]
                        if any(type(seed) is int and seed in HELD_OUT_SEEDS for seed in values):
                            return True
                if visit(value, seen):
                    return True
        elif isinstance(node, yaml.SequenceNode):
            return any(visit(value, seen) for value in node.value)
        return False

    return visit(root, set()) if root is not None else False


def _diff_file_context(root: Path, path: str) -> tuple[list[str], set[int]]:
    """Read full context so unchanged seed bounds participate in the gate."""
    file_path = root / path
    lines = file_path.read_text(errors="replace").splitlines() if file_path.is_file() else []
    return lines, _marked_block_lines(lines, path)


def _reference_finding(path: str, number: int, content: str, lines: list[str]) -> Finding | None:
    """Report misplaced markers and unreviewed named release-seed consumers."""
    if _eligible(path) and _misplaced_release_marker(path, content):
        return Finding(
            path,
            number,
            f"{content.strip()}  [release-evaluation marker outside release manifests]",
        )
    if _unallowlisted_sealed_reference(path, content) or _yaml_alias_holdout(
        path, number, content, lines
    ):
        return Finding(path, number, content.strip())
    return None


def check_diff(diff: str, root: Path) -> list[Finding]:
    """Return violations on added lines; root supplies complete file context."""
    findings: list[Finding] = []
    path = ""
    line_number = 0
    before: list[str] = []
    marked_block_lines: set[int] = set()
    file_lines: list[str] = []
    environment_seed_lines: set[int] = set()
    episode_seed_lines: set[int] = set()
    for row in diff.splitlines():
        if row.startswith("+++ b/"):
            path = row[6:]
            file_lines, marked_block_lines = _diff_file_context(root, path)
            environment_seed_lines = (
                _literal_environment_seed_lines(file_lines) if path.endswith(".py") else set()
            )
            episode_seed_lines = (
                _episode_seed_lines("\n".join(file_lines), root) if path.endswith(".py") else set()
            )
            before = []
        elif row.startswith("@@ "):
            match = HUNK.match(row)
            if match:
                line_number = int(match.group(1))
                if file_lines:
                    before = file_lines[: line_number - 1]
        elif row.startswith("+") and not row.startswith("+++ "):
            content = row[1:]
            episode_seed = line_number in episode_seed_lines or (
                not file_lines and _episode_range_context(content, before, [], root)
            )
            reference = _reference_finding(path, line_number, content, file_lines)
            if reference is not None:
                findings.append(reference)
            elif (
                _eligible(path)
                and path not in RELEASE_CONFIGS
                and line_number not in marked_block_lines
                and not _line_marker_exempts(path, content)
                and (
                    line_number in environment_seed_lines
                    or episode_seed
                    or SEED.search(_normalize_integer_spellings(RANGE.sub("", content)))
                    or _range_overlaps_holdout(content)
                    or _yaml_bounds_overlap(path, content, before, file_lines[line_number:])
                )
                and (
                    line_number in environment_seed_lines
                    or episode_seed
                    or _seed_context(
                        path,
                        _normalize_integer_spellings(content),
                        before,
                        file_lines[line_number:],
                    )
                )
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
        "robot_sf/",
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
