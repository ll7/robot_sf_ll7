"""Audit active YAML seed fields without executing training or evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

ROOT = Path(__file__).resolve().parents[2]
# Immutable release transports and paper schedules are not development inputs.
RELEASE_TRANSPORTS = {
    "configs/benchmarks/seed_sets_v1.yaml",
    "configs/benchmarks/seed_list_v1.yaml",
}
EXCLUSIONS = ROOT / "tests/fixtures/active_config_seed_exclusions.json"
VERSION = re.compile(r"(?:0[._]0[._][278])(?:\D|$)")
NUMBER = re.compile(r"(?<![\w.])\d+(?![\w.])")


def protected_config(path: Path) -> bool:
    """Keep frozen files, versioned release transports and sealed job inputs intact."""
    name = path.as_posix()
    return (
        name in RELEASE_TRANSPORTS
        or name in json.loads(EXCLUSIONS.read_text())
        or path.name.endswith("_frozen.yaml")
        or "releases" in path.parts
        or bool(VERSION.search(name))
        or "sealed" in name
    )


def restricted(value: int) -> bool:
    """Recognize the whole protected bands, not only the release seed inventory."""
    return 111 <= value <= 140 or 50_036 <= value <= 59_019


def seed_field(label: str, parent_seed: bool, inventory: bool) -> bool:
    """Identify seed values while avoiding sizes, paths and policy metadata."""
    if inventory or (parent_seed and label in {"min", "max", "low", "high", "start", "end"}):
        return True
    return "seed" in label.lower() and not any(
        word in label.lower()
        for word in (
            "count",
            "per_scenario",
            "seeds_per",
            "path",
            "file",
            "manifest",
            "steps",
            "budget",
            "independent",
            "rejected",
            "disjoint",
            "detail",
            "comparability",
            "checksums",
            "execution",
        )
    )


def role_predicate(keys):
    """Enforce numeric training inputs; explicit selection/evaluation fields keep their roles."""
    field = ".".join(keys).lower()
    if any(word in field for word in ("evaluation", "validation", "holdout", "eval_")):
        return lambda value: not 1101 <= value <= 1130
    if any(word in field for word in ("selection", "sampler")):
        return lambda value: not 1031 <= value <= 1060
    return lambda value: not 100000 <= value <= 199999


def scalar_spans(node: ScalarNode, text: str, seed_context: bool, forbidden=restricted):
    """Yield restricted numeric spans from seed scalars or embedded seed arguments."""
    if seed_context and node.tag.endswith(":int"):
        loader = yaml.SafeLoader("")
        try:
            value = loader.construct_object(node)
        finally:
            loader.dispose()
        if forbidden(value):
            yield node.start_mark.index, node.end_mark.index
        return
    raw = text[node.start_mark.index : node.end_mark.index]
    segments = (
        [(0, raw)]
        if seed_context
        else [
            (match.start(1), match.group(1))
            for match in re.finditer(r"--(?:[\w-]*-)?seeds?(?:\s+|=)([\d, \[\]]+)", raw)
        ]
    )
    for offset, segment in segments:
        for match in NUMBER.finditer(segment):
            if forbidden(int(match.group())):
                yield (
                    node.start_mark.index + offset + match.start(),
                    node.start_mark.index + offset + match.end(),
                )


def seed_hits(
    text: str, *, inventory: bool = False, training: bool = False
) -> list[tuple[int, int, int, str]]:
    """Return source spans and field paths; never include seed values in diagnostics."""
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    hits = []

    def walk(node, keys=(), seed_context=False, ancestors=frozenset()):
        if node is None or id(node) in ancestors:
            return
        ancestors = ancestors | {id(node)}
        if isinstance(node, MappingNode):
            for key, value in node.value:
                label = str(key.value)
                if label not in {"held_out_release_seed_range", "provenance"}:
                    walk(
                        value, (*keys, label), seed_field(label, seed_context, inventory), ancestors
                    )
        elif isinstance(node, SequenceNode):
            for value in node.value:
                walk(value, keys, seed_context, ancestors)
        elif isinstance(node, ScalarNode):
            for start, end in scalar_spans(
                node, text, seed_context, role_predicate(keys) if training else restricted
            ):
                hits.append((start, end, node.start_mark.line + 1, ".".join(keys)))

    walk(root, seed_context=inventory)
    return sorted(set(hits))


def config_hits(root: Path) -> list[tuple[str, int, str]]:
    """Scan every non-frozen YAML configuration, including inherited anchor values."""
    hits = []
    for path in sorted((root / "configs").rglob("*")):
        relative = path.relative_to(root)
        if path.suffix not in {".yaml", ".yml"} or protected_config(relative):
            continue
        inventory = "seed" in path.stem and (
            "policy_search" in path.parts or path.name.startswith("predictive_hard_seeds")
        )
        for _, _, line, field in seed_hits(path.read_text(), inventory=inventory):
            hits.append((relative.as_posix(), line, field))
    return hits


def test_active_configs_do_not_contain_restricted_seeds():
    """New or inherited active seed entries cannot silently reuse protected banks."""
    hits = config_hits(ROOT)
    assert not hits, "Restricted seed entries (values masked):\n" + "\n".join(
        f"{path}:{line}: {field}=<seed>" for path, line, field in hits
    )


# seed-holdout: synthetic-fixture begin
@pytest.mark.parametrize(
    "text",
    [
        "seed: 123\n",
        "seed: 0x7b\n",
        "seeds: ['111', 112]\n",
        "x: &s 111\nseed: *s\n",
        "defaults: &d {seed: 111}\nrun: {<<: *d}\n",
        "command: 'run --seeds 111 112'\n",
    ],
)
def test_scan_rejects_seed_literals_aliases_and_commands(text):
    """The scan covers transport forms used by active configuration consumers."""
    assert seed_hits(text)


def test_scan_preserves_unrelated_numbers_and_dev_seeds():
    """Batch sizes and budgets are not seeds, even below a seed-policy mapping."""
    assert not seed_hits(
        "batch_size: 128\nseed_policy: {episodes: 120}\nseeds: [1031, 1060, 1101, 1130]\n"
    )


# seed-holdout: synthetic-fixture end


def test_excluded_artifacts_keep_original_bytes():
    """An exclusion cannot be used to edit an immutable release or historical input."""
    for name, entry in json.loads(EXCLUSIONS.read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entry["sha256"], name


def test_active_configs_do_not_reference_retired_seed_transports():
    """Symbolic seed policies must resolve to the development inventories."""
    for path in (ROOT / "configs").rglob("*.yaml"):
        if not protected_config(path.relative_to(ROOT)):
            assert not re.search(r"seed_(?:sets|list)_v1\.yaml", path.read_text()), str(
                path.relative_to(ROOT)
            )


def test_scan_rejects_the_entire_sealed_band():
    """Interior and boundary values are rejected without exposing them in assertions."""
    for value in (50_000 + 36, 55_000, 59_000 + 19):
        assert seed_hits(f"seed: {value}\n"), "sealed value accepted"


SCRIPT_ROLES = ROOT / "tests/fixtures/active_seed_script_roles.json"


def assignment_seed_expressions(node):
    """Find ordinary assignments and diagnostic tuple seed columns."""
    if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
        return
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
        if name == "TARGET_EPISODES" and isinstance(node.value, (ast.List, ast.Tuple)):
            for row in node.value.elts:
                if isinstance(row, ast.Tuple) and len(row.elts) >= 2:
                    yield row.elts[1]
        elif seed_field(name, False, False):
            yield node.value


def call_seed_expressions(node):
    """Resolve literal get fallbacks and argparse seed defaults."""
    if not isinstance(node, ast.Call) or not node.args:
        return
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        name = first.value.lstrip("-").replace("-", "_")
        if seed_field(name, False, False):
            yield from node.args[1:]
            yield from (kw.value for kw in node.keywords if kw.arg == "default")


def function_seed_expressions(node):
    """Find positional and keyword-only default identities."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    defaults = node.args.defaults
    pairs = list(zip(node.args.args[-len(defaults) :], defaults, strict=True)) if defaults else []
    pairs += list(zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True))
    for arg, default in pairs:
        if default is not None and seed_field(arg.arg, False, False):
            yield default


def seed_expressions(tree):
    """Locate Python seed identities without treating arithmetic offsets as identities."""
    for node in ast.walk(tree):
        yield from assignment_seed_expressions(node)
        yield from call_seed_expressions(node)
        yield from function_seed_expressions(node)
        if isinstance(node, ast.keyword) and node.arg and seed_field(node.arg, False, False):
            yield node.value
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and seed_field(key.value, False, False)
                ):
                    yield value


def literal_seed_values(node):
    """Resolve bounded literal identities, fallback lists and half-open ranges."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        yield node.lineno, node.value
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for item in node.elts:
            yield from literal_seed_values(item)
    elif isinstance(node, ast.Call):
        yield from call_literal_seed_values(node)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            yield from literal_seed_values(value)
    elif isinstance(node, ast.Subscript):
        yield from literal_seed_values(node.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        yield from literal_seed_values(node.left)


def call_literal_seed_values(node):
    """Distinguish range bounds from literal arguments to conversion calls."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
        for arg in node.args:
            yield from literal_seed_values(arg)
        return
    try:
        bounds = [ast.literal_eval(arg) for arg in node.args]
    except (ValueError, TypeError):
        return
    if all(type(bound) is int for bound in bounds):
        for value in range(*bounds):
            yield node.lineno, value


def script_seed_hits(text, *, training=False):
    """Check literal script seed inputs; dynamic generation still needs runtime admission."""
    forbidden = role_predicate(()) if training else restricted
    return sorted(
        {
            line
            for expression in seed_expressions(ast.parse(text))
            for line, value in literal_seed_values(expression)
            if forbidden(value) and "# seed-holdout: setup-only" not in text.splitlines()[line - 1]
        }
    )


def test_training_inputs_use_the_training_pool():
    """Authored training identities cannot collide with selection/evaluation banks."""
    hits = []
    for path in sorted((ROOT / "configs/training").rglob("*.yaml")):
        if not protected_config(path.relative_to(ROOT)):
            for _, _, line, field in seed_hits(
                path.read_text(),
                training=True,
                inventory=path.name.endswith("_seed_manifest.yaml")
                or path.name.startswith("training_predictive_hard_seeds"),
            ):
                hits.append(f"{path.relative_to(ROOT)}:{line}: {field}=<seed>")
    assert not hits, "Training role violations:\n" + "\n".join(hits)


def test_active_script_seed_inputs_obey_their_roles():
    """All audited scripts and independently owned training inputs have static coverage."""
    roles = json.loads(SCRIPT_ROLES.read_text())
    for path in (ROOT / "scripts/training").glob("*.py"):
        if path.name != "train_ppo.py":
            roles.setdefault(path.relative_to(ROOT).as_posix(), "training")
    hits = []
    for name, role in roles.items():
        if protected_config(Path(name)):
            continue
        for line in script_seed_hits((ROOT / name).read_text(), training=role == "training"):
            hits.append(f"{name}:{line}: <seed>")
    assert not hits, "Script seed violations:\n" + "\n".join(hits)


# seed-holdout: synthetic-fixture begin
@pytest.mark.parametrize(
    "source",
    [
        "seed = 1043",
        "seed = 1101",
        "parser.add_argument('--seed', default=1043)",
        "def train(*, seed=1101): pass",
        "cfg.get('seed', 1043)",
    ],
)
def test_script_scan_refuses_selection_or_evaluation_training_defaults(source):
    """Regression probes cover defaults, fallback values and keyword-only inputs."""
    assert script_seed_hits(source, training=True)


def test_script_scan_checks_range_contents_and_skips_seed_offsets():
    """Whole ranges are checked, while valid training seed increments stay valid."""
    assert script_seed_hits("seeds = range(100, 150)")
    assert not script_seed_hits("seed = 100123 + worker_index; batch_size = 128", training=True)
    assert not script_seed_hits("parser.add_argument('--batch-size', default=128)", training=True)


# seed-holdout: synthetic-fixture end


# seed-holdout: synthetic-fixture begin
@pytest.mark.parametrize("text", ["seeds: [1043]", "experiment: {seed: 1101}", "seed: 42"])
def test_role_scan_rejects_training_pool_collisions(text):
    """The role predicate catches defaults the restricted-band-only scan accepted."""
    assert seed_hits(text, training=True)


def test_role_scan_accepts_separate_pools_and_ignores_counts():
    """Each authored role keeps its own pool; seed counts remain budgets."""
    assert not seed_hits(
        "seeds: [100000, 199999]\nevaluation: {seeds: [1101, 1130]}\nselection: {seed: 1031}\nrun_budget: {seeds_per_arm: 5}\n",
        training=True,
    )


# seed-holdout: synthetic-fixture end


def test_live_optuna_launcher_uses_training_reset_seeds():
    """Archival recipe preservation cannot redirect live training to retired inputs."""
    launcher = ROOT / "configs/training/ppo_imitation/optuna_expert_ppo.yaml"
    payload = yaml.safe_load(launcher.read_text())
    base = yaml.safe_load((launcher.parent / payload["base_config"]).read_text())
    assert base["seeds"] == [101043, 100231, 100777, 100992]
    assert all(100000 <= seed <= 199999 for seed in base["seeds"])


PREDICTIVE_HISTORICAL_CONTRACT = (
    "configs/training/predictive/predictive_ego_features_contract_v1.yaml"
)


def declared_config_dependencies(path: Path):
    """Follow actual YAML config references, including relative inherited inputs."""
    payload = yaml.safe_load(path.read_text())

    def walk(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        elif isinstance(value, str) and value.endswith((".yaml", ".yml")):
            target = ROOT / value if value.startswith("configs/") else path.parent / value
            if target.resolve().is_relative_to(ROOT / "configs"):
                assert target.is_file(), f"Missing historical config dependency: {target}"
                yield target.resolve()

    yield from walk(payload)


def test_protected_predictive_contract_dependencies_keep_original_bytes():
    """Preservation includes declared row configs and their transitive seed transports."""
    entries = json.loads(EXCLUSIONS.read_text())
    pending = [ROOT / PREDICTIVE_HISTORICAL_CONTRACT]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        name = path.relative_to(ROOT).as_posix()
        assert name in entries, f"Unpinned historical contract dependency: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entries[name]["sha256"], name
        pending.extend(declared_config_dependencies(path))


def test_retrospective_ablation_manifests_are_preserved():
    """Completed campaign inputs cannot be treated as fresh development proposals."""
    entries = json.loads(EXCLUSIONS.read_text())
    for name in (
        "configs/policy_search/ablation_manifests/issue_2104_component_ablation_pilot.yaml",
        "configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml",
    ):
        assert name in entries, f"Unpinned historical campaign manifest: {name}"
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == entries[name]["sha256"], (
            name
        )


def test_live_one_factor_launcher_uses_development_manifest():
    """The default executable pilot cannot reuse a completed campaign's seed identities."""
    from scripts.tools.run_one_factor_ablation_pilot import DEFAULT_MANIFEST

    assert DEFAULT_MANIFEST.name == "dev_one_factor_hybrid_component_manifest.yaml"
    seeds = yaml.safe_load((ROOT / DEFAULT_MANIFEST).read_text())["scenario_slice"]["seed_policy"][
        "seeds"
    ]
    assert seeds and all(1031 <= seed <= 1060 for seed in seeds), "selection seeds outside pool"
