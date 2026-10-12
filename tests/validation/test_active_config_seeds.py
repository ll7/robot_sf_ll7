"""Audit active YAML seed fields without executing training or evaluation."""

from __future__ import annotations

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
        word in label.lower() for word in ("count", "per_scenario", "path", "file")
    )


def scalar_spans(node: ScalarNode, text: str, seed_context: bool):
    """Yield restricted numeric spans from seed scalars or embedded seed arguments."""
    if seed_context and node.tag.endswith(":int"):
        loader = yaml.SafeLoader("")
        try:
            value = loader.construct_object(node)
        finally:
            loader.dispose()
        if restricted(value):
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
            if restricted(int(match.group())):
                yield (
                    node.start_mark.index + offset + match.start(),
                    node.start_mark.index + offset + match.end(),
                )


def seed_hits(text: str, *, inventory: bool = False) -> list[tuple[int, int, int, str]]:
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
            for start, end in scalar_spans(node, text, seed_context):
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
