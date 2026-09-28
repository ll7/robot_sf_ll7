"""Fail-closed checks for the issue #9748 pre-registered development split.

This checks input integrity, not planner quality or benchmark eligibility. It never
runs a simulation, writes a new freeze, or excuses a pre-protocol tuning record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SPLIT = "planner_development_v1"
SEEDS = tuple(range(1001, 1031))
RELEASE_SEEDS = frozenset(range(111, 141))
SPLIT_FILE = Path("configs/benchmarks/planner_development_split_v1.yaml")
FREEZE_FILE = Path("configs/benchmarks/planner_development_freeze_v1.json")
TUNING_DIR = Path("configs/benchmarks/planner_tuning")
SCENARIO_FILE = TUNING_DIR / "development_scenarios_v1.yaml"
EXPECTED = {
    "dev_v1__classic_doorway_medium": {"ped_density": 0.065, "route_spawn_jitter_frac": 0.30},
    "dev_v1__classic_group_crossing_medium": {"ped_density": 0.10},
    "dev_v1__francis2023_perpendicular_traffic": {"ped_density": 0.12},
    "dev_v1__francis2023_crowd_navigation": {"ped_density": 0.10},
}
_FILE_KEYS = frozenset(
    {
        "include", "includes", "extends", "base_config", "algo_config", "config_path",
        "config_file", "config", "base_config_path", "scenario_files", "scenario_matrix",
    }
)
_SEED_KEY = re.compile(
    r"(?:^|_)(?:seeds?|seed_start|seed_end|seed_range|seed_list|seed_min|seed_max)$"
)
_TEXT_SEEDS = re.compile(
    r"(?<![a-zA-Z])(?:[a-zA-Z]+_)?seeds?(?:_range)?[\"']?"
    r"\s*(?:[:=]\s*)?\[?\s*(\d+(?:\s*(?:[-–,:]|\s)\s*\d+)*)",
    re.IGNORECASE,
)


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently dropping a seed declaration."""


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    """Construct a mapping without allowing duplicate keys or merge-key ambiguity."""
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate key in tuning artifact: {key}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_data(path: Path) -> Any:
    """Read structured UTF-8 input; unsupported or malformed input fails closed."""
    if path.suffix.lower() not in {".yaml", ".yml", ".json", ".jsonl"}:
        raise ValueError(f"Unsupported tuning artifact format: {path}")
    text = path.read_text(encoding="utf-8")

    def parse(content: str) -> Any:
        """Parse through the duplicate-checking SafeLoader subclass."""
        loader = _UniqueLoader(content)
        try:
            return loader.get_single_data()
        finally:
            loader.dispose()

    if path.suffix.lower() == ".jsonl":
        return [parse(line) for line in text.splitlines() if line.strip()]
    value = parse(text)
    if not isinstance(value, (dict, list)):
        raise ValueError(f"Expected a structured mapping or list: {path}")
    return value


def sha256(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_sha256(row: Mapping[str, Any]) -> str:
    """Hash the portable raw row with the freeze's documented JSON encoding."""
    content = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def resolve_path(root: Path, owner: Path, reference: str) -> Path:
    """Resolve root-relative or file-relative references without ambiguous fallback."""
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Invalid file reference in {owner}: {reference!r}")
    candidates = {(root / reference).resolve(), (owner.parent / reference).resolve()}
    existing = {path for path in candidates if path.is_file()}
    if len(existing) != 1:
        raise ValueError(f"Missing or ambiguous file reference in {owner}: {reference}")
    path = existing.pop()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Reference escapes repository root: {path}")
    return path


def check_seeds(value: Any, location: str) -> list[int]:
    """Require explicit development seeds, including in integer strings and ranges."""
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a seed at {location}")
    if isinstance(value, int):
        seeds = [value]
    elif isinstance(value, (list, tuple)):
        seeds = [seed for item in value for seed in check_seeds(item, location)]
    elif isinstance(value, str):
        value = value.strip().strip("[]")
        match = re.fullmatch(r"(\d+)\s*[-–:]\s*(\d+)", value)
        if match:
            start, end = map(int, match.groups())
            if start > end or end - start > 10000:
                raise ValueError(f"Invalid seed range at {location}")
            seeds = list(range(start, end + 1))
        elif re.fullmatch(r"\d+(?:[\s,]+\d+)*", value):
            seeds = [int(item) for item in re.split(r"[\s,]+", value)]
        else:
            raise ValueError(f"Unresolved seed expression at {location}: {value!r}")
    else:
        raise ValueError(f"Invalid seeds at {location}: {value!r}")
    if not seeds:
        raise ValueError(f"Empty seed selection at {location}")
    leaked = sorted(set(seeds) & RELEASE_SEEDS)
    if leaked:
        raise ValueError(f"Release seeds forbidden at {location}: {leaked}")
    if not set(seeds) <= set(SEEDS):
        raise ValueError(f"Seeds outside the frozen development split at {location}: {seeds}")
    return seeds


def resolve_seed_policy(policy: Any, root: Path, owner: Path) -> list[int]:
    """Resolve only explicit fixed lists or the selected member of a named seed set."""
    if not isinstance(policy, dict):
        raise ValueError(f"Malformed seed_policy in {owner}")
    mode = policy.get("mode")
    if mode == "fixed-list":
        return check_seeds(policy.get("seeds"), str(owner))
    if mode == "seed-set":
        path = resolve_path(root, owner, policy.get("seed_sets_path"))
        sets = read_data(path)
        name = policy.get("seed_set")
        if not isinstance(name, str) or not isinstance(sets, dict) or name not in sets:
            raise ValueError(f"Unresolved named seed set in {owner}: {name!r}")
        return check_seeds(sets[name], f"{path}:{name}")
    raise ValueError(f"Unsupported tuning seed policy in {owner}: {mode!r}")


def audit_file(path: Path, root: Path, *, ancestors: frozenset[Path] = frozenset()) -> None:
    """Audit nested seeds, command strings and the referenced config/include closure."""
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Cyclic tuning config reference: {path}")
    ancestors = ancestors | {path}
    active: set[int] = set()

    def walk(value: Any, location: str) -> None:
        if isinstance(value, (dict, list)):
            if id(value) in active:
                raise ValueError(f"Cyclic YAML alias at {location}")
            active.add(id(value))
        if isinstance(value, dict):
            if "seed_policy" in value:
                resolve_seed_policy(value["seed_policy"], root, path)
            if "seed_set" in value and "mode" not in value:
                raise ValueError(f"Named seed sets require an explicit seed_policy at {location}")
            for key, child in value.items():
                key = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(key)).lower()
                here = f"{location}.{key}"
                if _SEED_KEY.search(key):
                    # A seed-set policy can carry an unused empty fixed-list field.
                    if not (key == "seeds" and child == [] and value.get("mode") == "seed-set"):
                        check_seeds(child, here)
                if key in _FILE_KEYS and isinstance(child, (str, list)):
                    references = child if isinstance(child, list) else [child]
                    for reference in references:
                        target = resolve_path(root, path, reference)
                        if key == "scenario_matrix" and target != (root / SCENARIO_FILE).resolve():
                            raise ValueError(
                                f"Tuning must use the frozen development scenarios: {here}"
                            )
                        audit_file(target, root, ancestors=ancestors)
                if key in {"scenario_ids", "select_scenarios"}:
                    if (
                        not isinstance(child, list) or not child
                        or not set(child) <= EXPECTED.keys()
                    ):
                        raise ValueError(f"Non-development scenario identity at {here}")
                if (
                    key in {"scenario_overrides", "scenario_overrides_by_name"}
                    and "algo" not in value
                ):
                    raise ValueError(f"Scenario overrides would change the frozen split: {here}")
                walk(child, here)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")
        elif isinstance(value, str):
            if re.search(r"--seed[-_](?:set|sets|policy|file|manifest)\b", value):
                raise ValueError(
                    f"Command seed indirection must use a structured seed_policy: {location}"
                )
            for match in _TEXT_SEEDS.finditer(value):
                check_seeds(match.group(1), location)
        if isinstance(value, (dict, list)):
            active.remove(id(value))

    data = read_data(path)
    # Check comments as well as parsed values; do not silently discard a seed reference.
    text = path.read_text(encoding="utf-8")
    for match in _TEXT_SEEDS.finditer(text):
        check_seeds(match.group(1), str(path))
    walk(data, str(path))


def validate_freeze(root: Path, *, check_maps: bool = True) -> list[dict[str, Any]]:
    """Verify committed definitions and unchanged map bytes against the pre-tuning record."""
    freeze = read_data(root / FREEZE_FILE)
    config = read_data(root / SPLIT_FILE)
    if freeze.get("schema_version") != "planner-development-freeze.v1":
        raise ValueError("Unsupported development freeze schema")
    if (
        config.get("schema_version") != "planner-development-split.v1"
        or config.get("name") != SPLIT
    ):
        raise ValueError("Unexpected development split identity")
    if config.get("paper_facing") is not False or config.get("purpose") != "planner_tuning":
        raise ValueError("Development split cannot be paper-facing")
    if config.get("scenario_matrix") != SCENARIO_FILE.as_posix():
        raise ValueError("Development scenario file changed")
    if config.get("freeze_manifest") != FREEZE_FILE.as_posix():
        raise ValueError("Development freeze reference changed")
    for key, expected_path in (("split_config", SPLIT_FILE), ("scenario_file", SCENARIO_FILE)):
        if freeze.get(key) != expected_path.as_posix():
            raise ValueError(f"Frozen {key} path changed")
        if sha256(root / expected_path) != freeze.get(f"{key}_sha256"):
            raise ValueError(f"Frozen {key} hash mismatch")
    if resolve_seed_policy(config.get("seed_policy"), root, root / SPLIT_FILE) != list(SEEDS):
        raise ValueError("Development split must contain exactly seeds 1001-1030 in order")
    rows = read_data(root / SCENARIO_FILE).get("scenarios")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED):
        raise ValueError("Expected exactly four development scenarios")
    if {row.get("name") for row in rows} != set(EXPECTED):
        raise ValueError("Development scenario identities changed")
    records = freeze.get("variants", [])
    if len(records) != len(rows) or {row.get("name") for row in records} != set(EXPECTED):
        raise ValueError("Freeze must identify each development variant exactly once")
    pinned = {row["name"]: row for row in records}
    for row in rows:
        name = row["name"]
        record = pinned[name]
        metadata = row.get("metadata", {})
        if metadata.get("development_only") is not True or metadata.get("split") != SPLIT:
            raise ValueError(f"Missing development-only identity: {name}")
        if metadata.get("source_scenario") != name.removeprefix("dev_v1__"):
            raise ValueError(f"Incorrect source scenario: {name}")
        if row.get("seeds") != list(SEEDS):
            raise ValueError(f"Incorrect scenario seeds: {name}")
        if any(row.get("simulation_config", {}).get(k) != v for k, v in EXPECTED[name].items()):
            raise ValueError(f"Author-approved parameters changed: {name}")
        if row_sha256(row) != record.get("definition_sha256"):
            raise ValueError(f"Frozen variant hash mismatch: {name}")
        if check_maps:
            path = resolve_path(root, root / SCENARIO_FILE, row.get("map_file"))
            if path.relative_to(root.resolve()).as_posix() != record.get("map_file"):
                raise ValueError(f"Frozen map path mismatch: {name}")
            content = path.read_bytes()
            git_blob = b"blob " + str(len(content)).encode("ascii") + b"\0" + content
            digest = hashlib.sha1(git_blob, usedforsecurity=False).hexdigest()
            if digest != record.get("map_git_blob_sha1"):
                raise ValueError(f"Frozen map bytes changed: {name}")
    return rows


def validate_tuning_config(path: Path, root: Path) -> None:
    """Require explicit scenario and seed choices in a tuning launch configuration."""
    data = read_data(path)
    if not isinstance(data, dict) or "scenario_matrix" not in data or "seed_policy" not in data:
        raise ValueError(
            f"Tuning launch config must declare scenario_matrix and seed_policy: {path}"
        )
    resolve_seed_policy(data["seed_policy"], root, path)
    audit_file(path, root)


def validate_log(path: Path, root: Path) -> None:
    """Validate a JSON/YAML trial ledger or JSONL trial records against the frozen split."""
    data = read_data(path)
    definition_hash = sha256(root / SCENARIO_FILE)
    if path.suffix == ".jsonl":
        trials = data
    else:
        if not isinstance(data, dict) or data.get("schema_version") != "planner-tuning-log.v1":
            raise ValueError(f"Unsupported tuning log schema: {path}")
        if data.get("split") != SPLIT or data.get("scenario_file_sha256") != definition_hash:
            raise ValueError(f"Tuning log is not bound to the frozen split: {path}")
        trials = data.get("trials")
    if not isinstance(trials, list):
        raise ValueError(f"Tuning log must contain a trials list: {path}")
    seen = set()
    for trial in trials:
        required = {"trial_id", "seeds", "scenario_ids", "config_path", "config_sha256", "changes"}
        if not isinstance(trial, dict) or not required <= trial.keys():
            raise ValueError(f"Incomplete tuning trial: {path}")
        if path.suffix == ".jsonl" and (
            trial.get("split") != SPLIT or trial.get("scenario_file_sha256") != definition_hash
        ):
            raise ValueError(f"Unbound JSONL tuning trial: {path}")
        trial_id = trial["trial_id"]
        if not isinstance(trial_id, str) or not trial_id or trial_id in seen:
            raise ValueError(f"Missing or duplicate trial identity: {path}")
        seen.add(trial_id)
        check_seeds(trial["seeds"], f"{path}:{trial_id}")
        ids = trial["scenario_ids"]
        if not isinstance(ids, list) or not ids or not set(ids) <= EXPECTED.keys():
            raise ValueError(f"Non-development scenario in tuning trial: {path}")
        if not isinstance(trial["changes"], str) or not trial["changes"].strip():
            raise ValueError(f"Trial must record what was tried: {path}")
        config = resolve_path(root, path, trial["config_path"])
        if sha256(config) != trial["config_sha256"]:
            raise ValueError(f"Tuning config hash mismatch: {path}:{trial_id}")
    audit_file(path, root)


def check_release_rows(rows: Sequence[Mapping[str, Any]], source: str) -> None:
    """Reject development identities or markers in an expanded release matrix."""
    for row in rows:
        metadata = row.get("metadata", {})
        if (
            row.get("name") in EXPECTED
            or metadata.get("development_only") is True
            or metadata.get("split") == SPLIT
        ):
            raise ValueError(
                f"Development scenario admitted to release matrix {source}: {row.get('name')}"
            )


def check_release_matrices(root: Path, extra: Sequence[Path] = ()) -> None:
    """Use the canonical loader for release includes, selection and override semantics."""
    from robot_sf.training.scenario_loader import load_scenarios

    paths = {root / "configs/scenarios/classic_interactions_francis2023.yaml"}
    paths.update((root / "configs/scenarios").glob("classic_interactions_francis2023*.yaml"))
    for manifest in (root / "configs/benchmarks/releases").glob("*.yaml"):
        # Release manifests are not tuning artifacts; preserve their YAML semantics.
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if isinstance(data, dict) and str(data.get("schema_version", "")).startswith(
            "benchmark-release-manifest."
        ):
            reference = data.get("scenario", {}).get("matrix_path")
            if reference:
                paths.add(resolve_path(root, manifest, reference))
    paths.update((root / path).resolve() for path in extra)
    for path in sorted(paths):
        check_release_rows(load_scenarios(path), str(path))


def main(argv: Sequence[str] | None = None) -> int:
    """Check the registered split and all new tuning artifacts before a tuning run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--release-matrix", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        validate_freeze(root)
        validate_tuning_config(root / SPLIT_FILE, root)
        validate_log(root / TUNING_DIR / "tuning_log_v1.json", root)
        for path in sorted((root / TUNING_DIR).rglob("*")):
            if path.is_file() and path.suffix in {".yaml", ".yml", ".json", ".jsonl"}:
                if "log" in path.stem or path.suffix == ".jsonl":
                    validate_log(path, root)
                else:
                    audit_file(path, root)
        for path in args.config:
            validate_tuning_config(root / path, root)
        for path in args.log:
            validate_log(root / path, root)
        check_release_matrices(root, args.release_matrix)
    except (OSError, ValueError, TypeError, KeyError, ImportError, yaml.YAMLError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: frozen development definitions, tuning seed isolation, and release separation")
    print(
        "Implementation-integrity check only; "
        "not benchmark evidence or approval to tune before review."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
