"""Validate the issue #9748 development split and tuning-log boundary.

The checker validates the checked-in campaign through the same camera-ready
campaign and scenario loaders used by the benchmark. When a structured tuning
log is supplied, only typed fields in the ``issue_9748.tuning_log.v2`` schema
are inspected for seed and scenario admission. Free-form strings such as a
rationale mentioning the held-out range are intentionally not scanned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml"
DEFAULT_RELEASE_CONFIG = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
DEFAULT_RELEASE_MATRIX = (
    ROOT / "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
)
DEFAULT_RELEASE_SEED_SETS = ROOT / "configs/benchmarks/seed_sets_v1.yaml"
RELEASE_SEED_SET_NAME = "paper_eval_s30"
EXPECTED_DEV_SEEDS = tuple(range(1001, 1031))
RELEASE_SEEDS = frozenset(range(111, 141))
EXPECTED_RELEASE_MATRIX_SHA256 = "03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c"
EXPECTED_RELEASE_CONFIG_SHA256 = "7dc9a2dd9df8585593c9bc8ecc001bed0d2ddff4ebb3803dfb92e8dad8762881"
EXPECTED_RELEASE_SEED_SETS_SHA256 = (
    "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6"
)
EXPECTED_RELEASE_SEED_SCHEDULE_SHA256 = (
    "ecbca1eaa1e3c0615d4d9eec8b3d59ec8432529fd9398482250e6f1c7e0abfe1"
)
EXPECTED_VARIANTS = {
    "issue_9748_dev_classic_doorway_medium": {
        "source_file": "configs/scenarios/archetypes/classic_doorway.yaml",
        "source_id": "classic_doorway_medium",
        "simulation_overrides": {"ped_density": 0.065, "route_spawn_jitter_frac": 0.30},
        "development_variant": "density_and_route_spawn_jitter",
    },
    "issue_9748_dev_classic_group_crossing_medium": {
        "source_file": "configs/scenarios/archetypes/classic_group_crossing.yaml",
        "source_id": "classic_group_crossing_medium",
        "simulation_overrides": {"ped_density": 0.10},
        "development_variant": "density",
    },
    "issue_9748_dev_francis2023_perpendicular_traffic": {
        "source_file": "configs/scenarios/single/francis2023_perpendicular_traffic.yaml",
        "source_id": "francis2023_perpendicular_traffic",
        "simulation_overrides": {"ped_density": 0.12},
        "development_variant": "density",
    },
    "issue_9748_dev_francis2023_crowd_navigation": {
        "source_file": "configs/scenarios/single/francis2023_crowd_navigation.yaml",
        "source_id": "francis2023_crowd_navigation",
        "simulation_overrides": {"ped_density": 0.10},
        "development_variant": "density",
    },
}
EXPECTED_SCENARIO_IDS = frozenset(EXPECTED_VARIANTS)
DEV_METADATA = {
    "development_only": True,
    "evidence_tier": "development_tuning_only",
    "claim_boundary": "no release or paper-facing evidence",
}
EXPECTED_PLANNER_CONFIGS = {
    "hybrid_rule_v4_fast_progress_static_escape_s30_h600_release": (
        ROOT
        / "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml"
    ),
    "hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release": (
        ROOT
        / "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml"
    ),
}
EXPECTED_PLANNER_ALGOS = dict.fromkeys(EXPECTED_PLANNER_CONFIGS, "hybrid_rule_local_planner")
TUNING_LOG_SCHEMA = "issue_9748.tuning_log.v2"


class ValidationError(ValueError):
    """Raised when a development-split admission contract is violated."""


def _repo_path(raw: str | Path, *, relative_to: Path | None = None) -> Path:
    """Resolve a repository-owned path from a config or the repository root."""
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    if relative_to is not None:
        from_config = (relative_to / candidate).resolve()
        if from_config.exists():
            return from_config
    return (ROOT / candidate).resolve()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValidationError(f"could not read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must be a YAML mapping: {path}")
    return payload


def _sha256_file(path: Path, *, label: str) -> str:
    """Return a file digest or fail closed when a bound input is unavailable."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValidationError(f"could not read {label} for checksum: {path}: {exc}") from exc


def _canonical_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value using the repository's stable compact encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _repo_relative_path(path: Path, *, label: str) -> str:
    """Return a stable repository-relative path for a frozen input."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationError(f"{label} must be tracked inside the repository") from exc


def _git_file_bytes(source_commit: str, relative_path: str, *, label: str) -> bytes:
    """Read one tracked input from the claimed immutable source commit."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative_path}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            f"source_commit {source_commit} does not contain frozen {label}: {relative_path}"
        ) from exc


def _validate_tuning_source_commit(
    source_commit: str, *, expected_bindings: Mapping[str, Any]
) -> None:
    """Require the source commit to exist and contain every frozen input byte."""
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            "tuning-log source_commit is not present in this Git repository"
        ) from exc

    file_bindings = (
        (
            "development_campaign_config_path",
            "development_campaign_config_sha256",
            "development campaign config",
        ),
        (
            "development_scenario_matrix_path",
            "development_scenario_matrix_sha256",
            "development scenario matrix",
        ),
        ("release_campaign_config_path", "release_campaign_config_sha256", "release config"),
        ("release_scenario_matrix_path", "release_scenario_matrix_sha256", "release matrix"),
        ("release_seed_set_path", "release_seed_set_file_sha256", "release seed-set file"),
    )
    for path_key, digest_key, label in file_bindings:
        relative_path = expected_bindings.get(path_key)
        expected_digest = expected_bindings.get(digest_key)
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise ValidationError(f"frozen {label} path or checksum is unavailable")
        observed = hashlib.sha256(
            _git_file_bytes(source_commit, relative_path, label=label)
        ).hexdigest()
        if observed != expected_digest:
            raise ValidationError(
                f"source_commit {source_commit} has different frozen {label} bytes"
            )

    candidate_paths = expected_bindings.get("candidate_template_paths")
    candidate_digests = expected_bindings.get("candidate_template_sha256")
    if not isinstance(candidate_paths, Mapping) or not isinstance(candidate_digests, Mapping):
        raise ValidationError("approved candidate template paths or checksums are unavailable")
    if set(candidate_paths) != set(candidate_digests):
        raise ValidationError("approved candidate template path and checksum keys differ")
    for candidate, relative_path in candidate_paths.items():
        expected_digest = candidate_digests[candidate]
        if not isinstance(relative_path, str) or not isinstance(expected_digest, str):
            raise ValidationError(f"approved candidate {candidate!r} path or checksum is invalid")
        observed = hashlib.sha256(
            _git_file_bytes(source_commit, relative_path, label=f"candidate {candidate}")
        ).hexdigest()
        if observed != expected_digest:
            raise ValidationError(
                f"source_commit {source_commit} has different approved candidate bytes: {candidate}"
            )


def _resolve_run_artifact(reference: str, *, label: str) -> Path:
    """Resolve a local immutable artifact file outside the repository checkout."""
    parsed = urlparse(reference)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or not parsed.path:
        raise ValidationError(f"{label} must be a local file:// URI to a durable artifact")
    path = Path(unquote(parsed.path)).resolve()
    if path == ROOT.resolve() or ROOT.resolve() in path.parents:
        raise ValidationError(f"{label} must be outside the repository worktree")
    if not path.is_file():
        raise ValidationError(f"{label} does not identify an existing artifact file: {path}")
    return path


def _load_frozen_release_bindings() -> dict[str, Any]:
    """Verify the accepted 0.0.7 campaign, scenario matrix, and effective seed set."""
    release_config_sha256 = _sha256_file(DEFAULT_RELEASE_CONFIG, label="frozen release config")
    matrix_sha256 = _sha256_file(DEFAULT_RELEASE_MATRIX, label="frozen release scenario matrix")
    seed_sets_sha256 = _sha256_file(DEFAULT_RELEASE_SEED_SETS, label="frozen release seed sets")
    expected_file_hashes = (
        (release_config_sha256, EXPECTED_RELEASE_CONFIG_SHA256, "release config"),
        (matrix_sha256, EXPECTED_RELEASE_MATRIX_SHA256, "release scenario matrix"),
        (seed_sets_sha256, EXPECTED_RELEASE_SEED_SETS_SHA256, "release seed-set file"),
    )
    for observed, expected, label in expected_file_hashes:
        if observed != expected:
            raise ValidationError(
                f"frozen 0.0.7 {label} checksum changed: expected {expected}, observed {observed}"
            )

    release_config = _load_mapping(DEFAULT_RELEASE_CONFIG, label="frozen 0.0.7 campaign config")
    configured_matrix = release_config.get("scenario_matrix")
    if (
        not isinstance(configured_matrix, str)
        or _repo_path(configured_matrix, relative_to=DEFAULT_RELEASE_CONFIG.parent)
        != DEFAULT_RELEASE_MATRIX.resolve()
    ):
        raise ValidationError("frozen 0.0.7 campaign config no longer selects its accepted matrix")

    seed_policy = release_config.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or seed_policy.get("mode") != "seed-set":
        raise ValidationError("frozen 0.0.7 campaign config must use its declared seed set")
    if seed_policy.get("seed_set") != RELEASE_SEED_SET_NAME:
        raise ValidationError("frozen 0.0.7 campaign config changed its seed-set identity")
    configured_seed_sets = seed_policy.get("seed_sets_path")
    if (
        not isinstance(configured_seed_sets, str)
        or _repo_path(configured_seed_sets, relative_to=DEFAULT_RELEASE_CONFIG.parent)
        != DEFAULT_RELEASE_SEED_SETS.resolve()
    ):
        raise ValidationError("frozen 0.0.7 campaign config changed its seed-set source")

    seed_sets = _load_mapping(DEFAULT_RELEASE_SEED_SETS, label="frozen 0.0.7 seed sets")
    raw_seeds = seed_sets.get(RELEASE_SEED_SET_NAME)
    if (
        not isinstance(raw_seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or tuple(raw_seeds) != tuple(sorted(RELEASE_SEEDS))
    ):
        raise ValidationError("frozen 0.0.7 paper_eval_s30 seed schedule changed")
    seed_schedule_sha256 = _canonical_sha256(
        {"seed_set": RELEASE_SEED_SET_NAME, "seeds": raw_seeds}
    )
    if seed_schedule_sha256 != EXPECTED_RELEASE_SEED_SCHEDULE_SHA256:
        raise ValidationError("frozen 0.0.7 paper_eval_s30 seed schedule checksum changed")

    return {
        "release_config_path": str(DEFAULT_RELEASE_CONFIG),
        "release_config_sha256": release_config_sha256,
        "release_scenario_matrix_path": str(DEFAULT_RELEASE_MATRIX),
        "release_scenario_matrix_sha256": matrix_sha256,
        "release_seed_set_path": str(DEFAULT_RELEASE_SEED_SETS),
        "release_seed_set_file_sha256": seed_sets_sha256,
        "release_seed_set_name": RELEASE_SEED_SET_NAME,
        "release_seed_schedule": list(raw_seeds),
        "release_seed_schedule_sha256": seed_schedule_sha256,
    }


def _scenario_id(scenario: Mapping[str, Any], *, label: str) -> str:
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValidationError(f"{label} contains a scenario without a non-empty ID")


def _load_scenario_rows(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    try:
        rows = load_scenarios(path, base_dir=path.parent)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(f"could not load {label} through scenario loader: {exc}") from exc
    if not rows:
        raise ValidationError(f"{label} resolved to no scenarios: {path}")
    return rows


def _require_exact_dev_seed_list(raw: Any, *, label: str) -> None:
    if not isinstance(raw, list) or any(isinstance(seed, bool) for seed in raw):
        raise ValidationError(f"{label} must be a list of integer seeds")
    if any(not isinstance(seed, int) for seed in raw):
        raise ValidationError(f"{label} must contain only integer seeds")
    if tuple(raw) != EXPECTED_DEV_SEEDS:
        raise ValidationError(f"{label} must be the ordered development seed list 1001–1030")


def _source_scenario_row(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    source_path = ROOT / spec["source_file"]
    source_id = spec["source_id"]
    source_rows = _load_scenario_rows(source_path, label=f"source scenario {source_id}")
    matches = [
        row for row in source_rows if _scenario_id(row, label="source scenario") == source_id
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"source scenario {source_id} must resolve exactly once in {source_path}"
        )
    return matches[0]


def _validate_variant_source(
    scenario_id: str,
    actual: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    source_id = spec["source_id"]
    source = _source_scenario_row(spec)
    source_simulation = source.get("simulation_config")
    source_metadata = source.get("metadata")
    if not isinstance(source_simulation, Mapping) or not isinstance(source_metadata, Mapping):
        raise ValidationError(f"source scenario {source_id} lacks simulation or metadata")

    expected = dict(source)
    expected["name"] = scenario_id
    expected["seeds"] = list(EXPECTED_DEV_SEEDS)
    expected["simulation_config"] = {
        **source_simulation,
        **spec["simulation_overrides"],
    }
    expected["metadata"] = {
        **source_metadata,
        **DEV_METADATA,
        "source_scenario": source_id,
        "development_variant": spec["development_variant"],
    }
    differences = sorted(
        key
        for key in expected.keys() | actual.keys()
        if key not in expected or key not in actual or expected[key] != actual[key]
    )
    if differences:
        raise ValidationError(
            f"{scenario_id} differs from its source outside approved development "
            f"overrides: {differences!r}"
        )


def _validate_scenario_parameters(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [_scenario_id(row, label="development scenario matrix") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValidationError("development scenario matrix contains duplicate scenario IDs")
    actual_ids = set(ids)
    if actual_ids != EXPECTED_SCENARIO_IDS:
        missing = sorted(EXPECTED_SCENARIO_IDS - actual_ids)
        unexpected = sorted(actual_ids - EXPECTED_SCENARIO_IDS)
        raise ValidationError(
            f"development scenario IDs drifted; missing={missing!r}, unexpected={unexpected!r}"
        )
    if len(rows) != len(EXPECTED_VARIANTS):
        raise ValidationError("development scenario matrix must contain exactly four variants")

    rows_by_id = dict(zip(ids, rows, strict=True))
    for scenario_id, spec in EXPECTED_VARIANTS.items():
        _validate_variant_source(scenario_id, rows_by_id[scenario_id], spec)


def _validate_planner_spec(
    planner: Any,
    *,
    config_path: Path,
    seen: set[str],
) -> None:
    if not isinstance(planner, Mapping):
        raise ValidationError("each campaign planner must be a mapping")
    key = planner.get("key")
    if not isinstance(key, str) or key not in EXPECTED_PLANNER_CONFIGS:
        raise ValidationError(f"planner key is not an approved v4 candidate: {key!r}")
    if planner.get("algo") != EXPECTED_PLANNER_ALGOS[key]:
        raise ValidationError(
            f"planner {key} must use approved algorithm {EXPECTED_PLANNER_ALGOS[key]!r}"
        )
    if planner.get("planner_group") != "experimental":
        raise ValidationError(f"planner {key} must remain in the experimental planner group")
    if planner.get("benchmark_profile") != "experimental":
        raise ValidationError(f"planner {key} must retain the experimental benchmark profile")
    if key in seen:
        raise ValidationError(f"duplicate planner key: {key}")
    seen.add(key)
    config_raw = planner.get("algo_config")
    if not isinstance(config_raw, str):
        raise ValidationError(f"planner {key} must declare algo_config")
    resolved = _repo_path(config_raw, relative_to=config_path.parent)
    expected = EXPECTED_PLANNER_CONFIGS[key].resolve()
    if resolved != expected:
        raise ValidationError(f"planner {key} must use existing candidate config {expected}")
    if not expected.is_file():
        raise ValidationError(f"approved v4 candidate config is missing: {expected}")
    _validate_planner_scenario_overrides(expected, key=key)


def _validate_planner_scenario_overrides(config_path: Path, *, key: str) -> None:
    planner_config = _load_mapping(config_path, label=f"planner config {key}")
    scenario_overrides = planner_config.get("scenario_overrides", {})
    if not isinstance(scenario_overrides, Mapping):
        raise ValidationError(f"planner config {key} scenario_overrides must be a mapping")
    overlap = EXPECTED_SCENARIO_IDS & scenario_overrides.keys()
    if overlap:
        raise ValidationError(
            f"planner config {key} applies scenario overrides to development IDs: "
            + ", ".join(sorted(overlap))
        )


def _validate_planner_specs(payload: Mapping[str, Any], *, config_path: Path) -> None:
    planners = payload.get("planners")
    if not isinstance(planners, list):
        raise ValidationError("campaign planners must be a list")
    if len(planners) != len(EXPECTED_PLANNER_CONFIGS):
        raise ValidationError(
            "development campaign must contain exactly the two approved v4 candidates"
        )
    seen: set[str] = set()
    for planner in planners:
        _validate_planner_spec(planner, config_path=config_path, seen=seen)
    if seen != set(EXPECTED_PLANNER_CONFIGS):
        raise ValidationError("development campaign planner roster is incomplete")


def _validate_config_metadata(payload: Mapping[str, Any]) -> None:
    if payload.get("issue") != 9748:
        raise ValidationError("development campaign config must declare issue: 9748")
    if (
        payload.get("development_only") is not True
        or payload.get("not_release_evidence") is not True
    ):
        raise ValidationError("development campaign must be explicitly marked not release evidence")
    if payload.get("evidence_class") != "development_tuning_only":
        raise ValidationError("development campaign evidence_class must be development_tuning_only")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise ValidationError("development campaign must declare a non-empty claim_boundary")


def _validate_config_seed_policy(payload: Mapping[str, Any]) -> None:
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or seed_policy.get("mode") != "fixed-list":
        raise ValidationError("development campaign must use a fixed-list seed policy")
    _require_exact_dev_seed_list(seed_policy.get("seeds"), label="seed_policy.seeds")
    typed_seed_values: list[int] = []
    for field, raw in _collect_typed_fields(payload, field_names=_SEED_FIELD_NAMES):
        values = _typed_values(raw)
        if not values:
            raise ValidationError(f"campaign seed field {field!r} has no typed integer seeds")
        typed_seed_values.extend(values)
    if set(typed_seed_values) & RELEASE_SEEDS:
        raise ValidationError("development seed policy admits a release seed")
    if set(typed_seed_values) - set(EXPECTED_DEV_SEEDS):
        raise ValidationError("development campaign admits a seed outside 1001–1030")
    _validate_typed_scenario_ids(payload, label="development campaign config")


def _load_development_rows(
    payload: Mapping[str, Any], *, config_path: Path
) -> tuple[Path, list[Mapping[str, Any]]]:
    scenario_raw = payload.get("scenario_matrix")
    if not isinstance(scenario_raw, str) or not scenario_raw.strip():
        raise ValidationError("development campaign must declare scenario_matrix")
    scenario_path = _repo_path(scenario_raw, relative_to=config_path.parent)
    rows = _load_scenario_rows(scenario_path, label="development scenario matrix")
    _validate_scenario_parameters(rows)
    for row in rows:
        scenario_id = _scenario_id(row, label="development scenario matrix")
        _require_exact_dev_seed_list(row.get("seeds"), label=f"{scenario_id}.seeds")
    return scenario_path, rows


def _validate_release_disjointness(*, release_matrix_path: Path) -> dict[str, Any]:
    release_bindings = _load_frozen_release_bindings()
    release_rows = _load_scenario_rows(release_matrix_path, label="frozen release scenario matrix")
    release_ids = {
        _scenario_id(row, label="frozen release scenario matrix") for row in release_rows
    }
    overlap = EXPECTED_SCENARIO_IDS & release_ids
    if overlap:
        raise ValidationError(
            "development scenario IDs overlap the frozen release matrix: "
            + ", ".join(sorted(overlap))
        )
    schedule: list[dict[str, Any]] = []
    matrix_seeds: set[int] = set()
    for row in release_rows:
        scenario_id = _scenario_id(row, label="frozen release scenario matrix")
        raw_seeds = row.get("seeds")
        if (
            not isinstance(raw_seeds, list)
            or not raw_seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        ):
            raise ValidationError(
                f"frozen release scenario {scenario_id} must declare typed integer seeds"
            )
        matrix_seeds.update(raw_seeds)
        schedule.append({"scenario_id": scenario_id, "seeds": sorted(set(raw_seeds))})
    seed_overlap = sorted(
        (set(EXPECTED_DEV_SEEDS) & RELEASE_SEEDS) | (set(EXPECTED_DEV_SEEDS) & matrix_seeds)
    )
    if seed_overlap:
        raise ValidationError(
            "development seeds overlap the frozen release seed schedule: "
            + ", ".join(str(seed) for seed in seed_overlap)
        )
    return {
        **release_bindings,
        "checked_release_matrix_path": str(release_matrix_path),
        "checked_release_matrix_sha256": _sha256_file(
            release_matrix_path, label="checked release scenario matrix"
        ),
        "release_scenario_seed_schedule_sha256": _canonical_sha256(schedule),
        "release_scenario_overlap": [],
        "release_seed_overlap": [],
    }


def _load_canonical_campaign_rows(
    config_path: Path,
) -> tuple[Any, list[Mapping[str, Any]]]:
    try:
        loaded = load_campaign_config(config_path, repository_root=ROOT)
        canonical_rows = load_scenarios(
            loaded.scenario_matrix_path,
            base_dir=loaded.scenario_matrix_path.parent,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(f"campaign config failed canonical loader validation: {exc}") from exc
    return loaded, canonical_rows


def _validate_campaign_config(config_path: Path, release_matrix_path: Path) -> dict[str, Any]:
    payload = _load_mapping(config_path, label="development campaign config")
    _validate_config_metadata(payload)
    _validate_config_seed_policy(payload)
    scenario_path, rows = _load_development_rows(payload, config_path=config_path)
    release_bindings = _validate_release_disjointness(release_matrix_path=release_matrix_path)
    _validate_planner_specs(payload, config_path=config_path)
    loaded, canonical_rows = _load_canonical_campaign_rows(config_path)
    if tuple(loaded.seed_policy.seeds) != EXPECTED_DEV_SEEDS:
        raise ValidationError("canonical campaign loader did not preserve development seeds")
    resolved_ids = {
        _scenario_id(row, label="canonical campaign scenarios") for row in canonical_rows
    }
    if resolved_ids != EXPECTED_SCENARIO_IDS:
        raise ValidationError("canonical campaign loader resolved an unexpected scenario set")
    candidate_hashes = {
        key: _sha256_file(candidate_path, label=f"approved candidate config {key}")
        for key, candidate_path in EXPECTED_PLANNER_CONFIGS.items()
    }
    tuning_log_bindings = {
        "development_campaign_config_path": _repo_relative_path(
            config_path, label="development campaign config"
        ),
        "development_campaign_config_sha256": _sha256_file(
            config_path, label="development campaign config"
        ),
        "development_scenario_matrix_path": _repo_relative_path(
            scenario_path, label="development scenario matrix"
        ),
        "development_scenario_matrix_sha256": _sha256_file(
            scenario_path, label="development scenario matrix"
        ),
        "release_campaign_config_path": _repo_relative_path(
            DEFAULT_RELEASE_CONFIG, label="release campaign config"
        ),
        "release_campaign_config_sha256": release_bindings["release_config_sha256"],
        "release_scenario_matrix_path": _repo_relative_path(
            DEFAULT_RELEASE_MATRIX, label="release scenario matrix"
        ),
        "release_scenario_matrix_sha256": release_bindings["release_scenario_matrix_sha256"],
        "release_seed_set_path": _repo_relative_path(
            DEFAULT_RELEASE_SEED_SETS, label="release seed-set file"
        ),
        "release_seed_set_file_sha256": release_bindings["release_seed_set_file_sha256"],
        "release_seed_schedule_sha256": release_bindings["release_seed_schedule_sha256"],
        "candidate_template_paths": {
            key: _repo_relative_path(path, label=f"approved candidate {key}")
            for key, path in EXPECTED_PLANNER_CONFIGS.items()
        },
        "candidate_template_sha256": candidate_hashes,
    }
    return {
        "config": str(config_path),
        "scenario_matrix": str(scenario_path),
        "scenario_count": len(rows),
        "scenario_ids": sorted(EXPECTED_SCENARIO_IDS),
        "seed_count": len(EXPECTED_DEV_SEEDS),
        "seed_range": [EXPECTED_DEV_SEEDS[0], EXPECTED_DEV_SEEDS[-1]],
        "planner_keys": sorted(EXPECTED_PLANNER_CONFIGS),
        "development_config_sha256": tuning_log_bindings["development_campaign_config_sha256"],
        "development_scenario_matrix_sha256": tuning_log_bindings[
            "development_scenario_matrix_sha256"
        ],
        "candidate_config_sha256": candidate_hashes,
        "release_bindings": release_bindings,
        "tuning_log_bindings": tuning_log_bindings,
        "release_scenario_overlap": release_bindings["release_scenario_overlap"],
        "release_seed_overlap": release_bindings["release_seed_overlap"],
    }


_SEED_FIELD_NAMES = frozenset(
    {
        "seed",
        "seeds",
        "scenario_seed",
        "scenario_seeds",
        "seed_range",
        "seed_ranges",
        "resolved_seeds",
        "tuning_seeds",
        "release_seed",
        "release_seeds",
        "evaluation_seed",
        "evaluation_seeds",
        "held_out_seed",
        "held_out_seeds",
        "release_seed_range",
        "evaluation_seed_range",
        "held_out_seed_range",
    }
)
_SCENARIO_FIELD_NAMES = frozenset(
    {
        "scenario_id",
        "scenario_ids",
        "scenario_name",
        "scenario_names",
        "scenario",
        "scenarios",
    }
)
_ENTRY_SCENARIO_FIELD_NAMES = frozenset({"scenario_id", "scenario_ids"})
_SCENARIO_LIST_FIELDS = frozenset({"scenario_ids", "scenario_names", "scenarios"})


def _typed_values(value: Any) -> list[int]:
    """Collect integer leaves from a typed field without parsing strings."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, Mapping):
        values: list[int] = []
        for nested in value.values():
            values.extend(_typed_values(nested))
        return values
    if isinstance(value, (list, tuple)):
        values: list[int] = []
        for nested in value:
            values.extend(_typed_values(nested))
        return values
    return []


def _collect_typed_fields(value: Any, *, field_names: frozenset[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in field_names:
                found.append((normalized, nested))
            found.extend(_collect_typed_fields(nested, field_names=field_names))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_collect_typed_fields(nested, field_names=field_names))
    return found


def _validate_tuning_log_seeds(payload: Mapping[str, Any]) -> int:
    seed_values: list[int] = []
    for field, raw in _collect_typed_fields(payload, field_names=_SEED_FIELD_NAMES):
        values = _typed_values(raw)
        if not values:
            raise ValidationError(
                f"structured tuning-log field {field!r} has no typed integer seeds"
            )
        seed_values.extend(values)
    if not seed_values:
        raise ValidationError("structured tuning log must contain at least one typed seed field")
    release_seed_values = sorted(set(seed_values) & RELEASE_SEEDS)
    if release_seed_values:
        raise ValidationError(
            "structured tuning log admits held-out release seeds: "
            + ", ".join(str(seed) for seed in release_seed_values)
        )
    invalid_seeds = sorted(set(seed_values) - set(EXPECTED_DEV_SEEDS))
    if invalid_seeds:
        raise ValidationError(
            "structured tuning log admits seeds outside the approved development set: "
            + ", ".join(str(seed) for seed in invalid_seeds)
        )
    return len(seed_values)


def _typed_scenario_values(field: str, raw: Any) -> list[str]:
    expected_type = list if field in _SCENARIO_LIST_FIELDS else str
    if not isinstance(raw, expected_type):
        type_name = "a list of scenario IDs" if expected_type is list else "a scenario ID string"
        raise ValidationError(f"structured field {field!r} must contain {type_name}")
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValidationError(f"structured field {field!r} must contain at least one scenario ID")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValidationError(f"structured field {field!r} contains invalid scenario IDs")
    return [value.strip() for value in values]


def _validate_typed_scenario_ids(payload: Mapping[str, Any], *, label: str) -> int:
    scenario_ids = [
        scenario_id
        for field, raw in _collect_typed_fields(payload, field_names=_SCENARIO_FIELD_NAMES)
        for scenario_id in _typed_scenario_values(field, raw)
    ]
    unexpected = sorted(set(scenario_ids) - EXPECTED_SCENARIO_IDS)
    if unexpected:
        raise ValidationError(
            f"{label} admits scenario IDs outside the development set: {unexpected!r}"
        )
    return len(scenario_ids)


def _validate_tuning_log_scenarios(
    payload: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> int:
    for index, entry in enumerate(entries):
        entry_values = [
            scenario_id
            for field, raw in _collect_typed_fields(entry, field_names=_ENTRY_SCENARIO_FIELD_NAMES)
            for scenario_id in _typed_scenario_values(field, raw)
        ]
        if not entry_values:
            raise ValidationError(
                f"structured tuning-log entry {index} must contain a non-empty typed "
                "scenario_id or scenario_ids field"
            )
    return _validate_typed_scenario_ids(payload, label="structured tuning log")


def _validate_tuning_log_trial(
    index: int,
    entry: Mapping[str, Any],
    *,
    candidate_hashes: Mapping[str, Any],
    trial_ids: set[str],
) -> None:
    candidate = entry.get("candidate")
    if not isinstance(candidate, str) or candidate not in EXPECTED_PLANNER_CONFIGS:
        raise ValidationError(
            f"structured tuning-log entry {index} names an unapproved candidate: {candidate!r}"
        )
    if entry.get("base_candidate_config_sha256") != candidate_hashes[candidate]:
        raise ValidationError(
            f"structured tuning-log entry {index} does not bind the approved candidate template"
        )
    trial_id = entry.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id.strip() or trial_id in trial_ids:
        raise ValidationError(f"structured tuning-log entry {index} needs a unique trial_id")
    trial_ids.add(trial_id)
    effective_config = entry.get("effective_candidate_config")
    if not isinstance(effective_config, Mapping) or not effective_config:
        raise ValidationError(
            f"structured tuning-log entry {index} must include effective_candidate_config"
        )
    effective_hash = _require_sha256(
        entry.get("effective_candidate_config_sha256"),
        label=f"tuning-log entry {index} effective_candidate_config_sha256",
    )
    if _canonical_sha256(effective_config) != effective_hash:
        raise ValidationError(
            f"structured tuning-log entry {index} effective candidate config checksum mismatch"
        )
    artifact_ref = entry.get("run_artifact_ref")
    if not isinstance(artifact_ref, str) or not artifact_ref.strip():
        raise ValidationError(f"structured tuning-log entry {index} needs a run_artifact_ref")
    artifact_digest = _require_sha256(
        entry.get("run_artifact_sha256"),
        label=f"tuning-log entry {index} run_artifact_sha256",
    )
    artifact_path = _resolve_run_artifact(
        artifact_ref,
        label=f"tuning-log entry {index} run_artifact_ref",
    )
    if _sha256_file(artifact_path, label=f"tuning-log entry {index} run artifact") != (
        artifact_digest
    ):
        raise ValidationError(f"structured tuning-log entry {index} run artifact checksum mismatch")


def _validate_tuning_log(path: Path, *, expected_bindings: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_mapping(path, label="structured tuning log")
    if payload.get("schema_version") != TUNING_LOG_SCHEMA:
        raise ValidationError(
            f"structured tuning log must declare schema_version: {TUNING_LOG_SCHEMA}"
        )
    source_commit = payload.get("source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValidationError(
            "structured tuning log must bind a lowercase 40-character source_commit"
        )
    if payload.get("input_bindings") != expected_bindings:
        raise ValidationError("structured tuning log input_bindings do not match frozen inputs")
    _validate_tuning_source_commit(source_commit, expected_bindings=expected_bindings)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValidationError("structured tuning log entries must be a non-empty list")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValidationError("structured tuning log entries must be mappings")
    candidate_hashes = expected_bindings.get("candidate_template_sha256")
    if not isinstance(candidate_hashes, Mapping):
        raise ValidationError("expected candidate template hashes are unavailable")
    trial_ids: set[str] = set()
    for index, entry in enumerate(entries):
        _validate_tuning_log_trial(
            index,
            entry,
            candidate_hashes=candidate_hashes,
            trial_ids=trial_ids,
        )
    typed_seed_count = _validate_tuning_log_seeds(payload)
    typed_scenario_count = _validate_tuning_log_scenarios(payload, entries)

    return {
        "path": str(path),
        "schema_version": TUNING_LOG_SCHEMA,
        "entry_count": len(entries),
        "typed_seed_count": typed_seed_count,
        "typed_scenario_count": typed_scenario_count,
        "release_seed_overlap": [],
        "release_scenario_overlap": [],
        "source_commit": source_commit,
        "input_bindings": dict(expected_bindings),
    }


def validate(
    config_path: Path = DEFAULT_CONFIG,
    *,
    release_matrix_path: Path = DEFAULT_RELEASE_MATRIX,
    tuning_logs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Validate the checked-in development split and optional tuning logs."""
    summary = _validate_campaign_config(config_path.resolve(), release_matrix_path.resolve())
    summary["tuning_logs"] = [
        _validate_tuning_log(path.resolve(), expected_bindings=summary["tuning_log_bindings"])
        for path in tuning_logs
    ]
    summary["ok"] = True
    summary["status"] = "development_protocol_validated_not_benchmark_evidence"
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-matrix", type=Path, default=DEFAULT_RELEASE_MATRIX)
    parser.add_argument(
        "--tuning-log",
        type=Path,
        action="append",
        default=[],
        help=f"Structured YAML/JSON log using schema {TUNING_LOG_SCHEMA}; repeatable.",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the issue #9748 validator and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        summary = validate(
            args.config,
            release_matrix_path=args.release_matrix,
            tuning_logs=args.tuning_log,
        )
    except ValidationError as exc:
        error = {"ok": False, "status": "invalid", "error": str(exc)}
        print(json.dumps(error, sort_keys=True) if args.json else f"INVALID: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True) if args.json else "VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
