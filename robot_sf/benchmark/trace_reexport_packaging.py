"""Fail-closed packaging for the issue #5756 trace re-export.

The packager verifies frozen inputs before reading episode rows, joins release and rerun rows by
``(planner, scenario_id, seed)``, and materializes renderer-neutral trace exports atomically.  It
does not fetch artifacts or know any machine-private path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
    simulation_trace_export_from_dict,
)
from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    load_campaign_config,
)
from robot_sf.benchmark.camera_ready._util import _hash_payload, _jsonable_repo_relative
from robot_sf.benchmark.utils import _config_hash
from scripts.tools.build_simulation_trace_export import build_simulation_trace_export

EXPECTED_OUTCOMES_SCHEMA = "issue_5756_expected_outcomes.v1"
MAPPING_RECEIPT_SCHEMA = "issue_5756_trace_reexport_mapping_receipt.v1"
PACKAGE_COMPLETE_SCHEMA = "issue_5756_trace_reexport_package_complete.v1"
EXECUTION_COMMIT = "a307ef276d701f8d14dead1aa0513f44ee97c0b0"
CANONICAL_CAMPAIGN_CONFIG_SHA256 = (
    "143ab63a235f40326c93c93044fba95e808388751f04d8ca979b89d1142ca465"
)
SCENARIO_MATRIX_SHA256 = "d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5"
SEED_SET_SHA256 = "3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6"
REQUEST_MANIFEST_SHA256 = "320190fd489797efeb194711d75f41d19f23eeef56107408270e62624b0e49e8"
PPO_CONFIG_SHA256 = "644b57e451cfc42b6ab5cf56ef6ec20fd6290a3bb9bae1be113a1aa6afb792ca"
PPO_CHECKPOINT_SHA256 = "2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d"
RELEASE_BUNDLE_SHA256 = "3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7"
RELEASE_GOAL_JSONL_SHA256 = "21702588cd197890fe2317f7214c71fc656a03009da7d2279df36ff1c21459e2"
RELEASE_PPO_JSONL_SHA256 = "c7b776a236254365eb71174070b4299af959423135707229e1af90dbe6e5fec1"
EXPECTED_OUTCOMES_SHA256 = "4d12c706c2475cc3adfd21f042d21a27afdb7833aeb387d430e0ae93a732a031"
PPO_MODEL_ID = "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"

_CANONICAL_CONFIG = Path("configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended.yaml")
_SCENARIO_MATRIX = Path("configs/scenarios/classic_interactions_francis2023.yaml")
_SEED_SETS = Path("configs/benchmarks/seed_sets_v1.yaml")
_PPO_CONFIG = Path("configs/baselines/ppo_15m_grid_socnav.yaml")
_MODEL_REGISTRY = Path("model/registry.yaml")
_CONFIGS = {
    "canary": Path("configs/benchmarks/issue_5756_trace90_ppo_canary.yaml"),
    "ppo": Path("configs/benchmarks/issue_5756_trace90_ppo.yaml"),
    "goal": Path("configs/benchmarks/issue_5756_trace90_goal.yaml"),
}
_OUTCOME_FIELDS = ("success", "route_complete", "collision_event", "timeout_event")


class TraceReexportPackagingError(ValueError):
    """Raised when any frozen packaging contract is not satisfied."""


@dataclass(frozen=True)
class FrozenTraceReexportContract:
    """Digest pins that may be replaced only by synthetic tests."""

    release_bundle_sha256: str = RELEASE_BUNDLE_SHA256
    request_manifest_sha256: str = REQUEST_MANIFEST_SHA256
    release_goal_jsonl_sha256: str = RELEASE_GOAL_JSONL_SHA256
    release_ppo_jsonl_sha256: str = RELEASE_PPO_JSONL_SHA256
    expected_outcomes_sha256: str = EXPECTED_OUTCOMES_SHA256


@dataclass(frozen=True)
class CampaignExpectation:
    """Resolved campaign provenance expected from one retrieved rerun output."""

    label: str
    name: str
    planner: str
    scenarios: tuple[str, ...]
    scenario_candidates: tuple[str, ...]
    seeds: tuple[int, ...]
    config_hash: str
    scenario_matrix_hash: str

    @property
    def tuples(self) -> set[tuple[str, str, int]]:
        """Return the exact planner/scenario/seed matrix for this output."""
        return {
            (self.planner, scenario, seed) for scenario in self.scenarios for seed in self.seeds
        }


def _canonical_bytes(payload: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return data + (b"\n" if newline else b"")


def canonical_sha256(payload: Any) -> str:
    """Hash a compact, key-sorted JSON representation.

    Returns:
        Full lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(actual: str, expected: str, label: str) -> None:
    if actual != expected:
        raise TraceReexportPackagingError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _read_json_object_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceReexportPackagingError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceReexportPackagingError(f"{label}: expected a JSON object")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        return _read_json_object_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise TraceReexportPackagingError(f"{path}: cannot read: {exc}") from exc


def _read_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise TraceReexportPackagingError(f"{label}: invalid UTF-8: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceReexportPackagingError(
                f"{label}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise TraceReexportPackagingError(f"{label}:{line_number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise TraceReexportPackagingError(f"{label}: contains no rows")
    return rows


def _verify_local_frozen_inputs(repo_root: Path) -> None:
    expected = {
        _CANONICAL_CONFIG: CANONICAL_CAMPAIGN_CONFIG_SHA256,
        _SCENARIO_MATRIX: SCENARIO_MATRIX_SHA256,
        _SEED_SETS: SEED_SET_SHA256,
        _PPO_CONFIG: PPO_CONFIG_SHA256,
    }
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file():
            raise TraceReexportPackagingError(f"required frozen input is missing: {relative}")
        _require_digest(_sha256_file(path), digest, relative.as_posix())

    registry_path = repo_root / _MODEL_REGISTRY
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TraceReexportPackagingError(f"cannot read {_MODEL_REGISTRY}: {exc}") from exc
    entries = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(entries, list):
        raise TraceReexportPackagingError("model/registry.yaml: models must be a list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("model_id") == PPO_MODEL_ID
    ]
    if len(matches) != 1:
        raise TraceReexportPackagingError(
            f"model registry must contain exactly one {PPO_MODEL_ID!r} entry"
        )
    github_release = matches[0].get("github_release")
    actual = github_release.get("sha256") if isinstance(github_release, dict) else None
    if actual != PPO_CHECKPOINT_SHA256:
        raise TraceReexportPackagingError("model registry PPO checkpoint SHA-256 mismatch")


def _archive_member_bytes(archive: tarfile.TarFile, suffix: str) -> tuple[str, bytes]:
    matches: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise TraceReexportPackagingError(f"unsafe release archive member: {member.name!r}")
        if member.name.endswith(suffix):
            matches.append(member)
    if len(matches) != 1:
        raise TraceReexportPackagingError(
            f"release archive must contain exactly one member ending in {suffix!r}; "
            f"found {len(matches)}"
        )
    member = matches[0]
    if not member.isfile():
        raise TraceReexportPackagingError(
            f"release archive member is not a regular file: {member.name}"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise TraceReexportPackagingError(f"cannot read release archive member: {member.name}")
    return member.name, handle.read()


def _load_release_rows(
    release_bundle: Path,
    *,
    contract: FrozenTraceReexportContract,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    _require_digest(_sha256_file(release_bundle), contract.release_bundle_sha256, "release bundle")
    try:
        with tarfile.open(release_bundle, "r:*") as archive:
            goal_name, goal_bytes = _archive_member_bytes(
                archive, "/payload/runs/goal__differential_drive/episodes.jsonl"
            )
            ppo_name, ppo_bytes = _archive_member_bytes(
                archive, "/payload/runs/ppo__differential_drive/episodes.jsonl"
            )
            _, manifest_bytes = _archive_member_bytes(archive, "/publication_manifest.json")
    except (OSError, tarfile.TarError) as exc:
        raise TraceReexportPackagingError(
            f"cannot read release bundle {release_bundle}: {exc}"
        ) from exc

    digests = {"goal": _sha256_bytes(goal_bytes), "ppo": _sha256_bytes(ppo_bytes)}
    _require_digest(digests["goal"], contract.release_goal_jsonl_sha256, "release goal JSONL")
    _require_digest(digests["ppo"], contract.release_ppo_jsonl_sha256, "release PPO JSONL")

    publication = _read_json_object_bytes(manifest_bytes, "publication_manifest.json")
    files = publication.get("files")
    if not isinstance(files, list):
        raise TraceReexportPackagingError("publication_manifest.json: files must be a list")
    signed: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise TraceReexportPackagingError(
                "publication_manifest.json: file entry must be an object"
            )
        path, digest = entry.get("path"), entry.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            if path in signed:
                raise TraceReexportPackagingError(f"duplicate publication manifest path: {path}")
            signed[path] = digest
    for member_name, expected in ((goal_name, digests["goal"]), (ppo_name, digests["ppo"])):
        marker = "/payload/"
        relative = member_name.split(marker, 1)[1]
        candidates = (relative, f"payload/{relative}")
        matches = [signed[path] for path in candidates if path in signed]
        if len(matches) != 1 or matches[0] != expected:
            raise TraceReexportPackagingError(
                f"publication manifest does not uniquely bind {relative} to its SHA-256"
            )

    return {
        "goal": _read_jsonl_bytes(goal_bytes, goal_name),
        "ppo": _read_jsonl_bytes(ppo_bytes, ppo_name),
    }, digests


def campaign_expectations(repo_root: Path) -> dict[str, CampaignExpectation]:
    """Resolve the exact config hashes and matrices emitted by the campaign runner.

    Returns:
        Expectations keyed by canary, PPO, and goal output label.
    """
    expectations: dict[str, CampaignExpectation] = {}
    for label, relative in _CONFIGS.items():
        cfg = load_campaign_config(repo_root / relative)
        scenarios = _load_campaign_scenarios(cfg)
        scenario_names = tuple(
            str(row.get("name") or row.get("scenario_id") or row.get("id")) for row in scenarios
        )
        seeds = tuple(sorted({int(seed) for row in scenarios for seed in row.get("seeds", [])}))
        expectations[label] = CampaignExpectation(
            label=label,
            name=cfg.name,
            planner=cfg.planners[0].key,
            scenarios=scenario_names,
            scenario_candidates=cfg.scenario_candidates.names,
            seeds=seeds,
            config_hash=_config_hash(_jsonable_repo_relative(asdict(cfg))),
            scenario_matrix_hash=_hash_payload(scenarios),
        )
    return expectations


def _tuple_from_request(row: Mapping[str, Any]) -> tuple[str, str, int]:
    planner = row.get("planner")
    scenario = row.get("scenario_id")
    seed = row.get("seed")
    if not isinstance(planner, str) or not planner.strip():
        raise TraceReexportPackagingError("request tuple planner must be a non-empty string")
    if not isinstance(scenario, str) or not scenario.strip():
        raise TraceReexportPackagingError("request tuple scenario_id must be a non-empty string")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise TraceReexportPackagingError("request tuple seed must be an integer") from exc
    if isinstance(seed, float) and not seed.is_integer():
        raise TraceReexportPackagingError("request tuple seed must be an integer")
    return planner.strip(), scenario.strip(), seed_int


def _load_request_manifest(
    path: Path, *, contract: FrozenTraceReexportContract
) -> tuple[dict[tuple[str, str, int], str], dict[str, Any]]:
    data = path.read_bytes()
    _require_digest(_sha256_bytes(data), contract.request_manifest_sha256, "request manifest")
    payload = _read_json_object_bytes(data, str(path))
    if payload.get("schema_version") != "issue_5446_trace_reexport_list.v1":
        raise TraceReexportPackagingError("unexpected request manifest schema_version")
    tuples = payload.get("tuples")
    if not isinstance(tuples, list) or len(tuples) != 90 or payload.get("n_tuples") != 90:
        raise TraceReexportPackagingError("request manifest must declare exactly 90 tuples")
    indexed: dict[tuple[str, str, int], str] = {}
    for row in tuples:
        if not isinstance(row, dict):
            raise TraceReexportPackagingError("request tuple must be an object")
        key = _tuple_from_request(row)
        episode_id = row.get("episode_id")
        if (
            row.get("episode_id_status") != "found"
            or not isinstance(episode_id, str)
            or not episode_id
        ):
            raise TraceReexportPackagingError(f"request tuple {key!r} lacks one found episode_id")
        if key in indexed:
            raise TraceReexportPackagingError(f"duplicate request tuple: {key!r}")
        indexed[key] = episode_id
    expected = {("ppo", "classic_doorway_medium", seed) for seed in range(111, 141)} | {
        (planner, "classic_realworld_double_bottleneck_high", seed)
        for planner in ("goal", "ppo")
        for seed in range(111, 141)
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise TraceReexportPackagingError(
            f"request tuple set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    return indexed, payload


def _row_tuple(row: Mapping[str, Any], *, planner_hint: str | None = None) -> tuple[str, str, int]:
    planner = row.get("algo")
    params = row.get("scenario_params")
    if not isinstance(planner, str) and isinstance(params, Mapping):
        planner = params.get("algo")
    if not isinstance(planner, str):
        planner = planner_hint
    scenario = row.get("scenario_id")
    seed = row.get("seed")
    if not isinstance(planner, str) or not isinstance(scenario, str):
        raise TraceReexportPackagingError("episode row lacks planner/algo or scenario_id")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise TraceReexportPackagingError("episode row seed must be an integer") from exc
    if isinstance(seed, float) and not seed.is_integer():
        raise TraceReexportPackagingError("episode row seed must be an integer")
    return planner, scenario, seed_int


def _strict_bool(value: Any, *, field: str, key: tuple[str, str, int]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise TraceReexportPackagingError(f"row {key!r} field {field} must be boolean or numeric 0/1")


def _outcome(row: Mapping[str, Any], key: tuple[str, str, int]) -> dict[str, bool]:
    metrics = row.get("metrics")
    outcome = row.get("outcome")
    if not isinstance(metrics, Mapping) or not isinstance(outcome, Mapping):
        raise TraceReexportPackagingError(f"row {key!r} lacks metrics/outcome objects")
    values = {
        "success": metrics.get("success"),
        "route_complete": outcome.get("route_complete"),
        "collision_event": outcome.get("collision_event"),
        "timeout_event": outcome.get("timeout_event"),
    }
    return {field: _strict_bool(values[field], field=field, key=key) for field in _OUTCOME_FIELDS}


def _index_rows(
    rows: Iterable[dict[str, Any]], *, planner_hint: str | None = None
) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = _row_tuple(row, planner_hint=planner_hint)
        if key in indexed:
            raise TraceReexportPackagingError(f"duplicate/ambiguous episode tuple: {key!r}")
        indexed[key] = row
    return indexed


def _verify_campaign_manifest(
    manifest: Mapping[str, Any], expectation: CampaignExpectation
) -> None:
    checks = {
        "name": expectation.name,
        "scenario_matrix": _SCENARIO_MATRIX.as_posix(),
        "scenario_matrix_hash": expectation.scenario_matrix_hash,
        "config_hash": expectation.config_hash,
        "scenario_candidates": list(expectation.scenario_candidates),
    }
    for field, expected in checks.items():
        if manifest.get(field) != expected:
            raise TraceReexportPackagingError(
                f"{expectation.label} campaign manifest {field} mismatch"
            )
    git = manifest.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != EXECUTION_COMMIT:
        raise TraceReexportPackagingError(
            f"{expectation.label} campaign manifest execution commit mismatch"
        )
    seed_policy = manifest.get("seed_policy")
    if not isinstance(seed_policy, Mapping) or seed_policy.get("resolved_seeds") != list(
        expectation.seeds
    ):
        raise TraceReexportPackagingError(
            f"{expectation.label} campaign manifest resolved seed set mismatch"
        )
    planners = manifest.get("planners")
    if (
        not isinstance(planners, list)
        or len(planners) != 1
        or planners[0].get("key") != expectation.planner
    ):
        raise TraceReexportPackagingError(f"{expectation.label} campaign manifest planner mismatch")
    if expectation.planner == "ppo":
        provenance = planners[0].get("checkpoint_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("checkpoint_sha256") != PPO_CHECKPOINT_SHA256
        ):
            raise TraceReexportPackagingError(
                f"{expectation.label} campaign manifest PPO checkpoint mismatch"
            )
        if (
            provenance.get("load_succeeded") is not True
            or provenance.get("fallback_triggered") is not False
        ):
            raise TraceReexportPackagingError(
                f"{expectation.label} campaign manifest PPO load/fallback provenance mismatch"
            )


def _load_rerun_output(
    root: Path, expectation: CampaignExpectation
) -> dict[tuple[str, str, int], dict[str, Any]]:
    manifest = _read_json_object(root / "campaign_manifest.json")
    _verify_campaign_manifest(manifest, expectation)
    episode_paths = sorted((root / "runs").glob("*/episodes.jsonl"))
    if len(episode_paths) != 1:
        raise TraceReexportPackagingError(
            f"{expectation.label} output must contain exactly one runs/*/episodes.jsonl; "
            f"found {len(episode_paths)}"
        )
    expected_dir = f"{expectation.planner}__differential_drive"
    if episode_paths[0].parent.name != expected_dir:
        raise TraceReexportPackagingError(
            f"{expectation.label} output run directory must be {expected_dir!r}"
        )
    rows = _read_jsonl_bytes(episode_paths[0].read_bytes(), str(episode_paths[0]))
    indexed = _index_rows(rows, planner_hint=expectation.planner)
    if set(indexed) != expectation.tuples:
        missing = sorted(expectation.tuples - set(indexed))
        extra = sorted(set(indexed) - expectation.tuples)
        raise TraceReexportPackagingError(
            f"{expectation.label} rerun tuple set mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    for key, row in indexed.items():
        _verify_rerun_row(row, key)
    return indexed


def _nested_mapping(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _verify_rerun_row(  # noqa: C901
    row: Mapping[str, Any], key: tuple[str, str, int]
) -> None:
    planner, _scenario, _seed = key
    if row.get("git_hash") != EXECUTION_COMMIT:
        raise TraceReexportPackagingError(f"rerun row {key!r} execution commit mismatch")
    params = row.get("scenario_params")
    if not isinstance(params, Mapping):
        raise TraceReexportPackagingError(f"rerun row {key!r} lacks scenario_params")
    if row.get("config_hash") != _config_hash(dict(params)):
        raise TraceReexportPackagingError(f"rerun row {key!r} scenario/config hash mismatch")
    required_params = {
        "algo": planner,
        "record_forces": True,
        "record_planner_decision_trace": True,
        "record_simulation_step_trace": True,
        "run_horizon": 600,
        "run_dt": 0.1,
    }
    for field, expected in required_params.items():
        if params.get(field) != expected:
            raise TraceReexportPackagingError(f"rerun row {key!r} scenario_params.{field} mismatch")
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise TraceReexportPackagingError(f"rerun row {key!r} lacks algorithm_metadata")
    kinematics = metadata.get("planner_kinematics")
    if (
        not isinstance(kinematics, Mapping)
        or kinematics.get("robot_kinematics") != "differential_drive"
    ):
        raise TraceReexportPackagingError(f"rerun row {key!r} is not differential-drive execution")
    trace = metadata.get("simulation_step_trace")
    if (
        not isinstance(trace, Mapping)
        or trace.get("schema_version") != "simulation-step-trace.v1"
        or not isinstance(trace.get("steps"), list)
        or not trace["steps"]
    ):
        raise TraceReexportPackagingError(f"rerun row {key!r} lacks a non-empty simulation trace")
    decision_trace = metadata.get("planner_decision_trace")
    if (
        not isinstance(decision_trace, Mapping)
        or decision_trace.get("schema_version") != "planner-decision-trace.v1"
        or not isinstance(decision_trace.get("steps"), list)
    ):
        raise TraceReexportPackagingError(f"rerun row {key!r} lacks a planner decision trace")
    if planner == "ppo":
        provenance = _nested_mapping(
            row, "algorithm_metadata", "planner_runtime", "checkpoint_provenance"
        )
        if provenance is None:
            provenance = _nested_mapping(row, "algorithm_metadata", "checkpoint_provenance")
        if provenance is None or provenance.get("checkpoint_sha256") != PPO_CHECKPOINT_SHA256:
            raise TraceReexportPackagingError(f"rerun row {key!r} PPO checkpoint mismatch")
        if (
            provenance.get("load_succeeded") is not True
            or provenance.get("fallback_triggered") is not False
        ):
            raise TraceReexportPackagingError(f"rerun row {key!r} PPO load/fallback mismatch")
    _outcome(row, key)


def _release_selection(
    release_rows: Mapping[str, list[dict[str, Any]]],
    requests: Mapping[tuple[str, str, int], str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    selected: dict[tuple[str, str, int], dict[str, Any]] = {}
    for planner, rows in release_rows.items():
        indexed = _index_rows(rows, planner_hint=planner)
        for key in (key for key in requests if key[0] == planner):
            row = indexed.get(key)
            if row is None:
                raise TraceReexportPackagingError(
                    f"release bundle is missing request tuple {key!r}"
                )
            if row.get("episode_id") != requests[key]:
                raise TraceReexportPackagingError(f"release episode_id mismatch for {key!r}")
            _outcome(row, key)
            selected[key] = row
    if set(selected) != set(requests):
        missing = sorted(set(requests) - set(selected))
        raise TraceReexportPackagingError(
            f"release rows do not cover request tuples: {missing[:3]}"
        )
    return selected


def _expected_outcomes_payload(
    release: Mapping[tuple[str, str, int], dict[str, Any]],
    *,
    contract: FrozenTraceReexportContract,
) -> dict[str, Any]:
    rows = []
    for key in sorted(release):
        planner, scenario, seed = key
        rows.append(
            {
                "planner": planner,
                "scenario_id": scenario,
                "seed": seed,
                "release_episode_id": str(release[key]["episode_id"]),
                **_outcome(release[key], key),
            }
        )
    return {
        "schema_version": EXPECTED_OUTCOMES_SCHEMA,
        "provenance": {
            "release_bundle_sha256": contract.release_bundle_sha256,
            "request_manifest_sha256": contract.request_manifest_sha256,
            "release_goal_jsonl_sha256": contract.release_goal_jsonl_sha256,
            "release_ppo_jsonl_sha256": contract.release_ppo_jsonl_sha256,
        },
        "rows": rows,
    }


def expected_outcomes_payload_for_rows(
    release: Mapping[tuple[str, str, int], dict[str, Any]],
    *,
    contract: FrozenTraceReexportContract,
) -> dict[str, Any]:
    """Build the canonical expected-outcome payload (primarily for synthetic fixtures).

    Returns:
        Versioned expected-outcome payload ready for canonical hashing.
    """
    return _expected_outcomes_payload(release, contract=contract)


def _row_sha256(row: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(row, newline=True))


def _trace_uri(key: tuple[str, str, int]) -> str:
    planner, scenario, seed = key
    return f"traces/{planner}/{scenario}/seed-{seed}.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload, newline=True))


def _validate_staged_package(  # noqa: C901
    root: Path, expected_rows: int
) -> dict[str, str]:
    outcomes = _read_json_object(root / "expected_outcomes.json")
    receipt = _read_json_object(root / "mapping_receipt.json")
    if outcomes.get("schema_version") != EXPECTED_OUTCOMES_SCHEMA:
        raise TraceReexportPackagingError("staged expected-outcome schema mismatch")
    declared_outcome_digest = outcomes.get("contract_sha256")
    outcome_contract = {key: value for key, value in outcomes.items() if key != "contract_sha256"}
    if (
        not isinstance(declared_outcome_digest, str)
        or canonical_sha256(outcome_contract) != declared_outcome_digest
    ):
        raise TraceReexportPackagingError("staged expected-outcome contract digest mismatch")
    outcome_rows = outcomes.get("rows")
    if not isinstance(outcome_rows, list) or len(outcome_rows) != expected_rows:
        raise TraceReexportPackagingError("staged expected-outcome row count mismatch")
    if (
        receipt.get("schema_version") != MAPPING_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
    ):
        raise TraceReexportPackagingError("staged mapping receipt is not complete")
    frozen = receipt.get("frozen_provenance")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("expected_outcomes_sha256") != declared_outcome_digest
    ):
        raise TraceReexportPackagingError("staged mapping receipt outcome provenance mismatch")
    rows = receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise TraceReexportPackagingError("staged mapping receipt row count mismatch")
    expected_tuples = {
        (str(row.get("planner")), str(row.get("scenario_id")), int(row.get("seed")))
        for row in outcome_rows
        if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
    }
    if len(expected_tuples) != expected_rows:
        raise TraceReexportPackagingError("staged expected outcomes contain duplicate identities")
    receipt_tuples: set[tuple[str, str, int]] = set()
    trace_uris: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TraceReexportPackagingError("staged mapping row must be an object")
        try:
            key = (str(row["planner"]), str(row["scenario_id"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceReexportPackagingError("staged mapping row identity is invalid") from exc
        if key in receipt_tuples:
            raise TraceReexportPackagingError(
                "staged mapping receipt contains duplicate identities"
            )
        receipt_tuples.add(key)
        uri = row.get("trace_uri")
        if (
            not isinstance(uri, str)
            or PurePosixPath(uri).is_absolute()
            or ".." in PurePosixPath(uri).parts
            or uri in trace_uris
        ):
            raise TraceReexportPackagingError("staged mapping trace_uri is not durable-relative")
        trace_uris.add(uri)
        trace_path = root / uri
        trace = load_simulation_trace_export(trace_path)
        if (trace.source.planner_id, trace.source.scenario_id, trace.source.seed) != key:
            raise TraceReexportPackagingError(f"trace {uri} source identity mismatch")
        if trace.source.episode_id != row.get("rerun_episode_id"):
            raise TraceReexportPackagingError(f"trace {uri} rerun episode ID mismatch")
        _require_digest(_sha256_file(trace_path), str(row.get("trace_sha256")), f"trace {uri}")
    if receipt_tuples != expected_tuples:
        raise TraceReexportPackagingError("staged mapping and expected-outcome identities differ")
    files = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "package_complete.json"
    }
    return files


def _verify_complete_output(root: Path) -> dict[str, Any] | None:
    marker_path = root / "package_complete.json"
    if not marker_path.exists():
        return None
    marker = _read_json_object(marker_path)
    if (
        marker.get("schema_version") != PACKAGE_COMPLETE_SCHEMA
        or marker.get("status") != "complete"
    ):
        raise TraceReexportPackagingError("existing package completion marker is invalid")
    expected_files = marker.get("files")
    if not isinstance(expected_files, dict):
        raise TraceReexportPackagingError("existing package completion marker files are invalid")
    if marker.get("trace_count") != 90:
        raise TraceReexportPackagingError(
            "existing package completion marker trace count is invalid"
        )
    actual = _validate_staged_package(root, expected_rows=90)
    if actual != expected_files:
        raise TraceReexportPackagingError("existing complete package file digests do not match")
    if marker.get("mapping_receipt_sha256") != actual.get("mapping_receipt.json"):
        raise TraceReexportPackagingError("existing package mapping receipt digest is invalid")
    outcomes = _read_json_object(root / "expected_outcomes.json")
    if marker.get("expected_outcomes_sha256") != outcomes.get("contract_sha256"):
        raise TraceReexportPackagingError("existing package expected-outcome digest is invalid")
    return marker


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_output_path(output_dir: Path, input_paths: Mapping[str, Path]) -> None:
    canonical_output = output_dir.resolve()
    for label, input_path in input_paths.items():
        canonical_input = input_path.resolve()
        if _paths_overlap(canonical_output, canonical_input):
            raise TraceReexportPackagingError(
                f"output path overlaps raw {label} input: {output_dir}"
            )

    if os.path.lexists(output_dir) and _verify_complete_output(canonical_output) is None:
        raise TraceReexportPackagingError(
            f"output path exists but is not a complete trace package: {output_dir}"
        )


def _install_staging(staging: Path, output_dir: Path) -> None:
    output_exists = os.path.lexists(output_dir)
    existing_marker = _verify_complete_output(output_dir) if output_exists else None
    if output_exists and existing_marker is None:
        raise TraceReexportPackagingError(
            f"output path exists but is not a complete trace package: {output_dir}"
        )
    staged_marker = _read_json_object(staging / "package_complete.json")
    if existing_marker == staged_marker:
        shutil.rmtree(staging)
        return
    backup = output_dir.with_name(f".{output_dir.name}.backup-{os.getpid()}")
    if backup.exists():
        raise TraceReexportPackagingError(f"atomic install backup already exists: {backup}")
    moved_existing = False
    try:
        if output_dir.exists():
            os.replace(output_dir, backup)
            moved_existing = True
        os.replace(staging, output_dir)
    except OSError:
        if moved_existing and not output_dir.exists() and backup.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _cleanup_staging(staging: Path, *, completed: bool) -> None:
    if not staging.exists():
        return
    try:
        shutil.rmtree(staging)
    except OSError:
        if completed:
            raise


def package_trace_reexport(  # noqa: PLR0913, PLR0915
    *,
    release_bundle: Path,
    request_manifest: Path,
    canary_output: Path,
    ppo_output: Path,
    goal_output: Path,
    output_dir: Path,
    repo_root: Path | None = None,
    contract: FrozenTraceReexportContract = FrozenTraceReexportContract(),
    trace_builder: Callable[..., dict[str, Any]] = build_simulation_trace_export,
) -> Path:
    """Validate all inputs and atomically materialize the complete 90-trace package.

    Returns:
        Path to the verified complete package directory.
    """
    repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    release_bundle = release_bundle.resolve()
    request_manifest = request_manifest.resolve()
    requested_output_dir = Path(output_dir)
    output_dir = requested_output_dir.resolve()
    _validate_output_path(
        requested_output_dir,
        {
            "release bundle": release_bundle,
            "request manifest": request_manifest,
            "canary": canary_output,
            "PPO": ppo_output,
            "goal": goal_output,
        },
    )

    # All frozen byte-level provenance is checked before episode rows are interpreted.
    _verify_local_frozen_inputs(repo_root)
    _require_digest(
        _sha256_file(request_manifest),
        contract.request_manifest_sha256,
        "request manifest",
    )
    release_rows, _release_digests = _load_release_rows(release_bundle, contract=contract)
    requests, _request_payload = _load_request_manifest(request_manifest, contract=contract)
    expectations = campaign_expectations(repo_root)

    canary = _load_rerun_output(canary_output.resolve(), expectations["canary"])
    ppo = _load_rerun_output(ppo_output.resolve(), expectations["ppo"])
    goal = _load_rerun_output(goal_output.resolve(), expectations["goal"])
    rerun = {**ppo, **goal}
    if len(rerun) != 90 or set(rerun) != set(requests):
        raise TraceReexportPackagingError(
            "combined full rerun does not contain exactly 90 requests"
        )

    release = _release_selection(release_rows, requests)
    expected_payload = _expected_outcomes_payload(release, contract=contract)
    expected_digest = canonical_sha256(expected_payload)
    _require_digest(expected_digest, contract.expected_outcomes_sha256, "expected-outcome contract")

    canary_key = ("ppo", "classic_doorway_medium", 113)
    if set(canary) != {canary_key}:
        raise TraceReexportPackagingError("canary output must contain only PPO doorway seed 113")
    if _outcome(canary[canary_key], canary_key) != _outcome(release[canary_key], canary_key):
        raise TraceReexportPackagingError("canary outcome does not match the frozen release")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    completed = False
    try:
        _write_json(
            staging / "expected_outcomes.json",
            {"contract_sha256": expected_digest, **expected_payload},
        )
        mapping_rows: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="issue-5756-isolated-") as isolated_dir:
            isolated_root = Path(isolated_dir)
            for key in sorted(requests):
                release_row = release[key]
                rerun_row = rerun[key]
                if _outcome(release_row, key) != _outcome(rerun_row, key):
                    raise TraceReexportPackagingError(f"rerun outcome mismatch for {key!r}")
                release_id = str(release_row.get("episode_id") or "")
                rerun_id = str(rerun_row.get("episode_id") or "")
                if not release_id or not rerun_id:
                    raise TraceReexportPackagingError(f"row {key!r} lacks release/rerun episode ID")
                isolated_path = isolated_root / f"row-{len(mapping_rows):03d}.jsonl"
                isolated_path.write_bytes(_canonical_bytes(rerun_row, newline=True))
                trace = trace_builder(
                    isolated_path,
                    planner_id=key[0],
                    scenario_id=key[1],
                    source_signature=_sha256_file(isolated_path),
                )
                simulation_trace_export_from_dict(trace, source=isolated_path)
                if trace["source"]["episode_id"] != rerun_id:
                    raise TraceReexportPackagingError(
                        f"trace source episode ID mismatch for {key!r}"
                    )
                uri = _trace_uri(key)
                trace_path = staging / uri
                _write_json(trace_path, trace)
                trace_digest = _sha256_file(trace_path)
                mapping_rows.append(
                    {
                        "planner": key[0],
                        "scenario_id": key[1],
                        "seed": key[2],
                        "release_episode_id": release_id,
                        "rerun_episode_id": rerun_id,
                        "release_row_sha256": _row_sha256(release_row),
                        "rerun_row_sha256": _row_sha256(rerun_row),
                        "trace_uri": uri,
                        "trace_sha256": trace_digest,
                    }
                )

        receipt = {
            "schema_version": MAPPING_RECEIPT_SCHEMA,
            "status": "complete",
            "frozen_provenance": {
                "execution_commit": EXECUTION_COMMIT,
                "canonical_campaign_config_sha256": CANONICAL_CAMPAIGN_CONFIG_SHA256,
                "scenario_matrix_sha256": SCENARIO_MATRIX_SHA256,
                "seed_set_sha256": SEED_SET_SHA256,
                "request_manifest_sha256": contract.request_manifest_sha256,
                "ppo_config_sha256": PPO_CONFIG_SHA256,
                "ppo_checkpoint_sha256": PPO_CHECKPOINT_SHA256,
                "release_bundle_sha256": contract.release_bundle_sha256,
                "release_goal_jsonl_sha256": contract.release_goal_jsonl_sha256,
                "release_ppo_jsonl_sha256": contract.release_ppo_jsonl_sha256,
                "expected_outcomes_sha256": expected_digest,
            },
            "canary": {
                "planner": canary_key[0],
                "scenario_id": canary_key[1],
                "seed": canary_key[2],
                "release_episode_id": str(release[canary_key]["episode_id"]),
                "rerun_episode_id": str(canary[canary_key]["episode_id"]),
                "rerun_row_sha256": _row_sha256(canary[canary_key]),
            },
            "rows": mapping_rows,
        }
        _write_json(staging / "mapping_receipt.json", receipt)
        files = _validate_staged_package(staging, expected_rows=90)
        marker = {
            "schema_version": PACKAGE_COMPLETE_SCHEMA,
            "status": "complete",
            "expected_outcomes_sha256": expected_digest,
            "mapping_receipt_sha256": files["mapping_receipt.json"],
            "trace_count": 90,
            "files": files,
        }
        _write_json(staging / "package_complete.json", marker)
        _verify_complete_output(staging)
        _install_staging(staging, output_dir)
        completed = True
    finally:
        _cleanup_staging(staging, completed=completed)
    return output_dir


__all__ = [
    "CampaignExpectation",
    "FrozenTraceReexportContract",
    "TraceReexportPackagingError",
    "campaign_expectations",
    "canonical_sha256",
    "expected_outcomes_payload_for_rows",
    "package_trace_reexport",
]
