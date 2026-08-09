# ruff: noqa: DOC201, RUF022

"""Fail-closed packaging for the issue #5756 trace re-export.

The packager verifies frozen inputs before reading episode rows, joins release and rerun rows by
``(planner, scenario_id, seed)``, and materializes renderer-neutral trace exports atomically.  It
does not fetch artifacts or know any machine-private path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
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
from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest
from robot_sf.benchmark.utils import _config_hash
from scripts.tools.build_simulation_trace_export import (
    build_simulation_trace_export,
    build_simulation_trace_export_with_receipt,
)

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
#: Frozen release tag the 0.0.3 publication bundle was published under.
RELEASE_TAG = "0.0.3"
#: Git commit the 0.0.3 release tag/report surface was built at (the publication commit,
#: distinct from the pinned re-execution commit). Carried into resolver provenance.
REPORT_COMMIT = "e2ac534c9d6bb750346b1e0724638c91306e410a"
#: Schema the candidate-trace resolver (#5615) consumes for the #5756 episode mapping.
RESOLVER_MAPPING_SCHEMA = "issue_5756_trace_mapping_receipt.v1"
REAL_REEXPORT_BINDING_SCHEMA = "issue_6411_real_reexport_binding.v1"
REAL_REEXPORT_RECEIPT_SCHEMA = "issue_6411_trace_transformation_receipt.v1"
REAL_REEXPORT_CONFIG_EVIDENCE_SCHEMA = "issue_6411_config_evidence.v1"

# This is the approved package identity from issue #6814.  The current tracked
# package deliberately fails this check (its SHA256SUMS digest is different),
# so the strict re-export path must stop before reading external artifacts.
ISSUE_6412_PACKAGE_SHA256SUMS_SHA256 = (
    "011c644bac469a1ce6255ddb8731c53c84bd310887759174f4c734b54d6bb543"
)
ISSUE_6814_EXECUTION_COMMIT = EXECUTION_COMMIT

REAL_REEXPORT_SEEDS = tuple(range(111, 141))
REAL_REEXPORT_EXCEPTION_SEEDS = (128, 130)
REAL_REEXPORT_ARM_SPECS = (
    {
        "key": "doorway_ppo",
        "job_id": "13483",
        "planner": "ppo",
        "scenario_id": "classic_doorway_medium",
        "config_name": "doorway_butterfly_trace_reexport",
        "config_path": "configs/benchmarks/doorway_butterfly_trace_reexport.yaml",
        "not_admitted_seeds": REAL_REEXPORT_EXCEPTION_SEEDS,
    },
    {
        "key": "double_bottleneck_goal",
        "job_id": "13487",
        "planner": "goal",
        "scenario_id": "classic_realworld_double_bottleneck_high",
        "config_name": "double_bottleneck_upset_goal",
        "config_path": "configs/benchmarks/dbneck_goal.yaml",
        "not_admitted_seeds": (),
    },
    {
        "key": "double_bottleneck_ppo",
        "job_id": "13488",
        "planner": "ppo",
        "scenario_id": "classic_realworld_double_bottleneck_high",
        "config_name": "double_bottleneck_upset_ppo",
        "config_path": "configs/benchmarks/dbneck_ppo.yaml",
        "not_admitted_seeds": (),
    },
)

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


class RealReexportBindingError(TraceReexportPackagingError):
    """Raised when a real #5756 arm cannot be provenance-bound safely."""


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


@dataclass(frozen=True)
class RealReexportArm:
    """Immutable identity contract for one real #5756 re-export arm."""

    key: str
    job_id: str
    planner: str
    scenario_id: str
    config_name: str
    config_path: str
    seeds: tuple[int, ...] = REAL_REEXPORT_SEEDS
    not_admitted_seeds: tuple[int, ...] = ()

    @property
    def tuples(self) -> set[tuple[str, str, int]]:
        """Return this arm's exact planner/scenario/seed tuple set."""

        return {(self.planner, self.scenario_id, seed) for seed in self.seeds}


@dataclass(frozen=True, slots=True)
class VerifiedRealReexportRowSource:
    """Immutable, source-bound record for one approved #6412 row."""

    arm: str
    job_id: str
    row_index: int
    episode_id: str
    scenario_id: str
    planner_id: str
    seed: int
    execution_commit: str
    raw_row: Mapping[str, object]
    raw_row_sha256: str
    prior_normalized_sha256: str
    episodes_sha256: str
    manifest_sha256: str
    run_summary_sha256: str
    preflight_sha256: str
    result_provenance_sha256: str | None
    result_provenance_row: Mapping[str, object] | None
    source_root_retrieval_key: str
    episodes_retrieval_key: str | None = None
    manifest_retrieval_key: str | None = None
    run_summary_retrieval_key: str | None = None
    preflight_retrieval_key: str | None = None
    manifest: Mapping[str, object] | None = None
    run_summary: Mapping[str, object] | None = None
    preflight: Mapping[str, object] | None = None
    result_provenance_manifest: Mapping[str, object] | None = None
    route_geometry: Mapping[str, object] | None = None
    conflict_geometry: Mapping[str, object] | None = None
    encounter_report: Mapping[str, object] | None = None
    route_geometry_sha256: str | None = None
    conflict_geometry_sha256: str | None = None
    encounter_report_sha256: str | None = None
    n_rows: int = 0


REAL_REEXPORT_ARMS = tuple(RealReexportArm(**spec) for spec in REAL_REEXPORT_ARM_SPECS)
REAL_REEXPORT_ARMS_BY_KEY = {arm.key: arm for arm in REAL_REEXPORT_ARMS}
REAL_REEXPORT_REQUEST_TUPLES = frozenset(
    tuple_value for arm in REAL_REEXPORT_ARMS for tuple_value in arm.tuples
)


def _canonical_bytes(payload: Any, *, newline: bool = False) -> bytes:
    """Serialize a payload to compact, key-sorted JSON bytes with an optional newline.

    Returns:
        The compact JSON bytes, with a trailing newline appended when requested.
    """
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return data + (b"\n" if newline else b"")


def canonical_sha256(payload: Any) -> str:
    """Hash a compact, key-sorted JSON representation.

    Returns:
        Full lowercase SHA-256 hex digest.
    """
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(actual: str, expected: str, label: str) -> None:
    """Raise a packaging error when an actual digest does not match the expected one."""
    if actual != expected:
        raise TraceReexportPackagingError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _read_json_object_bytes(data: bytes, label: str) -> dict[str, Any]:
    """Parse bytes as a JSON object, raising a packaging error on invalid input.

    Returns:
        The parsed JSON object as a dictionary.
    """
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceReexportPackagingError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceReexportPackagingError(f"{label}: expected a JSON object")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a file and parse it as a JSON object, wrapping read errors.

    Returns:
        The parsed JSON object as a dictionary.
    """
    try:
        return _read_json_object_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise TraceReexportPackagingError(f"{path}: cannot read: {exc}") from exc


def _read_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    """Parse JSON Lines bytes into a non-empty list of JSON object rows.

    Returns:
        A non-empty list of JSON object rows parsed from the bytes.
    """
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
    """Verify frozen config files and the model registry match their pinned digests."""
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
    """Extract the single safe archive member ending in a suffix as name and bytes.

    Returns:
        The matching member's name and raw bytes as a tuple.
    """
    matches: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise TraceReexportPackagingError(f"unsafe release archive member: {member.name!r}")
        normalized_name = f"/{member.name.lstrip('/')}"
        if normalized_name.endswith(suffix):
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


def _payload_relative_path(member_name: str) -> str:
    """Return an archive member's path relative to its single payload directory."""
    parts = PurePosixPath(member_name).parts
    if parts.count("payload") != 1:
        raise TraceReexportPackagingError(
            f"release archive member must contain exactly one payload directory: {member_name!r}"
        )
    payload_index = parts.index("payload")
    relative_parts = parts[payload_index + 1 :]
    if not relative_parts:
        raise TraceReexportPackagingError(
            f"release archive member has no path below payload: {member_name!r}"
        )
    return PurePosixPath(*relative_parts).as_posix()


def _load_release_rows(
    release_bundle: Path,
    *,
    contract: FrozenTraceReexportContract,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    """Load and verify release episode rows and their digests from the bundle.

    Returns:
        Verified release rows keyed by planner alongside their JSONL digests.
    """
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
        relative = _payload_relative_path(member_name)
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
    """Extract and validate a planner/scenario/seed tuple from a request row.

    Returns:
        The validated planner, scenario, and seed tuple.
    """
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
    """Load and verify the 90-tuple request manifest, indexing episode IDs by tuple.

    Returns:
        The indexed episode IDs by tuple and the raw manifest payload.
    """
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
    """Extract a planner/scenario/seed tuple from an episode row, with a planner hint.

    Returns:
        The planner, scenario, and seed tuple extracted from the row.
    """
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
    """Coerce a boolean or numeric 0/1 outcome field, rejecting other values.

    Returns:
        The coerced boolean value of the outcome field.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise TraceReexportPackagingError(f"row {key!r} field {field} must be boolean or numeric 0/1")


def _outcome(row: Mapping[str, Any], key: tuple[str, str, int]) -> dict[str, bool]:
    """Extract the canonical boolean outcome fields from an episode row.

    Returns:
        A mapping of canonical outcome field names to boolean values.
    """
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
    """Index episode rows by tuple, raising on duplicate or ambiguous keys.

    Returns:
        A dictionary indexing episode rows by their planner/scenario/seed tuple.
    """
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
    """Verify a rerun campaign manifest against the resolved campaign expectation."""
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
    """Load and verify one rerun output's episode rows against its expectation.

    Returns:
        A dictionary indexing the verified rerun episode rows by tuple.
    """
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
    """Walk nested mappings along a key path, returning None if any step is absent.

    Returns:
        The nested mapping at the key path, or None if any step is absent.
    """
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _verify_rerun_row(  # noqa: C901
    row: Mapping[str, Any], key: tuple[str, str, int]
) -> None:
    """Verify a rerun episode row's provenance, params, traces, and outcome."""
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


def real_reexport_arms() -> tuple[RealReexportArm, ...]:
    """Return the immutable three-arm #5756 provenance contract."""

    return REAL_REEXPORT_ARMS


def real_reexport_request_tuples() -> set[tuple[str, str, int]]:
    """Return the requested 90 planner/scenario/seed tuples."""

    tuples: set[tuple[str, str, int]] = set()
    for arm in REAL_REEXPORT_ARMS:
        tuples.update(arm.tuples)
    return tuples


def _first_manifest_value(manifest: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    """Return the first present value from a list of nested manifest paths."""

    for path in paths:
        current: Any = manifest
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            return current
    return None


def _required_manifest_text(
    manifest: Mapping[str, Any], label: str, *paths: tuple[str, ...]
) -> str:
    """Read one required manifest identity value as non-empty text.

    Returns:
        The stripped manifest value.
    """

    value = _first_manifest_value(manifest, *paths)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise RealReexportBindingError(f"real-arm manifest lacks {label}")
    normalized = str(value).strip()
    if not normalized:
        raise RealReexportBindingError(f"real-arm manifest lacks {label}")
    return normalized


def _manifest_seed_list(manifest: Mapping[str, Any]) -> list[int]:
    """Read the resolved seed list from a campaign manifest.

    Returns:
        The unique resolved seed values in manifest order.
    """

    value = _first_manifest_value(
        manifest,
        ("seed_policy", "resolved_seeds"),
        ("seed_policy", "seeds"),
        ("resolved_seeds",),
        ("seeds",),
    )
    if not isinstance(value, list) or any(isinstance(seed, bool) for seed in value):
        raise RealReexportBindingError("real-arm manifest lacks a list of resolved seeds")
    try:
        seeds = [int(seed) for seed in value]
    except (TypeError, ValueError) as exc:
        raise RealReexportBindingError("real-arm manifest resolved seeds are not integers") from exc
    if len(seeds) != len(set(seeds)):
        raise RealReexportBindingError("real-arm manifest resolved seeds contain duplicates")
    return seeds


def _manifest_planner(manifest: Mapping[str, Any]) -> str:
    """Read the single planner key from a campaign manifest.

    Returns:
        The planner key.
    """

    planners = manifest.get("planners")
    if isinstance(planners, list):
        if len(planners) != 1 or not isinstance(planners[0], Mapping):
            raise RealReexportBindingError("real-arm manifest must declare exactly one planner")
        value = planners[0].get("key") or planners[0].get("planner")
    else:
        value = _first_manifest_value(manifest, ("planner",), ("algorithm",))
    if not isinstance(value, str) or not value.strip():
        raise RealReexportBindingError("real-arm manifest lacks a planner key")
    return value.strip()


def _manifest_config_path(manifest: Mapping[str, Any]) -> str | None:
    """Extract the config path from manifest fields or the recorded invocation.

    Returns:
        The normalized repository-relative config path, or ``None`` when absent.
    """

    direct = _first_manifest_value(
        manifest,
        ("config_path",),
        ("config", "path"),
        ("config_file",),
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for field in ("invoked_command", "command"):
        command = manifest.get(field)
        if not isinstance(command, str):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise RealReexportBindingError(
                f"manifest {field} is not a valid shell command: {exc}"
            ) from exc
        for index, token in enumerate(tokens[:-1]):
            if token == "--config":
                value = tokens[index + 1].strip()
                if value:
                    return value
    return None


def _source_job_id(manifest: Mapping[str, Any], source_root: Path) -> str | None:
    """Recover a missing scheduler job id from the immutable retrieval root.

    Returns:
        The unique job id found in the source identity, or ``None`` when ambiguous.
    """

    candidates = [str(source_root)]
    for field in ("results_root", "output_root", "campaign_id", "invoked_command", "command"):
        value = manifest.get(field)
        if isinstance(value, str):
            candidates.append(value)
    job_ids = {
        match for value in candidates for match in re.findall(r"(?<!\d)job[-_](\d+)(?!\d)", value)
    }
    if len(job_ids) == 1:
        return next(iter(job_ids))
    return None


def _compact_source_evidence(source_root: Path, manifest_path: Path) -> list[dict[str, str]]:
    """Hash the compact source records that bind a real campaign invocation.

    Returns:
        Hash records for the manifest and available compact run/preflight records.
    """

    candidates = [manifest_path]
    summary = source_root / "run_summary.yaml"
    if summary.is_file():
        candidates.append(summary)
    preflight = sorted(source_root.rglob("preflight/validate_config.json"))
    if len(preflight) == 1:
        candidates.append(preflight[0])
    elif len(preflight) > 1:
        raise RealReexportBindingError(
            f"source root must contain at most one preflight validate record; found {len(preflight)}"
        )
    return [
        {
            "kind": "campaign_manifest" if path == manifest_path else path.name,
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for path in candidates
    ]


def _manifest_job_id(manifest: Mapping[str, Any], source_root: Path) -> str:
    """Read a scheduler job id, recovering it from immutable source metadata when needed.

    Returns:
        The normalized scheduler job id.
    """

    declared_job_id = _first_manifest_value(
        manifest,
        ("job_id",),
        ("slurm_job_id",),
        ("provenance", "job_id"),
        ("provenance", "slurm_job_id"),
        ("run_provenance", "job_id"),
        ("run_provenance", "slurm_job_id"),
    )
    if declared_job_id is None or (
        isinstance(declared_job_id, str) and not declared_job_id.strip()
    ):
        recovered_job_id = _source_job_id(manifest, source_root)
        if recovered_job_id is None:
            raise RealReexportBindingError("real-arm manifest lacks job_id")
        return recovered_job_id
    if isinstance(declared_job_id, bool) or not isinstance(declared_job_id, (str, int)):
        raise RealReexportBindingError("real-arm manifest lacks job_id")
    normalized = str(declared_job_id).strip()
    if not normalized:
        raise RealReexportBindingError("real-arm manifest lacks job_id")
    return normalized


def _read_config_evidence(  # noqa: C901, PLR0912, PLR0915
    evidence: Path | Mapping[str, Any] | None,
    *,
    arm: RealReexportArm,
    manifest: Mapping[str, Any],
    source_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Load and verify an independent config-path/hash evidence descriptor.

    Returns:
        A normalized, verified config provenance mapping.
    """

    descriptor: Mapping[str, Any] | None
    descriptor_path: Path | None = None
    auto_manifest_evidence = False
    if evidence is None:
        candidate = _first_manifest_value(
            manifest,
            ("config_provenance",),
            ("provenance", "config"),
        )
        descriptor = candidate if isinstance(candidate, Mapping) else None
        auto_manifest_evidence = descriptor is None
    elif isinstance(evidence, Mapping):
        descriptor = evidence
    else:
        descriptor_path = Path(evidence)
        if not descriptor_path.is_file():
            raise RealReexportBindingError(f"config evidence is unavailable: {descriptor_path}")
        try:
            parsed = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise RealReexportBindingError(
                f"cannot read config evidence {descriptor_path}: {exc}"
            ) from exc
        if not isinstance(parsed, Mapping):
            raise RealReexportBindingError("config evidence must contain an object")
        descriptor = parsed

    if descriptor is None and auto_manifest_evidence:
        descriptor = {}
    if descriptor is None:
        raise RealReexportBindingError(
            f"{arm.key} has no independent manifest/config evidence descriptor"
        )

    nested = descriptor.get("config_provenance")
    if isinstance(nested, Mapping):
        descriptor = nested
    config_name = descriptor.get("config_name") or descriptor.get("name")
    if config_name is not None and config_name != arm.config_name:
        raise RealReexportBindingError(f"{arm.key} config name evidence mismatch")
    config_path_value = (
        descriptor.get("config_path")
        or descriptor.get("path")
        or descriptor.get("file")
        or _first_manifest_value(
            descriptor,
            ("config_reconstruction", "reconstructed_file"),
        )
        or _manifest_config_path(manifest)
    )
    if not isinstance(config_path_value, str) or not config_path_value.strip():
        raise RealReexportBindingError(f"{arm.key} config path evidence is unavailable")
    if config_path_value != arm.config_path:
        raise RealReexportBindingError(f"{arm.key} config path evidence mismatch")

    expected_config_hash = _first_manifest_value(
        manifest,
        ("config_hash",),
        ("config", "hash"),
        ("provenance", "config_hash"),
        ("run_identification", "config_hash_from_run"),
    )
    evidence_config_hash = descriptor.get("config_hash") or descriptor.get("run_config_hash")
    if evidence_config_hash is None:
        evidence_config_hash = _first_manifest_value(
            descriptor,
            ("provenance", "config_hash"),
            ("run_identification", "config_hash_from_run"),
        )
    if not isinstance(expected_config_hash, str) or not expected_config_hash:
        raise RealReexportBindingError(f"{arm.key} manifest lacks config_hash")
    if evidence_config_hash is None and auto_manifest_evidence:
        evidence_config_hash = expected_config_hash
    if evidence_config_hash != expected_config_hash:
        raise RealReexportBindingError(f"{arm.key} config hash evidence mismatch")

    expected_sha = descriptor.get("sha256") or descriptor.get("config_sha256")
    if expected_sha is None:
        expected_sha = _first_manifest_value(
            descriptor,
            ("config_provenance", "sha256"),
            ("run_identification", "config_sha256"),
            ("config_reconstruction", "sha256"),
        )
    if expected_sha is None:
        expected_sha = _first_manifest_value(
            manifest,
            ("config_sha256",),
            ("config", "sha256"),
            ("provenance", "config_sha256"),
        )
    config_source_value = descriptor.get("source_path") or descriptor.get("local_path")
    config_path = Path(str(config_source_value or config_path_value))
    if not config_path.is_absolute():
        candidates = [
            config_path,
            source_root / config_path,
            descriptor_path.parent / config_path if descriptor_path else Path(),
        ]
        config_path = next(
            (candidate for candidate in candidates if candidate.is_file()), config_path
        )
    actual_sha: str | None = None
    evidence_status = "config_file"
    evidence_paths: list[dict[str, str]] = []
    if config_path.is_file() and not auto_manifest_evidence:
        actual_sha = _sha256_file(config_path)
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or len(expected_sha) != 64:
                raise RealReexportBindingError(
                    f"{arm.key} config evidence has an invalid SHA-256 digest"
                )
            _require_digest(actual_sha, expected_sha.lower(), f"{arm.key} config")
        evidence_paths = [
            {
                "kind": "config_file",
                "path": str(config_path),
                "sha256": actual_sha,
            }
        ]
    elif auto_manifest_evidence:
        evidence_status = "campaign_manifest_invocation"
        evidence_paths = _compact_source_evidence(source_root, manifest_path)
        if not any(item["kind"] == "campaign_manifest" for item in evidence_paths):
            raise RealReexportBindingError(f"{arm.key} campaign manifest evidence is unavailable")
    else:
        raise RealReexportBindingError(f"{arm.key} config source is unavailable: {config_path}")
    return {
        "schema_version": REAL_REEXPORT_CONFIG_EVIDENCE_SCHEMA,
        "config_name": arm.config_name,
        "config_path": arm.config_path,
        "config_hash": expected_config_hash,
        "config_sha256": actual_sha,
        "evidence_status": evidence_status,
        "evidence_paths": evidence_paths,
        "evidence_path": str(descriptor_path) if descriptor_path else None,
    }


def _discover_real_arm_inputs(
    root: Path, arm: RealReexportArm
) -> tuple[Path, Path, dict[str, Any]]:
    """Find exactly one campaign manifest and one arm episode JSONL.

    Returns:
        The manifest path, episode JSONL path, and parsed manifest.
    """

    root = root.resolve()
    if not root.is_dir():
        raise RealReexportBindingError(f"{arm.key} source root is unavailable: {root}")
    manifests = sorted(root.rglob("campaign_manifest.json"))
    if len(manifests) != 1:
        raise RealReexportBindingError(
            f"{arm.key} source must contain exactly one campaign_manifest.json; found {len(manifests)}"
        )
    manifest_path = manifests[0]
    manifest = _read_json_object(manifest_path)
    episodes = sorted(root.rglob("runs/*/episodes.jsonl"))
    if len(episodes) != 1:
        raise RealReexportBindingError(
            f"{arm.key} source must contain exactly one runs/*/episodes.jsonl; found {len(episodes)}"
        )
    return manifest_path, episodes[0], manifest


def _read_real_rows_with_raw_bytes(
    data: bytes, label: str
) -> list[tuple[dict[str, Any], bytes, int]]:
    """Parse real-arm JSONL while retaining each source line for digest binding.

    Returns:
        Parsed row, exact UTF-8 source line bytes, and one-based source line number.
    """

    rows: list[tuple[dict[str, Any], bytes, int]] = []
    for line_number, raw_line in enumerate(data.splitlines(keepends=True), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RealReexportBindingError(f"{label}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise RealReexportBindingError(f"{label}:{line_number}: expected a JSON object")
        rows.append((row, raw_line, line_number))
    if not rows:
        raise RealReexportBindingError(f"{label}: contains no rows")
    return rows


def _verify_real_arm_manifest(
    manifest: Mapping[str, Any], *, arm: RealReexportArm, source_root: Path
) -> dict[str, str]:
    """Verify job, commit, config, planner, scenario, and seed identity.

    Returns:
        The normalized identity fields used by row-level checks and receipts.
    """

    job_id = _manifest_job_id(manifest, source_root)
    if job_id != arm.job_id:
        raise RealReexportBindingError(
            f"{arm.key} job mismatch: expected {arm.job_id}, got {job_id}"
        )
    commit = _required_manifest_text(
        manifest,
        "execution commit",
        ("git", "commit"),
        ("execution_commit",),
        ("git_hash",),
        ("provenance", "execution_commit"),
    )
    if commit != EXECUTION_COMMIT:
        raise RealReexportBindingError(f"{arm.key} execution commit mismatch")
    config_name = _required_manifest_text(
        manifest,
        "config name",
        ("name",),
        ("config_name",),
        ("config", "name"),
    )
    if config_name != arm.config_name:
        raise RealReexportBindingError(
            f"{arm.key} config name mismatch: expected {arm.config_name}, got {config_name}"
        )
    config_path = _manifest_config_path(manifest)
    if config_path != arm.config_path:
        raise RealReexportBindingError(f"{arm.key} config path mismatch")
    scenarios = _first_manifest_value(
        manifest,
        ("scenario_candidates",),
        ("scenarios",),
    )
    if scenarios != [arm.scenario_id]:
        raise RealReexportBindingError(f"{arm.key} scenario candidate mismatch")
    seeds = _manifest_seed_list(manifest)
    if seeds != list(arm.seeds):
        raise RealReexportBindingError(f"{arm.key} resolved seed set mismatch")
    planner = _manifest_planner(manifest)
    if planner != arm.planner:
        raise RealReexportBindingError(f"{arm.key} planner mismatch")
    scenario_matrix = _first_manifest_value(
        manifest,
        ("scenario_matrix",),
        ("scenario", "matrix"),
    )
    if scenario_matrix != _SCENARIO_MATRIX.as_posix():
        raise RealReexportBindingError(f"{arm.key} scenario matrix mismatch")
    config_hash = _first_manifest_value(
        manifest,
        ("config_hash",),
        ("config", "hash"),
        ("provenance", "config_hash"),
        ("run_identification", "config_hash_from_run"),
    )
    if not isinstance(config_hash, str) or not config_hash.strip():
        raise RealReexportBindingError(f"{arm.key} manifest lacks config_hash")
    return {
        "job_id": job_id,
        "execution_commit": commit,
        "config_name": config_name,
        "config_hash": config_hash.strip(),
        "campaign": str(manifest.get("campaign_id") or manifest.get("name") or arm.key),
    }


def _real_outcome_index(  # noqa: C901
    expected_outcomes: Path | Mapping[Any, Any],
) -> dict[tuple[str, str, int], dict[str, bool]]:
    """Load release outcome evidence keyed by the requested tuple.

    Returns:
        Canonical outcome booleans keyed by planner/scenario/seed.
    """

    if isinstance(expected_outcomes, Path):
        payload = _read_json_object(expected_outcomes)
        rows = payload.get("rows")
    else:
        rows = expected_outcomes.get("rows") if isinstance(expected_outcomes, Mapping) else None
        if rows is None:
            rows = expected_outcomes
    if isinstance(rows, Mapping):
        indexed: dict[tuple[str, str, int], dict[str, bool]] = {}
        for raw_key, value in rows.items():
            if (
                not isinstance(raw_key, tuple)
                or len(raw_key) != 3
                or not isinstance(value, Mapping)
            ):
                raise RealReexportBindingError(
                    "expected outcome mapping keys must be tuple identities"
                )
            key = (str(raw_key[0]), str(raw_key[1]), int(raw_key[2]))
            if set(_OUTCOME_FIELDS) <= set(value):
                indexed[key] = {
                    field: _strict_bool(value[field], field=field, key=key)
                    for field in _OUTCOME_FIELDS
                }
            else:
                raise RealReexportBindingError(f"expected outcome row {key!r} lacks outcome fields")
        return indexed
    if not isinstance(rows, list):
        raise RealReexportBindingError("expected outcomes must contain a rows list")
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RealReexportBindingError("expected outcome row must be an object")
        key = _tuple_from_request(row)
        if key in indexed:
            raise RealReexportBindingError(f"duplicate expected outcome tuple: {key!r}")
        if isinstance(row.get("metrics"), Mapping) and isinstance(row.get("outcome"), Mapping):
            indexed[key] = _outcome(row, key)
        else:
            if not set(_OUTCOME_FIELDS) <= set(row):
                raise RealReexportBindingError(f"expected outcome row {key!r} lacks outcome fields")
            indexed[key] = {
                field: _strict_bool(row[field], field=field, key=key) for field in _OUTCOME_FIELDS
            }
    return indexed


def _verify_real_row_config_hash(
    row: Mapping[str, Any], *, key: tuple[str, str, int], params: Mapping[str, Any]
) -> None:
    """Verify the row hash binds the exact scenario parameters."""

    row_config_hash = row.get("config_hash")
    if not isinstance(row_config_hash, str) or not row_config_hash.strip():
        raise RealReexportBindingError(f"real rerun row {key!r} lacks scenario config hash")
    if row_config_hash != _config_hash(dict(params)):
        raise RealReexportBindingError(f"real rerun row {key!r} scenario/config hash mismatch")


def _verify_real_algorithm_config(
    metadata: Mapping[str, Any], *, key: tuple[str, str, int]
) -> None:
    """Verify optional algorithm configuration provenance when the row provides it."""

    algorithm_config_hash = metadata.get("config_hash")
    algorithm_config = metadata.get("config")
    if algorithm_config_hash is None:
        return
    if not isinstance(algorithm_config_hash, str) or not algorithm_config_hash.strip():
        raise RealReexportBindingError(f"real rerun row {key!r} algorithm config hash is invalid")
    if (
        isinstance(algorithm_config, Mapping)
        and _config_hash(dict(algorithm_config)) != algorithm_config_hash
    ):
        raise RealReexportBindingError(f"real rerun row {key!r} algorithm config hash mismatch")


def _verify_real_rerun_row(row: Mapping[str, Any], *, key: tuple[str, str, int]) -> None:
    """Verify a real row's identity, pinned commit, native trace, and config binding."""

    if row.get("git_hash") != EXECUTION_COMMIT:
        raise RealReexportBindingError(f"real rerun row {key!r} execution commit mismatch")
    params = row.get("scenario_params")
    if not isinstance(params, Mapping) or params.get("algo") != key[0]:
        raise RealReexportBindingError(f"real rerun row {key!r} planner/config mismatch")
    _verify_real_row_config_hash(row, key=key, params=params)
    metadata = row.get("algorithm_metadata")
    if not isinstance(metadata, Mapping):
        raise RealReexportBindingError(f"real rerun row {key!r} lacks algorithm_metadata")
    kinematics = metadata.get("planner_kinematics")
    if (
        not isinstance(kinematics, Mapping)
        or kinematics.get("robot_kinematics") != "differential_drive"
    ):
        raise RealReexportBindingError(
            f"real rerun row {key!r} is not differential-drive execution"
        )
    trace = metadata.get("simulation_step_trace")
    if (
        not isinstance(trace, Mapping)
        or trace.get("schema_version") != "simulation-step-trace.v1"
        or not isinstance(trace.get("steps"), list)
        or not trace["steps"]
    ):
        raise RealReexportBindingError(f"real rerun row {key!r} lacks a non-empty simulation trace")
    _verify_real_algorithm_config(metadata, key=key)
    _outcome(row, key)


def _real_row_trace_receipt(  # noqa: PLR0913
    row: Mapping[str, Any],
    *,
    arm: RealReexportArm,
    source_root: Path,
    episodes_path: Path,
    row_index: int,
    raw_row_bytes: bytes,
    manifest_identity: Mapping[str, str],
    config_evidence: Mapping[str, Any],
    normalized_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one real row and return its payload plus enriched receipt.

    Returns:
        The normalized trace payload and its transformation receipt.
    """

    key = (arm.planner, arm.scenario_id, int(row["seed"]))
    raw_row_sha256 = _sha256_bytes(raw_row_bytes)
    provenance = {
        "job_id": arm.job_id,
        "campaign": manifest_identity["campaign"],
        "config_name": arm.config_name,
        "config_path": arm.config_path,
        "config_hash": config_evidence["config_hash"],
        "config_sha256": config_evidence["config_sha256"],
        "execution_commit": EXECUTION_COMMIT,
    }
    metadata = row.get("algorithm_metadata")
    algorithm_config_hash = metadata.get("config_hash") if isinstance(metadata, Mapping) else None
    with tempfile.TemporaryDirectory(prefix="issue-6411-row-") as isolated_dir:
        isolated_path = Path(isolated_dir) / "episode.jsonl"
        isolated_path.write_bytes(raw_row_bytes)
        payload, transform_receipt = build_simulation_trace_export_with_receipt(
            isolated_path,
            planner_id=arm.planner,
            scenario_id=arm.scenario_id,
            source_signature=raw_row_sha256,
            provenance=provenance,
        )
    trace = load_simulation_trace_export_from_payload(payload)
    if trace.source.episode_id != str(row.get("episode_id")):
        raise RealReexportBindingError(f"real rerun row {key!r} trace episode ID mismatch")
    if normalized_path is not None:
        _write_json(normalized_path, payload)
        _require_digest(
            _sha256_file(normalized_path),
            transform_receipt["normalized_trace_sha256"],
            f"normalized trace {key!r}",
        )
    removed_fields = transform_receipt["removed_fields"]
    removed_field_representatives: dict[str, dict[str, Any]] = {}
    removed_field_counts: dict[str, int] = {}
    for item in removed_fields:
        field = str(item["field"])
        removed_field_counts[field] = removed_field_counts.get(field, 0) + 1
        removed_field_representatives.setdefault(field, dict(item))
    compact_removed_fields = [
        removed_field_representatives[field] for field in sorted(removed_field_representatives)
    ]
    removed_field_paths_sha256 = _sha256_bytes(_canonical_bytes(removed_fields))
    enriched = {
        "schema_version": REAL_REEXPORT_RECEIPT_SCHEMA,
        "status": "complete",
        "arm": arm.key,
        "job_id": arm.job_id,
        "execution_commit": EXECUTION_COMMIT,
        "campaign": manifest_identity["campaign"],
        "config": dict(config_evidence),
        "source": {
            "root": str(source_root),
            "episodes_path": str(episodes_path),
            "episodes_sha256": _sha256_file(episodes_path),
            "row_index": row_index,
            "episode_id": str(row["episode_id"]),
        },
        "planner": key[0],
        "scenario_id": key[1],
        "seed": key[2],
        "raw_trace_sha256": raw_row_sha256,
        "row_config_hash": str(row["config_hash"]),
        "algorithm_config_hash": algorithm_config_hash,
        "normalized_trace_sha256": transform_receipt["normalized_trace_sha256"],
        "trace_schema_version": transform_receipt["trace_schema_version"],
        "normalization_policy": transform_receipt["normalization_policy"],
        "removed_fields": compact_removed_fields,
        "removed_field_counts": removed_field_counts,
        "removed_field_paths_sha256": removed_field_paths_sha256,
        "removed_field_count": len(removed_fields),
        "semantic_payload_unchanged": transform_receipt["semantic_payload_unchanged"],
        "normalized_trace_path": str(normalized_path) if normalized_path else None,
    }
    return payload, enriched


def load_simulation_trace_export_from_payload(payload: Mapping[str, Any]) -> Any:
    """Validate a normalized payload without exposing the schema helper publicly.

    Returns:
        The typed simulation trace export.
    """

    return simulation_trace_export_from_dict(payload)


def bind_real_reexport_arms(  # noqa: C901, PLR0912, PLR0915
    arm_roots: Mapping[str, Path],
    *,
    expected_outcomes: Path | Mapping[Any, Any],
    config_evidence: Mapping[str, Path | Mapping[str, Any]] | None = None,
    request_manifest: Path | None = None,
    normalized_output_dir: Path | None = None,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Bind and normalize the three real #5756 arms without making a package.

    Every arm must provide one unambiguous campaign manifest and one episode JSONL
    tree.  The manifest/config descriptors must independently identify the pinned
    job, config, planner, scenario, 30-seed matrix, and execution commit.  Release
    outcomes are required so the two doorway exceptions are represented as
    ``outcome_mismatch`` / ``not_admitted`` rather than silently dropped or relabeled.
    When ``normalized_output_dir`` is supplied, only normalized traces are written;
    this function never creates a complete 90-row package.

    Returns:
        A compact binding receipt covering all 90 source rows and the explicit 88+2
        outcome boundary.
    """

    expected_keys = {arm.key for arm in REAL_REEXPORT_ARMS}
    if set(arm_roots) != expected_keys:
        raise RealReexportBindingError(
            f"real arm roots must cover exactly {sorted(expected_keys)}; got {sorted(arm_roots)}"
        )
    if config_evidence is not None and set(config_evidence) != expected_keys:
        raise RealReexportBindingError("config evidence must cover exactly the three real arms")

    request_tuples = real_reexport_request_tuples()
    request_digest = None
    if request_manifest is not None:
        request_manifest = request_manifest.resolve()
        if not request_manifest.is_file():
            raise RealReexportBindingError(f"request manifest is unavailable: {request_manifest}")
        request_payload = _read_json_object(request_manifest)
        if request_payload.get("schema_version") != "issue_5446_trace_reexport_list.v1":
            raise RealReexportBindingError("request manifest schema mismatch")
        rows = request_payload.get("tuples")
        if not isinstance(rows, list) or request_payload.get("n_tuples") != 90:
            raise RealReexportBindingError("request manifest must declare exactly 90 tuples")
        indexed = {_tuple_from_request(row) for row in rows if isinstance(row, Mapping)}
        if indexed != request_tuples or len(rows) != 90:
            raise RealReexportBindingError(
                "request manifest does not prove the exact 90-tuple contract"
            )
        request_digest = _sha256_file(request_manifest)

    release_outcomes = _real_outcome_index(expected_outcomes)
    if set(release_outcomes) != request_tuples:
        raise RealReexportBindingError(
            "release outcome evidence does not cover the exact 90 tuples"
        )

    staging_dir: Path | None = None
    final_output: Path | None = None
    if normalized_output_dir is not None:
        final_output = normalized_output_dir.resolve()
        if final_output.exists():
            raise RealReexportBindingError(
                f"normalized output already exists; refusing to overwrite: {final_output}"
            )
        final_output.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{final_output.name}.staging-", dir=final_output.parent)
        )

    receipt_rows: list[dict[str, Any]] = []
    arm_receipts: list[dict[str, Any]] = []
    try:
        for arm in REAL_REEXPORT_ARMS:
            source_root = Path(arm_roots[arm.key]).resolve()
            manifest_path, episodes_path, manifest = _discover_real_arm_inputs(source_root, arm)
            manifest_identity = _verify_real_arm_manifest(
                manifest, arm=arm, source_root=source_root
            )
            config_descriptor = config_evidence.get(arm.key) if config_evidence else None
            config = _read_config_evidence(
                config_descriptor,
                arm=arm,
                manifest=manifest,
                source_root=source_root,
                manifest_path=manifest_path,
            )
            rows_with_raw = _read_real_rows_with_raw_bytes(
                episodes_path.read_bytes(), str(episodes_path)
            )
            rows = [row for row, _raw_bytes, _line_number in rows_with_raw]
            indexed = _index_rows(rows, planner_hint=arm.planner)
            raw_by_key = {
                _row_tuple(row, planner_hint=arm.planner): (raw_bytes, line_number)
                for row, raw_bytes, line_number in rows_with_raw
            }
            if set(indexed) != arm.tuples:
                missing = sorted(arm.tuples - set(indexed))
                extra = sorted(set(indexed) - arm.tuples)
                raise RealReexportBindingError(
                    f"{arm.key} tuple set mismatch; missing={missing[:3]}, extra={extra[:3]}"
                )
            arm_rows: list[dict[str, Any]] = []
            for key in sorted(arm.tuples):
                row = indexed[key]
                raw_row_bytes, source_line_number = raw_by_key[key]
                _verify_real_rerun_row(row, key=key)
                rerun_outcome = _outcome(row, key)
                release_outcome = release_outcomes[key]
                mismatch = rerun_outcome != release_outcome
                expected_exception = key[2] in arm.not_admitted_seeds
                if mismatch != expected_exception:
                    raise RealReexportBindingError(
                        f"{key!r} outcome boundary mismatch; expected exception={expected_exception}"
                    )
                outcome_status = "outcome_mismatch" if mismatch else "outcome_match"
                admission_status = "not_admitted" if mismatch else "admitted"
                normalized_path = None
                if staging_dir is not None:
                    normalized_path = (
                        staging_dir / arm.key / arm.scenario_id / f"seed-{key[2]}.json"
                    )
                    normalized_path.parent.mkdir(parents=True, exist_ok=True)
                _payload, row_receipt = _real_row_trace_receipt(
                    row,
                    arm=arm,
                    source_root=source_root,
                    episodes_path=episodes_path,
                    row_index=source_line_number,
                    raw_row_bytes=raw_row_bytes,
                    manifest_identity=manifest_identity,
                    config_evidence=config,
                    normalized_path=normalized_path,
                )
                if (
                    normalized_path is not None
                    and staging_dir is not None
                    and final_output is not None
                ):
                    row_receipt["normalized_trace_path"] = str(
                        final_output / normalized_path.relative_to(staging_dir)
                    )
                row_receipt.update(
                    {
                        "release_outcome": _canonical_outcome_from_row(release_outcome),
                        "rerun_outcome": _canonical_outcome_from_row(rerun_outcome),
                        "outcome_status": outcome_status,
                        "admission_status": admission_status,
                    }
                )
                arm_rows.append(row_receipt)
                receipt_rows.append(row_receipt)
            arm_receipts.append(
                {
                    "arm": arm.key,
                    "job_id": arm.job_id,
                    "planner": arm.planner,
                    "scenario_id": arm.scenario_id,
                    "config": config,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "episodes_path": str(episodes_path),
                    "episodes_sha256": _sha256_file(episodes_path),
                    "n_rows": len(arm_rows),
                }
            )

        mismatch_rows = [row for row in receipt_rows if row["outcome_status"] == "outcome_mismatch"]
        expected_mismatches = {
            ("ppo", "classic_doorway_medium", seed) for seed in REAL_REEXPORT_EXCEPTION_SEEDS
        }
        observed_mismatches = {
            (row["planner"], row["scenario_id"], row["seed"]) for row in mismatch_rows
        }
        if observed_mismatches != expected_mismatches or len(receipt_rows) != 90:
            raise RealReexportBindingError("real re-export exception boundary is not exactly 88+2")
        receipt: dict[str, Any] = {
            "schema_version": REAL_REEXPORT_BINDING_SCHEMA,
            "status": "complete",
            "execution_commit": EXECUTION_COMMIT,
            "trace_schema_version": receipt_rows[0]["trace_schema_version"],
            "normalization_policy": receipt_rows[0]["normalization_policy"],
            "request_contract": {
                "schema_version": "issue_5446_trace_reexport_list.v1",
                "n_tuples": len(request_tuples),
                "sha256": request_digest,
            },
            "arms": arm_receipts,
            "rows": receipt_rows,
            "summary": {
                "n_rows": len(receipt_rows),
                "n_admitted": len(receipt_rows) - len(mismatch_rows),
                "n_not_admitted": len(mismatch_rows),
            },
            "exception_boundary": [
                {
                    "planner": "ppo",
                    "scenario_id": "classic_doorway_medium",
                    "seed": seed,
                    "outcome_status": "outcome_mismatch",
                    "admission_status": "not_admitted",
                }
                for seed in REAL_REEXPORT_EXCEPTION_SEEDS
            ],
            "package_status": "not_created; package assembly belongs to issue #6412",
        }
        if receipt_path is not None:
            _write_json(receipt_path.resolve(), receipt)
        if staging_dir is not None and final_output is not None:
            os.replace(staging_dir, final_output)
            staging_dir = None
        return receipt
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)


def _release_selection(
    release_rows: Mapping[str, list[dict[str, Any]]],
    requests: Mapping[tuple[str, str, int], str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Select and verify the release rows that cover every requested tuple.

    Returns:
        A dictionary of selected release rows indexed by requested tuple.
    """
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
    """Build the versioned expected-outcome payload from selected release rows.

    Returns:
        The versioned expected-outcome payload with provenance and rows.
    """
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
    """Return the canonical SHA-256 digest of a single episode row."""
    return _sha256_bytes(_canonical_bytes(row, newline=True))


def _trace_uri(key: tuple[str, str, int]) -> str:
    """Return the durable relative trace URI for a planner/scenario/seed tuple."""
    planner, scenario, seed = key
    return f"traces/{planner}/{scenario}/seed-{seed}.json"


def _write_json(path: Path, payload: Any) -> None:
    """Write a payload as canonical newline-terminated JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload, newline=True))


def _validate_staged_package(  # noqa: C901
    root: Path, expected_rows: int
) -> dict[str, str]:
    """Validate a staged package's receipts and traces, returning all file digests.

    Returns:
        A mapping of every staged file's relative path to its SHA-256 digest.
    """
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
    """Return a verified completion marker for an existing package, or None if absent."""
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
    """Report whether two paths are equal or one is nested inside the other.

    Returns:
        True when the paths are equal or one is nested inside the other.
    """
    return left == right or left in right.parents or right in left.parents


def _validate_output_path(output_dir: Path, input_paths: Mapping[str, Path]) -> None:
    """Reject output paths that overlap inputs or exist without a complete package."""
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
    """Atomically install a staged package, backing up and restoring on failure."""
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
    """Remove the staging directory, surfacing errors only when the run completed."""
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


def _canonical_outcome_from_row(outcome_row: Mapping[str, Any]) -> str:
    """Reduce four release-outcome booleans to one canonical resolver label.

    Mirrors the resolver's outcome priority (collision > timeout > route > success)
    so the emitted ``expected_release_outcome`` / ``rerun_outcome`` labels match the
    labels the resolver itself derives.  The packager already proved the packaged row's
    release and rerun outcomes are identical, so a single label is authoritative here.

    Returns:
        The canonical outcome label.
    """
    for key in ("collision_event", "timeout_event", "route_complete", "success"):
        if outcome_row.get(key) is True:
            return key
    raise TraceReexportPackagingError(
        f"expected-outcome row has no canonical true flag: {dict(outcome_row)!r}"
    )


def _index_outcome_rows(
    outcomes_payload: Mapping[str, Any],
) -> dict[tuple[str, str, int], str]:
    """Index expected-outcome rows by ``(planner, scenario, seed)`` to canonical labels.

    Returns:
        Canonical outcome label per episode tuple.
    """
    outcome_by_key: dict[tuple[str, str, int], str] = {}
    for row in outcomes_payload.get("rows", []):
        if not isinstance(row, Mapping):
            raise TraceReexportPackagingError("expected-outcome row must be an object")
        key = (str(row["planner"]), str(row["scenario_id"]), int(row["seed"]))
        outcome_by_key[key] = _canonical_outcome_from_row(row)
    return outcome_by_key


def default_resolver_mapping_path(package_dir: Path) -> Path:
    """Return a resolver-receipt path beside, never inside, a complete package.

    The completion marker covers every package file. Writing a derived receipt below
    ``package_dir`` would therefore invalidate the package immediately after a
    successful conversion.

    Returns:
        A sibling path named ``<package>.resolver_mapping_receipt.json``.
    """
    package_dir = package_dir.resolve()
    return package_dir.with_name(f"{package_dir.name}.resolver_mapping_receipt.json")


def _validate_resolver_output_path(package_dir: Path, output_path: Path | None) -> Path | None:
    if output_path is None:
        return None
    resolved_output = output_path.resolve()
    if resolved_output == package_dir or package_dir in resolved_output.parents:
        raise TraceReexportPackagingError(
            "resolver mapping receipt must be written outside the immutable complete package"
        )
    return resolved_output


def build_resolver_mapping_receipt(
    package_dir: Path, *, output_path: Path | None = None
) -> dict[str, Any]:
    """Convert a complete #5756 trace package into the resolver's mapping receipt.

    The packager emits ``issue_5756_trace_reexport_mapping_receipt.v1`` (release/rerun
    row digests and relative trace URIs).  The candidate-trace resolver (#5615) and the
    worked-example renderer (#5756) consume ``issue_5756_trace_mapping_receipt.v1``
    (canonical per-row outcomes, absolute trace artifact URIs, resolver provenance).
    This fail-closed adapter joins the two: it requires a complete package, derives each
    row's canonical release outcome from ``expected_outcomes.json`` (the packager
    already verified release==rerun outcome for every packaged row), rewrites the
    relative ``trace_uri`` to the absolute ``trace_artifact_uri`` the resolver opens,
    and rebuilds the resolver-pinned provenance from the package's frozen provenance.

    Args:
        package_dir: A directory previously produced by :func:`package_trace_reexport`.
        output_path: When given, write the JSON receipt here (canonical form). The
            destination must be outside ``package_dir`` so the package completion
            marker remains valid.

    Returns:
        The ``issue_5756_trace_mapping_receipt.v1`` payload.
    """
    package_dir = package_dir.resolve()
    if _verify_complete_output(package_dir) is None:
        raise TraceReexportPackagingError(f"package is not a complete trace package: {package_dir}")
    output_path = _validate_resolver_output_path(package_dir, output_path)
    receipt = _read_json_object(package_dir / "mapping_receipt.json")
    if receipt.get("schema_version") != MAPPING_RECEIPT_SCHEMA:
        raise TraceReexportPackagingError("package mapping receipt has an unexpected schema")
    frozen = receipt.get("frozen_provenance")
    if not isinstance(frozen, Mapping):
        raise TraceReexportPackagingError("package mapping receipt has no frozen provenance")
    outcomes_payload = _read_json_object(package_dir / "expected_outcomes.json")
    if outcomes_payload.get("schema_version") != EXPECTED_OUTCOMES_SCHEMA:
        raise TraceReexportPackagingError("package expected-outcomes schema mismatch")
    outcome_by_key = _index_outcome_rows(outcomes_payload)
    mapping_rows: list[dict[str, Any]] = []
    for row in receipt.get("rows", []):
        if not isinstance(row, Mapping):
            raise TraceReexportPackagingError("mapping receipt row must be an object")
        key = (str(row["planner"]), str(row["scenario_id"]), int(row["seed"]))
        if key not in outcome_by_key:
            raise TraceReexportPackagingError(
                f"package receipt row {key!r} has no expected-outcome entry"
            )
        trace_uri = str(row.get("trace_uri") or "")
        trace_path = package_dir / trace_uri
        if not trace_path.is_file():
            raise TraceReexportPackagingError(
                f"package receipt row {key!r} trace artifact is missing: {trace_uri!r}"
            )
        outcome = outcome_by_key[key]
        mapping_rows.append(
            {
                "scenario_id": key[1],
                "planner": key[0],
                "seed": key[2],
                "episode_id": str(row["rerun_episode_id"]),
                "release_episode_id": str(row["release_episode_id"]),
                "expected_release_outcome": outcome,
                "rerun_outcome": outcome,
                "trace_artifact_uri": str(trace_path),
                "trace_sha256": str(row["trace_sha256"]).lower(),
            }
        )
    resolver_receipt: dict[str, Any] = {
        "schema_version": RESOLVER_MAPPING_SCHEMA,
        "n_rows": len(mapping_rows),
        "provenance": {
            "release_tag": RELEASE_TAG,
            "release_bundle_sha256": str(frozen["release_bundle_sha256"]),
            "report_commit": REPORT_COMMIT,
            "execution_commit": str(frozen["execution_commit"]),
            "canonical_campaign_config_sha256": str(frozen["canonical_campaign_config_sha256"]),
            "scenario_matrix_sha256": str(frozen["scenario_matrix_sha256"]),
            "checkpoint_sha256": str(frozen["ppo_checkpoint_sha256"]),
            "request_manifest_sha256": str(frozen["request_manifest_sha256"]),
        },
        "rows": mapping_rows,
    }
    # Round-trip through the resolver loader to fail closed on any drift before
    # emitting the canonical receipt bytes. The adapter validates the mapping against
    # the provenance derived from the package, not the production pins, so synthetic
    # fixtures (whose digests differ from the release pins) still exercise the path.
    from robot_sf.benchmark.candidate_trace_resolution import (  # noqa: PLC0415
        load_episode_mapping,
    )

    derived_provenance = dict(resolver_receipt["provenance"])
    # Validate an isolated canonical receipt before writing the requested destination.
    # Otherwise a validation failure leaves a caller-visible but unusable receipt behind.
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as handle:
        handle.write(_canonical_bytes(resolver_receipt, newline=True))
        temp_receipt = Path(handle.name)
    try:
        load_episode_mapping(temp_receipt, expected_provenance=derived_provenance)
    finally:
        temp_receipt.unlink(missing_ok=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, resolver_receipt)
    return resolver_receipt


def _issue_6814_identity_value(identity: object, field: str) -> object:
    """Read an identity field from a mapping or a small immutable record."""

    if isinstance(identity, Mapping):
        return identity.get(field)
    return getattr(identity, field, None)


def _issue_6814_required_text(value: object, field: str) -> str:
    """Require one non-empty identity string."""

    if not isinstance(value, str) or not value.strip():
        raise RealReexportBindingError(f"issue #6814 {field} must be a non-empty string")
    return value.strip()


def _issue_6814_sha256(value: object, field: str) -> str:
    """Require one lowercase SHA-256 digest."""

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RealReexportBindingError(f"issue #6814 {field} must be lowercase SHA-256")
    return value


def _issue_6814_read_package_json(package_root: Path, name: str) -> dict[str, Any]:
    """Read one immutable package JSON object."""

    path = package_root / name
    if not path.is_file():
        raise RealReexportBindingError(f"issue #6814 package artifact is unavailable: {name}")
    return _read_json_object(path)


def _parse_issue_6814_sha256sums(text: str) -> list[tuple[int, str, str]]:
    """Parse non-comment SHA256SUMS entries with their source line numbers."""

    entries: list[tuple[int, str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise RealReexportBindingError(f"invalid issue #6412 SHA256SUMS line {line_number}")
        relative = parts[1].lstrip("*").strip()
        if not relative:
            raise RealReexportBindingError(f"invalid issue #6412 SHA256SUMS line {line_number}")
        entries.append((line_number, parts[0], relative))
    if not entries:
        raise RealReexportBindingError("issue #6412 SHA256SUMS is empty")
    return entries


def _issue_6814_verify_package(  # noqa: C901, PLR0912, PLR0915
    package_root: Path,
    *,
    expected_identity: object,
    expected_package_sha256: str,
) -> dict[str, Any]:
    """Verify the complete #6412 compact package before external retrieval."""

    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise RealReexportBindingError(f"issue #6412 package is unavailable: {package_root}")
    complete = _issue_6814_read_package_json(package_root, "package_complete.json")
    if complete.get("schema_version") != "issue_6412_package_complete.v1":
        raise RealReexportBindingError("issue #6412 package_complete schema mismatch")
    if (
        complete.get("status") != "complete"
        or complete.get("visualization_only") is not True
        or complete.get("n_requested") != 90
        or complete.get("n_admitted") != 88
        or complete.get("n_excluded") != 2
    ):
        raise RealReexportBindingError("issue #6412 package count or evidence boundary mismatch")

    sums_path = package_root / "SHA256SUMS"
    if not sums_path.is_file():
        raise RealReexportBindingError("issue #6412 SHA256SUMS is unavailable")
    actual_sums_sha = _sha256_file(sums_path)
    _require_digest(actual_sums_sha, expected_package_sha256, "issue #6412 SHA256SUMS")
    declared_sums_sha = complete.get("sha256sums_sha256")
    if declared_sums_sha != expected_package_sha256:
        raise RealReexportBindingError("issue #6412 package_complete SHA256SUMS identity mismatch")

    try:
        sums_text = sums_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RealReexportBindingError(f"cannot read issue #6412 SHA256SUMS: {exc}") from exc
    listed_paths: set[str] = set()
    for line_number, digest, relative in _parse_issue_6814_sha256sums(sums_text):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RealReexportBindingError(f"unsafe issue #6412 package path: {relative}")
        listed_paths.add(relative_path.as_posix())
        artifact = package_root / relative_path
        if not artifact.is_file():
            raise RealReexportBindingError(
                f"issue #6412 package artifact is unavailable: {relative}"
            )
        _require_digest(_sha256_file(artifact), digest, f"issue #6412 package {relative}")
    required_package_paths = {
        "package_manifest.json",
        "source_pointer.json",
        "mapping_receipt.json",
    }
    if not required_package_paths <= listed_paths:
        missing = sorted(required_package_paths - listed_paths)
        raise RealReexportBindingError(
            f"issue #6412 SHA256SUMS omits required package artifacts: {missing}"
        )

    manifest = _issue_6814_read_package_json(package_root, "package_manifest.json")
    if (
        manifest.get("schema_version") != "issue_6412_real_reexport_package.v1"
        or manifest.get("execution_commit") != ISSUE_6814_EXECUTION_COMMIT
        or manifest.get("n_requested") != 90
        or manifest.get("n_admitted") != 88
        or manifest.get("n_excluded") != 2
        or manifest.get("visualization_only") is not True
    ):
        raise RealReexportBindingError("issue #6412 package_manifest identity mismatch")
    if manifest.get("excluded_tuples") != [
        ["ppo", "classic_doorway_medium", 128],
        ["ppo", "classic_doorway_medium", 130],
    ]:
        raise RealReexportBindingError("issue #6412 excluded tuple boundary mismatch")

    source_pointer = _issue_6814_read_package_json(package_root, "source_pointer.json")
    mapping = _issue_6814_read_package_json(package_root, "mapping_receipt.json")
    if mapping.get("schema_version") != "issue_6412_trace_reexport_mapping_receipt.v1":
        raise RealReexportBindingError("issue #6412 mapping receipt schema mismatch")
    rows = mapping.get("rows")
    if mapping.get("n_rows") != 90 or not isinstance(rows, list) or len(rows) != 90:
        raise RealReexportBindingError("issue #6412 mapping receipt must contain 90 rows")

    expected_arm = _issue_6814_required_text(
        _issue_6814_identity_value(expected_identity, "arm"), "arm"
    )
    expected_planner = _issue_6814_required_text(
        _issue_6814_identity_value(expected_identity, "planner_id"), "planner_id"
    )
    expected_scenario = _issue_6814_required_text(
        _issue_6814_identity_value(expected_identity, "scenario_id"), "scenario_id"
    )
    expected_seed = _issue_6814_identity_value(expected_identity, "seed")
    if type(expected_seed) is not int:
        raise RealReexportBindingError("issue #6814 seed identity must be an integer")
    selected = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("planner") == expected_planner
        and row.get("scenario_id") == expected_scenario
        and row.get("seed") == expected_seed
    ]
    if len(selected) != 1:
        raise RealReexportBindingError("issue #6412 selected row identity is not unique")
    selected_row = dict(selected[0])
    source_provenance = selected_row.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        raise RealReexportBindingError("issue #6412 selected row lacks source provenance")
    if source_provenance.get("arm") != expected_arm:
        raise RealReexportBindingError("issue #6412 selected row arm mismatch")
    row_identity = {
        "episode_id": selected_row.get("episode_id"),
        "row_index": source_provenance.get("source_row_index"),
        "job_id": source_provenance.get("job_id"),
        "execution_commit": source_provenance.get("execution_commit"),
        "raw_trace_sha256": selected_row.get("raw_trace_sha256"),
        "prior_normalized_sha256": selected_row.get("normalized_trace_sha256"),
    }
    for field in (
        "episode_id",
        "row_index",
        "job_id",
        "execution_commit",
        "raw_trace_sha256",
        "prior_normalized_sha256",
    ):
        expected = _issue_6814_identity_value(expected_identity, field)
        if expected is not None and row_identity[field] != expected:
            raise RealReexportBindingError(f"issue #6412 selected row {field} mismatch")
    for field in ("raw_trace_sha256", "prior_normalized_sha256"):
        _issue_6814_sha256(row_identity[field], f"selected row {field}")
    pointer_arms = source_pointer.get("arms")
    if not isinstance(pointer_arms, list):
        raise RealReexportBindingError("issue #6412 source_pointer arms are unavailable")
    pointer = next(
        (
            entry
            for entry in pointer_arms
            if isinstance(entry, Mapping) and entry.get("arm") == expected_arm
        ),
        None,
    )
    if not isinstance(pointer, Mapping):
        raise RealReexportBindingError("issue #6412 selected source arm is unavailable")
    for field, source_field in (
        ("episodes_sha256", "source_episodes_sha256"),
        ("manifest_sha256", "source_manifest_sha256"),
        ("run_summary_sha256", "run_summary_sha256"),
        ("preflight_sha256", "preflight_sha256"),
    ):
        if source_provenance.get(source_field) != pointer.get(field):
            raise RealReexportBindingError(f"issue #6412 {field} disagrees across package records")
        _issue_6814_sha256(pointer.get(field), f"{expected_arm} {field}")
    if pointer.get("job_id") != source_provenance.get("job_id"):
        raise RealReexportBindingError("issue #6412 source job identity disagrees")
    if (
        pointer.get("planner") != expected_planner
        or pointer.get("scenario_id") != expected_scenario
    ):
        raise RealReexportBindingError("issue #6412 source arm identity disagrees")
    return {
        "package_root": package_root,
        "package_sha256sums_sha256": expected_package_sha256,
        "package_manifest": manifest,
        "package_manifest_sha256": _sha256_file(package_root / "package_manifest.json"),
        "package_complete": complete,
        "package_complete_sha256": _sha256_file(package_root / "package_complete.json"),
        "source_pointer": source_pointer,
        "mapping_receipt": mapping,
        "mapping_receipt_sha256": _sha256_file(package_root / "mapping_receipt.json"),
        "selected_row": selected_row,
        "source_provenance": dict(source_provenance),
        "source_pointer_arm": dict(pointer),
    }


def _issue_6814_one_external_file(
    root: Path,
    *,
    label: str,
    direct_names: tuple[str, ...],
    glob_patterns: tuple[str, ...],
) -> Path:
    """Resolve one unambiguous external compact artifact."""

    candidates = _issue_6814_external_candidates(
        root, direct_names=direct_names, glob_patterns=glob_patterns
    )
    if len(candidates) != 1:
        raise RealReexportBindingError(
            f"issue #6814 {label} must resolve to one file; found {len(candidates)}"
        )
    return candidates[0]


def _issue_6814_artifact_retrieval_key(
    source_root_retrieval_key: str, root: Path, artifact: Path
) -> str:
    """Bind a verified external artifact to its root retrieval namespace."""

    try:
        relative = artifact.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RealReexportBindingError(
            "issue #6814 external artifact is outside its verified arm root"
        ) from exc
    return f"{source_root_retrieval_key.rstrip('/')}/{relative}"


def _issue_6814_external_candidates(
    root: Path,
    *,
    direct_names: tuple[str, ...],
    glob_patterns: tuple[str, ...],
) -> list[Path]:
    """Resolve direct external artifact names before recursive glob matches."""

    direct = [root / name for name in direct_names if (root / name).is_file()]
    if direct:
        return direct
    return sorted({path for pattern in glob_patterns for path in root.glob(pattern)})


def _issue_6814_optional_external_json(
    root: Path,
    *,
    label: str,
    direct_names: tuple[str, ...],
    glob_patterns: tuple[str, ...],
    schema_versions: tuple[str, ...],
) -> tuple[dict[str, Any], str] | None:
    """Load one optional semantic owner without inventing a missing record."""

    candidates = _issue_6814_external_candidates(
        root, direct_names=direct_names, glob_patterns=glob_patterns
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RealReexportBindingError(
            f"issue #6814 {label} owner must resolve to one file; found {len(candidates)}"
        )
    payload = _read_json_object(candidates[0])
    if payload.get("schema_version") not in schema_versions:
        raise RealReexportBindingError(f"issue #6814 {label} owner schema mismatch")
    return payload, _sha256_file(candidates[0])


def _issue_6814_row_value(row: Mapping[str, Any], *paths: tuple[str, ...]) -> object:
    """Read the first present nested row value."""

    for path in paths:
        current: object = row
        for part in path:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                break
        else:
            return current
    return None


def _issue_6814_external_row_identity(
    row: Mapping[str, Any], *, expected: Mapping[str, Any], raw_line: bytes
) -> None:
    """Verify selected external row identity and both row-level hashes."""

    expected_fields = {
        "episode_id": expected["episode_id"],
        "scenario_id": expected["scenario_id"],
        "planner_id": expected["planner_id"],
        "seed": expected["seed"],
    }
    planner = _issue_6814_row_value(
        row, ("planner_id",), ("planner",), ("algo",), ("scenario_params", "algo")
    )
    actual = {
        "episode_id": row.get("episode_id"),
        "scenario_id": row.get("scenario_id"),
        "planner_id": planner,
        "seed": row.get("seed"),
    }
    if actual != expected_fields:
        raise RealReexportBindingError(
            f"issue #6814 selected external row identity mismatch: expected {expected_fields}, got {actual}"
        )
    if _sha256_bytes(raw_line) != expected["raw_trace_sha256"]:
        raise RealReexportBindingError("issue #6814 selected external raw row SHA-256 mismatch")
    row_config_hash = expected.get("row_config_hash")
    if row_config_hash is not None:
        actual_row_config = _issue_6814_row_value(
            row,
            ("row_config_hash",),
            ("config_hash",),
            ("result_provenance", "config_hash"),
        )
        if actual_row_config != row_config_hash:
            raise RealReexportBindingError("issue #6814 selected external row config hash mismatch")
    algorithm_config_hash = expected.get("algorithm_config_hash")
    if algorithm_config_hash is not None:
        actual_algorithm_hash = _issue_6814_row_value(
            row,
            ("algorithm_config_hash",),
            ("algorithm_metadata", "config_hash"),
            ("result_provenance", "row_config", "algorithm_config_hash"),
        )
        if actual_algorithm_hash != algorithm_config_hash:
            raise RealReexportBindingError(
                "issue #6814 selected external algorithm config hash mismatch"
            )
    commit = _issue_6814_row_value(
        row,
        ("git_hash",),
        ("repo_commit",),
        ("execution_commit",),
        ("result_provenance", "repo_commit"),
    )
    if commit != ISSUE_6814_EXECUTION_COMMIT:
        raise RealReexportBindingError("issue #6814 selected external execution commit mismatch")


def load_verified_real_reexport_row_source(  # noqa: C901, PLR0912, PLR0915
    *,
    package_root: Path,
    external_arm_root: Path,
    expected_identity: object,
    expected_package_sha256: str = ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
) -> VerifiedRealReexportRowSource:
    """Verify the complete #6412 lineage before exposing one row's fields."""

    package = _issue_6814_verify_package(
        package_root,
        expected_identity=expected_identity,
        expected_package_sha256=expected_package_sha256,
    )
    source_provenance = package["source_provenance"]
    pointer = package["source_pointer_arm"]
    root = Path(external_arm_root).resolve()
    if not root.is_dir():
        raise RealReexportBindingError(f"issue #6814 external arm root is unavailable: {root}")
    episodes_path = _issue_6814_one_external_file(
        root,
        label="episodes.jsonl",
        direct_names=("episodes.jsonl",),
        glob_patterns=("**/episodes.jsonl",),
    )
    manifest_path = _issue_6814_one_external_file(
        root,
        label="arm manifest",
        direct_names=("manifest.json", "campaign_manifest.json"),
        glob_patterns=("**/manifest.json", "**/campaign_manifest.json"),
    )
    run_summary_path = _issue_6814_one_external_file(
        root,
        label="run summary",
        direct_names=("run_summary.yaml", "run_summary.yml"),
        glob_patterns=("**/run_summary.yaml", "**/run_summary.yml"),
    )
    preflight_path = _issue_6814_one_external_file(
        root,
        label="preflight",
        direct_names=("preflight/validate_config.json", "validate_config.json"),
        glob_patterns=("**/preflight/validate_config.json", "**/validate_config.json"),
    )
    artifact_paths = {
        "episodes_sha256": episodes_path,
        "manifest_sha256": manifest_path,
        "run_summary_sha256": run_summary_path,
        "preflight_sha256": preflight_path,
    }
    for field, path in artifact_paths.items():
        _require_digest(_sha256_file(path), pointer[field], f"issue #6814 {field}")

    expected = {
        "episode_id": package["selected_row"]["episode_id"],
        "scenario_id": package["selected_row"]["scenario_id"],
        "planner_id": package["selected_row"]["planner"],
        "seed": package["selected_row"]["seed"],
        "raw_trace_sha256": package["selected_row"]["raw_trace_sha256"],
        "row_config_hash": source_provenance.get("row_config_hash"),
        "algorithm_config_hash": source_provenance.get("algorithm_config_hash"),
    }
    raw_lines = episodes_path.read_bytes().splitlines(keepends=True)
    n_rows = sum(1 for raw_line in raw_lines if raw_line.strip())
    row_index = int(source_provenance["source_row_index"])
    if row_index < 1 or row_index > len(raw_lines):
        raise RealReexportBindingError("issue #6814 selected external row index is out of range")
    raw_line = raw_lines[row_index - 1]
    try:
        row = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealReexportBindingError("issue #6814 selected external row is invalid JSON") from exc
    if not isinstance(row, Mapping):
        raise RealReexportBindingError("issue #6814 selected external row must be an object")
    _issue_6814_external_row_identity(row, expected=expected, raw_line=raw_line)

    manifest = _read_json_object(manifest_path)
    manifest_planner = _issue_6814_row_value(
        manifest, ("planner_id",), ("planner",), ("planners", "0", "key")
    )
    if manifest.get("job_id", manifest.get("slurm_job_id")) != source_provenance["job_id"]:
        raise RealReexportBindingError("issue #6814 external manifest job mismatch")
    if manifest_planner is not None and manifest_planner != expected["planner_id"]:
        raise RealReexportBindingError("issue #6814 external manifest planner mismatch")
    manifest_scenario = manifest.get("scenario_id")
    if manifest_scenario is not None and manifest_scenario != expected["scenario_id"]:
        raise RealReexportBindingError("issue #6814 external manifest scenario mismatch")
    try:
        run_summary_raw = yaml.safe_load(run_summary_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RealReexportBindingError("issue #6814 run summary is not valid YAML") from exc
    if not isinstance(run_summary_raw, Mapping):
        raise RealReexportBindingError("issue #6814 run summary must be an object")
    preflight = _read_json_object(preflight_path)
    route_geometry_artifact = _issue_6814_optional_external_json(
        root,
        label="route geometry",
        direct_names=("process_trace_geometry_registry.json", "geometry_registry.json"),
        glob_patterns=(
            "**/process_trace_geometry_registry*.json",
            "**/geometry_registry*.json",
        ),
        schema_versions=("process_trace_geometry_registry.v1",),
    )
    conflict_geometry_artifact = _issue_6814_optional_external_json(
        root,
        label="conflict geometry",
        direct_names=("conflict_registry.json", "process_trace_conflict_registry.json"),
        glob_patterns=("**/conflict_registry*.json", "**/process_trace_conflict_registry*.json"),
        schema_versions=("conflict_geometry.v1", "process_trace_conflict_registry.v1"),
    )
    encounter_report_artifact = _issue_6814_optional_external_json(
        root,
        label="encounter report",
        direct_names=("near_miss_encounter.json", "encounter_report.json"),
        glob_patterns=("**/near_miss_encounter*.json", "**/encounter_report*.json"),
        schema_versions=("near_miss_encounter.v1",),
    )

    provenance_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    if not provenance_path.is_file():
        matches = sorted(root.glob("**/episodes.jsonl.provenance.json"))
        if len(matches) > 1:
            raise RealReexportBindingError("issue #6814 result provenance sidecar is ambiguous")
        provenance_path = matches[0] if matches else provenance_path
    result_provenance_sha256: str | None = None
    result_provenance_row: Mapping[str, object] | None = None
    result_provenance_manifest: Mapping[str, object] | None = None
    if provenance_path.is_file():
        provenance_payload = _read_json_object(provenance_path)
        rows = provenance_payload.get("rows")
        if not isinstance(rows, list):
            raise RealReexportBindingError(
                "issue #6814 result provenance sidecar rows are unavailable"
            )
        try:
            validate_result_provenance_manifest(provenance_payload)
        except (ValueError, TypeError) as exc:
            raise RealReexportBindingError(
                "issue #6814 result provenance sidecar failed validation"
            ) from exc
        result_provenance_sha256 = _sha256_file(provenance_path)
        result_provenance_manifest = dict(provenance_payload)
        matches = [
            candidate
            for candidate in rows
            if isinstance(candidate, Mapping)
            and candidate.get("episode_id") == expected["episode_id"]
            and candidate.get("scenario_id") == expected["scenario_id"]
            and candidate.get("seed") == expected["seed"]
        ]
        if len(matches) != 1:
            raise RealReexportBindingError(
                "issue #6814 result provenance sidecar row link is not unique"
            )
        result_provenance_row = dict(matches[0])
        provenance_commit = result_provenance_row.get("repo_commit")
        if provenance_commit != ISSUE_6814_EXECUTION_COMMIT:
            raise RealReexportBindingError("issue #6814 result provenance commit mismatch")
        jsonl_line = result_provenance_row.get("jsonl_line")
        if jsonl_line is not None and jsonl_line != row_index:
            raise RealReexportBindingError("issue #6814 result provenance row index mismatch")
        artifact_sha = result_provenance_row.get("trace_artifact_sha256")
        if artifact_sha is not None and artifact_sha != pointer["episodes_sha256"]:
            raise RealReexportBindingError("issue #6814 result provenance artifact hash mismatch")
        artifact_path = result_provenance_row.get("raw_artifact") or result_provenance_row.get(
            "trace_artifact_path"
        )
        if artifact_path is not None and not str(artifact_path).endswith("episodes.jsonl"):
            raise RealReexportBindingError("issue #6814 result provenance artifact path mismatch")
        simulator_settings = result_provenance_row.get("simulator_settings")
        if not isinstance(simulator_settings, Mapping):
            raise RealReexportBindingError(
                "issue #6814 result provenance row lacks simulator settings"
            )
    source_root_retrieval_key = source_provenance.get("source_retrieval_key")
    if not isinstance(source_root_retrieval_key, str) or not source_root_retrieval_key.strip():
        source_root_retrieval_key = pointer.get("retrieval_key")
    if not isinstance(source_root_retrieval_key, str) or not source_root_retrieval_key.strip():
        source_root_retrieval_key = package["source_pointer"].get("retrieval_key")
    if not isinstance(source_root_retrieval_key, str) or not source_root_retrieval_key.strip():
        raise RealReexportBindingError(
            "issue #6814 source retrieval key is unavailable in verified provenance"
        )
    source_root_retrieval_key = source_root_retrieval_key.strip()
    return VerifiedRealReexportRowSource(
        arm=str(source_provenance["arm"]),
        job_id=str(source_provenance["job_id"]),
        row_index=row_index,
        episode_id=str(expected["episode_id"]),
        scenario_id=str(expected["scenario_id"]),
        planner_id=str(expected["planner_id"]),
        seed=int(expected["seed"]),
        execution_commit=ISSUE_6814_EXECUTION_COMMIT,
        raw_row=dict(row),
        raw_row_sha256=str(expected["raw_trace_sha256"]),
        prior_normalized_sha256=str(package["selected_row"]["normalized_trace_sha256"]),
        episodes_sha256=str(pointer["episodes_sha256"]),
        manifest_sha256=str(pointer["manifest_sha256"]),
        run_summary_sha256=str(pointer["run_summary_sha256"]),
        preflight_sha256=str(pointer["preflight_sha256"]),
        result_provenance_sha256=result_provenance_sha256,
        result_provenance_row=result_provenance_row,
        source_root_retrieval_key=source_root_retrieval_key,
        episodes_retrieval_key=_issue_6814_artifact_retrieval_key(
            source_root_retrieval_key, root, episodes_path
        ),
        manifest_retrieval_key=_issue_6814_artifact_retrieval_key(
            source_root_retrieval_key, root, manifest_path
        ),
        run_summary_retrieval_key=_issue_6814_artifact_retrieval_key(
            source_root_retrieval_key, root, run_summary_path
        ),
        preflight_retrieval_key=_issue_6814_artifact_retrieval_key(
            source_root_retrieval_key, root, preflight_path
        ),
        manifest=dict(manifest),
        run_summary=dict(run_summary_raw),
        preflight=dict(preflight),
        result_provenance_manifest=result_provenance_manifest,
        route_geometry=(route_geometry_artifact[0] if route_geometry_artifact else None),
        conflict_geometry=(conflict_geometry_artifact[0] if conflict_geometry_artifact else None),
        encounter_report=(encounter_report_artifact[0] if encounter_report_artifact else None),
        route_geometry_sha256=(route_geometry_artifact[1] if route_geometry_artifact else None),
        conflict_geometry_sha256=(
            conflict_geometry_artifact[1] if conflict_geometry_artifact else None
        ),
        encounter_report_sha256=(
            encounter_report_artifact[1] if encounter_report_artifact else None
        ),
        n_rows=n_rows,
    )


__all__ = [
    "REAL_REEXPORT_ARMS",
    "REAL_REEXPORT_ARMS_BY_KEY",
    "REAL_REEXPORT_EXCEPTION_SEEDS",
    "REAL_REEXPORT_REQUEST_TUPLES",
    "ISSUE_6412_PACKAGE_SHA256SUMS_SHA256",
    "ISSUE_6814_EXECUTION_COMMIT",
    "CampaignExpectation",
    "FrozenTraceReexportContract",
    "RealReexportArm",
    "VerifiedRealReexportRowSource",
    "RealReexportBindingError",
    "TraceReexportPackagingError",
    "bind_real_reexport_arms",
    "build_resolver_mapping_receipt",
    "campaign_expectations",
    "canonical_sha256",
    "default_resolver_mapping_path",
    "expected_outcomes_payload_for_rows",
    "package_trace_reexport",
    "real_reexport_arms",
    "real_reexport_request_tuples",
    "load_verified_real_reexport_row_source",
]
