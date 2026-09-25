"""Build deterministic, replay-checked visual bundles from adversarial search results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.archive import curate_failure_archive
from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.bundle import compute_effective_scenario_hash
from robot_sf.adversarial.certification_types import CertificationStatus
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
from robot_sf.adversarial.objectives import (
    constraints_first_outcome_projection,
    get_objective,
)
from robot_sf.adversarial.search import DEFAULT_SCHEMA_PATH
from robot_sf.benchmark.episode_replay_figure import (
    EpisodeRow,
    replay_episode_and_generate_figures,
)
from robot_sf.benchmark.fallback_policy import (
    availability_payload,
    resolve_execution_mode,
    runtime_fallback_or_degraded_marker,
)
from robot_sf.benchmark.runner import run_batch
from robot_sf.training import scenario_loader

GALLERY_SCHEMA_VERSION = "adversarial-replay-gallery.v1"
SEARCH_MANIFEST_SCHEMA_VERSION = "adversarial-search-manifest.v1"
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MAX_PED_TRACK_SPEED_MPS = 12.0
_ADMISSIBLE_CLASSIFICATIONS = frozenset({"valid", "hard_but_solvable"})
_CANONICAL_OUTCOME_FIELDS = (
    "route_complete",
    "collision",
    "collision_event",
    "timeout",
    "timeout_event",
    "severe_intrusion",
    "severe_intrusion_event",
)
_CANONICAL_METRIC_FIELDS = (
    "success",
    "collisions",
    "near_misses",
    "min_distance",
    "snqi",
    "path_efficiency",
)


@dataclass(frozen=True)
class _ReplayContext:
    """Immutable settings shared by each selected case in one gallery run."""

    config: dict[str, Any]
    objective: Any
    objective_name: str
    policy: str
    search_method: str
    source_revision: str | None
    source_manifest_sha256: str
    output_dir: Path
    tolerance: float
    video: bool
    render: bool
    root: Path
    checkout_revision: str | None
    checkout_clean: bool
    checkout_dirty_paths: tuple[str, ...]


def build_replay_gallery(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    top_k: int = 5,
    tolerance: float = 1e-6,
    video: bool = True,
    render: bool = True,
) -> dict[str, Any]:
    """Materialize, replay, compare, and render a bounded set of search candidates.

    The search manifest and its candidate bundle files remain read-only. The output
    directory must not already exist, so an earlier replay bundle cannot be replaced
    accidentally. A candidate is selected only when its certificate, analysis
    eligibility, source episode, scenario inputs, and objective are available.

    Returns:
        The deterministic ``adversarial-replay-gallery.v1`` manifest.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1 or top_k > 20:
        raise ValueError("top_k must be an integer between 1 and 20")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    source_manifest = Path(manifest_path).expanduser().resolve()
    root = _repository_root()
    destination = _validated_output_directory(output_dir, root=root)
    source_root = _source_repository_root(source_manifest, root)
    payload, source_manifest_snapshot = _read_search_manifest_snapshot(source_manifest)
    checkout_state = _git_checkout_state(root)
    source_manifest_sha256 = hashlib.sha256(source_manifest_snapshot).hexdigest()
    source_revision = _manifest_revision(payload)
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    objective_name = str(config.get("objective") or "")
    if not objective_name:
        raise ValueError("search manifest config.objective must be non-empty")
    objective = get_objective(objective_name)
    policy = str(config.get("policy") or "")
    if not policy:
        raise ValueError("search manifest config.policy must be non-empty")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("search manifest candidates must be a list")

    # Reuse the repository's mechanism clustering for attributed failures. The
    # selector below still ranks each cluster by the search objective, rather than
    # substituting the archive's perturbation-minimal representative.
    archive_clusters, archive_cluster_status = _failure_cluster_by_candidate(
        source_manifest_snapshot
    )
    eligible: list[dict[str, Any]] = []
    accounting: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(candidates):
        entry, reason = _prepare_candidate(
            index=index,
            raw_candidate=raw_candidate,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            archive_cluster=archive_clusters.get(index),
            root=root,
            source_root=source_root,
        )
        if entry is None:
            accounting.append(reason)
            continue
        eligible.append(entry)
        accounting.append(entry["accounting"])

    selected = _select_top_candidates(eligible, top_k=top_k)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source_search_manifest.json").write_bytes(source_manifest_snapshot)
    replay_context = _ReplayContext(
        config=config,
        objective=objective,
        objective_name=objective_name,
        policy=policy,
        search_method=_search_method(payload, config),
        source_revision=source_revision,
        source_manifest_sha256=source_manifest_sha256,
        output_dir=destination,
        tolerance=tolerance,
        video=video,
        render=render,
        root=root,
        checkout_revision=checkout_state["revision"],
        checkout_clean=checkout_state["clean"],
        checkout_dirty_paths=tuple(checkout_state["dirty_paths"]),
    )
    cases: list[dict[str, Any]] = []
    for selected_candidate in selected:
        case = _run_selected_candidate(selected_candidate, replay_context)
        cases.append(case)
        accounting[selected_candidate["index"]]["case_id"] = case["case_id"]
        accounting[selected_candidate["index"]]["disposition"] = "selected"
        accounting[selected_candidate["index"]]["replay_match"] = case["replay_match"]

    summary_counts = Counter(str(item.get("disposition", "unknown")) for item in accounting)
    result = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "source": {
            "manifest_path": _display_path(source_manifest, source_root),
            "manifest_sha256": source_manifest_sha256,
            "source_revision": source_revision,
            "revision_status": "known" if source_revision else "unknown",
            "search_method": _search_method(payload, config),
            "search_seed": config.get("seed"),
            "search_budget": config.get("budget"),
            "search_space_sha256": hashlib.sha256(
                _stable_json(config.get("search_space")).encode("utf-8")
            ).hexdigest()
            if isinstance(config.get("search_space"), dict)
            else None,
            "gallery_checkout": {
                "revision": replay_context.checkout_revision,
                "clean": replay_context.checkout_clean,
                "dirty_paths": list(replay_context.checkout_dirty_paths),
            },
        },
        "selection": {
            "top_k": top_k,
            "objective": objective_name,
            "ordering": "objective_value_desc_then_source_candidate_index_asc",
            "duplicate_policy": (
                "available_failure_mechanism_cluster_then_exact_effective_scenario_hash"
            ),
            "mechanism_cluster_deduplication": archive_cluster_status,
            "replay_objective_absolute_tolerance": tolerance,
        },
        "replay": {
            "policy": policy,
            "record_simulation_step_trace": True,
            "video_enabled": bool(video),
            "video_renderer": "synthetic" if video else "none",
            "rendering": "existing_episode_replay_figure" if render else "disabled",
            "instrumentation_note": (
                "Replay enables simulation step traces and optional synthetic video; "
                "these outputs are presentation artifacts and are not source-run evidence."
            ),
        },
        "summary": {
            "source_candidate_count": len(candidates),
            "selected_case_count": len(cases),
            "replay_match_count": sum(case["replay_match"] == "match" for case in cases),
            "replay_mismatch_count": sum(case["replay_match"] == "mismatch" for case in cases),
            "replay_unavailable_count": sum(
                case["replay_match"] == "unavailable" for case in cases
            ),
            "dispositions": dict(sorted(summary_counts.items())),
        },
        "candidates": accounting,
        "cases": cases,
    }
    _write_json(destination / "gallery_manifest.json", result)
    _write_gallery_readme(destination, cases, result["summary"])
    return result


def _validated_output_directory(output_dir: str | Path, *, root: Path) -> Path:
    """Require gallery output to stay below this checkout's ignored output/ tree."""
    destination = Path(output_dir).expanduser().resolve()
    output_boundary = root / "output"
    if output_boundary.is_symlink():
        raise ValueError("repository output/ must not be a symlink")
    output_root = output_boundary.resolve()
    try:
        relative = destination.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(
            f"gallery output must be inside the repository ignored output/ directory: {output_root}"
        ) from exc
    if not relative.parts:
        raise ValueError("gallery output must be a new child directory inside output/")
    return destination


def _read_search_manifest_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read, validate, and retain one immutable source-manifest byte snapshot."""
    try:
        source_bytes = path.read_bytes()
        payload = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read search manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("search manifest must be a JSON object")
    if payload.get("schema_version") != SEARCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported search manifest schema: "
            f"{payload.get('schema_version')!r}; expected {SEARCH_MANIFEST_SCHEMA_VERSION!r}"
        )
    return payload, source_bytes


def _prepare_candidate(  # noqa: C901, PLR0912, PLR0915 - keep ordered validation gates together
    *,
    index: int,
    raw_candidate: Any,
    source_manifest: Path,
    source_manifest_sha256: str,
    archive_cluster: dict[str, Any] | None,
    root: Path,
    source_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve and validate one candidate without modifying its source bundle."""
    candidate_payload = raw_candidate if isinstance(raw_candidate, dict) else {}
    candidate = candidate_payload.get("candidate")
    if not isinstance(candidate, dict):
        return None, _accounting_row(index, candidate_payload, "missing_candidate_parameters")
    case_id = _stable_case_id(index, candidate, source_manifest_sha256)
    row = _accounting_row(index, candidate_payload, "eligible", case_id=case_id)
    if candidate_payload.get("error"):
        return None, _accounting_row(index, candidate_payload, "evaluation_failed")
    eligibility = candidate_payload.get("analysis_eligibility")
    if not isinstance(eligibility, dict) or eligibility.get("eligible") is not True:
        return None, _accounting_row(index, candidate_payload, "analysis_ineligible")
    classification = _certification_classification(candidate_payload.get("certification_status"))
    if classification not in _ADMISSIBLE_CLASSIFICATIONS:
        disposition = (
            f"certificate_{classification}"
            if classification is not None
            else "certificate_classification_unknown"
        )
        return None, _accounting_row(index, candidate_payload, disposition)
    objective_value = _finite_number(candidate_payload.get("objective_value"))
    if objective_value is None:
        return None, _accounting_row(index, candidate_payload, "objective_unavailable")

    episode_path = _resolve_artifact_path(
        candidate_payload.get("episode_record_path"),
        source_manifest=source_manifest,
        root=root,
        source_root=source_root,
    )
    scenario_path = _resolve_artifact_path(
        candidate_payload.get("scenario_yaml_path"),
        source_manifest=source_manifest,
        root=root,
        source_root=source_root,
    )
    if episode_path is None or not episode_path.is_file():
        return None, _accounting_row(index, candidate_payload, "episode_record_missing")
    if scenario_path is None or not scenario_path.is_file():
        return None, _accounting_row(index, candidate_payload, "scenario_input_missing")
    try:
        source_episode_bytes = episode_path.read_bytes()
    except OSError:
        return None, _accounting_row(index, candidate_payload, "episode_record_unreadable")
    source_episode_sha256 = hashlib.sha256(source_episode_bytes).hexdigest()
    source_record = _read_single_episode_bytes(source_episode_bytes)
    if source_record is None:
        return None, _accounting_row(
            index, candidate_payload, "episode_record_invalid_or_ambiguous"
        )
    try:
        scenario_bytes = scenario_path.read_bytes()
    except OSError:
        return None, _accounting_row(index, candidate_payload, "scenario_input_unreadable")
    scenario_sha256 = hashlib.sha256(scenario_bytes).hexdigest()
    scenario_identity, scenario_error = _scenario_identity_bytes(scenario_bytes)
    if scenario_error is not None or scenario_identity is None:
        return None, _accounting_row(
            index, candidate_payload, scenario_error or "scenario_identity_unavailable"
        )
    map_id_snapshot, map_id_error = _snapshot_map_id_input(
        scenario_identity,
        scenario_path=scenario_path,
        episode_path=episode_path,
        source_record=source_record,
        root=root,
        source_root=source_root,
    )
    if map_id_error is not None:
        return None, _accounting_row(index, candidate_payload, map_id_error)
    map_file_declared, map_file_sha256, map_file_error = _scenario_file_binding(
        scenario_identity,
        "map_file",
        scenario_path=scenario_path,
        source_root=source_root,
        root=root,
    )
    if map_file_error is not None:
        return None, _accounting_row(index, candidate_payload, map_file_error)
    effective_hash, hash_error = _source_effective_scenario_hash(
        scenario_path, scenario_identity, source_root=source_root, root=root
    )
    try:
        scenario_unchanged = _sha256_file(scenario_path) == scenario_sha256
    except OSError:
        scenario_unchanged = False
    if not scenario_unchanged:
        return None, _accounting_row(index, candidate_payload, "scenario_changed_during_selection")
    _, current_map_sha256, map_file_error = _scenario_file_binding(
        scenario_identity,
        "map_file",
        scenario_path=scenario_path,
        source_root=source_root,
        root=root,
    )
    if map_file_error is not None:
        return None, _accounting_row(index, candidate_payload, map_file_error)
    if current_map_sha256 != map_file_sha256:
        return None, _accounting_row(
            index, candidate_payload, "scenario_map_changed_during_selection"
        )
    try:
        if hashlib.sha256(episode_path.read_bytes()).hexdigest() != source_episode_sha256:
            return None, _accounting_row(
                index, candidate_payload, "episode_record_changed_during_selection"
            )
    except OSError:
        return None, _accounting_row(index, candidate_payload, "episode_record_unreadable")
    if hash_error is not None:
        return None, _accounting_row(index, candidate_payload, hash_error)
    declared_hash = candidate_payload.get("effective_scenario_hash")
    if not isinstance(declared_hash, str) or declared_hash != effective_hash:
        return None, _accounting_row(
            index, candidate_payload, "effective_scenario_hash_missing_or_mismatch"
        )
    if not _source_identity_matches(candidate, source_record, scenario_identity):
        return None, _accounting_row(index, candidate_payload, "source_candidate_identity_mismatch")
    source_availability_problem = _source_availability_problem(candidate_payload, source_record)
    if source_availability_problem is not None:
        return None, _accounting_row(index, candidate_payload, source_availability_problem)
    try:
        _candidate_spec(candidate)
    except (TypeError, ValueError, OverflowError):
        return None, _accounting_row(index, candidate_payload, "candidate_parameters_invalid")
    if attribution_from_episode_record(source_record).primary_failure == "success":
        return None, _accounting_row(index, candidate_payload, "source_episode_not_a_failure")
    if not _manifest_attribution_matches(candidate_payload, source_record):
        return None, _accounting_row(
            index, candidate_payload, "source_failure_attribution_mismatch"
        )

    source_revision = _record_revision(source_record)
    effective_hash = candidate_payload.get("effective_scenario_hash")
    dedupe_key = str(effective_hash) if effective_hash else _stable_json(candidate)
    mechanism_cluster = _stable_json(archive_cluster) if archive_cluster else None
    row["disposition"] = "eligible_for_selection"
    return {
        "index": index,
        "case_id": case_id,
        "candidate": candidate,
        "candidate_payload": candidate_payload,
        "objective_value": objective_value,
        "failure_attribution": candidate_payload.get("failure_attribution"),
        "episode_path": episode_path,
        "scenario_path": scenario_path,
        "source_record": source_record,
        "source_episode_sha256": source_episode_sha256,
        "source_revision": source_revision,
        "source_root": source_root,
        "scenario_sha256": scenario_sha256,
        "map_file_declared": map_file_declared,
        "map_file_sha256": map_file_sha256,
        "map_id_snapshot": map_id_snapshot,
        "effective_scenario_hash": effective_hash,
        "dedupe_key": dedupe_key,
        "mechanism_cluster": mechanism_cluster,
        "accounting": row,
    }, None


def _select_top_candidates(candidates: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    """Select objective-ranked candidates while collapsing exact and mechanism duplicates."""
    ordered = sorted(candidates, key=lambda row: (-row["objective_value"], row["index"]))
    selected: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_mechanisms: set[str] = set()
    for row in ordered:
        if row["dedupe_key"] in seen_exact:
            row["accounting"]["disposition"] = "duplicate_effective_scenario"
            continue
        cluster = row["mechanism_cluster"]
        if cluster is not None and cluster in seen_mechanisms:
            row["accounting"]["disposition"] = "duplicate_failure_mechanism"
            continue
        seen_exact.add(row["dedupe_key"])
        if cluster is not None:
            seen_mechanisms.add(cluster)
        selected.append(row)
        if len(selected) >= top_k:
            break
    selected_ids = {row["index"] for row in selected}
    for row in ordered:
        if (
            row["index"] not in selected_ids
            and row["accounting"]["disposition"] == "eligible_for_selection"
        ):
            row["accounting"]["disposition"] = "ranked_below_top_k"
    return selected


def _run_selected_candidate(selected: dict[str, Any], context: _ReplayContext) -> dict[str, Any]:
    """Materialize immutable inputs, run one canonical episode, compare, and render."""
    case_id = str(selected["case_id"])
    case_dir = context.output_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    materialization = _materialize_scenario(
        selected["scenario_path"],
        case_dir / "inputs",
        root=context.root,
        source_root=selected["source_root"],
        selected_map_id_snapshot=selected.get("map_id_snapshot"),
    )
    try:
        source_episode_bytes = selected["episode_path"].read_bytes()
    except OSError:
        source_episode_bytes = None
    if source_episode_bytes is not None:
        (case_dir / "source_episode.jsonl").write_bytes(source_episode_bytes)
    result = _initial_case_result(selected, context, materialization)
    source_episode_sha256 = (
        hashlib.sha256(source_episode_bytes).hexdigest()
        if source_episode_bytes is not None
        else None
    )
    if source_episode_sha256 != selected["source_episode_sha256"]:
        result["verification_status"] = "not_replayed_source_episode_changed_after_selection"
        result["materialization"]["source_episode_selection_binding"] = {
            "status": "mismatch" if source_episode_bytes is not None else "unavailable",
            "selected_source_sha256": selected["source_episode_sha256"],
            "materialized_source_sha256": source_episode_sha256,
        }
        return _complete_case(result, case_dir, context.output_dir)
    result["materialization"]["source_episode_selection_binding"] = {
        "status": "bound",
        "source_episode_sha256": selected["source_episode_sha256"],
    }
    if materialization["status"] != "materialized":
        result["verification_status"] = "not_replayed_inputs_unavailable"
        return _complete_case(result, case_dir, context.output_dir)
    selection_error, selection_binding = _materialized_selection_binding(
        selected, materialization, case_dir=case_dir
    )
    result["materialization"]["selection_binding"] = selection_binding
    if selection_error is not None:
        result["verification_status"] = selection_error
        return _complete_case(result, case_dir, context.output_dir)

    replay_dir = case_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=False)
    episode_records = replay_dir / "episode_records.jsonl"
    runner_config, config_error, execution_config = _runner_config(
        context.config,
        case_dir=case_dir,
        root=context.root,
        source_root=selected["source_root"],
    )
    result["execution_config"] = execution_config
    if config_error is not None or runner_config is None:
        result["verification_status"] = "not_replayed_config_unavailable"
        result["replay_error"] = config_error or "runner_config_unavailable"
        return _complete_case(result, case_dir, context.output_dir)
    result["source_input_binding"] = _source_input_binding(
        materialization,
        execution_config,
        effective_scenario_hash=selected["effective_scenario_hash"],
        root=context.root,
        case_dir=case_dir,
    )

    replay_summary, replay_error = _run_one_episode(
        case_dir / materialization["scenario_path"],
        episode_records,
        runner_config,
        context,
        map_registry_path=(
            case_dir / materialization["runner_map_registry_path"]
            if isinstance(materialization.get("runner_map_registry_path"), str)
            else None
        ),
    )
    if replay_error is not None:
        result["verification_status"] = "replay_execution_failed"
        result["replay_error"] = replay_error
        result["replay"] = {"error": replay_error}
        return _complete_case(result, case_dir, context.output_dir)

    replay_record = _read_single_episode(episode_records)
    replay_availability, replay_availability_error = _replay_availability(replay_summary)
    if replay_record is None:
        result["verification_status"] = (
            "replay_execution_unavailable"
            if replay_availability_error is not None
            else "replay_record_missing_or_ambiguous"
        )
        result["replay"] = {
            "summary": replay_summary,
            "episode_record_path": "replay/episode_records.jsonl",
            "benchmark_availability": replay_availability,
            "availability_error": replay_availability_error,
        }
        return _complete_case(result, case_dir, context.output_dir)
    replay_record_availability_error = _replay_record_availability(replay_record)
    if replay_availability_error is None:
        replay_availability_error = replay_record_availability_error

    _record_replay_comparison(
        selected,
        context,
        result,
        replay_record,
        materialization=materialization,
        replay_dir=replay_dir,
        episode_records=episode_records,
        case_dir=case_dir,
        replay_summary=replay_summary or {},
        replay_availability=replay_availability,
        replay_availability_error=replay_availability_error,
    )
    return _complete_case(result, case_dir, context.output_dir)


def _materialized_selection_binding(  # noqa: C901 - preserve ordered snapshot checks
    selected: dict[str, Any], materialization: dict[str, Any], *, case_dir: Path
) -> tuple[str | None, dict[str, Any]]:
    """Bind source and effective scenario inputs again at the replay boundary."""
    source_digest = selected["scenario_sha256"]
    materialized_digest = materialization.get("source_scenario_sha256")
    if materialized_digest != source_digest:
        return "not_replayed_scenario_changed_after_selection", {
            "status": "mismatch",
            "selected_source_sha256": source_digest,
            "materialized_source_sha256": materialized_digest,
        }
    effective_hash = selected["effective_scenario_hash"]
    materialized_hash = materialization.get("effective_scenario_hash")
    if materialized_hash != effective_hash:
        return "not_replayed_effective_scenario_changed_after_selection", {
            "status": "mismatch",
            "selected_effective_scenario_hash": effective_hash,
            "materialized_effective_scenario_hash": materialized_hash,
        }
    if selected.get("map_file_declared") is True:
        map_asset = next(
            (
                asset
                for asset in materialization.get("assets", [])
                if isinstance(asset, dict) and asset.get("field") == "map_file"
            ),
            None,
        )
        materialized_map_sha256 = (
            map_asset.get("source_sha256") if isinstance(map_asset, dict) else None
        )
        selected_map_sha256 = selected.get("map_file_sha256")
        if not isinstance(selected_map_sha256, str):
            return "not_replayed_map_input_unavailable_at_selection", {
                "status": "unavailable",
                "selected_source_sha256": selected_map_sha256,
                "materialized_source_sha256": materialized_map_sha256,
            }
        if materialized_map_sha256 != selected_map_sha256:
            return "not_replayed_map_changed_after_selection", {
                "status": "mismatch",
                "selected_source_sha256": selected_map_sha256,
                "materialized_source_sha256": materialized_map_sha256,
            }
    map_id_snapshot = selected.get("map_id_snapshot")
    if isinstance(map_id_snapshot, dict):
        resolution = materialization.get("map_resolution")
        if not isinstance(resolution, dict) or resolution.get("status") != "materialized":
            return "not_replayed_map_id_input_unavailable", {
                "status": "unavailable",
                "reason": "materialized_map_id_resolution_missing",
            }
        if (
            resolution.get("map_id") != map_id_snapshot.get("map_id")
            or resolution.get("source_registry_sha256") != map_id_snapshot.get("registry_sha256")
            or resolution.get("resolved_map_sha256") != map_id_snapshot.get("map_sha256")
        ):
            return "not_replayed_map_id_input_changed_after_selection", {
                "status": "mismatch",
                "selected_map_id": map_id_snapshot.get("map_id"),
                "materialized_map_id": resolution.get("map_id"),
            }
        runner_registry_bundle_path = resolution.get("runner_registry_bundle_path")
        runner_registry_sha256 = resolution.get("runner_registry_sha256")
        if (
            not isinstance(runner_registry_bundle_path, str)
            or materialization.get("runner_map_registry_path") != runner_registry_bundle_path
            or not isinstance(runner_registry_sha256, str)
        ):
            return "not_replayed_map_id_input_unavailable", {
                "status": "mismatch",
                "reason": "runner_map_registry_path_or_digest_mismatch",
            }
        for field in ("resolved_map_id", "map_registry_source", "runner_map_registry"):
            asset = next(
                (
                    item
                    for item in materialization.get("assets", [])
                    if isinstance(item, dict) and item.get("field") == field
                ),
                None,
            )
            if not isinstance(asset, dict):
                return "not_replayed_map_id_input_unavailable", {
                    "status": "unavailable",
                    "reason": f"{field}_asset_missing",
                }
            expected_sha256 = {
                "resolved_map_id": map_id_snapshot.get("map_sha256"),
                "map_registry_source": map_id_snapshot.get("registry_sha256"),
                "runner_map_registry": runner_registry_sha256,
            }[field]
            if asset.get("source_sha256") != expected_sha256 or (
                field == "runner_map_registry"
                and asset.get("bundle_path") != runner_registry_bundle_path
            ):
                return "not_replayed_map_id_input_changed_after_selection", {
                    "status": "mismatch",
                    "reason": f"{field}_selected_digest_or_path_mismatch",
                }
            bundle_check = _bundled_input_binding(
                case_dir=case_dir,
                field=field,
                bundle_path=asset.get("bundle_path"),
                expected_sha256=asset.get("source_sha256"),
            )
            if bundle_check.get("status") != "bound":
                return "not_replayed_map_id_input_changed_after_selection", {
                    "status": bundle_check.get("status"),
                    "reason": bundle_check.get("reason") or "bundled_map_input_changed",
                    "input": field,
                }
    return None, {
        "status": "bound",
        "source_scenario_sha256": source_digest,
        "effective_scenario_hash": effective_hash,
        "map_file_sha256": selected.get("map_file_sha256"),
    }


def _initial_case_result(
    selected: dict[str, Any], context: _ReplayContext, materialization: dict[str, Any]
) -> dict[str, Any]:
    """Create the case receipt before execution so early exits remain auditable."""
    config = context.config
    return {
        "case_id": selected["case_id"],
        "source_candidate_index": selected["index"],
        "candidate": selected["candidate"],
        "objective": {
            "name": context.objective_name,
            "source_value": selected["objective_value"],
            "absolute_tolerance": context.tolerance,
        },
        "failure_attribution": selected["failure_attribution"],
        "feasibility_verdict": _feasibility_verdict(
            selected["candidate_payload"].get("certification_status")
        ),
        "source": {
            "revision": selected["source_revision"] or context.source_revision,
            "manifest_revision": context.source_revision,
            "episode_revision": selected["source_revision"],
            "revision_conflict": (
                selected["source_revision"] is not None
                and context.source_revision is not None
                and selected["source_revision"] != context.source_revision
            ),
            "search_method": context.search_method,
            "search_seed": config.get("seed"),
            "search_budget": config.get("budget"),
            "manifest_sha256": context.source_manifest_sha256,
            "episode_record_path": _display_path(selected["episode_path"], selected["source_root"]),
            "episode_record_sha256": selected["source_episode_sha256"],
            "bundled_episode_record_path": "source_episode.jsonl",
            "scenario_yaml_path": _display_path(selected["scenario_path"], selected["source_root"]),
            "scenario_yaml_sha256": selected["scenario_sha256"],
            "episode_id": selected["source_record"].get("episode_id"),
            "scenario_id": selected["source_record"].get("scenario_id"),
            "seed": selected["source_record"].get("seed"),
            "effective_scenario_hash": selected["effective_scenario_hash"],
            "gallery_checkout": {
                "revision": context.checkout_revision,
                "clean": context.checkout_clean,
                "dirty_paths": list(context.checkout_dirty_paths),
            },
        },
        "materialization": materialization,
        "execution_config": None,
        "replay_match": "unavailable",
        "verification_status": "not_replayed",
        "replay": None,
        "video_status": _video_status(context.video, [], attempted=False),
        "rendering": {"status": "not_attempted", "artifacts": []},
    }


def _run_one_episode(
    scenario_path: Path,
    episode_records: Path,
    runner_config: dict[str, Any],
    context: _ReplayContext,
    *,
    map_registry_path: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one canonical replay and turn execution errors into explicit case data."""
    previous_registry = os.environ.get("ROBOT_SF_MAP_REGISTRY")
    if map_registry_path is not None:
        os.environ["ROBOT_SF_MAP_REGISTRY"] = str(map_registry_path.resolve())
        scenario_loader._load_map_registry.cache_clear()
    try:
        replay_summary = run_batch(
            scenario_path,
            out_path=episode_records,
            schema_path=DEFAULT_SCHEMA_PATH,
            horizon=runner_config["horizon"],
            dt=runner_config["dt"],
            record_forces=runner_config["record_forces"],
            snqi_weights=runner_config["snqi_weights"],
            snqi_baseline=runner_config["snqi_baseline"],
            algo=context.policy,
            algo_config_path=runner_config["algo_config_path"],
            benchmark_profile=runner_config["benchmark_profile"],
            workers=runner_config["workers"],
            resume=False,
            append=False,
            fail_fast=False,
            video_enabled=context.video,
            video_renderer="synthetic" if context.video else "none",
            record_simulation_step_trace=True,
        )
    except Exception as exc:  # noqa: BLE001 - preserve per-candidate failures in the bundle
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if map_registry_path is not None:
            if previous_registry is None:
                os.environ.pop("ROBOT_SF_MAP_REGISTRY", None)
            else:
                os.environ["ROBOT_SF_MAP_REGISTRY"] = previous_registry
            scenario_loader._load_map_registry.cache_clear()
    return replay_summary, None


def _record_replay_comparison(  # noqa: PLR0913 - preserve explicit replay diagnostics
    selected: dict[str, Any],
    context: _ReplayContext,
    result: dict[str, Any],
    replay_record: dict[str, Any],
    *,
    materialization: dict[str, Any],
    replay_dir: Path,
    episode_records: Path,
    case_dir: Path,
    replay_summary: dict[str, Any],
    replay_availability: dict[str, Any] | None,
    replay_availability_error: str | None,
) -> None:
    """Compare one replay to its source and add available diagnostic artifacts."""
    objective_name = context.objective_name
    root = context.root
    output_dir = context.output_dir
    tolerance = context.tolerance

    replay_revision = _record_revision(replay_record) or _git_revision(root)
    identity_match, identity_details = _episode_identity_matches(
        selected["source_record"], replay_record, policy=context.policy
    )
    outcome_match, outcome_details = _outcomes_match(
        selected["source_record"], replay_record, objective_name, tolerance=tolerance
    )
    replay_objective = _objective_value(
        context.objective,
        selected["candidate"],
        replay_record,
        episode_records,
        selected["candidate_payload"].get("certification_status"),
    )
    objective_match = replay_objective is not None and math.isclose(
        selected["objective_value"], replay_objective, rel_tol=0.0, abs_tol=tolerance
    )
    source_input_binding = result.get("source_input_binding")
    if not isinstance(source_input_binding, dict):
        source_input_binding = {"status": "unknown", "checks": []}
    algorithm_config_binding = _algorithm_config_binding(selected["source_record"], replay_record)
    input_checks = source_input_binding.get("checks")
    if not isinstance(input_checks, list):
        input_checks = []
    input_checks.append(algorithm_config_binding)
    source_input_binding["checks"] = input_checks
    source_input_binding["status"] = _combined_binding_status(input_checks)
    result["source_input_binding"] = source_input_binding
    source_revision_for_case = selected["source_revision"] or context.source_revision
    checkout_after_replay = _git_checkout_state(root)
    result["source"]["gallery_checkout_after_replay"] = {
        "revision": checkout_after_replay["revision"],
        "clean": checkout_after_replay["clean"],
        "dirty_paths": checkout_after_replay["dirty_paths"],
    }
    result["replay_match"], result["verification_status"] = _replay_verification_status(
        replay_availability_error=replay_availability_error,
        source_input_status=source_input_binding["status"],
        identity_match=identity_match,
        outcome_match=outcome_match,
        objective_match=objective_match,
        source_revision_conflict=result["source"]["revision_conflict"],
        source_revision=source_revision_for_case,
        manifest_revision=context.source_revision,
        replay_revision=replay_revision,
        checkout_revision=checkout_after_replay["revision"],
        checkout_clean=(
            context.checkout_clean
            and checkout_after_replay["clean"]
            and checkout_after_replay["revision"] == context.checkout_revision
        ),
    )

    video_artifacts = _video_artifacts(replay_dir, output_dir)
    video_status = _video_status(context.video, video_artifacts, attempted=True)
    result["video_status"] = video_status
    result["replay"] = {
        "summary": replay_summary,
        "episode_record_path": "replay/episode_records.jsonl",
        "episode_record_sha256": _sha256_file(episode_records),
        "episode_id": replay_record.get("episode_id"),
        "scenario_id": replay_record.get("scenario_id"),
        "seed": replay_record.get("seed"),
        "revision": replay_revision,
        "benchmark_availability": replay_availability,
        "availability_error": replay_availability_error,
        "revision_matches_source": (
            source_revision_for_case is not None and replay_revision == source_revision_for_case
        ),
        "source_revision_for_verification": source_revision_for_case,
        "objective_value": replay_objective,
        "objective_matches": objective_match,
        "identity_matches": identity_match,
        "identity_comparison": identity_details,
        "outcome_matches": outcome_match,
        "outcome_comparison": outcome_details,
        "failure_attribution_matches": outcome_details.get("failure_attribution_matches"),
        "source_input_binding": source_input_binding,
        "algorithm_config_binding": algorithm_config_binding,
        "video_artifacts": video_artifacts,
        "video_status": video_status,
    }
    _write_json(case_dir / "replay_comparison.json", result["replay"])
    if context.render:
        result["rendering"] = _render_replay(
            replay_record,
            episode_records,
            replay_dir,
            case_dir / "figures",
            output_dir,
            map_path=_materialized_map_path(materialization, case_dir),
        )


def _replay_verification_status(  # noqa: C901, PLR0913 - preserve explicit provenance gates
    *,
    replay_availability_error: str | None,
    source_input_status: str,
    identity_match: bool,
    outcome_match: bool,
    objective_match: bool,
    source_revision_conflict: bool,
    source_revision: str | None,
    manifest_revision: str | None,
    replay_revision: str | None,
    checkout_revision: str | None,
    checkout_clean: bool,
) -> tuple[str, str]:
    """Return replay match and verification labels without weakening provenance gates."""
    if replay_availability_error is not None:
        return "unavailable", "replay_execution_unavailable"
    if source_input_status == "mismatch":
        return "mismatch", "replay_input_mismatch"
    if not (identity_match and outcome_match and objective_match):
        return "mismatch", "replay_mismatch"
    if source_revision_conflict or (
        source_revision is not None
        and manifest_revision is not None
        and source_revision != manifest_revision
    ):
        return "match", "source_revision_conflict"
    if source_revision is None:
        return "match", "outcome_reproduced_source_revision_unknown"
    if replay_revision is None:
        return "match", "outcome_reproduced_replay_revision_unknown"
    if source_revision != replay_revision:
        return "match", "outcome_reproduced_revision_changed"
    if checkout_revision != source_revision:
        return "match", "outcome_reproduced_checkout_revision_changed"
    if not checkout_clean:
        return "match", "outcome_reproduced_checkout_dirty"
    if source_input_status != "bound":
        return "match", "outcome_reproduced_source_inputs_unbound"
    return "match", "verified"


def _replay_availability(  # noqa: C901 - keep independent runner evidence gates explicit
    summary: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Require a successful canonical runner receipt before treating a replay as evidence."""
    if not isinstance(summary, dict):
        return None, "replay_summary_missing_or_malformed"

    failures = summary.get("failures")
    if not isinstance(failures, list):
        return None, "replay_failures_missing_or_malformed"
    if failures:
        return None, "replay_runner_reported_failures"

    preflight = summary.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("status") != "ok":
        return None, "replay_preflight_not_successful"

    algorithm_contract = summary.get("algorithm_metadata_contract")
    if not isinstance(algorithm_contract, dict) or algorithm_contract.get("status") != "ok":
        return None, "replay_algorithm_metadata_unavailable"
    execution_mode = resolve_execution_mode(algorithm_contract)
    if execution_mode == "unknown":
        return None, "replay_execution_mode_unknown"
    if execution_mode != "native":
        return None, "replay_execution_mode_not_native"

    try:
        expected = availability_payload(summary)
    except (TypeError, ValueError, OverflowError):
        return None, "replay_benchmark_availability_unavailable"
    reported = summary.get("benchmark_availability")
    if not isinstance(reported, dict):
        return None, "replay_benchmark_availability_missing_or_malformed"
    if any(reported.get(key) != value for key, value in expected.items()):
        return expected, "replay_benchmark_availability_mismatch"
    if expected["benchmark_success"] is not True or expected["availability_status"] != "available":
        return expected, "replay_benchmark_unavailable"
    if expected["readiness_status"] in {"fallback", "degraded"}:
        return expected, "replay_benchmark_degraded"
    return expected, None


def _replay_record_availability(record: dict[str, Any]) -> str | None:
    """Reject episode-row execution metadata that contradicts native replay provenance."""
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, dict) or str(metadata.get("status", "")).strip().lower() != "ok":
        return "replay_episode_algorithm_metadata_unavailable"
    if resolve_execution_mode(metadata) != "native":
        return "replay_episode_execution_mode_not_native"
    if _runtime_algorithm_fallback_marker(record) is not None:
        return "replay_episode_runtime_fallback_or_degraded"
    return None


def _snapshot_map_id_input(  # noqa: C901, PLR0912 - keep source and map snapshot gates explicit
    scenario: dict[str, Any],
    *,
    scenario_path: Path,
    episode_path: Path,
    source_record: dict[str, Any],
    root: Path,
    source_root: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Capture map_id registry and map bytes before candidate selection completes."""
    raw_map_id = scenario.get("map_id")
    if raw_map_id is None:
        return None, None
    if not isinstance(raw_map_id, str) or not raw_map_id.strip():
        return None, "scenario_map_id_invalid"

    registry_override = os.environ.get("ROBOT_SF_MAP_REGISTRY")
    registry_path = scenario_loader._resolve_map_registry_path()
    if registry_path is None:
        return None, "scenario_map_registry_unavailable"
    registry_path = registry_path.expanduser().resolve()
    try:
        required_profile = scenario_loader._resolve_required_map_profile(
            scenario, source=scenario_path
        )
    except ValueError as exc:
        return None, f"scenario_map_profile_invalid: {exc}"
    try:
        registry_bytes = registry_path.read_bytes()
        registry_payload = yaml.safe_load(registry_bytes.decode("utf-8"))
        if not isinstance(registry_payload, dict):
            return None, "scenario_map_registry_invalid_shape"
        scenario_loader._validate_catalog_header(registry_payload, registry_path=registry_path)
        map_registry = {}
        for resolved_map_id, row in scenario_loader._iter_map_registry_entries(
            registry_payload, registry_path=registry_path
        ):
            scenario_loader._register_map_entry(
                map_registry,
                map_id=resolved_map_id,
                row=row,
                registry_path=registry_path,
            )
        map_path = scenario_loader._resolve_map_id(
            raw_map_id.strip(),
            map_registry=map_registry,
            source=scenario_path,
            required_profile=required_profile,
        ).resolve()
        map_bytes = map_path.read_bytes()
        registry_bytes_after = registry_path.read_bytes()
        map_bytes_after = map_path.read_bytes()
    except (OSError, ValueError, TypeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        return None, f"scenario_map_id_resolution_failed: {type(exc).__name__}: {exc}"
    if registry_bytes != registry_bytes_after or map_bytes != map_bytes_after:
        return None, "scenario_map_registry_or_map_changed_during_selection"
    if os.environ.get("ROBOT_SF_MAP_REGISTRY") != registry_override:
        return None, "scenario_map_registry_environment_changed_during_selection"

    entries = registry_payload.get("maps", registry_payload)
    if isinstance(entries, list):
        selected_registry_row = next(
            (
                dict(entry)
                for entry in entries
                if isinstance(entry, dict)
                and (entry.get("map_id") or entry.get("id")) == raw_map_id.strip()
            ),
            None,
        )
    elif isinstance(entries, dict):
        raw_entry = entries.get(raw_map_id.strip())
        selected_registry_row = (
            {"map_id": raw_map_id.strip(), "path": raw_entry}
            if isinstance(raw_entry, str)
            else None
        )
    else:
        selected_registry_row = None
    if selected_registry_row is None:
        return None, "scenario_map_registry_entry_not_materializable"

    source_episode_map_binding: dict[str, Any] = {
        "status": "unknown",
        "reason": "source_episode_map_file_missing",
    }
    scenario_params = source_record.get("scenario_params")
    raw_source_map = scenario_params.get("map_file") if isinstance(scenario_params, dict) else None
    if isinstance(raw_source_map, str) and raw_source_map.strip():
        source_episode_map = _resolve_referenced_file(
            raw_source_map, episode_path.parent, source_root, root
        )
        if source_episode_map is None or not source_episode_map.is_file():
            source_episode_map_binding = {
                "status": "unknown",
                "reason": "source_episode_map_file_unresolvable",
            }
        else:
            try:
                source_episode_map_sha256 = _sha256_file(source_episode_map)
            except OSError:
                source_episode_map_sha256 = None
            source_episode_map_binding = {
                "status": (
                    "bound"
                    if source_episode_map_sha256 == hashlib.sha256(map_bytes).hexdigest()
                    else "mismatch"
                    if source_episode_map_sha256 is not None
                    else "unknown"
                ),
                "source_path": _display_path(source_episode_map, source_root),
                "source_path_repository_relative": _is_repository_relative_path(
                    source_episode_map, source_root
                ),
                "source_sha256": source_episode_map_sha256,
                "resolved_map_sha256": hashlib.sha256(map_bytes).hexdigest(),
            }

    return {
        "map_id": raw_map_id.strip(),
        "required_profile": required_profile,
        "registry_path": registry_path,
        "registry_path_display": _display_path(registry_path, source_root),
        "registry_path_repository_relative": _is_repository_relative_path(
            registry_path, source_root
        ),
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "registry_bytes": registry_bytes,
        "registry_row": selected_registry_row,
        "registry_override_used": bool(registry_override),
        "source_registry_binding": _source_recorded_map_registry_binding(
            source_record,
            registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        ),
        "map_path": map_path,
        "map_path_display": _display_path(map_path, source_root),
        "map_path_repository_relative": _is_repository_relative_path(map_path, source_root),
        "map_sha256": hashlib.sha256(map_bytes).hexdigest(),
        "map_bytes": map_bytes,
        "source_episode_map_binding": source_episode_map_binding,
    }, None


def _source_recorded_map_registry_binding(
    source_record: dict[str, Any], *, registry_sha256: str
) -> dict[str, Any]:
    """Check any registry digest attested by the source episode or its provenance."""
    containers: list[dict[str, Any]] = [source_record]
    for key in ("provenance", "scenario_params", "algorithm_metadata"):
        value = source_record.get(key)
        if isinstance(value, dict):
            containers.append(value)
            nested = value.get("provenance")
            if isinstance(nested, dict):
                containers.append(nested)
    for container in containers:
        registry = container.get("map_registry")
        registry_info = registry if isinstance(registry, dict) else {}
        recorded_sha256 = container.get("map_registry_sha256") or registry_info.get("sha256")
        if not isinstance(recorded_sha256, str):
            continue
        if recorded_sha256.lower() != registry_sha256.lower():
            return {
                "status": "mismatch",
                "reason": "source_episode_map_registry_digest_differs",
                "source_sha256": recorded_sha256,
                "resolved_sha256": registry_sha256,
            }
        return {"status": "bound", "source_sha256": recorded_sha256}
    return {
        "status": "unknown",
        "reason": "source_episode_map_registry_digest_missing",
    }


def _materialize_scenario(  # noqa: C901 - ordered fail-closed provenance gates
    source: Path,
    input_dir: Path,
    *,
    root: Path,
    source_root: Path,
    selected_map_id_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a generated scenario and its declared file inputs into a stable bundle."""
    try:
        source_bytes = source.read_bytes()
        payload = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {"status": "unavailable", "reason": f"scenario_read_failed: {exc}"}
    except UnicodeDecodeError as exc:
        return {"status": "unavailable", "reason": f"scenario_read_failed: {exc}"}
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        return {"status": "unavailable", "reason": "scenario_matrix_shape_invalid"}
    input_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = input_dir / "assets"
    assets_dir.mkdir()
    copied_assets: list[dict[str, str]] = []
    effective_hashes: list[str] = []
    referenced_asset_bytes: dict[tuple[int, str], bytes] = {}
    map_id_materialization: dict[str, Any] | None = None
    for scenario_index, scenario in enumerate(payload["scenarios"]):
        if not isinstance(scenario, dict):
            return {"status": "unavailable", "reason": f"scenario_{scenario_index}_invalid"}
        route_payload, route_bytes, route_error = _materialized_route_payload(
            scenario, source=source, root=root, source_root=source_root
        )
        if route_error is not None or route_payload is None:
            return {
                "status": "unavailable",
                "reason": route_error or "route_overrides_file_invalid",
            }
        if route_bytes is not None:
            referenced_asset_bytes[(scenario_index, "route_overrides_file")] = route_bytes
        if scenario.get("map_id") is not None:
            if selected_map_id_snapshot is None:
                return {
                    "status": "unavailable",
                    "reason": "scenario_map_id_input_not_snapshotted",
                }
            if scenario.get("map_id") != selected_map_id_snapshot.get("map_id"):
                return {"status": "unavailable", "reason": "scenario_map_id_changed"}
            map_id_materialization, map_error = _materialize_map_id_input(
                selected_map_id_snapshot,
                assets_dir=assets_dir,
                input_dir=input_dir,
            )
            if map_error is not None or map_id_materialization is None:
                return {
                    "status": "unavailable",
                    "reason": map_error or "scenario_map_id_materialization_failed",
                }
            copied_assets.extend(map_id_materialization["assets"])
        try:
            effective_hashes.append(compute_effective_scenario_hash(scenario, route_payload))
        except (TypeError, ValueError):
            return {"status": "unavailable", "reason": "effective_scenario_hash_unavailable"}
        for field in ("map_file", "route_overrides_file"):
            raw_ref = scenario.get(field)
            if raw_ref is None:
                continue
            asset, asset_name, asset_error = _copy_materialized_asset(
                raw_ref,
                field=field,
                source=source,
                assets_dir=assets_dir,
                source_root=source_root,
                root=root,
                pinned_bytes=referenced_asset_bytes.get((scenario_index, field)),
            )
            if asset_error is not None or asset is None or asset_name is None:
                return {"status": "unavailable", "reason": asset_error or f"{field}_missing"}
            scenario[field] = (Path("assets") / asset_name).as_posix()
            copied_assets.append(asset)
    scenario_path = input_dir / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "status": "materialized",
        "scenario_path": "inputs/scenario.yaml",
        "scenario_sha256": _sha256_file(scenario_path),
        "source_scenario_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "effective_scenario_hash": effective_hashes[0] if len(effective_hashes) == 1 else None,
        "assets": copied_assets,
        "map_resolution": (
            map_id_materialization["receipt"]
            if map_id_materialization is not None
            else _implicit_map_resolution_status(payload["scenarios"])
        ),
        "runner_map_registry_path": (
            map_id_materialization["receipt"]["runner_registry_bundle_path"]
            if map_id_materialization is not None
            else None
        ),
    }


def _implicit_map_resolution_status(scenarios: list[Any]) -> dict[str, Any]:
    """Describe map inputs that are not explicitly materialized by the gallery."""
    rows = [scenario for scenario in scenarios if isinstance(scenario, dict)]
    if rows and all(isinstance(scenario.get("map_file"), str) for scenario in rows):
        return {"status": "explicit_map_file_materialized"}
    if any(isinstance(scenario.get("map_id"), str) for scenario in rows):
        return {
            "status": "unknown",
            "reason": "scenario_map_id_resolution_snapshot_missing",
        }
    return {
        "status": "unknown",
        "reason": "implicit_default_map_pool_not_materialized",
    }


def _materialize_map_id_input(
    snapshot: dict[str, Any],
    *,
    assets_dir: Path,
    input_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Bundle the selected map and registry, pinning map_id replay resolution."""
    map_bytes = snapshot.get("map_bytes")
    registry_bytes = snapshot.get("registry_bytes")
    map_path = snapshot.get("map_path")
    registry_path = snapshot.get("registry_path")
    map_sha256 = snapshot.get("map_sha256")
    registry_sha256 = snapshot.get("registry_sha256")
    if (
        not all(isinstance(value, bytes) for value in (map_bytes, registry_bytes))
        or not isinstance(map_path, Path)
        or not isinstance(registry_path, Path)
    ):
        return None, "scenario_map_id_snapshot_incomplete"
    if hashlib.sha256(map_bytes).hexdigest() != map_sha256:
        return None, "scenario_map_snapshot_digest_mismatch"
    if hashlib.sha256(registry_bytes).hexdigest() != registry_sha256:
        return None, "scenario_map_registry_snapshot_digest_mismatch"

    map_name = f"{map_sha256[:12]}-{map_path.name}"
    bundled_map = assets_dir / map_name
    bundled_map.write_bytes(map_bytes)
    source_registry_name = f"{registry_sha256[:12]}-map_registry.yaml"
    bundled_source_registry = assets_dir / source_registry_name
    bundled_source_registry.write_bytes(registry_bytes)
    try:
        registry_payload = yaml.safe_load(registry_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, f"scenario_map_registry_invalid: {type(exc).__name__}: {exc}"
    if not isinstance(registry_payload, dict):
        return None, "scenario_map_registry_entry_not_materializable"
    selected_row = snapshot.get("registry_row")
    selected_row = dict(selected_row) if isinstance(selected_row, dict) else None
    if selected_row is None:
        return None, "scenario_map_registry_entry_not_materializable"
    selected_row["path"] = str(bundled_map.resolve())
    selected_row["source_sha256"] = map_sha256
    replay_registry_payload = dict(registry_payload)
    original_entries = registry_payload.get("maps")
    if isinstance(original_entries, list):
        replay_registry_payload["maps"] = [selected_row]
    elif isinstance(original_entries, dict):
        replay_registry_payload["maps"] = {snapshot["map_id"]: str(bundled_map.resolve())}
    else:
        replay_registry_payload = {snapshot["map_id"]: str(bundled_map.resolve())}
    runner_registry = input_dir / "map_registry.yaml"
    runner_registry_bytes = yaml.safe_dump(
        replay_registry_payload, sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    runner_registry.write_bytes(runner_registry_bytes)
    runner_registry_sha256 = hashlib.sha256(runner_registry_bytes).hexdigest()
    assets = [
        {
            "field": "resolved_map_id",
            "source_path": snapshot["map_path_display"],
            "source_path_repository_relative": snapshot["map_path_repository_relative"],
            "source_sha256": map_sha256,
            "bundle_path": (Path("inputs") / "assets" / map_name).as_posix(),
        },
        {
            "field": "map_registry_source",
            "source_path": snapshot["registry_path_display"],
            "source_path_repository_relative": snapshot["registry_path_repository_relative"],
            "source_sha256": registry_sha256,
            "bundle_path": (Path("inputs") / "assets" / source_registry_name).as_posix(),
        },
        {
            "field": "runner_map_registry",
            "source_path": None,
            "source_path_repository_relative": False,
            "source_sha256": runner_registry_sha256,
            "bundle_path": "inputs/map_registry.yaml",
        },
    ]
    return {
        "assets": assets,
        "receipt": {
            "status": "materialized",
            "kind": "map_id_registry_resolution",
            "map_id": snapshot["map_id"],
            "required_profile": snapshot["required_profile"],
            "source_registry_path": snapshot["registry_path_display"],
            "source_registry_sha256": registry_sha256,
            "source_registry_override_used": snapshot["registry_override_used"],
            "source_registry_binding": snapshot["source_registry_binding"],
            "resolved_map_path": snapshot["map_path_display"],
            "resolved_map_sha256": map_sha256,
            "source_episode_map_binding": snapshot["source_episode_map_binding"],
            "runner_registry_bundle_path": "inputs/map_registry.yaml",
            "runner_registry_sha256": runner_registry_sha256,
        },
    }, None


def _materialized_route_payload(
    scenario: dict[str, Any], *, source: Path, root: Path, source_root: Path
) -> tuple[dict[str, Any] | None, bytes | None, str | None]:
    """Read route overrides once so the hash and bundled bytes describe one input."""
    route_ref = scenario.get("route_overrides_file")
    if route_ref is None:
        route_payload = scenario.get("route_overrides") or {}
        if not isinstance(route_payload, dict):
            return None, None, "route_overrides_file_invalid"
        return route_payload, None, None
    route_path = _resolve_referenced_file(route_ref, source.parent, source_root, root)
    if route_path is None or not route_path.is_file():
        return None, None, "route_overrides_file_missing"
    try:
        route_bytes = route_path.read_bytes()
        route_payload = yaml.safe_load(route_bytes.decode("utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        return None, None, f"route_overrides_file_invalid: {exc}"
    if not isinstance(route_payload, dict):
        return None, None, "route_overrides_file_invalid"
    return route_payload, route_bytes, None


def _copy_materialized_asset(
    raw_ref: Any,
    *,
    field: str,
    source: Path,
    assets_dir: Path,
    source_root: Path,
    root: Path,
    pinned_bytes: bytes | None,
) -> tuple[dict[str, str] | None, str | None, str | None]:
    """Copy one scenario asset and return its immutable bundle and source receipt."""
    asset_path = _resolve_referenced_file(raw_ref, source.parent, source_root, root)
    if asset_path is None or not asset_path.is_file():
        return None, None, f"{field}_missing"
    try:
        asset_bytes = pinned_bytes if pinned_bytes is not None else asset_path.read_bytes()
    except OSError as exc:
        return None, None, f"{field}_read_failed: {exc}"
    digest = hashlib.sha256(asset_bytes).hexdigest()
    name = f"{digest[:12]}-{asset_path.name}"
    destination = assets_dir / name
    if not destination.exists():
        destination.write_bytes(asset_bytes)
    return (
        {
            "field": field,
            "source_path": _display_path(asset_path, source_root),
            "source_path_repository_relative": _is_repository_relative_path(
                asset_path, source_root
            ),
            "source_sha256": digest,
            "bundle_path": (Path("inputs") / "assets" / name).as_posix(),
        },
        name,
        None,
    )


def _runner_config(
    config: dict[str, Any], *, case_dir: Path, root: Path, source_root: Path
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    """Copy and hash file-backed settings, then record the effective runner configuration."""
    paths: dict[str, Path | None] = {}
    file_inputs: list[dict[str, str]] = []
    for name in ("algo_config_path", "snqi_weights_path", "snqi_baseline_path"):
        raw_path = config.get(name)
        if raw_path in (None, ""):
            paths[name] = None
            continue
        resolved = _resolve_repo_file(raw_path, source_root, root)
        if resolved is None or not resolved.is_file():
            error = f"{name}_missing"
            return (
                None,
                error,
                {"status": "unavailable", "reason": error, "file_inputs": file_inputs},
            )
        digest = _sha256_file(resolved)
        bundle_path = Path("inputs") / "config" / f"{digest[:12]}-{resolved.name}"
        destination = case_dir / bundle_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(resolved, destination)
        paths[name] = destination
        file_inputs.append(
            {
                "field": name,
                "source_path": _display_path(resolved, source_root),
                "source_path_repository_relative": _is_repository_relative_path(
                    resolved, source_root
                ),
                "source_sha256": digest,
                "bundle_path": bundle_path.as_posix(),
            }
        )
    try:
        snqi_weights = _load_json_file(paths["snqi_weights_path"])
        snqi_baseline = _load_json_file(paths["snqi_baseline_path"])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        error = f"snqi_configuration_invalid: {exc}"
        return None, error, {"status": "unavailable", "reason": error, "file_inputs": file_inputs}
    try:
        horizon = _positive_integer_setting(config, "horizon", default=100)
        dt = _positive_float_setting(config, "dt", default=0.1)
        workers = _positive_integer_setting(config, "workers", default=1)
        record_forces = _boolean_setting(config, "record_forces", default=True)
        benchmark_profile = _text_setting(config, "benchmark_profile", default="baseline-safe")
    except ValueError as exc:
        error = f"runner_configuration_invalid: {exc}"
        return None, error, {"status": "unavailable", "reason": error, "file_inputs": file_inputs}
    execution_config = {
        "status": "ready",
        "horizon": horizon,
        "dt": dt,
        "record_forces": record_forces,
        "benchmark_profile": benchmark_profile,
        "workers": workers,
        "file_inputs": file_inputs,
    }
    return (
        {
            "horizon": horizon,
            "dt": dt,
            "record_forces": record_forces,
            "snqi_weights": snqi_weights,
            "snqi_baseline": snqi_baseline,
            "algo_config_path": str(paths["algo_config_path"])
            if paths["algo_config_path"]
            else None,
            "benchmark_profile": benchmark_profile,
            "workers": workers,
        },
        None,
        execution_config,
    )


def _source_input_binding(  # noqa: C901, PLR0912 - explicit evidence checks stay together
    materialization: dict[str, Any],
    execution_config: dict[str, Any],
    *,
    effective_scenario_hash: str,
    root: Path,
    case_dir: Path,
) -> dict[str, Any]:
    """Bind materialized map/config files to tracked inputs at the source revision."""
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "input": "effective_scenario_hash",
            "status": "bound"
            if isinstance(effective_scenario_hash, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", effective_scenario_hash)
            else "unknown",
            "evidence": "recomputed against source manifest during candidate selection",
        }
    )
    assets = materialization.get("assets")
    if not isinstance(assets, list):
        checks.append({"input": "materialized_assets", "status": "unknown"})
    else:
        for asset in assets:
            if not isinstance(asset, dict):
                checks.append({"input": "materialized_asset", "status": "unknown"})
                continue
            field = asset.get("field")
            checks.append(
                _bundled_input_binding(
                    case_dir=case_dir,
                    field=str(field or "materialized_asset"),
                    bundle_path=asset.get("bundle_path"),
                    expected_sha256=asset.get("source_sha256"),
                )
            )
            if field == "map_file":
                checks.append(
                    _tracked_source_file_binding(
                        root=root,
                        field="map_file",
                        source_path=asset.get("source_path"),
                        repository_relative=asset.get("source_path_repository_relative"),
                        expected_sha256=asset.get("source_sha256"),
                    )
                )
            elif field == "resolved_map_id":
                checks.append(
                    _tracked_source_file_binding(
                        root=root,
                        field="resolved_map_file",
                        source_path=asset.get("source_path"),
                        repository_relative=asset.get("source_path_repository_relative"),
                        expected_sha256=asset.get("source_sha256"),
                    )
                )
            elif field == "map_registry_source":
                checks.append(
                    _tracked_source_file_binding(
                        root=root,
                        field="map_registry",
                        source_path=asset.get("source_path"),
                        repository_relative=asset.get("source_path_repository_relative"),
                        expected_sha256=asset.get("source_sha256"),
                    )
                )
            elif field == "runner_map_registry":
                # This derived registry is the exact one passed to load_scenarios;
                # its source registry and selected map are checked separately.
                pass
            elif field == "route_overrides_file":
                checks.append(
                    _tracked_source_file_binding(
                        root=root,
                        field="route_overrides_file",
                        source_path=asset.get("source_path"),
                        repository_relative=asset.get("source_path_repository_relative"),
                        expected_sha256=asset.get("source_sha256"),
                    )
                )
                checks.append(
                    {
                        "input": field,
                        "status": "bound" if checks[0]["status"] == "bound" else "unknown",
                        "evidence": "included in the recomputed effective scenario hash",
                    }
                )
            else:
                checks.append({"input": str(field or "materialized_asset"), "status": "unknown"})

    map_resolution = materialization.get("map_resolution")
    if (
        isinstance(map_resolution, dict)
        and map_resolution.get("kind") == "map_id_registry_resolution"
    ):
        source_episode_binding = map_resolution.get("source_episode_map_binding")
        if isinstance(source_episode_binding, dict):
            checks.append(
                {
                    "input": "source_episode_map_file",
                    **source_episode_binding,
                }
            )
        else:
            checks.append(
                {
                    "input": "source_episode_map_file",
                    "status": "unknown",
                    "reason": "source_episode_map_binding_missing",
                }
            )
        source_registry_binding = map_resolution.get("source_registry_binding")
        if isinstance(source_registry_binding, dict):
            checks.append(
                {
                    "input": "source_episode_map_registry",
                    **source_registry_binding,
                }
            )
        else:
            checks.append(
                {
                    "input": "source_episode_map_registry",
                    "status": "unknown",
                    "reason": "source_episode_map_registry_binding_missing",
                }
            )
        if map_resolution.get("source_registry_override_used") is True:
            checks.append(
                {
                    "input": "source_map_registry_environment",
                    "status": "unknown",
                    "reason": "source_registry_override_not_attested_by_episode_provenance",
                }
            )
        elif map_resolution.get("source_registry_override_used") is not False:
            checks.append(
                {
                    "input": "source_map_registry_environment",
                    "status": "unknown",
                    "reason": "source_registry_environment_status_unknown",
                }
            )
    elif not (
        isinstance(map_resolution, dict)
        and map_resolution.get("status") == "explicit_map_file_materialized"
    ):
        checks.append(
            {
                "input": "effective_map_resolution",
                "status": "unknown",
                "reason": (
                    map_resolution.get("reason", "map_resolution_unknown")
                    if isinstance(map_resolution, dict)
                    else "map_resolution_receipt_missing"
                ),
            }
        )

    file_inputs = execution_config.get("file_inputs")
    if not isinstance(file_inputs, list):
        checks.append({"input": "runner_config_files", "status": "unknown"})
    else:
        for file_input in file_inputs:
            if not isinstance(file_input, dict):
                checks.append({"input": "runner_config_file", "status": "unknown"})
                continue
            checks.append(
                _bundled_input_binding(
                    case_dir=case_dir,
                    field=str(file_input.get("field") or "runner_config_file"),
                    bundle_path=file_input.get("bundle_path"),
                    expected_sha256=file_input.get("source_sha256"),
                )
            )
            checks.append(
                _tracked_source_file_binding(
                    root=root,
                    field=str(file_input.get("field") or "runner_config_file"),
                    source_path=file_input.get("source_path"),
                    repository_relative=file_input.get("source_path_repository_relative"),
                    expected_sha256=file_input.get("source_sha256"),
                )
            )
    return {"status": _combined_binding_status(checks), "checks": checks}


def _bundled_input_binding(
    *, case_dir: Path, field: str, bundle_path: Any, expected_sha256: Any
) -> dict[str, Any]:
    """Verify the exact copied bytes passed to the replay runner."""
    if not isinstance(bundle_path, str) or not bundle_path.strip():
        return {"input": field, "status": "unknown", "reason": "bundle_path_missing"}
    relative_path = Path(bundle_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return {"input": field, "status": "unknown", "reason": "unsafe_bundle_path"}
    resolved = (case_dir / relative_path).resolve()
    try:
        resolved.relative_to(case_dir.resolve())
    except ValueError:
        return {"input": field, "status": "unknown", "reason": "bundle_path_outside_case"}
    if not resolved.is_file():
        return {"input": field, "status": "unknown", "reason": "bundled_input_missing"}
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
    ):
        return {"input": field, "status": "unknown", "reason": "source_digest_missing_or_invalid"}
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256.lower() != expected_sha256.lower():
        return {
            "input": field,
            "status": "mismatch",
            "source_sha256": expected_sha256,
            "bundle_sha256": actual_sha256,
        }
    return {"input": field, "status": "bound", "bundle_sha256": actual_sha256}


def _tracked_source_file_binding(
    *,
    root: Path,
    field: str,
    source_path: Any,
    repository_relative: Any,
    expected_sha256: Any,
) -> dict[str, Any]:
    """Check one materialized input against a tracked file in the exact-source checkout."""
    if (
        repository_relative is not True
        or not isinstance(source_path, str)
        or not source_path.strip()
    ):
        return {"input": field, "status": "unknown", "reason": "source_path_not_repo_relative"}
    relative_path = Path(source_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return {"input": field, "status": "unknown", "reason": "unsafe_source_path"}
    resolved = (root / relative_path).resolve()
    try:
        normalized = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return {"input": field, "status": "unknown", "reason": "source_path_outside_checkout"}
    if not resolved.is_file():
        return {"input": field, "status": "unknown", "reason": "tracked_source_file_missing"}
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
    ):
        return {"input": field, "status": "unknown", "reason": "source_digest_missing_or_invalid"}
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", normalized],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if tracked.returncode != 0:
        return {"input": field, "status": "unknown", "reason": "source_file_not_git_tracked"}
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256.lower() != expected_sha256.lower():
        return {
            "input": field,
            "status": "mismatch",
            "source_sha256": expected_sha256,
            "checkout_sha256": actual_sha256,
        }
    return {"input": field, "status": "bound", "source_sha256": expected_sha256}


def _combined_binding_status(checks: list[dict[str, Any]]) -> str:
    """Combine input-binding evidence without promoting missing inputs to a match."""
    statuses = [check.get("status") for check in checks]
    if "mismatch" in statuses:
        return "mismatch"
    if not checks or any(status != "bound" for status in statuses):
        return "unknown"
    return "bound"


def _algorithm_config_binding(source: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    """Compare the canonical planner configuration hashes recorded by both episodes."""
    source_metadata = source.get("algorithm_metadata")
    replay_metadata = replay.get("algorithm_metadata")
    source_hash = source_metadata.get("config_hash") if isinstance(source_metadata, dict) else None
    replay_hash = replay_metadata.get("config_hash") if isinstance(replay_metadata, dict) else None
    if not isinstance(source_hash, str) or not source_hash.strip():
        return {
            "input": "algorithm_config_hash",
            "status": "unknown",
            "reason": "source_hash_missing",
        }
    if not isinstance(replay_hash, str) or not replay_hash.strip():
        return {
            "input": "algorithm_config_hash",
            "status": "unknown",
            "reason": "replay_hash_missing",
        }
    if source_hash != replay_hash:
        return {
            "input": "algorithm_config_hash",
            "status": "mismatch",
            "source_config_hash": source_hash,
            "replay_config_hash": replay_hash,
        }
    return {
        "input": "algorithm_config_hash",
        "status": "bound",
        "source_config_hash": source_hash,
    }


def _positive_integer_setting(config: dict[str, Any], name: str, *, default: int) -> int:
    """Read one positive integer runner setting without coercing booleans or strings."""
    value = config.get(name)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float_setting(config: dict[str, Any], name: str, *, default: float) -> float:
    """Read one finite, positive numeric runner setting without coercing strings."""
    value = config.get(name)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return parsed


def _boolean_setting(config: dict[str, Any], name: str, *, default: bool) -> bool:
    """Read a boolean runner setting without truthiness coercion."""
    value = config.get(name)
    if value is None:
        value = default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _text_setting(config: dict[str, Any], name: str, *, default: str) -> str:
    """Read a non-empty text setting without stringifying malformed values."""
    value = config.get(name)
    if value is None:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _render_replay(
    record: dict[str, Any],
    episode_records: Path,
    replay_dir: Path,
    figure_dir: Path,
    output_dir: Path,
    *,
    map_path: Path | None,
) -> dict[str, Any]:
    """Render the exact replay trace through existing figure infrastructure."""
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
    if not isinstance(trace, dict) or trace.get("schema_version") != "simulation-step-trace.v1":
        return {"status": "unavailable", "reason": "replay_trace_missing", "artifacts": []}
    if not isinstance(trace.get("steps"), list):
        return {"status": "unavailable", "reason": "replay_trace_steps_missing", "artifacts": []}
    replay_steps, critical_step, smallest_clearance, continuity = _replay_steps_from_trace(trace)
    if len(replay_steps) < 2:
        return {"status": "unavailable", "reason": "replay_trace_too_short", "artifacts": []}

    row_payload = dict(record)
    row_payload["replay_steps"] = replay_steps
    row_payload["replay_dt"] = trace.get("dt")
    map_context = _renderer_map_context(map_path)
    render_map_path = map_path if map_context["status"] == "renderable" else None
    row_payload["replay_map_path"] = str(render_map_path) if render_map_path is not None else None
    episode_row = EpisodeRow.from_dict(row_payload)
    frame_steps = [critical_step] if critical_step is not None else [len(replay_steps) - 1]
    rendered, error = _generate_replay_figures(
        episode_row,
        figure_dir=figure_dir,
        frame_steps=frame_steps,
        episode_records=episode_records,
    )
    if error is not None or rendered is None:
        reason = (
            f"figure_generation_failed: {error}" if error else "figure_generation_missing_result"
        )
        return {"status": "failed", "reason": reason, "artifacts": []}
    artifacts: list[str] = []
    for raw_path in rendered.get("artifact_paths", []):
        try:
            artifacts.append(Path(raw_path).resolve().relative_to(output_dir.resolve()).as_posix())
        except (ValueError, TypeError):
            artifacts.append(str(raw_path))
    if render_map_path is not None and _sha256_file(render_map_path) != map_context["sha256"]:
        map_context.update(
            status="unavailable",
            reason="map_bytes_changed_during_render",
        )
    elif map_context["status"] == "renderable":
        map_context["status"] = "overlay_rendered"
    return {
        "status": "rendered",
        "determinism_check_status": rendered.get("determinism_check_status"),
        "critical_frame_step": critical_step,
        "smallest_surface_clearance_m": smallest_clearance,
        "track_continuity": continuity,
        "map_context": {
            "status": map_context["status"],
            "reason": map_context.get("reason"),
            "path": _relative_to(str(map_path), output_dir) if map_path is not None else None,
            "sha256": map_context.get("sha256"),
            "renderer_error": map_context.get("renderer_error"),
        },
        "artifacts": artifacts,
        "provenance_sidecar": _relative_to(rendered.get("provenance_sidecar"), output_dir),
        "caption_fragment": _relative_to(rendered.get("caption_fragment"), output_dir),
    }


def _materialized_map_path(materialization: dict[str, Any], case_dir: Path) -> Path | None:
    """Resolve the copied map asset for the existing renderer, keeping it inside the case."""
    assets = materialization.get("assets")
    if not isinstance(assets, list):
        return None
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("field") not in {
            "map_file",
            "resolved_map_id",
        }:
            continue
        bundle_path = asset.get("bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path.strip():
            return None
        path = (case_dir / bundle_path).resolve()
        try:
            path.relative_to(case_dir.resolve())
        except ValueError:
            return None
        return path if path.is_file() else None
    return None


def _renderer_map_context(map_path: Path | None) -> dict[str, Any]:
    """Preflight the exact map bytes with the image reader used by the renderer."""
    if map_path is None:
        return {
            "status": "unavailable",
            "reason": "no_materialized_map_asset",
            "sha256": None,
        }
    from robot_sf.common.optional_import import try_import  # noqa: PLC0415

    try:
        digest = _sha256_file(map_path)
    except OSError as exc:
        return {
            "status": "unavailable",
            "reason": "existing_renderer_cannot_decode_map_overlay",
            "renderer_error": f"{type(exc).__name__}: {exc}",
            "sha256": None,
        }
    mpimg = try_import("matplotlib.image")
    if mpimg is None:
        return {
            "status": "unavailable",
            "reason": "existing_renderer_cannot_decode_map_overlay",
            "renderer_error": "ImportError: matplotlib.image is unavailable",
            "sha256": digest,
        }
    try:
        mpimg.imread(map_path)
    except (OSError, SyntaxError, ValueError) as exc:
        return {
            "status": "unavailable",
            "reason": "existing_renderer_cannot_decode_map_overlay",
            "renderer_error": f"{type(exc).__name__}: {exc}",
            "sha256": digest,
        }
    return {"status": "renderable", "sha256": digest}


def _replay_steps_from_trace(
    trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None, float | None, dict[str, Any]]:
    """Convert the canonical runner trace to the replay-figure row shape."""
    trace_steps = trace.get("steps")
    if not isinstance(trace_steps, list):
        return (
            [],
            None,
            None,
            {
                "status": "unavailable",
                "reason": "trace_steps_missing",
                "discontinuity_split_count": 0,
                "unknown_identity_position_count": 0,
            },
        )
    replay_steps: list[dict[str, Any]] = []
    previous_tracks: dict[str, tuple[float, float, float, int]] = {}
    critical_step: int | None = None
    smallest_clearance = math.inf
    split_count = 0
    unknown_identity_count = 0
    for trace_index, step in enumerate(trace_steps):
        converted, step_clearance = _trace_step_to_replay_step(step)
        if converted is None:
            continue
        positions = converted["ped_positions"]
        actor_ids = converted.pop("_pedestrian_actor_ids")
        track_ids: list[str] = []
        for ped_index, (position, actor_id) in enumerate(zip(positions, actor_ids, strict=True)):
            if actor_id is None:
                track_ids.append(f"unknown-step-{trace_index}-actor-{ped_index}")
                unknown_identity_count += 1
                continue
            prior = previous_tracks.get(actor_id)
            segment = prior[3] if prior is not None else 0
            if prior is not None:
                previous_x, previous_y, previous_time, segment = prior
                elapsed = converted["t"] - previous_time
                distance = math.hypot(position[0] - previous_x, position[1] - previous_y)
                speed = distance / elapsed if elapsed > 0.0 else math.inf
                if not math.isfinite(speed) or speed > _MAX_PED_TRACK_SPEED_MPS:
                    segment += 1
                    split_count += 1
            track_id = actor_id if segment == 0 else f"{actor_id}#segment-{segment}"
            track_ids.append(track_id)
            previous_tracks[actor_id] = (
                position[0],
                position[1],
                converted["t"],
                segment,
            )
        converted["pedestrian_ids"] = track_ids
        if step_clearance is not None and step_clearance < smallest_clearance:
            smallest_clearance = step_clearance
            critical_step = len(replay_steps)
        replay_steps.append(converted)
    return (
        replay_steps,
        critical_step,
        smallest_clearance if math.isfinite(smallest_clearance) else None,
        {
            "status": (
                "discontinuities_split"
                if split_count
                else "unknown_identity_isolated"
                if unknown_identity_count
                else "continuous"
            ),
            "maximum_assumed_track_speed_mps": _MAX_PED_TRACK_SPEED_MPS,
            "discontinuity_split_count": split_count,
            "unknown_identity_position_count": unknown_identity_count,
            "note": (
                "positions without stable actor identity are isolated; reused actor IDs are "
                "split when consecutive displacement exceeds the visualization threshold"
            ),
        },
    )


def _trace_step_to_replay_step(step: Any) -> tuple[dict[str, Any] | None, float | None]:
    """Convert one trace step and return its closest pedestrian clearance."""
    if not isinstance(step, dict):
        return None, None
    robot = step.get("robot")
    if not isinstance(robot, dict):
        return None, None
    position = robot.get("position")
    x = _finite_number(position[0]) if isinstance(position, list | tuple) and position else None
    y = (
        _finite_number(position[1])
        if isinstance(position, list | tuple) and len(position) > 1
        else None
    )
    heading = _finite_number(robot.get("heading"))
    time_s = _finite_number(step.get("time_s"))
    if x is None or y is None or heading is None or time_s is None:
        return None, None

    pedestrian_positions: list[list[float]] = []
    pedestrian_actor_ids: list[str | None] = []
    smallest_clearance: float | None = None
    pedestrians = step.get("pedestrians")
    for pedestrian in pedestrians if isinstance(pedestrians, list) else []:
        if not isinstance(pedestrian, dict):
            continue
        ped_position = pedestrian.get("position")
        ped_x = (
            _finite_number(ped_position[0])
            if isinstance(ped_position, list | tuple) and ped_position
            else None
        )
        ped_y = (
            _finite_number(ped_position[1])
            if isinstance(ped_position, list | tuple) and len(ped_position) > 1
            else None
        )
        if ped_x is None or ped_y is None:
            continue
        pedestrian_positions.append([ped_x, ped_y])
        raw_actor_id = pedestrian.get("id", pedestrian.get("actor_id"))
        pedestrian_actor_ids.append(
            str(raw_actor_id).strip()
            if isinstance(raw_actor_id, str | int) and str(raw_actor_id).strip()
            else None
        )
        clearance = _finite_number(pedestrian.get("surface_clearance_m"))
        if clearance is not None and (smallest_clearance is None or clearance < smallest_clearance):
            smallest_clearance = clearance

    speed = _robot_speed(robot.get("velocity"))
    return {
        "t": time_s,
        "x": x,
        "y": y,
        "heading": heading,
        "speed": speed,
        "ped_positions": pedestrian_positions,
        "_pedestrian_actor_ids": pedestrian_actor_ids,
    }, smallest_clearance


def _robot_speed(velocity: Any) -> float | None:
    """Return the magnitude of one finite planar velocity vector."""
    if not isinstance(velocity, list | tuple) or len(velocity) < 2:
        return None
    vx = _finite_number(velocity[0])
    vy = _finite_number(velocity[1])
    return math.hypot(vx, vy) if vx is not None and vy is not None else None


def _generate_replay_figures(
    episode_row: EpisodeRow,
    *,
    figure_dir: Path,
    frame_steps: list[int],
    episode_records: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the existing renderer and keep its failure available to the case receipt."""
    try:
        rendered = replay_episode_and_generate_figures(
            episode_row,
            outputs=["still", "filmstrip", "trajectory"],
            out_dir=figure_dir,
            tolerance_m=0.0,
            frame_steps=frame_steps,
            episodes_jsonl_path=episode_records,
            scenario_matrix_path=None,
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return rendered, None


def _objective_value(
    objective: Any,
    candidate: dict[str, Any],
    record: dict[str, Any],
    record_path: Path,
    certification_payload: Any,
) -> float | None:
    """Evaluate a replay row through the search's registered objective function."""
    candidate_spec = _candidate_spec(candidate)
    certification = _certification_object(certification_payload)
    evaluation = CandidateEvaluation(
        candidate=candidate_spec,
        certification_status=certification,
        objective_value=None,
        failure_attribution=attribution_from_episode_record(record),
        episode_record_path=record_path,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )
    value = objective(evaluation)
    return _finite_number(value)


def _outcomes_match(
    source: dict[str, Any], replay: dict[str, Any], objective_name: str, *, tolerance: float
) -> tuple[bool, dict[str, Any]]:
    """Compare raw outcome categories, failure attribution, and objective projection."""
    raw_comparison = _raw_categorical_outcome_comparison(source, replay)
    if objective_name == "constraints_first_lexicographic_v1":
        source_projection = constraints_first_outcome_projection(source)
        replay_projection = constraints_first_outcome_projection(replay)
        projection_match = _json_values_match(
            source_projection, replay_projection, tolerance=tolerance
        )
        differences = list(raw_comparison["differences"])
        if not projection_match:
            differences.append("objective_projection")
        differences = list(dict.fromkeys(differences))
        return not differences, {
            "contract": objective_name,
            "source": source_projection,
            "replay": replay_projection,
            "projection_matches": projection_match,
            "raw_categorical": raw_comparison,
            "failure_attribution_matches": raw_comparison["failure_attribution_matches"],
            "differences": differences,
        }
    differences: list[str] = []
    if source.get("status") != replay.get("status"):
        differences.append("status")
    if source.get("termination_reason") != replay.get("termination_reason"):
        differences.append("termination_reason")
    for container_name, fields in (
        ("outcome", _CANONICAL_OUTCOME_FIELDS),
        ("metrics", _CANONICAL_METRIC_FIELDS),
    ):
        left = source.get(container_name)
        right = replay.get(container_name)
        if not isinstance(left, dict) or not isinstance(right, dict):
            differences.append(container_name)
            continue
        for field in fields:
            if field in left or field in right:
                if (
                    field not in left
                    or field not in right
                    or not _json_values_match(left[field], right[field], tolerance=tolerance)
                ):
                    differences.append(f"{container_name}.{field}")
    differences.extend(raw_comparison["differences"])
    differences = list(dict.fromkeys(differences))
    return not differences, {
        "contract": "canonical_outcome_fields.v1",
        "differences": differences,
        "raw_categorical": raw_comparison,
        "failure_attribution_matches": raw_comparison["failure_attribution_matches"],
    }


def _raw_categorical_outcome_comparison(
    source: dict[str, Any], replay: dict[str, Any]
) -> dict[str, Any]:
    """Compare raw outcome labels and canonical failure-attribution category exactly."""
    differences: list[str] = []
    for field in ("status", "termination_reason"):
        if field in source or field in replay:
            if (
                field not in source
                or field not in replay
                or type(source[field]) is not type(replay[field])
                or source[field] != replay[field]
            ):
                differences.append(field)

    source_outcome = source.get("outcome")
    replay_outcome = replay.get("outcome")
    if not isinstance(source_outcome, dict) or not isinstance(replay_outcome, dict):
        differences.append("outcome")
    else:
        for field in _CANONICAL_OUTCOME_FIELDS:
            if field not in source_outcome and field not in replay_outcome:
                continue
            if field not in source_outcome or field not in replay_outcome:
                differences.append(f"outcome.{field}")
                continue
            source_value = source_outcome[field]
            replay_value = replay_outcome[field]
            if type(source_value) is not type(replay_value) or source_value != replay_value:
                differences.append(f"outcome.{field}")

    source_failure = attribution_from_episode_record(source).primary_failure
    replay_failure = attribution_from_episode_record(replay).primary_failure
    failure_matches = source_failure == replay_failure
    if not failure_matches:
        differences.append("failure_attribution.primary_failure")
    return {
        "source_status": source.get("status"),
        "replay_status": replay.get("status"),
        "source_termination_reason": source.get("termination_reason"),
        "replay_termination_reason": replay.get("termination_reason"),
        "failure_attribution": {
            "source_primary_failure": source_failure,
            "replay_primary_failure": replay_failure,
        },
        "failure_attribution_matches": failure_matches,
        "differences": differences,
    }


def _episode_identity_matches(
    source: dict[str, Any], replay: dict[str, Any], *, policy: str
) -> tuple[bool, dict[str, Any]]:
    """Require scenario, seed, and planner identity to match the source episode."""
    comparisons = {
        "scenario_id": (source.get("scenario_id"), replay.get("scenario_id")),
        "seed": (source.get("seed"), replay.get("seed")),
        "planner": (source.get("algo"), replay.get("algo")),
    }
    mismatches = [name for name, (left, right) in comparisons.items() if left != right]
    if source.get("algo") != policy or replay.get("algo") != policy:
        mismatches.append("configured_policy")
    return not mismatches, {
        name: {"source": left, "replay": right} for name, (left, right) in comparisons.items()
    } | {"mismatches": mismatches}


def _source_identity_matches(
    candidate: dict[str, Any], record: dict[str, Any], scenario_identity: dict[str, Any]
) -> bool:
    """Check candidate, generated scenario, and source episode identity before selection."""
    candidate_seed = candidate.get("scenario_seed")
    record_seed = record.get("seed")
    if (
        isinstance(candidate_seed, bool)
        or not isinstance(candidate_seed, int)
        or candidate_seed < 0
        or isinstance(record_seed, bool)
        or not isinstance(record_seed, int)
        or record_seed < 0
    ):
        return False
    scenario_names = {
        str(scenario_identity.get("name", "")),
        str(scenario_identity.get("scenario_id", "")),
    }
    scenario_names.discard("")
    metadata = scenario_identity.get("metadata")
    candidate_metadata = (
        metadata.get("adversarial_candidate") if isinstance(metadata, dict) else None
    )
    return (
        candidate_seed == record_seed
        and str(record.get("scenario_id", "")) in scenario_names
        and isinstance(candidate_metadata, dict)
        and all(candidate_metadata.get(key) == value for key, value in candidate.items())
    )


def _source_availability_problem(
    candidate_payload: dict[str, Any], record: dict[str, Any]
) -> str | None:
    """Reject a critical source row unless canonical availability says it ran natively."""
    eligibility = candidate_payload.get("analysis_eligibility")
    attribution = candidate_payload.get("failure_attribution")
    details = attribution.get("details") if isinstance(attribution, dict) else None
    if not isinstance(eligibility, dict) or not isinstance(details, dict):
        return "source_availability_missing_or_malformed"

    execution_mode = details.get("execution_mode")
    readiness_status = details.get("readiness_status")
    availability_status = details.get("availability_status")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (execution_mode, readiness_status, availability_status)
    ):
        return "source_availability_missing_or_malformed"
    if availability_status != "available":
        normalized = availability_status.strip().lower().replace("-", "_")
        return f"source_availability_{normalized}"
    if readiness_status != "native":
        return "source_readiness_not_native"
    if execution_mode != "native" or eligibility.get("execution_mode") != execution_mode:
        return "source_execution_mode_not_native_or_mismatched"

    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, dict) or str(metadata.get("status", "")).strip().lower() != "ok":
        return "source_algorithm_metadata_unavailable"
    if resolve_execution_mode(metadata) != execution_mode:
        return "source_execution_mode_mismatch"
    if _runtime_algorithm_fallback_marker(record) is not None:
        return "source_runtime_fallback_or_degraded"
    return None


def _runtime_algorithm_fallback_marker(record: dict[str, Any]) -> tuple[str, str] | None:
    """Inspect planner runtime metadata without treating unavailable metrics as fallback.

    Episode records also carry diagnostic metric subtrees whose ``status`` can be
    ``unavailable`` when a metric lacks support. Those are not execution-mode
    failures. Runtime fallback and degradation are reported under the algorithm
    metadata contract, which is the evidence this gallery must gate.
    """
    metadata = record.get("algorithm_metadata")
    if not isinstance(metadata, dict):
        return None
    runtime_metadata = _without_unavailable_diagnostic_statuses(metadata)
    return runtime_fallback_or_degraded_marker(runtime_metadata)


def _without_unavailable_diagnostic_statuses(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Drop unsupported diagnostic statuses while retaining every runtime marker.

    The canonical episode metadata includes status-bearing diagnostic products
    such as paired metric coverage and step-trace reset metadata. Their
    ``unavailable`` status does not describe planner execution. Keep all other
    fields, including any explicit fallback or degraded markers in those
    products, for the shared runtime marker check.
    """
    if isinstance(value, dict):
        normalized_path = tuple(part.lower() for part in path)
        diagnostic_status = any(
            part in {"paired_effect_metric_producer", "simulation_step_trace"}
            for part in normalized_path
        )
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if (
                diagnostic_status
                and key == "status"
                and isinstance(item, str)
                and item.strip().lower().replace("-", "_") in {"unavailable", "not_available"}
            ):
                continue
            result[key] = _without_unavailable_diagnostic_statuses(item, path=(*path, key))
        return result
    if isinstance(value, list):
        return [
            _without_unavailable_diagnostic_statuses(item, path=(*path, str(index)))
            for index, item in enumerate(value)
        ]
    return value


def _source_effective_scenario_hash(
    scenario_path: Path,
    scenario: dict[str, Any],
    *,
    source_root: Path,
    root: Path,
) -> tuple[str | None, str | None]:
    """Recompute the canonical effective hash from the persisted scenario inputs."""
    route_file = scenario.get("route_overrides_file")
    if route_file is None:
        route_payload = scenario.get("route_overrides") or {}
    else:
        route_path = _resolve_referenced_file(route_file, scenario_path.parent, source_root, root)
        if route_path is None or not route_path.is_file():
            return None, "route_overrides_input_missing"
        try:
            route_payload = yaml.safe_load(route_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return None, "route_overrides_input_invalid"
    if not isinstance(route_payload, dict):
        return None, "route_overrides_input_invalid"
    try:
        return compute_effective_scenario_hash(scenario, route_payload), None
    except (TypeError, ValueError):
        return None, "effective_scenario_hash_unavailable"


def _scenario_file_binding(
    scenario: dict[str, Any],
    field: str,
    *,
    scenario_path: Path,
    source_root: Path,
    root: Path,
) -> tuple[bool, str | None, str | None]:
    """Capture one scenario file input's bytes so later materialization cannot redefine it."""
    raw_ref = scenario.get(field)
    if raw_ref is None or raw_ref == "":
        return False, None, None
    source_path = _resolve_referenced_file(raw_ref, scenario_path.parent, source_root, root)
    if source_path is None or not source_path.is_file():
        return True, None, f"scenario_{field}_missing"
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return True, None, f"scenario_{field}_unreadable"
    return True, hashlib.sha256(source_bytes).hexdigest(), None


def _manifest_attribution_matches(candidate: dict[str, Any], record: dict[str, Any]) -> bool:
    """Reject a search manifest whose failure label disagrees with its source episode."""
    supplied = candidate.get("failure_attribution")
    if not isinstance(supplied, dict) or supplied.get("status") != "attributed":
        return False
    derived = attribution_from_episode_record(record).to_json()
    if supplied.get("primary_failure") != derived.get("primary_failure"):
        return False
    details = supplied.get("details")
    if not isinstance(details, dict):
        return False
    source_termination = details.get("termination_reason")
    return source_termination is None or source_termination == record.get("termination_reason")


def _scenario_identity(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a one-scenario generated search bundle without interpreting rendered output."""
    try:
        scenario_bytes = path.read_bytes()
    except OSError:
        return None, "scenario_yaml_invalid"
    return _scenario_identity_bytes(scenario_bytes)


def _scenario_identity_bytes(raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one generated scenario from the exact bytes whose digest is recorded."""
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        return None, "scenario_yaml_invalid"
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        return None, "scenario_yaml_not_single_candidate"
    return scenarios[0], None


def _read_single_episode(path: Path) -> dict[str, Any] | None:
    """Return one JSONL episode record only when the file contains exactly one row."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return _read_single_episode_bytes(raw)


def _read_single_episode_bytes(raw: bytes) -> dict[str, Any] | None:
    """Parse one JSONL episode from the exact bytes whose digest is retained."""
    try:
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        record = json.loads(lines[0])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _failure_cluster_by_candidate(
    manifest_snapshot: bytes,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Cluster failures from the same immutable manifest snapshot as selection."""
    try:
        with tempfile.TemporaryDirectory(prefix="rsf-replay-gallery-") as temp_dir:
            snapshot_path = Path(temp_dir) / "source_search_manifest.json"
            snapshot_path.write_bytes(manifest_snapshot)
            archive = curate_failure_archive(
                [snapshot_path], output_path=Path(temp_dir) / "failure_archive.json"
            )
    except Exception as exc:  # noqa: BLE001 - preserve deduplication fallback in the manifest
        return {}, {"status": "unavailable", "reason": type(exc).__name__}
    if not isinstance(archive, dict):
        return {}, {"status": "unavailable", "reason": "archive_payload_malformed"}
    clusters = archive.get("clusters", [])
    if not isinstance(clusters, list):
        return {}, {"status": "unavailable", "reason": "archive_clusters_malformed"}
    index_to_cluster: dict[int, dict[str, Any]] = {}
    entry_by_id = {
        str(entry.get("archive_id")): entry
        for entry in archive.get("entries", [])
        if isinstance(entry, dict)
    }
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_key = cluster.get("cluster_key")
        if not isinstance(cluster_key, dict):
            continue
        for archive_id in cluster.get("member_archive_ids", []):
            entry = entry_by_id.get(str(archive_id))
            if isinstance(entry, dict):
                try:
                    index_to_cluster[int(entry["source_candidate_index"])] = cluster_key
                except (KeyError, TypeError, ValueError):
                    continue
    return index_to_cluster, {
        "status": "available",
        "cluster_count": len(clusters),
        "candidate_count_with_cluster": len(index_to_cluster),
    }


def _certification_classification(payload: Any) -> str | None:
    """Extract the most specific route certificate classification."""
    if not isinstance(payload, dict):
        return None
    direct = payload.get("classification")
    if isinstance(direct, str) and direct:
        return direct.lower()
    details = payload.get("details")
    certificates = details.get("certificates") if isinstance(details, dict) else None
    if isinstance(certificates, list) and certificates:
        values = [
            str(item.get("classification")).lower()
            for item in certificates
            if isinstance(item, dict) and item.get("classification")
        ]
        if values and all(value == values[0] for value in values):
            return values[0]
        if values:
            return "mixed"
    return None


def _feasibility_verdict(payload: Any) -> dict[str, Any]:
    """Preserve the repository's certificate result without promoting it to a proof oracle."""
    classification = _certification_classification(payload)
    if classification in _ADMISSIBLE_CLASSIFICATIONS:
        return {"status": "admissible_by_source_certificate", "source_certificate": payload}
    if classification in {
        "invalid",
        "geometrically_infeasible",
        "kinodynamically_infeasible",
        "dynamically_overconstrained",
        "infeasible",
        "impossible",
    }:
        return {"status": "invalid_or_infeasible", "source_certificate": payload}
    if classification == "knife_edge":
        return {"status": "stress_only", "source_certificate": payload}
    return {
        "status": "unknown",
        "reason": "certificate classification does not establish feasibility",
        "source_certificate": payload,
    }


def _certification_object(payload: Any) -> Any:
    """Convert manifest certification fields into the objective's inert evaluation slot."""
    if isinstance(payload, dict):
        return CertificationStatus(
            schema_version=str(payload.get("schema_version", "scenario_certificate.v1")),
            status=str(payload.get("status", "unknown")),
            reason=str(payload.get("reason", "source manifest")),
            details=dict(payload.get("details") or {}),
        )
    return CertificationStatus("scenario_certificate.v1", "unknown", "missing", {})


def _candidate_spec(candidate: dict[str, Any]) -> CandidateSpec:
    """Build the existing candidate type without dropping unsupported parameters."""
    supported = {
        "start",
        "goal",
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
        "scenario_seed",
        "pedestrian_acceleration_mps2",
        "group_size",
        "vru_profile",
    }
    if not candidate.keys() <= supported:
        raise ValueError("candidate contains fields unsupported by the replay objective adapter")
    start = candidate.get("start")
    goal = candidate.get("goal")
    if not isinstance(start, dict) or not isinstance(goal, dict):
        raise ValueError("candidate start and goal must be objects")
    seed = candidate.get("scenario_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("candidate scenario_seed must be a non-negative integer")
    acceleration = candidate.get("pedestrian_acceleration_mps2")
    if acceleration is not None:
        acceleration = _candidate_float(acceleration, "pedestrian_acceleration_mps2")
    group_size = candidate.get("group_size")
    if group_size is not None and (
        isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1
    ):
        raise ValueError("candidate group_size must be a positive integer")
    vru_profile = candidate.get("vru_profile")
    if vru_profile is not None and (not isinstance(vru_profile, str) or not vru_profile.strip()):
        raise ValueError("candidate vru_profile must be non-empty text")
    return CandidateSpec(
        start=Pose2D(
            _candidate_float(start.get("x"), "start.x"),
            _candidate_float(start.get("y"), "start.y"),
            _candidate_float(start.get("theta", 0.0), "start.theta"),
        ),
        goal=Pose2D(
            _candidate_float(goal.get("x"), "goal.x"),
            _candidate_float(goal.get("y"), "goal.y"),
            _candidate_float(goal.get("theta", 0.0), "goal.theta"),
        ),
        spawn_time_s=_candidate_float(candidate.get("spawn_time_s"), "spawn_time_s"),
        pedestrian_speed_mps=_candidate_float(
            candidate.get("pedestrian_speed_mps"), "pedestrian_speed_mps"
        ),
        pedestrian_delay_s=_candidate_float(
            candidate.get("pedestrian_delay_s"), "pedestrian_delay_s"
        ),
        scenario_seed=seed,
        pedestrian_acceleration_mps2=acceleration,
        group_size=group_size,
        vru_profile=vru_profile.strip() if isinstance(vru_profile, str) else None,
    )


def _candidate_float(value: Any, name: str) -> float:
    """Read one finite candidate number without accepting booleans or strings."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"candidate {name} must be numeric")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"candidate {name} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"candidate {name} must be finite")
    return parsed


def _accounting_row(
    index: int, candidate: dict[str, Any], disposition: str, *, case_id: str | None = None
) -> dict[str, Any]:
    """Return stable summary accounting for one source candidate."""
    return {
        "source_candidate_index": index,
        "case_id": case_id,
        "disposition": disposition,
        "objective_value": _finite_number(candidate.get("objective_value")),
        "failure_attribution": candidate.get("failure_attribution"),
        "error": candidate.get("error"),
    }


def _resolve_artifact_path(
    raw: Any, *, source_manifest: Path, root: Path, source_root: Path
) -> Path | None:
    """Resolve a manifest artifact path using explicit source and repository anchors."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [source_root / path, root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_referenced_file(
    raw: Any, source_dir: Path, source_root: Path, root: Path
) -> Path | None:
    """Resolve a scenario's relative file from its source matrix before repository root."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    candidates = (
        [path] if path.is_absolute() else [source_dir / path, source_root / path, root / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_repo_file(raw: Any, source_root: Path, root: Path) -> Path | None:
    """Resolve a config file path from the current repository or its stored path."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [source_root / path, root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _load_json_file(path: Path | None) -> dict[str, Any] | None:
    """Load an optional JSON config file."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _manifest_revision(payload: dict[str, Any]) -> str | None:
    """Read exact source revision fields from the search manifest when available."""
    for container in (payload, payload.get("provenance"), payload.get("run_metadata")):
        if not isinstance(container, dict):
            continue
        for name in (
            "source_revision",
            "source_search_commit",
            "experiment_source_commit",
            "git_hash",
        ):
            revision = container.get(name)
            if isinstance(revision, str) and _REVISION_RE.fullmatch(revision):
                return revision.lower()
    return None


def _record_revision(record: dict[str, Any]) -> str | None:
    """Read a full source or replay commit from a canonical episode row."""
    for container in (record, record.get("provenance"), record.get("result_provenance")):
        if not isinstance(container, dict):
            continue
        for name in ("git_hash", "commit_hash", "source_revision", "revision"):
            revision = container.get(name)
            if isinstance(revision, str) and _REVISION_RE.fullmatch(revision):
                return revision.lower()
    return None


def _git_revision(root: Path) -> str | None:
    """Return the current full repository commit when Git is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision.lower() if _REVISION_RE.fullmatch(revision) else None


def _git_checkout_state(root: Path) -> dict[str, Any]:
    """Capture the exact checkout revision and whether tracked/untracked inputs are clean."""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "revision": _git_revision(root),
            "clean": False,
            "dirty_paths": ["git_status_unavailable"],
        }
    lines = [line for line in status.stdout.splitlines() if line]
    return {
        "revision": _git_revision(root),
        "clean": not lines,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in lines],
    }


def _video_artifacts(replay_dir: Path, output_dir: Path) -> list[str]:
    """Return stable relative paths for non-empty video outputs."""
    paths = sorted(
        path.relative_to(output_dir).as_posix()
        for path in replay_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".webm", ".gif"}
        and path.stat().st_size > 0
    )
    return paths


def _video_status(requested: bool, artifacts: list[str], *, attempted: bool) -> dict[str, Any]:
    """Report whether the canonical replay actually emitted an optional video."""
    status = "disabled"
    reason = None
    if requested and not attempted:
        status = "not_attempted"
        reason = "replay_did_not_complete"
    elif requested and artifacts:
        status = "rendered"
    elif requested:
        status = "unavailable"
        reason = "canonical_runner_did_not_emit_video_artifact"
    return {
        "requested": bool(requested),
        "renderer": "synthetic" if requested else "none",
        "status": status,
        "reason": reason,
        "artifacts": artifacts,
    }


def _json_values_match(left: Any, right: Any, *, tolerance: float) -> bool:
    """Recursively compare JSON values with explicit absolute float tolerance."""
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, int | float) and isinstance(right, int | float):
        left_value, right_value = float(left), float(right)
        return (
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=tolerance)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_values_match(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_values_match(lvalue, rvalue, tolerance=tolerance)
            for lvalue, rvalue in zip(left, right, strict=True)
        )
    return left == right


def _certification_classification_from_status(payload: Any) -> str | None:
    """Public compatibility helper kept for tests and downstream callers."""
    return _certification_classification(payload)


def _stable_case_id(index: int, candidate: dict[str, Any], manifest_sha256: str) -> str:
    """Build a location-independent stable identifier for a manifest candidate."""
    identity = {"candidate": candidate, "manifest_sha256": manifest_sha256, "index": index}
    digest = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()[:12]
    return f"case_{index:04d}_{digest}"


def _stable_json(payload: Any) -> str:
    """Serialize a JSON-compatible value canonically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _finite_number(value: Any) -> float | None:
    """Return a finite numeric value without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root() -> Path:
    """Return the package's source tree root."""
    return Path(__file__).resolve().parents[2]


def _complete_case(result: dict[str, Any], case_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Persist one case receipt on every terminal path."""
    case_manifest_path = case_dir / "case_manifest.json"
    result["case_manifest_path"] = case_manifest_path.relative_to(output_dir).as_posix()
    _write_json(case_manifest_path, result)
    return result


def _search_method(payload: dict[str, Any], config: dict[str, Any]) -> str:
    """Return a declared search method or the explicit value ``unknown``."""
    direct = config.get("search_method") or config.get("sampler") or payload.get("search_method")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return "unknown"


def _source_repository_root(manifest: Path, fallback: Path) -> Path:
    """Find the checkout containing a manifest whose bundle paths are relative."""
    for parent in manifest.parents:
        if (parent / "AGENTS.md").is_file() and (parent / ".git").exists():
            return parent
    return fallback


def _display_path(path: Path, root: Path) -> str:
    """Render repository-local paths relatively and external paths by basename."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _is_repository_relative_path(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves inside ``root``."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _relative_to(raw: Any, root: Path) -> str | None:
    """Convert an output path to a stable relative path when possible."""
    if not isinstance(raw, str):
        return None
    try:
        return Path(raw).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return Path(raw).name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_gallery_readme(
    output_dir: Path, cases: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    """Write a compact human-readable index over replay outputs."""
    lines = [
        "# Adversarial replay gallery",
        "",
        "Generated from a persisted search manifest. Each case records source and replay identity,",
        "objective comparison, feasibility classification, and available diagnostic artifacts.",
        "",
        f"Cases selected: {summary['selected_case_count']}",
        f"Replay outcome matches: {summary['replay_match_count']}",
        f"Replay mismatches: {summary['replay_mismatch_count']}",
        f"Replay unavailable: {summary['replay_unavailable_count']}",
        "",
    ]
    if not cases:
        lines.append(
            "No candidates met the required provenance, certificate, and replay-input checks."
        )
    for case in cases:
        lines.extend(
            [
                f"## {case['case_id']}",
                "",
                f"- Replay match: `{case['replay_match']}`",
                f"- Verification: `{case['verification_status']}`",
                f"- Objective: `{case['objective']['source_value']}` → "
                f"`{(case.get('replay') or {}).get('objective_value')}`",
                f"- Feasibility: `{case['feasibility_verdict']['status']}`",
            ]
        )
        for artifact in case.get("rendering", {}).get("artifacts", []):
            if str(artifact).endswith(".png"):
                lines.append(f"\n![{case['case_id']} diagnostic]({artifact})")
            else:
                lines.append(f"\n- Artifact: [{Path(str(artifact)).name}]({artifact})")
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


__all__ = ["GALLERY_SCHEMA_VERSION", "build_replay_gallery"]
