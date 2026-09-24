"""Build deterministic, replay-checked visual bundles from adversarial search results."""

from __future__ import annotations

import hashlib
import json
import math
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
from robot_sf.benchmark.runner import run_batch

GALLERY_SCHEMA_VERSION = "adversarial-replay-gallery.v1"
SEARCH_MANIFEST_SCHEMA_VERSION = "adversarial-search-manifest.v1"
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_ADMISSIBLE_CLASSIFICATIONS = frozenset({"valid", "hard_but_solvable"})
_CANONICAL_OUTCOME_FIELDS = (
    "route_complete",
    "collision",
    "collision_event",
    "timeout",
    "timeout_event",
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
    source_root = _source_repository_root(source_manifest, root)
    payload = _load_search_manifest(source_manifest)
    destination = Path(output_dir).expanduser().resolve()
    source_manifest_sha256 = _sha256_file(source_manifest)
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
    archive_clusters, archive_cluster_status = _failure_cluster_by_candidate(source_manifest)
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
    shutil.copyfile(source_manifest, destination / "source_search_manifest.json")
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


def _load_search_manifest(path: Path) -> dict[str, Any]:
    """Load one manifest with the supported adversarial search schema version."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read search manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("search manifest must be a JSON object")
    if payload.get("schema_version") != SEARCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported search manifest schema: "
            f"{payload.get('schema_version')!r}; expected {SEARCH_MANIFEST_SCHEMA_VERSION!r}"
        )
    return payload


def _prepare_candidate(  # noqa: C901 - keep the ordered row-disposition gates together
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
    source_record = _read_single_episode(episode_path)
    if source_record is None:
        return None, _accounting_row(
            index, candidate_payload, "episode_record_invalid_or_ambiguous"
        )
    scenario_identity, scenario_error = _scenario_identity(scenario_path)
    if scenario_error is not None or scenario_identity is None:
        return None, _accounting_row(
            index, candidate_payload, scenario_error or "scenario_identity_unavailable"
        )
    effective_hash, hash_error = _source_effective_scenario_hash(
        scenario_path, scenario_identity, source_root=source_root, root=root
    )
    if hash_error is not None:
        return None, _accounting_row(index, candidate_payload, hash_error)
    declared_hash = candidate_payload.get("effective_scenario_hash")
    if not isinstance(declared_hash, str) or declared_hash != effective_hash:
        return None, _accounting_row(
            index, candidate_payload, "effective_scenario_hash_missing_or_mismatch"
        )
    if not _source_identity_matches(candidate, source_record, scenario_identity):
        return None, _accounting_row(index, candidate_payload, "source_candidate_identity_mismatch")
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
        "source_revision": source_revision,
        "source_root": source_root,
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
    )
    shutil.copyfile(selected["episode_path"], case_dir / "source_episode.jsonl")
    result = _initial_case_result(selected, context, materialization)
    if materialization["status"] != "materialized":
        result["verification_status"] = "not_replayed_inputs_unavailable"
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

    replay_summary, replay_error = _run_one_episode(
        case_dir / materialization["scenario_path"],
        episode_records,
        runner_config,
        context,
    )
    if replay_error is not None:
        result["verification_status"] = "replay_execution_failed"
        result["replay_error"] = replay_error
        result["replay"] = {"error": replay_error}
        return _complete_case(result, case_dir, context.output_dir)

    replay_record = _read_single_episode(episode_records)
    if replay_record is None:
        result["verification_status"] = "replay_record_missing_or_ambiguous"
        result["replay"] = {
            "summary": replay_summary,
            "episode_record_path": "replay/episode_records.jsonl",
        }
        return _complete_case(result, case_dir, context.output_dir)

    _record_replay_comparison(
        selected,
        context,
        result,
        replay_record,
        replay_dir=replay_dir,
        episode_records=episode_records,
        case_dir=case_dir,
        replay_summary=replay_summary or {},
    )
    return _complete_case(result, case_dir, context.output_dir)


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
            "episode_record_sha256": _sha256_file(selected["episode_path"]),
            "bundled_episode_record_path": "source_episode.jsonl",
            "scenario_yaml_path": _display_path(selected["scenario_path"], selected["source_root"]),
            "scenario_yaml_sha256": _sha256_file(selected["scenario_path"]),
            "episode_id": selected["source_record"].get("episode_id"),
            "scenario_id": selected["source_record"].get("scenario_id"),
            "seed": selected["source_record"].get("seed"),
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
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one canonical replay and turn execution errors into explicit case data."""
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
    return replay_summary, None


def _record_replay_comparison(
    selected: dict[str, Any],
    context: _ReplayContext,
    result: dict[str, Any],
    replay_record: dict[str, Any],
    *,
    replay_dir: Path,
    episode_records: Path,
    case_dir: Path,
    replay_summary: dict[str, Any],
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
    source_revision_for_case = selected["source_revision"] or context.source_revision
    if identity_match and outcome_match and objective_match:
        result["replay_match"] = "match"
        if result["source"]["revision_conflict"]:
            result["verification_status"] = "source_revision_conflict"
        elif source_revision_for_case is None:
            result["verification_status"] = "outcome_reproduced_source_revision_unknown"
        elif replay_revision is None:
            result["verification_status"] = "outcome_reproduced_replay_revision_unknown"
        elif source_revision_for_case != replay_revision:
            result["verification_status"] = "outcome_reproduced_revision_changed"
        else:
            result["verification_status"] = "verified"
    else:
        result["replay_match"] = "mismatch"
        result["verification_status"] = "replay_mismatch"

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
        "revision_matches_source": (
            source_revision_for_case is not None and replay_revision == source_revision_for_case
        ),
        "objective_value": replay_objective,
        "objective_matches": objective_match,
        "identity_matches": identity_match,
        "identity_comparison": identity_details,
        "outcome_matches": outcome_match,
        "outcome_comparison": outcome_details,
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
        )


def _materialize_scenario(
    source: Path, input_dir: Path, *, root: Path, source_root: Path
) -> dict[str, Any]:
    """Copy a generated scenario and its declared file inputs into a stable bundle."""
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {"status": "unavailable", "reason": f"scenario_read_failed: {exc}"}
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        return {"status": "unavailable", "reason": "scenario_matrix_shape_invalid"}
    input_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = input_dir / "assets"
    assets_dir.mkdir()
    copied_assets: list[dict[str, str]] = []
    for scenario_index, scenario in enumerate(payload["scenarios"]):
        if not isinstance(scenario, dict):
            return {"status": "unavailable", "reason": f"scenario_{scenario_index}_invalid"}
        for field in ("map_file", "route_overrides_file"):
            raw_ref = scenario.get(field)
            if raw_ref is None:
                continue
            asset_path = _resolve_referenced_file(raw_ref, source.parent, source_root, root)
            if asset_path is None or not asset_path.is_file():
                return {
                    "status": "unavailable",
                    "reason": f"{field}_missing",
                    "reference": str(raw_ref),
                }
            digest = _sha256_file(asset_path)
            name = f"{digest[:12]}-{asset_path.name}"
            destination = assets_dir / name
            if not destination.exists():
                shutil.copyfile(asset_path, destination)
            scenario[field] = (Path("assets") / name).as_posix()
            copied_assets.append(
                {
                    "field": field,
                    "source_path": _display_path(asset_path, source_root),
                    "source_sha256": digest,
                    "bundle_path": (Path("inputs") / "assets" / name).as_posix(),
                }
            )
    scenario_path = input_dir / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return {
        "status": "materialized",
        "scenario_path": "inputs/scenario.yaml",
        "scenario_sha256": _sha256_file(scenario_path),
        "source_scenario_sha256": _sha256_file(source),
        "assets": copied_assets,
    }


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
) -> dict[str, Any]:
    """Render the exact replay trace through existing figure infrastructure."""
    metadata = record.get("algorithm_metadata")
    trace = metadata.get("simulation_step_trace") if isinstance(metadata, dict) else None
    if not isinstance(trace, dict) or trace.get("schema_version") != "simulation-step-trace.v1":
        return {"status": "unavailable", "reason": "replay_trace_missing", "artifacts": []}
    if not isinstance(trace.get("steps"), list):
        return {"status": "unavailable", "reason": "replay_trace_steps_missing", "artifacts": []}
    replay_steps, critical_step, smallest_clearance = _replay_steps_from_trace(trace)
    if len(replay_steps) < 2:
        return {"status": "unavailable", "reason": "replay_trace_too_short", "artifacts": []}

    row_payload = dict(record)
    row_payload["replay_steps"] = replay_steps
    row_payload["replay_dt"] = trace.get("dt")
    row_payload["replay_map_path"] = None
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
    return {
        "status": "rendered",
        "determinism_check_status": rendered.get("determinism_check_status"),
        "critical_frame_step": critical_step,
        "smallest_surface_clearance_m": smallest_clearance,
        "artifacts": artifacts,
        "provenance_sidecar": _relative_to(rendered.get("provenance_sidecar"), output_dir),
        "caption_fragment": _relative_to(rendered.get("caption_fragment"), output_dir),
    }


def _replay_steps_from_trace(
    trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], int | None, float | None]:
    """Convert the canonical runner trace to the replay-figure row shape."""
    trace_steps = trace.get("steps")
    if not isinstance(trace_steps, list):
        return [], None, None
    replay_steps: list[dict[str, Any]] = []
    critical_step: int | None = None
    smallest_clearance = math.inf
    for step in trace_steps:
        converted, step_clearance = _trace_step_to_replay_step(step)
        if converted is None:
            continue
        if step_clearance is not None and step_clearance < smallest_clearance:
            smallest_clearance = step_clearance
            critical_step = len(replay_steps)
        replay_steps.append(converted)
    return (
        replay_steps,
        critical_step,
        smallest_clearance if math.isfinite(smallest_clearance) else None,
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
    """Compare canonical outcome fields and the objective's strict projection."""
    if objective_name == "constraints_first_lexicographic_v1":
        source_projection = constraints_first_outcome_projection(source)
        replay_projection = constraints_first_outcome_projection(replay)
        match = _json_values_match(source_projection, replay_projection, tolerance=tolerance)
        return match, {
            "contract": objective_name,
            "source": source_projection,
            "replay": replay_projection,
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
    return not differences, {"contract": "canonical_outcome_fields.v1", "differences": differences}


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
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, "scenario_yaml_invalid"
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else None
    if not isinstance(scenarios, list) or len(scenarios) != 1 or not isinstance(scenarios[0], dict):
        return None, "scenario_yaml_not_single_candidate"
    return scenarios[0], None


def _read_single_episode(path: Path) -> dict[str, Any] | None:
    """Return one JSONL episode record only when the file contains exactly one row."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        record = json.loads(lines[0])
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _failure_cluster_by_candidate(
    manifest_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Reuse existing failure-archive clusters and retain any fallback status."""
    try:
        with tempfile.TemporaryDirectory(prefix="rsf-replay-gallery-") as temp_dir:
            archive = curate_failure_archive(
                [manifest_path], output_path=Path(temp_dir) / "failure_archive.json"
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
