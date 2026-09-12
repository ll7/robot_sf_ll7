#!/usr/bin/env python3
"""Fail-closed trace and matched-packet preflight for issue #8849."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/analysis/issue_8849_risk_calibration_preflight.yaml"
CONFIG_SCHEMA = "risk_calibration_preflight_config.v1"
REPORT_SCHEMA = "risk_calibration_preflight.v1"
_MISSING = object()
_PRIVATE_MARKER = re.compile(r"(?:/home/|/tmp/|\\Users\\|ssh |slurm|password|token)", re.I)
_SOURCE_KINDS = {"trace_series", "episode_jsonl"}


PreflightError = ValueError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be a mapping")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value.strip()


def _relative_path(value: Any, field: str) -> str:
    path = _nonempty_string(value, field)
    parsed = Path(path)
    _require(
        not parsed.is_absolute() and ".." not in parsed.parts, f"{field} must be repo-relative"
    )
    _require(not _PRIVATE_MARKER.search(path), f"{field} contains a private locator")
    return parsed.as_posix()


def _load_yaml(path: Path, field: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"cannot read {field} {path}: {exc}") from exc
    return _mapping(payload, field)


def _load_config(config_path: Path, root: Path) -> dict[str, Any]:
    config = _load_yaml(config_path, "config")
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema_version mismatch")
    target = _mapping(config.get("target"), "target")
    for field in "target_distribution_id collision_predicate_id required_execution_mode scenario_matrix".split():
        _nonempty_string(target.get(field), f"target.{field}")
    _require(
        isinstance(target.get("horizon_steps"), int)
        and target["horizon_steps"] > 0
        and isinstance(target.get("dt_s"), (int, float))
        and target["dt_s"] > 0
        and isinstance(target.get("minimum_eligible_samples"), int)
        and target["minimum_eligible_samples"] > 0,
        "target numeric contract is invalid",
    )
    target["scenario_matrix"] = _relative_path(target["scenario_matrix"], "target.scenario_matrix")

    packet = _mapping(config.get("packet"), "packet")
    for field in "estimators calibration_split weighting primary_metrics runtime_cap_ms compute_estimate stop_rule".split():
        _require(field in packet, f"packet.{field} is required")
    _require(
        all(
            isinstance(packet[field], list) and packet[field]
            for field in ("estimators", "primary_metrics")
        ),
        "packet roster/metrics are empty",
    )
    sources = config.get("sources")
    _require(isinstance(sources, list) and sources, "sources is empty")
    seen_ids: set[str] = set()
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, f"sources[{index}]")
        source_id = _nonempty_string(source.get("id"), f"sources[{index}].id")
        _require(source_id not in seen_ids, f"duplicate source id: {source_id}")
        seen_ids.add(source_id)
        _require(source.get("kind") in _SOURCE_KINDS, f"sources[{index}].kind is unsupported")
        source["glob"] = _relative_path(source.get("glob"), f"sources[{index}].glob")
        _nonempty_string(source.get("expected_schema"), f"sources[{index}].expected_schema")
    matrix_path = root / target["scenario_matrix"]
    _require(
        matrix_path.is_file(), f"target scenario matrix is missing: {target['scenario_matrix']}"
    )
    return config


def _path_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _first(value: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        candidate = _path_value(value, path)
        if candidate is not _MISSING and candidate is not None:
            return candidate
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _usable_text(value: Any) -> str | None:
    text = _text(value)
    if text is None or text.lower().startswith(("missing:", "unknown", "unavailable")):
        return None
    return text


def _scenario_index(root: Path, matrix_rel: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    from loguru import logger

    logger.disable("robot_sf.training.scenario_loader")
    try:
        scenarios = load_scenarios(root / matrix_rel)
    except (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise PreflightError(f"cannot expand scenario matrix: {exc}") from exc
    finally:
        logger.enable("robot_sf.training.scenario_loader")
    for scenario_value in scenarios:
        scenario = _mapping(scenario_value, "scenario")
        scenario_id = _text(scenario.get("name") or scenario.get("scenario_id"))
        if scenario_id is None:
            continue
        metadata = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
        simulation = (
            scenario.get("simulation_config")
            if isinstance(scenario.get("simulation_config"), dict)
            else {}
        )
        index[scenario_id] = {
            "scenario_family": _text(metadata.get("archetype") or metadata.get("scenario_family")),
            "density": _text(metadata.get("density")) or _text(simulation.get("ped_density")),
        }
    return index


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _repo_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PreflightError(f"source path escapes repository: {path}") from exc


def _read_source(path: Path, kind: str) -> list[dict[str, Any]]:
    try:
        if kind == "trace_series":
            value = json.loads(path.read_text(encoding="utf-8"))
            return [_mapping(value, f"trace source {path}")]
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            records.append(_mapping(value, f"episode source {path}:{line_number}"))
        return records
    except (OSError, json.JSONDecodeError, PreflightError) as exc:
        raise PreflightError(f"cannot parse {kind} source {path}: {exc}") from exc


def _source_records(
    root: Path, source: dict[str, Any]
) -> tuple[list[tuple[Path, int, dict[str, Any]]], list[str]]:
    matches = sorted(root.glob(str(source["glob"])))
    records: list[tuple[Path, int, dict[str, Any]]] = []
    errors: list[str] = []
    for path in matches:
        try:
            records.extend(
                (path, index, record)
                for index, record in enumerate(_read_source(path, source["kind"]))
            )
        except PreflightError as exc:
            del exc
            errors.append(f"parse_error:{_repo_path(path, root)}")
    return records, errors


def _present(value: Any) -> bool:
    return value is not None and (not isinstance(value, (str, list, dict)) or bool(value))


def _identifier(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("id") or value.get("profile_id") or value.get("name")
    return _usable_text(value)


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _frames(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = _first(record, ("frames",), ("trace", "frames"), ("steps",))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _frame_field(frames: list[dict[str, Any]], *paths: tuple[str, ...]) -> bool:
    return any(_present(_first(frame, *paths)) for frame in frames)


_PATHS: dict[str, str] = {
    "scenario_id": "scenario_id metadata.scenario_id scenario.id",
    "planner_id": "planner_id metadata.planner planner algorithm_metadata.planner_id algorithm_metadata.algorithm algorithm_metadata.canonical_algorithm",
    "seed": "seed metadata.seed",
    "episode_id": "episode_id metadata.episode_id episode.id",
    "source_commit": "git_commit git_hash metadata.git_commit provenance.git_commit source.git_commit",
    "horizon": "horizon run_horizon metadata.horizon scenario_params.run_horizon",
    "dt_s": "dt_s dt run_dt metadata.dt_s scenario_params.run_dt",
    "execution_mode": "execution_mode planner_kinematics.execution_mode algorithm_metadata.planner_kinematics.execution_mode algorithm_metadata.execution_mode",
    "fallback_count": "fallback_count planner_runtime.fallback_count algorithm_metadata.planner_runtime.fallback_count fallback.count",
    "degraded": "degraded execution.degraded quality.degraded",
    "candidate_action_trajectory": "candidate_action_trajectories candidate_trajectories action_candidates risk.candidate_action_trajectories",
    "forecast_inputs": "forecast_inputs forecast_input prediction.forecast_input risk.forecast_inputs",
    "collision_label": "outcome.collision_event collision_event label.collision_event labels.collision_event",
    "action_conditioned_label": "action_conditioned_label outcome.action_conditioned label.action_conditioned labels.action_conditioned",
    "collision_predicate": "collision_predicate_id collision_predicate metric_parameters.threshold_profile.profile_id",
    "prediction_model": "prediction_model forecast.prediction_model prediction.model algorithm_metadata.prediction_model",
    "target_distribution": "target_distribution_id target_distribution provenance.target_distribution_id risk.target_distribution_id",
    "sampling_provenance": "sampling_provenance provenance.sampling_provenance sampling",
    "proposal_weight": "proposal_weight sampling.proposal_weight provenance.proposal_weight weights.proposal",
    "leakage_check": "leakage_check validation.future_leakage provenance.future_leakage_check",
    "footprints": "footprints footprint_contract robot_radius_m robot_radius",
}


def _field(record: dict[str, Any], name: str) -> Any:
    return _first(record, *(tuple(path.split(".")) for path in _PATHS[name].split()))


def _record_values(record: dict[str, Any]) -> dict[str, Any]:
    frames = _frames(record)
    values = {key: _field(record, key) for key in _PATHS}
    values.update(
        scenario_id=_identifier(values["scenario_id"]),
        planner_id=_identifier(values["planner_id"]),
        episode_id=_identifier(values["episode_id"]),
        source_commit=_usable_text(values["source_commit"]),
        horizon=_number(values["horizon"]),
        dt_s=_number(values["dt_s"]),
        execution_mode=_identifier(values["execution_mode"]),
        fallback_count=_number(values["fallback_count"]),
        degraded=values["degraded"] if isinstance(values["degraded"], bool) else None,
        collision_predicate=_identifier(values["collision_predicate"]),
        prediction_model=_identifier(values["prediction_model"]),
        target_distribution=_identifier(values["target_distribution"]),
    )
    action = _first(record, ("action",), ("selected_action",), ("planner", "selected_action"))
    values["history"] = _present(
        _first(record, ("history",), ("observation_history",), ("frames",), ("trace", "frames"))
    )
    values["action_channel"] = _present(action) or _frame_field(
        frames, ("planner", "selected_action"), ("selected_action",), ("action",)
    )
    for key in (
        "candidate_action_trajectory",
        "forecast_inputs",
        "collision_label",
        "sampling_provenance",
        "footprints",
    ):
        values[key] = _present(values[key])
    return values


def _field_presence(values: dict[str, Any]) -> dict[str, bool]:
    identity_fields = "scenario_id planner_id seed episode_id source_commit horizon dt_s execution_mode fallback_count degraded"
    return {
        key: _present(value) for key, value in values.items() if key not in identity_fields.split()
    }


def _inspect_record(  # noqa: C901
    record: dict[str, Any],
    source: dict[str, Any],
    scenario_index: dict[str, dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    values = _record_values(record)
    reasons: list[str] = []
    if (record.get("schema_version") or record.get("version")) != source["expected_schema"]:
        reasons.append("schema_mismatch")
    required = (
        ("scenario_id", "missing_scenario_id"),
        ("planner_id", "missing_planner_id"),
        ("episode_id", "missing_episode_id"),
        ("source_commit", "missing_source_commit"),
        ("horizon", "missing_horizon"),
        ("dt_s", "missing_dt"),
        ("execution_mode", "execution_mode_unavailable"),
    )
    reasons.extend(reason for field, reason in required if not _present(values[field]))
    if not isinstance(values["seed"], int) or isinstance(values["seed"], bool):
        reasons.append("missing_seed")
    scenario = scenario_index.get(values["scenario_id"])
    if scenario is None or not scenario.get("scenario_family") or not scenario.get("density"):
        reasons.append("scenario_metadata_unavailable")
    if values["horizon"] is not None and values["horizon"] != target["horizon_steps"]:
        reasons.append("horizon_mismatch")
    if values["dt_s"] is not None and abs(float(values["dt_s"]) - float(target["dt_s"])) > 1e-9:
        reasons.append("dt_mismatch")
    if (
        values["execution_mode"] is not None
        and values["execution_mode"] != target["required_execution_mode"]
    ):
        reasons.append("non_native_execution")
    if values["fallback_count"] is None and values["degraded"] is None:
        reasons.append("fallback_status_unverifiable")
    if values["fallback_count"] is not None and values["fallback_count"] > 0:
        reasons.append("fallback_or_degraded_execution")
    if values["degraded"] is True:
        reasons.append("fallback_or_degraded_execution")
    required_fields = (
        ("history", "missing_history"),
        ("action_channel", "missing_action_channel"),
        ("candidate_action_trajectory", "missing_candidate_action_trajectory"),
        ("forecast_inputs", "missing_forecast_inputs"),
        ("collision_label", "missing_collision_label"),
        ("footprints", "missing_footprints"),
        ("prediction_model", "missing_prediction_model"),
        ("target_distribution", "missing_target_distribution"),
        ("sampling_provenance", "missing_sampling_provenance"),
        ("proposal_weight", "missing_proposal_weight"),
    )
    reasons.extend(reason for field, reason in required_fields if not values[field])
    action_label = values["action_conditioned_label"]
    if action_label is not True:
        reasons.append("missing_action_conditioned_label")
    predicate = values["collision_predicate"]
    if predicate is None:
        reasons.append("missing_collision_predicate")
    elif predicate != target["collision_predicate_id"]:
        reasons.append("inconsistent_collision_predicate")
    if (
        values["target_distribution"] is not None
        and values["target_distribution"] != target["target_distribution_id"]
    ):
        reasons.append("inconsistent_target_distribution")
    leakage = values["leakage_check"]
    if leakage is False:
        reasons.append("future_leakage_detected")
    elif leakage is not True:
        reasons.append("future_leakage_unverified")
    dimensions = {
        "scenario_family": scenario.get("scenario_family") if scenario else None,
        "density": scenario.get("density") if scenario else None,
        "planner_id": values["planner_id"],
        "prediction_model": values["prediction_model"],
        "horizon_steps": values["horizon"],
        "dt_s": values["dt_s"],
        "target_distribution_id": values["target_distribution"],
        "collision_predicate_id": values["collision_predicate"],
        "execution_mode": values["execution_mode"],
        "action_conditioned_label": action_label,
    }
    return {
        "values": values,
        "reasons": sorted(set(reasons)),
        "field_presence": _field_presence(values),
        "dimensions": dimensions,
        "status": "eligible" if not reasons else "ineligible",
    }


def _source_status(records: list[dict[str, Any]], errors: list[str], matched_files: int) -> str:
    if not matched_files:
        return "unavailable"
    if errors:
        return "unknown"
    if records and all(record["status"] == "eligible" for record in records):
        return "eligible"
    if any("missing_source_commit" in record["reasons"] for record in records):
        return "stale"
    return "ineligible"


def _increment(target: dict[str, int], keys: list[str]) -> None:
    for key in keys:
        target[key] = target.get(key, 0) + 1


def _sorted_counts(values: dict[str, int]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_public(value: Any, path: str = "report") -> None:
    if isinstance(value, str):
        _require(not _PRIVATE_MARKER.search(value), f"{path} contains a private locator")
    elif isinstance(value, (dict, list)):
        children = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in children:
            _assert_public(child, f"{path}.{key}")


def build_report(  # noqa: C901
    root: Path = REPO_ROOT, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Build a deterministic report from the retained stores under ``root``."""
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = _load_config(config_path, root)
    target = dict(config["target"])
    scenario_index = _scenario_index(root, target["scenario_matrix"])
    pending: list[tuple[dict[str, Any], Path, int, dict[str, Any]]] = []
    source_summaries: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    file_hashes: dict[Path, str] = {}

    for source_value in config["sources"]:
        source = _mapping(source_value, "source")
        records, errors = _source_records(root, source)
        matches = sorted(root.glob(str(source["glob"])))
        files = [path for path in matches if path.is_file()]
        for path in files:
            file_hashes[path] = _file_hash(path)
        inspected: list[dict[str, Any]] = []
        for path, record_index, record in records:
            inspection = _inspect_record(record, source, scenario_index, target)
            pending.append((source, path, record_index, inspection))
            inspected.append(inspection)
            _increment(reason_counts, inspection["reasons"])
            _increment(
                field_counts,
                [field for field, present in inspection["field_presence"].items() if present],
            )
        source_summaries.append(
            {
                "id": source["id"],
                "kind": source["kind"],
                "expected_schema": source["expected_schema"],
                "glob": source["glob"],
                "matched_file_count": len(files),
                "record_count": len(inspected),
                "_records": inspected,
                "disposition": _source_status(inspected, errors, len(files)),
                "files": [
                    {"path": _repo_path(path, root), "sha256": file_hashes[path]} for path in files
                ],
                "errors": errors,
            }
        )

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for _, _, _, inspection in pending:
        episode_id = inspection["values"]["episode_id"]
        if episode_id:
            by_episode.setdefault(episode_id, []).append(inspection)
    for duplicate_rows in by_episode.values():
        if len(duplicate_rows) > 1:
            for inspection in duplicate_rows:
                inspection["reasons"] = sorted(
                    set(inspection["reasons"]) | {"duplicate_episode_id"}
                )
                inspection["status"] = "ineligible"
                reason_counts["duplicate_episode_id"] = (
                    reason_counts.get("duplicate_episode_id", 0) + 1
                )
    for summary in source_summaries:
        inspected = summary.pop("_records")
        summary["eligible_count"] = sum(row["status"] == "eligible" for row in inspected)
        summary["disposition"] = _source_status(
            inspected, summary["errors"], summary["matched_file_count"]
        )
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for source, path, record_index, inspection in pending:
        values = inspection["values"]
        identity = {
            "source_id": source["id"],
            "path": _repo_path(path, root),
            "record_index": record_index,
            "sha256": file_hashes.get(path),
            "scenario_id": values["scenario_id"],
            "planner_id": values["planner_id"],
            "seed": values["seed"],
            "episode_id": values["episode_id"],
            "source_commit": values["source_commit"],
        }
        identities.append(identity)
        rows.append(
            {
                **identity,
                "status": inspection["status"],
                "reasons": inspection["reasons"],
                "field_presence": inspection["field_presence"],
                "dimensions": inspection["dimensions"],
            }
        )

    def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
        return row["source_id"], row["path"], row["record_index"]

    key = row_key
    identities.sort(key=key)
    rows.sort(key=key)
    eligible_count = sum(row["status"] == "eligible" for row in rows)
    unavailable_sources = [
        source["id"]
        for source in source_summaries
        if source["disposition"] in {"unavailable", "unknown"}
    ]
    decision = (
        "blocked"
        if unavailable_sources
        else "admitted_for_private_execution"
        if eligible_count >= target["minimum_eligible_samples"]
        else "insufficient_eligible_traces"
    )
    staging = {
        "bundle_id": "issue8849-risk-calibration-matched-inputs-v1",
        "identity_digest": _canonical_digest(identities),
        "files": sorted(
            {item["path"]: item["sha256"] for item in identities if item["sha256"]}.items()
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "ok",
        "decision": decision,
        "config_path": _repo_path(config_path, root),
        "target": target,
        "frozen_packet": config["packet"],
        "record_count": len(rows),
        "eligible_count": eligible_count,
        "minimum_eligible_samples": target["minimum_eligible_samples"],
        "source_dispositions": source_summaries,
        "field_coverage": _sorted_counts(field_counts),
        "reason_counts": _sorted_counts(reason_counts),
        "expected_identities": identities,
        "staging_bundle": staging,
        "findings": (
            [
                {
                    "code": "unavailable_or_unknown_source",
                    "sources": unavailable_sources,
                    "blocking": True,
                }
            ]
            if unavailable_sources
            else []
        ),
        "rows": rows,
    }
    _assert_public(report)
    return report


def _invalid_report(config_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "invalid",
        "decision": "blocked",
        "config_path": config_path.name,
        "findings": [
            {"code": "invalid_preflight", "message": type(error).__name__, "blocking": True}
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), type=Path)
    parser.add_argument("--root", default=REPO_ROOT, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("json",), default="json")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic preflight and return its admission status."""
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    try:
        report = build_report(root, config_path)
    except (OSError, PreflightError, ValueError) as exc:
        report = _invalid_report(config_path, exc)
    if args.compact and "rows" in report:
        report = {key: value for key, value in report.items() if key != "rows"}
    if args.output:
        from robot_sf.evidence.writers import write_json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    if not args.check:
        return 0
    if report.get("status") != "ok" or report.get("decision") == "blocked":
        return 2
    return 0 if report.get("decision") == "admitted_for_private_execution" else 1


if __name__ == "__main__":
    raise SystemExit(main())
