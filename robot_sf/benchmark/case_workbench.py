"""Deterministic case discovery and explanation workbench.

This module owns the machine proposal ledger.  It intentionally stops at a
digest-bound proposal/admission package: author approval is an explicit overlay,
not an implicit side effect of ranking.
"""

# The workbench is intentionally a compact orchestration boundary.  Its input
# adapters keep optional analytics dependencies lazy, and the package emits
# reviewable return payloads rather than public library objects.
# ruff: noqa: DOC201, PLC0415

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.analysis_trace import canonical_json, trace_coverage
from robot_sf.benchmark.parquet_export import (
    derive_episode_metrics,
    export_campaign_result_store_v2,
)
from robot_sf.common.optional_import import try_import

SCHEMA_VERSION = "case-workbench.v1"
METRIC_PROFILE_VERSION = "case-workbench-metrics.v1"
ADMISSION_SCHEMA_VERSION = "case-admission-overlay.v1"
INELIGIBLE_STATUSES = {"fallback", "degraded", "failed", "unavailable", "partial"}


def load_workbench_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the compact workbench configuration."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"configuration must declare {SCHEMA_VERSION}")
    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, dict) or not isinstance(portfolio.get("roles"), list):
        raise ValueError("portfolio.roles must be a list")
    return payload


def analyze_cases(
    *,
    config_path: str | Path,
    result_store: str | Path,
    output: str | Path,
    check_determinism: bool = False,
) -> dict[str, Any]:
    """Build a deterministic proposal/admission package from a result store."""

    config = load_workbench_config(config_path)
    input_path = Path(result_store)
    output_path = Path(output)
    records = _load_records(input_path)
    candidates = [_candidate(record) for record in records]
    proposal = _build_proposal(candidates, config=config)
    if check_determinism:
        repeat = _build_proposal(candidates, config=config)
        if canonical_json(proposal) != canonical_json(repeat):
            raise RuntimeError("case-workbench selection is not deterministic")

    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        try:
            export_campaign_result_store_v2(
                input_path,
                output_path / "campaign-result-store.v2",
                study_id="case-workbench",
                command=f"analyze-cases --result-store {input_path}",
                overwrite=True,
            )
        except RuntimeError:
            # The proposal still remains useful in a lean environment; the
            # package states that the normalized store could not be materialized.
            (output_path / "campaign-result-store.v2.unavailable").write_text(
                "PyArrow is required to materialize campaign-result-store.v2\n",
                encoding="utf-8",
            )
    (output_path / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output_path / "proposal.json").write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overlay = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "proposal_sha256": _sha256_json(proposal),
        "status": str(config.get("admission", {}).get("status", "proposed")),
        "decisions": [],
    }
    (output_path / "admission_overlay.json").write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_case_files(output_path, proposal)
    _write_viewer_blueprint(output_path, proposal)
    _write_audit_dossier(output_path, proposal)
    _write_publication_preview(output_path, proposal)
    (output_path / "review_memo.md").write_text(_review_memo(proposal), encoding="utf-8")
    manifest = _manifest(output_path, proposal, input_path, config_path)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "SHA256SUMS").write_text(_checksums(output_path), encoding="utf-8")
    return proposal


def apply_admission_overlay(  # noqa: C901
    proposal: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply a digest-bound author overlay while retaining machine recommendations."""

    if overlay.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise ValueError(f"overlay must declare {ADMISSION_SCHEMA_VERSION}")
    expected_digest = _sha256_json(proposal)
    if overlay.get("proposal_sha256") != expected_digest:
        raise ValueError("admission overlay proposal_sha256 does not match the proposal")
    decisions = overlay.get("decisions", [])
    if not isinstance(decisions, list):
        raise ValueError("admission overlay decisions must be a list")
    machine_portfolio = [
        dict(case) for case in proposal.get("portfolio", []) if isinstance(case, Mapping)
    ]
    machine_portfolio = json.loads(canonical_json(machine_portfolio))
    working_portfolio = json.loads(canonical_json(machine_portfolio))
    by_id = {str(case.get("case_id")): case for case in working_portfolio}
    final_portfolio = working_portfolio
    admission_records: list[dict[str, Any]] = []
    for decision in sorted(
        (item for item in decisions if isinstance(item, Mapping)),
        key=lambda item: (str(item.get("case_id")), str(item.get("decision"))),
    ):
        case_id = str(decision.get("case_id") or "")
        action = str(decision.get("decision") or "").lower()
        rationale = str(decision.get("rationale") or "").strip()
        if case_id not in by_id:
            raise ValueError(f"admission overlay references unknown proposed case: {case_id}")
        if action not in {"approve", "reject", "replace"}:
            raise ValueError(f"unsupported admission decision for {case_id}: {action}")
        if not rationale:
            raise ValueError(f"admission rationale is required for {case_id}")
        if action == "replace":
            replacement = decision.get("replacement")
            if not isinstance(replacement, Mapping) or not replacement.get("case_id"):
                raise ValueError(f"replacement case is required for {case_id}")
            replacement_case = dict(replacement)
            replacement_provenance = replacement_case.get("provenance")
            if not isinstance(replacement_provenance, Mapping) or not replacement_provenance.get(
                "artifact_sha256"
            ):
                raise ValueError(f"replacement case provenance is required for {case_id}")
            replacement_case["author_status"] = "replacement"
            replacement_case["machine_recommendation"] = case_id
            replacement_case["author_rationale"] = rationale
            final_portfolio = [
                case for case in final_portfolio if str(case.get("case_id")) != case_id
            ]
            replacement_id = str(replacement_case["case_id"])
            if any(str(case.get("case_id")) == replacement_id for case in final_portfolio):
                raise ValueError(
                    f"replacement case id is already in the portfolio: {replacement_id}"
                )
            final_portfolio.append(replacement_case)
        else:
            selected = by_id[case_id]
            selected["author_status"] = "approved" if action == "approve" else "rejected"
            selected["author_rationale"] = rationale
            if action == "reject":
                final_portfolio = [
                    case for case in final_portfolio if str(case.get("case_id")) != case_id
                ]
        admission_records.append(
            {
                "case_id": case_id,
                "decision": action,
                "rationale": rationale,
            }
        )
    result = json.loads(canonical_json(proposal))
    result["machine_portfolio"] = machine_portfolio
    result["portfolio"] = final_portfolio
    result["author_admission"] = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "status": str(overlay.get("status") or "proposed"),
        "overlay_sha256": _sha256_json(overlay),
        "decisions": admission_records,
    }
    return result


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load source records from JSONL or a v2 Parquet result store."""

    if path.is_file():
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if not path.is_dir():
        raise FileNotFoundError(f"result store does not exist: {path}")
    jsonl_candidates = [path / "episodes.jsonl", path / "records.jsonl"]
    for candidate in jsonl_candidates:
        if candidate.is_file():
            return _load_records(candidate)
    episodes_path = path / "episodes.parquet"
    if not episodes_path.is_file():
        raise ValueError(f"result store has no episodes.jsonl or episodes.parquet: {path}")
    return _load_v2_records(path, episodes_path)


def _load_v2_records(store: Path, episodes_path: Path) -> list[dict[str, Any]]:
    """Rehydrate v2 episode, step, actor, event, and feature tables."""

    episode_rows = _read_parquet_rows(episodes_path)
    step_rows = (
        _read_parquet_rows(store / "steps.parquet") if (store / "steps.parquet").is_file() else []
    )
    actor_rows = (
        _read_parquet_rows(store / "actors.parquet") if (store / "actors.parquet").is_file() else []
    )
    event_rows = (
        _read_parquet_rows(store / "events.parquet") if (store / "events.parquet").is_file() else []
    )
    feature_rows = (
        _read_parquet_rows(store / "features.parquet")
        if (store / "features.parquet").is_file()
        else []
    )
    by_episode_steps: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode_actors: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_episode_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_episode_features: dict[str, dict[str, float]] = defaultdict(dict)
    for row in step_rows:
        by_episode_steps[str(row.get("episode_id"))].append(row)
    for row in actor_rows:
        by_episode_actors[(str(row.get("episode_id")), int(row.get("step") or 0))].append(row)
    for row in event_rows:
        by_episode_events[str(row.get("episode_id"))].append(row)
    for row in feature_rows:
        value = row.get("value_number")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            by_episode_features[str(row.get("episode_id"))][str(row.get("feature_name"))] = float(
                value
            )
    result: list[dict[str, Any]] = []
    for row in episode_rows:
        episode_id = str(row.get("episode_id") or "")
        trace_steps: list[dict[str, Any]] = []
        for step in sorted(
            by_episode_steps.get(episode_id, []), key=lambda item: int(item.get("step") or 0)
        ):
            robot = {
                "actor_id": "robot",
                "position": [step.get("robot_x"), step.get("robot_y")],
                "heading": step.get("heading_rad"),
                "velocity": [step.get("robot_vx"), step.get("robot_vy")],
                "radius_m": None,
            }
            pedestrians = []
            for actor in by_episode_actors.get((episode_id, int(step.get("step") or 0)), []):
                if str(actor.get("actor_kind")) == "robot":
                    robot["radius_m"] = actor.get("radius_m")
                    continue
                pedestrians.append(
                    {
                        "actor_id": actor.get("actor_id"),
                        "position": [actor.get("x"), actor.get("y")],
                        "velocity": [actor.get("vx"), actor.get("vy")],
                        "radius_m": actor.get("radius_m"),
                    }
                )
            trace_steps.append(
                {
                    "step": step.get("step"),
                    "time_s": step.get("time_s"),
                    "robot": robot,
                    "pedestrians": pedestrians,
                    "controls": {
                        "requested": {
                            "linear_m_s": step.get("requested_linear_m_s"),
                            "turn_rate_rad_s": step.get("requested_turn_rate_rad_s"),
                        },
                        "applied": {
                            "linear_m_s": step.get("applied_linear_m_s"),
                            "turn_rate_rad_s": step.get("applied_turn_rate_rad_s"),
                        },
                    },
                    "events": [],
                }
            )
        coverage = _decode_json(row.get("trace_coverage_json")) or {"status": "unavailable"}
        provenance = _decode_json(row.get("provenance_json")) or {}
        first_step = by_episode_steps.get(episode_id, [None])[0]
        units = _decode_json(first_step.get("units_json")) if isinstance(first_step, dict) else None
        analysis_trace = {
            "schema_version": "analysis-trace.v1",
            "scenario_id": row.get("scenario_id"),
            "planner": row.get("planner"),
            "map_digest": provenance.get("map_digest"),
            "config_hash": row.get("config_hash") or provenance.get("config_hash"),
            "git_hash": provenance.get("git_hash"),
            "coordinate_frame": (
                first_step.get("coordinate_frame") if isinstance(first_step, dict) else None
            ),
            "units": units,
            "steps": trace_steps,
            "events": by_episode_events.get(episode_id, []),
        }
        result.append(
            {
                "episode_id": episode_id,
                "scenario_id": row.get("scenario_id"),
                "seed": row.get("seed"),
                "algo": row.get("planner"),
                "status": row.get("execution_status"),
                "row_status": row.get("row_status"),
                "outcome": _decode_json(row.get("outcome_json")) or {},
                "provenance": provenance,
                "config_hash": row.get("config_hash") or provenance.get("config_hash"),
                "git_hash": provenance.get("git_hash"),
                "metrics": by_episode_features.get(episode_id, {}),
                "trace_coverage": coverage,
                "algorithm_metadata": {
                    "analysis_trace": analysis_trace,
                },
            }
        )
    return result


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read a Parquet table through DuckDB or pandas."""

    duckdb = try_import("duckdb")
    if duckdb is not None:
        return (
            duckdb.connect(database=":memory:")
            .execute("SELECT * FROM read_parquet(?)", [str(path)])
            .fetchdf()
            .to_dict(orient="records")
        )
    import pandas as pd

    return pd.read_parquet(path).to_dict(orient="records")


def _decode_json(value: Any) -> Any:
    """Decode a stored JSON column."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return value


def _row_from_v2_store(row: Mapping[str, Any], store: Path) -> dict[str, Any]:
    """Rehydrate the episode-level fields required by proposal selection."""

    def decode(name: str) -> Any:
        value = row.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        try:
            return json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            return value

    return {
        "episode_id": row.get("episode_id"),
        "scenario_id": row.get("scenario_id"),
        "seed": row.get("seed"),
        "algo": row.get("planner"),
        "status": row.get("execution_status"),
        "row_status": row.get("row_status"),
        "outcome": decode("outcome_json") or {},
        "provenance": decode("provenance_json") or {"artifact_uri": str(store)},
        "algorithm_metadata": {"analysis_trace": None},
        "trace_coverage": decode("trace_coverage_json") or {"status": "unavailable"},
    }


def _candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one episode into an eligibility and metric candidate."""

    coverage = record.get("trace_coverage")
    if not isinstance(coverage, Mapping):
        coverage = trace_coverage(dict(record))
    provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    row_status = str(record.get("row_status") or record.get("status") or "native")
    blockers: list[str] = []
    if row_status in INELIGIBLE_STATUSES:
        blockers.append(f"execution_status:{row_status}")
    if coverage.get("status") != "complete":
        blockers.append(f"trace_coverage:{coverage.get('reason') or 'incomplete'}")
    if not provenance.get("artifact_sha256"):
        blockers.append("provenance:artifact_sha256_missing")
    scenario_id = str(record.get("scenario_id") or "unknown")
    planner = str(record.get("algo") or record.get("planner") or "unknown")
    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    success = bool(outcome.get("success") or outcome.get("route_complete"))
    collision = bool(outcome.get("collision"))
    return {
        "episode_id": str(record.get("episode_id") or ""),
        "scenario_id": scenario_id,
        "planner": planner,
        "seed": _int(record.get("seed")),
        "row_status": row_status,
        "coverage": dict(coverage),
        "provenance": dict(provenance),
        "outcome": {"success": success, "collision": collision, "label": _outcome_label(record)},
        "metrics": _episode_metrics(record),
        "eligible": not blockers,
        "exclusion_reasons": blockers,
        "interestingness_score": _interestingness(record),
        "raw": dict(record),
    }


def _episode_metrics(record: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract interpretable metrics without fabricating missing values."""

    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    derived = derive_episode_metrics(record)

    def value(*keys: str) -> float | None:
        """Prefer recorded aggregate values, then v2 trace-derived features."""

        recorded = _first_number(metrics, *keys)
        return recorded if recorded is not None else _first_number(derived, *keys)

    return {
        "surface_clearance_min": value(
            "surface_clearance_min", "min_surface_clearance", "min_separation"
        ),
        "progress": value("progress", "route_progress", "distance_travelled"),
        "control_effort": value("control_effort", "action_effort"),
        "event_time": value("event_time", "first_collision_time"),
        "ttc_min": value("ttc_min"),
        "cpa_min": value("cpa_min"),
        "closing_speed_max": value("closing_speed_max"),
        "braking_response_time": value("braking_response_time"),
        "turning_response_time": value("turning_response_time"),
        "critical_duration_integral": value("critical_duration_integral"),
        "stall_duration": value("stall_duration"),
        "reversal_count": value("reversal_count"),
        "detour_ratio": value("detour_ratio"),
        "clipping_steps": value("clipping_steps"),
        "fallback_steps": value("fallback_steps"),
        "outcome_score": value("outcome_score"),
    }


def _interestingness(record: Mapping[str, Any]) -> float:
    """Use a transparent scalar only for broad exploratory triage."""

    metrics = _episode_metrics(record)
    clearance = metrics.get("surface_clearance_min")
    score = 0.0
    if clearance is not None:
        score += max(0.0, 2.0 - float(clearance))
    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    if outcome.get("collision"):
        score += 1.0
    if outcome.get("success"):
        score += 0.25
    effort = metrics.get("control_effort")
    if effort is not None:
        score += min(1.0, abs(float(effort)) / 10.0)
    return round(score, 9)


def _build_proposal(
    candidates: list[dict[str, Any]], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the complete proposed portfolio and reason ledger."""

    eligible = [item for item in candidates if item["eligible"]]
    roles = [str(role) for role in config.get("portfolio", {}).get("roles", [])]
    max_cases = int(config.get("portfolio", {}).get("max_cases", 12))
    role_candidates = {role: _role_candidates(role, eligible) for role in roles}
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    unavailable_roles: list[dict[str, Any]] = []
    for role in roles:
        options = role_candidates[role]
        if not options:
            unavailable_roles.append(
                {"role": role, "reason": _role_unavailable_reason(role, candidates)}
            )
            continue
        pareto = _pareto_front(options)
        if role in {"seed_sensitivity", "planner_upset"}:
            # Exact pair coverage may deliberately retain a dominated member: a
            # success/collision or planner contrast is itself the role signal.
            pair = _comparison_pair(role, options)
            if pair is not None:
                pair_ids = [str(item["episode_id"]) for item in pair]
                for chosen in pair:
                    if chosen["episode_id"] in selected_ids:
                        continue
                    selected_ids.add(chosen["episode_id"])
                    selected.append(
                        _selection_record(
                            role,
                            chosen,
                            options,
                            pareto,
                            comparison_pair_ids=pair_ids,
                        )
                    )
                continue
        chosen = sorted(pareto, key=lambda item: (-_role_score(role, item), item["episode_id"]))[0]
        if chosen["episode_id"] in selected_ids:
            continue
        selected_ids.add(chosen["episode_id"])
        selected.append(_selection_record(role, chosen, options, pareto))
    selected = selected[:max_cases]
    selected_ids = {item["case_id"] for item in selected}
    excluded = []
    for candidate in candidates:
        if candidate["episode_id"] in selected_ids:
            continue
        reason = candidate["exclusion_reasons"] or ["not_selected_after_role_coverage"]
        excluded.append(
            {
                "case_id": candidate["episode_id"],
                "eligible": candidate["eligible"],
                "reasons": reason,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_profile_version": METRIC_PROFILE_VERSION,
        "selection_authority": "machine_proposal_plus_author_admission_overlay",
        "claim_boundary": "descriptive observed cases; no causal divergence or planner ranking",
        "portfolio": selected,
        "excluded": sorted(excluded, key=lambda item: item["case_id"]),
        "unavailable_roles": unavailable_roles,
        "runner_ups": _runner_up_ledger(role_candidates, selected),
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
    }


def _runner_up_ledger(
    role_candidates: Mapping[str, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Return stable runner-up explanations for every configured role."""

    selected_by_role: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        selected_by_role[str(item.get("primary_role"))].add(str(item.get("case_id")))
    ledger: dict[str, list[dict[str, Any]]] = {}
    for role, options in role_candidates.items():
        chosen_ids = selected_by_role.get(role, set())
        ledger[role] = [
            _runner_up(item, {})
            for item in sorted(
                options, key=lambda item: (-_role_score(role, item), item["episode_id"])
            )
            if item["episode_id"] not in chosen_ids
        ][:3]
    return ledger


def _role_candidates(role: str, candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates satisfying a role's local predicate."""

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_scenario[candidate["scenario_id"]].append(candidate)
    if role == "seed_sensitivity":
        return [
            item
            for group in by_scenario.values()
            if len({entry["seed"] for entry in group}) > 1
            for item in group
        ]
    if role == "planner_upset":
        return [
            item
            for group in by_scenario.values()
            if len({entry["planner"] for entry in group}) > 1
            for item in group
        ]
    if role == "safety_boundary":
        return [
            item for item in candidates if item["metrics"].get("surface_clearance_min") is not None
        ]
    if role == "metric_disagreement":
        return [
            item
            for item in candidates
            if item["metrics"].get("surface_clearance_min") is not None
            and item["metrics"].get("progress") is not None
        ]
    if role == "cross_cell_inversion":
        return [
            item
            for item in candidates
            if item["outcome"]["success"] or item["outcome"]["collision"]
        ]
    if role == "representative_control":
        return [item for item in candidates if item["metrics"].get("control_effort") is not None]
    return []


def _comparison_pair(role: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Choose a stable two-member comparison pair for a pair-valued role."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["scenario_id"]].append(candidate)
    pair_options: list[tuple[float, tuple[str, str], str, list[dict[str, Any]]]] = []
    dimension = "seed" if role == "seed_sensitivity" else "planner"
    for scenario_id, group in grouped.items():
        ranked = sorted(group, key=lambda item: (-_role_score(role, item), item["episode_id"]))
        for left_index, left in enumerate(ranked):
            for right in ranked[left_index + 1 :]:
                if left[dimension] == right[dimension]:
                    continue
                selected = sorted([left, right], key=lambda item: item["episode_id"])
                ids = (str(selected[0]["episode_id"]), str(selected[1]["episode_id"]))
                pair_options.append(
                    (_role_score(role, left) + _role_score(role, right), ids, scenario_id, selected)
                )
    if not pair_options:
        return None
    return sorted(pair_options, key=lambda item: (-item[0], item[1], item[2]))[0][3]


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute a stable nondominated front over role-local scalar dimensions."""

    if len(candidates) < 2:
        return list(candidates)
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            a = _vector(candidate)
            b = _vector(other)
            if all(x <= y for x, y in zip(a, b, strict=True)) and any(
                x < y for x, y in zip(a, b, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["episode_id"])


def _vector(candidate: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return comparable dimensions (lower clearance is more interesting)."""

    metrics = candidate["metrics"]
    clearance = metrics.get("surface_clearance_min")
    effort = metrics.get("control_effort")
    progress = metrics.get("progress")
    outcome = 1.0 if candidate["outcome"]["collision"] else 0.0
    return (
        float(clearance) if clearance is not None else 1.0e9,
        -outcome,
        -float(progress) if progress is not None else 1.0e9,
        -float(effort) if effort is not None else 1.0e9,
    )


def _role_score(role: str, candidate: Mapping[str, Any]) -> float:
    """Rank candidates locally after Pareto filtering."""

    metrics = candidate["metrics"]
    clearance = metrics.get("surface_clearance_min")
    score = float(candidate.get("interestingness_score", 0.0))
    if clearance is not None:
        score += max(0.0, 1.0 - float(clearance))
    if role == "seed_sensitivity":
        score += 0.5 if candidate["outcome"]["collision"] else 0.0
    if role == "planner_upset":
        score += 0.5
    return score


def _selection_record(
    role: str,
    chosen: Mapping[str, Any],
    options: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    *,
    comparison_pair_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a selection ledger row with runner-up explanation."""

    ranked = sorted(options, key=lambda item: (-_role_score(role, item), item["episode_id"]))
    runner_up = next((item for item in ranked if item["episode_id"] != chosen["episode_id"]), None)
    return {
        "case_id": chosen["episode_id"],
        "primary_role": role,
        "scenario_id": chosen["scenario_id"],
        "planner": chosen["planner"],
        "seed": chosen["seed"],
        "selection_reason": {
            "role": role,
            "pareto_status": "nondominated"
            if chosen["episode_id"] in {item["episode_id"] for item in pareto}
            else "dominated",
            "interestingness_score": chosen["interestingness_score"],
            "why_selected_over_runner_up": (
                (
                    "paired role coverage across distinct dimensions; stable episode-id tie-break"
                    if comparison_pair_ids
                    else "higher role-local score; stable episode-id tie-break"
                )
                if runner_up is not None
                else (
                    "paired role coverage across distinct dimensions"
                    if comparison_pair_ids
                    else "only eligible candidate for this role"
                )
            ),
            "comparison_pair_ids": comparison_pair_ids or [],
        },
        "outcome": chosen["outcome"],
        "metrics": chosen["metrics"],
        "coverage": chosen["coverage"],
        "provenance": chosen["provenance"],
        "trace": (
            chosen.get("raw", {}).get("algorithm_metadata", {}).get("analysis_trace")
            if isinstance(chosen.get("raw"), Mapping)
            and isinstance(chosen.get("raw", {}).get("algorithm_metadata"), Mapping)
            else None
        ),
        "shared_prefix": False,
        "comparison_pair_ids": comparison_pair_ids or [],
        "author_status": "proposed",
    }


def _runner_up(candidate: Mapping[str, Any], chosen: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact runner-up explanation."""

    return {
        "case_id": candidate.get("episode_id"),
        "score": candidate.get("interestingness_score"),
        "reason_not_selected": "lower role-local score or stable tie-break",
    }


def _role_unavailable_reason(role: str, candidates: list[dict[str, Any]]) -> str:
    """Explain why a role has no eligible candidate."""

    if not candidates:
        return "no_candidates"
    if any(not item["eligible"] for item in candidates):
        return "all_candidates_failed_eligibility"
    return "role_predicate_not_satisfied"


def _write_case_files(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write one compact case record per proposed case."""

    cases = output / "cases"
    cases.mkdir(exist_ok=True)
    for case in proposal.get("portfolio", []):
        case_id = str(case["case_id"])
        (cases / f"{case_id}.json").write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _write_viewer_blueprint(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write the stable synchronized-view blueprint consumed by the viewer adapter."""

    payload = {
        "schema_version": "case-workbench-viewer-blueprint.v1",
        "layout": "world_plus_absolute_time_tracks",
        "case_ids": [
            str(case.get("case_id"))
            for case in proposal.get("portfolio", [])
            if isinstance(case, Mapping)
        ],
        "views": [
            {"id": "world", "kind": "spatial", "coordinate_frame": "world", "synchronized": True},
            {
                "id": "clearance",
                "kind": "time_series",
                "field": "surface_clearance",
                "units": "m",
                "synchronized": True,
            },
            {
                "id": "speed",
                "kind": "time_series",
                "field": "applied_linear_speed",
                "units": "m/s",
                "synchronized": True,
            },
            {
                "id": "turn_rate",
                "kind": "time_series",
                "field": "applied_turn_rate",
                "units": "rad/s",
                "synchronized": True,
            },
            {"id": "events", "kind": "event_timeline", "synchronized": True},
            {"id": "state", "kind": "state_inspector", "synchronized": True},
        ],
        "time_policy": "absolute_recorded_time; no normalized-duration alignment",
    }
    (output / "viewer_blueprint.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_audit_dossier(output: Path, proposal: Mapping[str, Any]) -> None:
    """Write a complete, machine-readable audit dossier beside the reduced figure."""

    dossier = {
        "schema_version": "case-workbench-audit-dossier.v1",
        "evidence_status": "proposed_not_admitted",
        "claim_boundary": proposal.get("claim_boundary"),
        "portfolio": proposal.get("portfolio", []),
        "excluded": proposal.get("excluded", []),
        "unavailable_roles": proposal.get("unavailable_roles", []),
        "runner_ups": proposal.get("runner_ups", {}),
        "selection_authority": proposal.get("selection_authority"),
    }
    (output / "audit_dossier.json").write_text(
        json.dumps(dossier, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Case workbench audit dossier",
        "",
        "This is the full review ledger. It is not a dissertation figure and does not admit evidence.",
        "",
        f"- evidence status: `{dossier['evidence_status']}`",
        f"- candidates: `{proposal.get('candidate_count', 0)}`",
        f"- eligible: `{proposal.get('eligible_count', 0)}`",
        f"- selected: `{len(proposal.get('portfolio', []))}`",
        "- shared prefix: `false` unless a matched-start receipt is supplied",
        "",
        "## Complete proposed portfolio",
    ]
    for case in proposal.get("portfolio", []):
        if not isinstance(case, Mapping):
            continue
        lines.extend(
            [
                "",
                f"### `{case.get('case_id')}`",
                f"- role: `{case.get('primary_role')}`",
                f"- planner/seed: `{case.get('planner')}` / `{case.get('seed')}`",
                f"- selection reason: {case.get('selection_reason', {}).get('why_selected_over_runner_up')}",
                f"- provenance: `{case.get('provenance', {}).get('artifact_sha256')}`",
                f"- metrics: `{json.dumps(case.get('metrics', {}), sort_keys=True)}`",
            ]
        )
    lines.extend(["", "## Unavailable roles"])
    for role in proposal.get("unavailable_roles", []):
        lines.append(f"- `{role.get('role')}`: {role.get('reason')}")
    lines.extend(["", "## Exclusions"])
    for item in proposal.get("excluded", []):
        lines.append(
            f"- `{item.get('case_id')}`: {', '.join(str(reason) for reason in item.get('reasons', []))}"
        )
    (output / "audit_dossier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_publication_preview(output: Path, proposal: Mapping[str, Any]) -> None:
    """Render a reduced preview when plotting dependencies are available."""

    publication = output / "publication"
    publication.mkdir(exist_ok=True)
    if not proposal.get("portfolio"):
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps({"status": "unavailable", "reason": "no_proposed_cases"}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if try_import("matplotlib") is None:
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": "renderer_unavailable:matplotlib_missing",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    try:
        from robot_sf.benchmark.case_publication_figure import render_publication_figure

        render_publication_figure(output, output=publication / "figure.pdf", output_format="pdf")
    except RuntimeError as exc:
        (publication / "UNAVAILABLE.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "reason": f"renderer_unavailable:{exc.__class__.__name__}",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _review_memo(proposal: Mapping[str, Any]) -> str:
    """Render a deterministic author review memo."""

    lines = [
        "# Case workbench review memo",
        "",
        "Machine proposal only. Author admission is required before publication.",
        "",
        f"- candidates: {proposal.get('candidate_count', 0)}",
        f"- eligible: {proposal.get('eligible_count', 0)}",
        f"- selected: {len(proposal.get('portfolio', []))}",
        "- shared prefix: false unless a matched-start receipt is supplied",
        "",
        "## Proposed cases",
    ]
    for case in proposal.get("portfolio", []):
        lines.append(
            f"- `{case['case_id']}` ({case['primary_role']}): {case['selection_reason']['why_selected_over_runner_up']}"
        )
    lines.extend(["", "## Unavailable roles"])
    for role in proposal.get("unavailable_roles", []):
        lines.append(f"- `{role['role']}`: {role['reason']}")
    return "\n".join(lines) + "\n"


def _manifest(
    output: Path, proposal: Mapping[str, Any], source: Path, config: str | Path
) -> dict[str, Any]:
    """Build package provenance manifest."""

    return {
        "schema_version": "case-workbench-package.v1",
        "workbench_schema_version": SCHEMA_VERSION,
        "proposal_sha256": _sha256_json(proposal),
        "source": {
            "path": str(source),
            "sha256": _sha256_file(source) if source.is_file() else None,
        },
        "config": {"path": str(config), "sha256": _sha256_file(Path(config))},
        "evidence_status": "proposed_not_admitted",
        "source_integrity_gate": {
            "status": "blocked_pending_exact_source_restore",
            "robot_sf_issues": [
                "https://github.com/ll7/robot_sf_ll7/issues/6792",
                "https://github.com/ll7/robot_sf_ll7/issues/6814",
            ],
            "dissertation_issue": "https://github.com/ll7/diss/issues/698",
        },
        "claim_boundary": "No causal divergence point; no planner ranking; source integrity remains separate.",
        "files": sorted(
            str(path.relative_to(output))
            for path in output.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        ),
    }


def _checksums(output: Path) -> str:
    """Return deterministic checksums for package files."""

    paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    return "\n".join(f"{_sha256_file(path)}  {path.relative_to(output)}" for path in paths) + "\n"


def _outcome_label(record: Mapping[str, Any]) -> str:
    """Return a stable descriptive outcome label."""

    outcome = record.get("outcome") if isinstance(record.get("outcome"), Mapping) else {}
    if outcome.get("collision"):
        return "collision"
    if outcome.get("success") or outcome.get("route_complete"):
        return "success"
    return str(record.get("termination_reason") or record.get("status") or "unknown")


def _first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    """Return the first finite numeric value under the given keys."""

    for key in keys:
        value = mapping.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    return None


def _int(value: Any) -> int | None:
    """Coerce an integer-like value."""

    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sha256_json(value: Any) -> str:
    """Hash canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ADMISSION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "analyze_cases",
    "apply_admission_overlay",
    "load_workbench_config",
]
