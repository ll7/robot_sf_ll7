#!/usr/bin/env python3
"""Build a deterministic, fail-closed plan for a finite compute window."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Keep the data-shaping literals compact to honor issue #8829's hard net-line cap.
# Ruff lint remains active; the formatter is scoped out for this deliberately dense helper.
# fmt: off
INPUT_SCHEMA = "compute_window_capacity_inventory.v1"
PLAN_SCHEMA = "compute_window_capacity_plan.v1"
LANES = ("cpu", "gpu", "carla", "host_pinned")
WEIGHTS = {
    "time_to_loss": 0.20,
    "expected_wall_time": 0.10,
    "queue_uncertainty": 0.10,
    "resource_footprint": 0.10,
    "transfer_volume": 0.10,
    "completion_probability": 0.20,
    "unlock_value": 0.15,
    "executable_later": 0.05,
}
READY = frozenset(("ready", "admitted", "eligible", "planned"))
SATISFIED = frozenset(("satisfied", "met", "complete", "available", "resolved"))
REQUIRED = frozenset(("required", "pending", "unmet", "blocked"))
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DEFAULT_OUTPUT = Path("output/compute_window_capacity_planner/compute_window_plan.json")
CLAIM_BOUNDARY = (
    "Operational capacity planning only: this plan ranks admitted workload metadata and does "
    "not establish benchmark quality, scientific priority, scheduler admission, or job success."
)


def _problem(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "location": path, "message": message}


def _first(value: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _slug(value: Any) -> str | None:
    return value if isinstance(value, str) and SLUG_RE.fullmatch(value) else None


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def _render(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _range(value: Any, path: str, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        expected = _number(value.get("expected"))
        upper = _number(_first(value, "upper", "maximum", "conservative"))
    else:
        expected = upper = _number(value)
    if expected is None or upper is None:
        errors.append(_problem("missing_estimate", path, "expected and upper are required"))
        return None
    if upper < expected:
        errors.append(_problem("invalid_estimate", path, "upper must be >= expected"))
        return None
    return {"expected": expected, "upper": upper}


def _probability(value: Any, path: str, errors: list[dict[str, str]]) -> dict[str, Any] | None:
    raw = value if isinstance(value, Mapping) else {"expected": value}
    expected = _number(raw.get("expected"))
    lower = _number(raw.get("lower", expected))
    upper = _number(raw.get("upper", expected))
    if any(item is None or item > 1 for item in (expected, lower, upper)):
        errors.append(_problem("missing_estimate", path, "probability must be in [0, 1]"))
        return None
    if lower > expected or expected > upper:
        errors.append(_problem("invalid_estimate", path, "probability bounds are inconsistent"))
        return None
    return {"lower": lower, "expected": expected, "upper": upper}


def _resources(value: Any, errors: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    capacities: dict[str, int] = {}
    queues: dict[str, int] = {}
    if not isinstance(value, Mapping):
        errors.append(_problem("missing_resources", "/resources", "lane limits are required"))
        return capacities, queues
    for lane in LANES:
        record = value.get(lane)
        for field, target, code in (("capacity", capacities, "missing_resource_capacity"), ("queue_limit", queues, "missing_queue_limit")):
            item = record.get(field) if isinstance(record, Mapping) else None
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                errors.append(_problem(code, f"/resources/{lane}/{field}", "non-negative integer required"))
            else:
                target[lane] = item
    return capacities, queues


def _storage(value: Any, errors: list[dict[str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(value, Mapping):
        errors.append(_problem("missing_storage", "/storage", "storage limits are required"))
        return result
    for name in ("available_bytes", "transfer_capacity_bytes"):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            errors.append(_problem("missing_storage_capacity", f"/storage/{name}", "non-negative integer required"))
        else:
            result[name] = item
    return result


def _dependencies(value: Any, errors: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        errors.append(_problem("missing_dependencies", "/dependencies", "fresh dependency graph required"))
        return []
    if value.get("freshness") != "fresh":
        errors.append(_problem("stale_dependencies", "/dependencies/freshness", "graph is not fresh"))
    raw_edges = value.get("edges")
    if not isinstance(raw_edges, list):
        errors.append(_problem("missing_dependency_edges", "/dependencies/edges", "edge list required"))
        return []
    result: list[dict[str, str]] = []
    for index, raw in enumerate(raw_edges):
        path = f"/dependencies/edges[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(_problem("invalid_dependency", path, "edge must be an object"))
            continue
        child = _slug(_first(raw, "workload_id", "workload", "child"))
        parent = _slug(_first(raw, "depends_on", "dependency", "parent"))
        status, satisfied = raw.get("status"), raw.get("satisfied")
        known = satisfied is True or satisfied is False or status in SATISFIED or status in REQUIRED
        if child is None or parent is None or child == parent or not known:
            errors.append(_problem("invalid_dependency", path, "distinct ids and known status required"))
        elif satisfied is False or status in REQUIRED:
            result.append({"child": child, "parent": parent})
    return result


def _context(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:  # noqa: C901 - independent fail-closed gates are intentionally explicit.
    errors: list[dict[str, str]] = []
    if payload.get("schema") != INPUT_SCHEMA:
        errors.append(_problem("schema_mismatch", "/schema", f"must be {INPUT_SCHEMA}"))
    as_of = _time(payload.get("as_of"))
    if as_of is None:
        errors.append(_problem("missing_as_of", "/as_of", "timezone-aware timestamp required"))
    inventory = payload.get("inventory")
    if not isinstance(inventory, Mapping) or inventory.get("status") != "fresh":
        errors.append(_problem("stale_source", "/inventory/status", "fresh inventory required"))
    admission = payload.get("admission")
    if not isinstance(admission, Mapping):
        errors.append(_problem("missing_admission", "/admission", "live admission required"))
    else:
        if admission.get("status") not in {"admitted", "live_admitted"}:
            errors.append(_problem("admission_unavailable", "/admission/status", "admission is not live"))
        if admission.get("freshness") != "fresh":
            errors.append(_problem("stale_admission", "/admission/freshness", "admission is not fresh"))
    deadline_raw = payload.get("deadline")
    deadline = _time(deadline_raw.get("timestamp")) if isinstance(deadline_raw, Mapping) else None
    if deadline is None:
        errors.append(_problem("missing_deadline", "/deadline/timestamp", "timezone-aware timestamp required"))
    if not isinstance(deadline_raw, Mapping) or _slug(deadline_raw.get("source")) is None:
        errors.append(_problem("missing_deadline_source", "/deadline/source", "sanitized source slug required"))
    if not isinstance(deadline_raw, Mapping) or deadline_raw.get("freshness") != "fresh":
        errors.append(_problem("stale_deadline", "/deadline/freshness", "deadline is not fresh"))
    if as_of and deadline and deadline <= as_of:
        errors.append(_problem("deadline_expired", "/deadline/timestamp", "access deadline has passed"))
    capacities, queues = _resources(payload.get("resources"), errors)
    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        errors.append(_problem("missing_workloads", "/workloads", "workload list required"))
        workloads = []
    return {"as_of": as_of, "deadline": deadline, "capacities": capacities, "queues": queues, "storage": _storage(payload.get("storage"), errors), "edges": _dependencies(payload.get("dependencies"), errors), "workloads": workloads}, errors


def _row_gates(row: Mapping[str, Any], path: str) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    status = row.get("status")
    if status in {"blocked", "decision_required"}:
        errors.append(_problem("blocked", f"{path}/status", "row is blocked"))
    elif status not in READY:
        errors.append(_problem("missing_status", f"{path}/status", "ready status required"))
    checks = (
        (row.get("decision_required") is not True and row.get("decision_status") != "required", "decision_required", "maintainer decision required"),
        (row.get("duplicate") is not True and row.get("duplicate_of") in (None, ""), "duplicate", "equivalent workload already exists"),
        (row.get("source_status", row.get("source")) == "fresh", "stale_source", "fresh source required"),
        (row.get("input_status", row.get("inputs_status", row.get("inputs_ready"))) in {"complete", "ready", True}, "missing_input", "complete inputs required"),
        (row.get("authorization", row.get("authorized")) in {"authorized", True}, "unauthorized", "authorization required"),
    )
    for valid, code, message in checks:
        if not valid:
            errors.append(_problem(code, path, message))
    execution = row.get("execution_state", row.get("state"))
    if row.get("already_running") is True or execution in {"running", "queued", "pending", "configuring"}:
        errors.append(_problem("already_running", f"{path}/execution_state", "equivalent work is running or queued"))
    elif execution == "completed":
        errors.append(_problem("already_complete", f"{path}/execution_state", "workload is complete"))
    elif execution != "not_running":
        errors.append(_problem("missing_execution_state", f"{path}/execution_state", "not_running state required"))
    return errors


def _demand(row: Mapping[str, Any], path: str) -> tuple[str | None, dict[str, int], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    lane = row.get("lane", row.get("resource_class"))
    if not isinstance(lane, str) or lane not in LANES:
        errors.append(_problem("invalid_lane", f"{path}/lane", "known lane required"))
        lane = None
    raw, demand = row.get("resource_demand", row.get("resources")), {}
    if not isinstance(raw, Mapping):
        errors.append(_problem("missing_resource_demand", f"{path}/resource_demand", "all lane demands required"))
    else:
        for name in LANES:
            value = raw.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(_problem("missing_resource_demand", f"{path}/resource_demand/{name}", "non-negative integer required"))
            else:
                demand[name] = value
    if lane and demand.get(lane, 0) <= 0:
        errors.append(_problem("missing_resource_demand", f"{path}/resource_demand/{lane}", "selected lane demand must be positive"))
    return lane, demand, errors


def _estimates(row: Mapping[str, Any], path: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    raw = row.get("estimates", row.get("estimate", row))
    if not isinstance(raw, Mapping):
        return {}, [_problem("missing_estimate", f"{path}/estimates", "estimate mapping required")]
    aliases = {"wall": ("wall_seconds", "runtime_seconds", "expected_wall_seconds"), "queue": ("queue_seconds",), "transfer": ("transfer_bytes", "output_bytes"), "unlock": ("unlock_value",), "loss": ("time_to_loss_seconds",)}
    result = {name: _range(_first(raw, *names), f"{path}/estimates/{name}", errors) for name, names in aliases.items()}
    result["probability"] = _probability(_first(raw, "completion_probability", "probability_before_deadline"), f"{path}/estimates/completion_probability", errors)
    result["later"] = raw.get("later_executable", raw.get("executable_later"))
    if not isinstance(result["later"], bool):
        errors.append(_problem("missing_estimate", f"{path}/estimates/later_executable", "boolean value required"))
        result["later"] = None
    return result, errors


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _component(expected: Any, upper: Any, normalized: float | None, weight: float) -> dict[str, Any]:
    return {"expected": expected, "upper": upper, "normalized": normalized, "weight": weight, "weighted": None if normalized is None else round(normalized * weight, 6)}


def _score(estimates: Mapping[str, Any], demand: Mapping[str, int], ctx: Mapping[str, Any]) -> dict[str, Any]:
    names = ("wall", "queue", "transfer", "probability", "unlock", "loss")
    if estimates.get("later") is None or any(estimates.get(name) is None for name in names):
        return {"total": None, "components": {name: _component(None, None, None, weight) for name, weight in WEIGHTS.items()}}
    if not ctx.get("as_of") or not ctx.get("deadline"):
        return {"total": None, "components": {name: _component(None, None, None, weight) for name, weight in WEIGHTS.items()}}
    horizon = max((ctx["deadline"] - ctx["as_of"]).total_seconds(), 1.0)
    wall, queue, transfer = estimates["wall"], estimates["queue"], estimates["transfer"]
    probability, unlock, loss = estimates["probability"], estimates["unlock"], estimates["loss"]
    caps, storage = ctx["capacities"], ctx["storage"]
    footprint = max((demand[name] / caps[name] for name in LANES if caps.get(name, 0)), default=1.0)
    normalized = {"time_to_loss": _clamp(1 - loss["expected"] / horizon), "expected_wall_time": _clamp(1 - wall["expected"] / horizon), "queue_uncertainty": _clamp((queue["upper"] - queue["expected"]) / max(queue["upper"], 1)), "resource_footprint": _clamp(footprint), "transfer_volume": _clamp(transfer["upper"] / max(storage.get("transfer_capacity_bytes", 1), 1)), "completion_probability": probability["expected"], "unlock_value": _clamp(unlock["expected"] / 100), "executable_later": 0.0 if estimates["later"] else 1.0}
    values = {"time_to_loss": (loss["expected"], loss["upper"]), "expected_wall_time": (wall["expected"], wall["upper"]), "queue_uncertainty": (queue["expected"], queue["upper"]), "resource_footprint": (footprint, None), "transfer_volume": (transfer["expected"], transfer["upper"]), "completion_probability": (probability["expected"], probability["upper"]), "unlock_value": (unlock["expected"], unlock["upper"]), "executable_later": (estimates["later"], None)}
    components = {name: _component(*values[name], normalized[name], weight) for name, weight in WEIGHTS.items()}
    return {"total": round(sum(item["weighted"] for item in components.values()), 6), "components": components}


def _deadline(estimates: Mapping[str, Any], ctx: Mapping[str, Any], errors: list[dict[str, str]], path: str) -> dict[str, Any]:
    result = {"verdict": "unknown", "expected_slack_seconds": None, "conservative_slack_seconds": None}
    if not estimates.get("wall") or not estimates.get("queue") or not ctx.get("as_of") or not ctx.get("deadline"):
        errors.append(_problem("deadline_unknown", path, "wall, queue, and deadline are required"))
        return result
    start = ctx["as_of"].timestamp()
    expected = start + estimates["queue"]["expected"] + estimates["wall"]["expected"]
    conservative = start + estimates["queue"]["upper"] + estimates["wall"]["upper"]
    expected_slack = ctx["deadline"].timestamp() - expected
    conservative_slack = ctx["deadline"].timestamp() - conservative
    result.update({"verdict": "conservative_fit" if conservative_slack >= 0 else "expected_only" if expected_slack >= 0 else "misses", "expected_finish_utc": _render(datetime.fromtimestamp(expected, UTC)), "conservative_finish_utc": _render(datetime.fromtimestamp(conservative, UTC)), "expected_slack_seconds": round(expected_slack, 6), "conservative_slack_seconds": round(conservative_slack, 6)})
    if conservative_slack < 0:
        errors.append(_problem("deadline_infeasible", path, "conservative finish misses deadline"))
    return result


def _row(raw: Any, index: int, ctx: Mapping[str, Any], global_ok: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    path, row = f"/workloads[{index}]", raw if isinstance(raw, Mapping) else {}
    errors = _row_gates(row, path)
    workload_id = _slug(_first(row, "id", "workload_id"))
    if workload_id is None:
        errors.append(_problem("invalid_workload_id", f"{path}/id", "lowercase workload id required"))
    lane, demand, demand_errors = _demand(row, path)
    errors.extend(demand_errors)
    estimates, estimate_errors = _estimates(row, path)
    errors.extend(estimate_errors)
    deadline = _deadline(estimates, ctx, errors, f"{path}/estimates")
    if lane and demand and any(demand[name] > ctx["capacities"].get(name, -1) for name in LANES):
        errors.append(_problem("resource_infeasible", f"{path}/resource_demand", "demand exceeds capacity"))
    transfer = estimates.get("transfer")
    if transfer and any(transfer["upper"] > ctx["storage"].get(name, -1) for name in ("available_bytes", "transfer_capacity_bytes")):
        errors.append(_problem("storage_infeasible", f"{path}/estimates/transfer_bytes", "transfer exceeds storage"))
    if not global_ok:
        errors.append(_problem("global_input_unavailable", "/", "required planning input is unavailable"))
    reasons = sorted({item["code"] for item in errors})
    uncertainty = {name: estimates.get(name) for name in ("wall", "queue", "transfer", "probability")}
    uncertainty["level"] = "high" if any(value and value["upper"] > value["expected"] for value in (estimates.get("wall"), estimates.get("queue"))) else "low"
    report = {"workload_id": workload_id, "lane": lane, "resource_demand": demand or None, "feasibility": {"eligible": not reasons, "blocking_reasons": reasons}, "deadline": deadline, "score": _score(estimates, demand, ctx), "uncertainty": uncertainty, "selection": {"primary": False, "packing_plans": []}}
    return report, {"id": workload_id, "lane": lane, "demand": demand, "estimates": estimates, "report": report}


def _mark(report: dict[str, Any], reason: str) -> None:
    report["feasibility"]["blocking_reasons"] = sorted(set(report["feasibility"]["blocking_reasons"]) | {reason})
    report["feasibility"]["eligible"] = False


def _dependencies_for(internals: Mapping[str, Mapping[str, Any]], reports: Sequence[dict[str, Any]], edges: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:
    by_id = {report["workload_id"]: report for report in reports if report.get("workload_id")}
    deps = {name: [] for name in internals}
    for edge in edges:
        child, parent = edge["child"], edge["parent"]
        if child not in by_id:
            continue
        if parent not in internals:
            _mark(by_id[child], "missing_dependency")
        elif parent not in deps[child]:
            deps[child].append(parent)
    return deps


def _apply_dependencies(internals: Mapping[str, Mapping[str, Any]], reports: Sequence[dict[str, Any]], edges: Sequence[Mapping[str, str]]) -> dict[str, list[str]]:  # noqa: C901 - cycle detection and propagation are separate bounded passes.
    deps = _dependencies_for(internals, reports, edges)
    state: dict[str, int] = {}
    cycles: set[str] = set()

    def visit(node: str, path: list[str]) -> None:
        if state.get(node) == 1:
            cycles.update(path[path.index(node) :])
            return
        if state.get(node) == 2:
            return
        state[node] = 1
        for parent in deps.get(node, ()):
            visit(parent, [*path, parent])
        state[node] = 2

    for node in sorted(deps):
        visit(node, [node])
    for node in cycles:
        _mark(internals[node]["report"], "dependency_cycle")
    for _ in range(len(internals) + 1):
        changed = False
        for node, parents in deps.items():
            if any(not internals[parent]["report"]["feasibility"]["eligible"] for parent in parents):
                before = tuple(internals[node]["report"]["feasibility"]["blocking_reasons"])
                _mark(internals[node]["report"], "dependency_blocked")
                changed |= before != tuple(internals[node]["report"]["feasibility"]["blocking_reasons"])
        if not changed:
            break
    return deps


def _order(items: Mapping[str, Mapping[str, Any]], strategy: str) -> list[str]:
    def key(name: str) -> tuple[Any, ...]:
        report = items[name]["report"]
        score = report["score"]["total"] or 0.0
        slack = report["deadline"].get("conservative_slack_seconds")
        unlock = report["score"]["components"].get("unlock_value", {}).get("normalized") or 0.0
        if strategy == "deadline_first":
            return (slack if slack is not None else float("inf"), -score, name)
        if strategy == "unlock_first":
            return (-unlock, -score, name)
        return (-score, name)

    return sorted(items, key=key)


def _closure(node: str, candidates: Mapping[str, Mapping[str, Any]], deps: Mapping[str, Sequence[str]], selected: set[str]) -> list[str] | None:
    result: list[str] = []
    active: set[str] = set()

    def visit(current: str) -> bool:
        if current in selected or current in result:
            return True
        if current in active or current not in candidates:
            return False
        active.add(current)
        if not all(visit(parent) for parent in deps.get(current, ())):
            return False
        active.remove(current)
        result.append(current)
        return True

    return result if visit(node) else None


def _fits(addition: Sequence[str], candidates: Mapping[str, Mapping[str, Any]], ctx: Mapping[str, Any], state: Mapping[str, Any]) -> str | None:
    counts, demand, storage = dict(state["counts"]), dict(state["demand"]), state["storage"]
    for name in addition:
        item = candidates[name]
        lane = item["lane"]
        counts[lane] += 1
        if counts[lane] > ctx["queues"].get(lane, -1):
            return "queue_limit"
        storage += item["estimates"]["transfer"]["upper"]
        if storage > ctx["storage"].get("available_bytes", -1) or storage > ctx["storage"].get("transfer_capacity_bytes", -1):
            return "storage_overflow"
        for resource in LANES:
            demand[resource] += item["demand"][resource]
            if demand[resource] > ctx["capacities"].get(resource, -1):
                return "resource_conflict"
    return None


def _pack(internals: Mapping[str, Mapping[str, Any]], ctx: Mapping[str, Any], deps: Mapping[str, Sequence[str]], strategy: str) -> dict[str, Any]:
    candidates = {name: item for name, item in internals.items() if item["report"]["feasibility"]["eligible"]}
    state = {"counts": dict.fromkeys(LANES, 0), "demand": dict.fromkeys(LANES, 0), "storage": 0}
    selected: set[str] = set()
    order: list[str] = []
    excluded: list[dict[str, str]] = []
    for name in _order(candidates, strategy):
        if name in selected:
            continue
        addition = _closure(name, candidates, deps, selected)
        reason = "dependency_not_packable" if addition is None else _fits(addition, candidates, ctx, state)
        if reason:
            excluded.append({"workload_id": name, "reason": reason})
            continue
        for item_id in addition:
            item = candidates[item_id]
            selected.add(item_id)
            order.append(item_id)
            state["counts"][item["lane"]] += 1
            state["storage"] += item["estimates"]["transfer"]["upper"]
            for resource in LANES:
                state["demand"][resource] += item["demand"][resource]
    lanes = {lane: {"workload_ids": [name for name in order if candidates[name]["lane"] == lane], "count": state["counts"][lane], "queue_limit": ctx["queues"].get(lane), "resource_used": state["demand"][lane], "resource_capacity": ctx["capacities"].get(lane)} for lane in LANES}
    return {"strategy": strategy, "workload_ids": order, "lanes": lanes, "storage": {"planned_bytes": state["storage"], **ctx["storage"]}, "excluded": sorted(excluded, key=lambda item: (item["workload_id"], item["reason"])), "total_score": round(sum(internals[name]["report"]["score"]["total"] for name in order), 6)}


def _empty_plan(ctx: Mapping[str, Any]) -> dict[str, Any]:
    return {"workload_ids": [], "lanes": {lane: {"workload_ids": [], "count": 0, "queue_limit": ctx["queues"].get(lane), "resource_used": 0, "resource_capacity": ctx["capacities"].get(lane)} for lane in LANES}}


def build_plan(payload: Any, *, explain: bool = False) -> dict[str, Any]:
    """Return a deterministic capacity plan without mutating input or services."""
    if not isinstance(payload, Mapping):
        raise ValueError("inventory must be a mapping")
    ctx, global_errors = _context(payload)
    reports: list[dict[str, Any]] = []
    internals: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, raw in enumerate(ctx["workloads"]):
        report, internal = _row(raw, index, ctx, not global_errors)
        reports.append(report)
        if internal["id"] is not None:
            if internal["id"] in internals:
                duplicate_ids.add(internal["id"])
            else:
                internals[internal["id"]] = internal
    for name in duplicate_ids:
        for report in reports:
            if report["workload_id"] == name:
                _mark(report, "duplicate")
        _mark(internals[name]["report"], "duplicate")
    deps = _apply_dependencies(internals, reports, ctx["edges"])
    plans = [] if global_errors else [_pack(internals, ctx, deps, strategy) for strategy in ("score_order", "deadline_first", "unlock_first")]
    primary = plans[0] if plans else _empty_plan(ctx)
    chosen = set(primary["workload_ids"])
    for report in reports:
        report["selection"] = {"primary": report["workload_id"] in chosen, "packing_plans": [plan["strategy"] for plan in plans if report["workload_id"] in plan["workload_ids"]]}
    status = "incomplete" if global_errors else "ready" if primary["workload_ids"] else "blocked"
    result = {"schema": PLAN_SCHEMA, "claim_boundary": CLAIM_BOUNDARY, "status": status, "ok": status == "ready", "as_of_utc": _render(ctx["as_of"]), "access_deadline_utc": _render(ctx["deadline"]), "feasibility_gates": {"global_input": "pass" if not global_errors else "fail", "deadline": "pass" if ctx["deadline"] and not any(item["code"].startswith("deadline") for item in global_errors) else "fail", "resources": "pass" if len(ctx["capacities"]) == len(LANES) and len(ctx["queues"]) == len(LANES) else "fail", "storage": "pass" if len(ctx["storage"]) == 2 else "fail", "dependency_graph": "pass" if isinstance(payload.get("dependencies"), Mapping) and not any("dependency" in item["code"] for item in global_errors) else "fail"}, "score_policy": {"weights": WEIGHTS, "note": "Operational heuristics only; scores are not evidence quality."}, "issues": sorted(global_errors, key=lambda item: (item["location"], item["code"], item["message"])), "workloads": sorted(reports, key=lambda item: (str(item["workload_id"]), str(item["lane"]))), "ordered_plan": primary["workload_ids"], "lanes": primary["lanes"], "packing_plans": plans}
    if explain:
        result["explain"] = {"method": "Filter hard gates, then greedily pack dependency closures using score, deadline, and unlock orderings.", "excluded": [{"workload_id": item["workload_id"], "reasons": item["feasibility"]["blocking_reasons"]} for item in result["workloads"] if not item["feasibility"]["eligible"]]}
    return result


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("inventory must be a mapping")
    return value


def _text(result: Mapping[str, Any]) -> str:
    codes = ",".join(item["code"] for item in result["issues"]) or "-"
    return f"status={result['status']} eligible={sum(item['feasibility']['eligible'] for item in result['workloads'])} selected={len(result['ordered_plan'])}\nordered_plan={','.join(result['ordered_plan'])}\nissues={codes}\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the planner CLI; check mode never writes an output file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="print without writing")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = build_plan(_load(args.inventory), explain=args.explain)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        result = {"schema": PLAN_SCHEMA, "claim_boundary": CLAIM_BOUNDARY, "status": "incomplete", "ok": False, "issues": [_problem("unreadable_input", "/inventory", str(exc))], "workloads": [], "ordered_plan": [], "lanes": {lane: {"workload_ids": []} for lane in LANES}, "packing_plans": []}
    rendered = _text(result) if args.format == "text" and not args.json else json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.check:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sys.stdout.write(f"wrote {args.output}\n")
    return 0 if result["ok"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
