"""Acceptance logic of the frozen trace-example selection protocol (exsel.protocol.v1).

Every rule here is the protocol's own text applied to one serialized episode row:

* outcome S: metrics.success is true; C: not S and outcome.collision_event is true;
  T: not S, not C and outcome.timeout_event is true; otherwise "?" (no class matches).
* collision_kind pedestrian: C and metrics.ped_collision_count > 0; obstacle: C, no
  pedestrian collision, and metrics.obstacle_collision_count > 0 or
  metrics.wall_collisions > 0 (release rows).
* min_exposure_steps k: metrics.near_misses >= k (never matches a row without it).
* min_surface_clearance_m: safety_predicates.occlusion_near_miss_predicate.fields.
  actual_minimum_separation_m minus the 1.4 m envelope radius sum; absent when the
  value is missing or is the no-pedestrian sentinel (>= 1e6).
* exposure_steps: metrics.near_misses.
* time_to_goal_s: round(steps * scenario_params.run_dt, 6), successes only.
* acceptance (typical claim): every arm matches its class and every banded measure
  with a frozen band lies inside it, inclusive; an absent value is outside. A band of
  null is not checked.
* screening order: the band sorted by the hex SHA-256 of "<selection_rng_seed>:<seed>",
  first ``budget`` seeds.

The projection mirrors ``episode_record``/``claim_record_from_release``/``class_matches``/
``acceptance_count`` of ll7/diss ``scripts/dev/diss_worked_example_nomination.py`` at the
protocol commit named in each contract.yaml.txt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml

ENVELOPE_RADIUS_SUM_M = 1.4
NO_PEDESTRIAN_SENTINEL = 1e6
MEASURES = ("min_surface_clearance_m", "exposure_steps", "time_to_goal_s")
RETIRED_BAND = (111, 140)
DEV_BAND = (1001, 1200)


def _get(d: Any, *path: str) -> Any:
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def load_claim(contract: dict[str, Any]) -> dict[str, Any]:
    """Parse the verbatim protocol claim block stored in a contract.

    Returns:
        The claim mapping.
    """
    parsed = yaml.safe_load(contract["protocol_claim_verbatim"])
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError("protocol_claim_verbatim must hold exactly one claim list item")
    claim = parsed[0]
    if claim["id"] != contract["claim_id"]:
        raise ValueError("contract claim_id differs from the verbatim claim block")
    return claim


def screening_order(band: list[int], rng_seed: str, budget: int) -> list[int]:
    """The committed order: band sorted by SHA-256 hex of '<rng_seed>:<seed>'.

    Returns:
        The first ``budget`` seeds in that order.
    """
    lo, hi = int(band[0]), int(band[1])
    for seed in range(lo, hi + 1):
        if RETIRED_BAND[0] <= seed <= RETIRED_BAND[1] or not DEV_BAND[0] <= seed <= DEV_BAND[1]:
            raise ValueError(f"seed {seed} is not a plain development seed")
    ordered = sorted(
        range(lo, hi + 1), key=lambda s: hashlib.sha256(f"{rng_seed}:{s}".encode()).hexdigest()
    )
    return ordered[:budget]


def project(raw: dict[str, Any]) -> dict[str, Any]:
    """Project one serialized episode row onto the protocol fields.

    Returns:
        Flat record with raw fields, derived class fields and measures.
    """
    m = raw.get("metrics") or {}
    out = raw.get("outcome") or {}
    scp = raw.get("scenario_params") or {}
    sep = _get(
        raw,
        "safety_predicates",
        "occlusion_near_miss_predicate",
        "fields",
        "actual_minimum_separation_m",
    )
    if not isinstance(sep, (int, float)) or sep >= NO_PEDESTRIAN_SENTINEL:
        sep = None
    success = bool(m.get("success"))
    collision = bool(out.get("collision_event"))
    timeout = bool(out.get("timeout_event"))
    outcome = "S" if success else "C" if collision else "T" if timeout else "?"
    ped = m.get("ped_collision_count") or 0
    obst = m.get("obstacle_collision_count") or 0
    wall = m.get("wall_collisions") or 0
    kind = None
    if outcome == "C":
        if ped:
            kind = "pedestrian"
        elif obst or wall:
            kind = "obstacle"
    steps = raw.get("steps")
    dt = scp.get("run_dt")
    ttg = None
    if outcome == "S" and isinstance(steps, (int, float)) and isinstance(dt, (int, float)):
        ttg = round(steps * dt, 6)
    return {
        "scenario_id": raw.get("scenario_id"),
        "seed": raw.get("seed"),
        "git_hash": raw.get("git_hash"),
        "config_hash": raw.get("config_hash"),
        "success": m.get("success"),
        "collision_event": out.get("collision_event"),
        "timeout_event": out.get("timeout_event"),
        "termination_reason": raw.get("termination_reason"),
        "steps": steps,
        "run_dt": dt,
        "ped_collision_count": m.get("ped_collision_count"),
        "obstacle_collision_count": m.get("obstacle_collision_count"),
        "wall_collisions": m.get("wall_collisions"),
        "near_misses": m.get("near_misses"),
        "min_ped_separation_m": sep,
        "outcome": outcome,
        "collision_kind": kind,
        "min_surface_clearance_m": None if sep is None else sep - ENVELOPE_RADIUS_SUM_M,
        "exposure_steps": m.get("near_misses"),
        "time_to_goal_s": ttg,
    }


def mode_of(raw: dict[str, Any]) -> str:
    """Execution mode from the row's own markers: native, fallback or degraded.

    ``fallback`` when the planner status or pedestrian model reports a fallback, or the
    planner metadata records a fallback count above zero; ``degraded`` when the effective
    view is degraded or the pedestrian model is not native; ``native`` otherwise.

    Returns:
        One of "native", "fallback", "degraded".
    """
    meta = raw.get("algorithm_metadata") or {}
    status = str(meta.get("status", "")).lower()
    ped_status = str(_get(raw, "pedestrian_model", "fallback_degraded_status") or "").lower()
    counters = [
        v
        for k, v in _flat(meta)
        if "fallback" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if "fallback" in status or ped_status == "fallback" or any(c > 0 for c in counters):
        return "fallback"
    if _get(raw, "integrity", "effective_view", "degraded") or ped_status not in ("native", ""):
        return "degraded"
    return "native"


def mode_evidence(raw: dict[str, Any]) -> dict[str, Any]:
    """The raw markers behind ``mode_of``.

    Returns:
        Marker values as recorded (None when absent).
    """
    meta = raw.get("algorithm_metadata") or {}
    counters = sorted(
        (k, v)
        for k, v in _flat(meta)
        if "fallback" in k.lower() and isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    return {
        "algorithm_status": meta.get("status"),
        "pedestrian_model_status": _get(raw, "pedestrian_model", "fallback_degraded_status"),
        "effective_view_degraded": _get(raw, "integrity", "effective_view", "degraded"),
        "planner_execution_mode": _get(meta, "planner_kinematics", "execution_mode"),
        "fallback_counters": ";".join(f"{k}={v}" for k, v in counters) or "none_recorded",
    }


def _flat(d: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        for k in sorted(d):
            out.extend(_flat(d[k], f"{prefix}.{k}" if prefix else str(k)))
    else:
        out.append((prefix, d))
    return out


def class_verdicts(rec: dict[str, Any], cls: dict[str, Any]) -> dict[str, bool]:
    """Verdict per class predicate key present in the frozen class.

    Returns:
        Mapping predicate name to pass/fail.
    """
    allowed = {"outcome", "collision_kind", "min_exposure_steps"}
    unknown = set(cls) - allowed
    if unknown:
        raise ValueError(f"unknown class keys {sorted(unknown)}")
    v = {"outcome": rec["outcome"] == cls["outcome"]}
    if "collision_kind" in cls:
        v["collision_kind"] = rec["collision_kind"] == cls["collision_kind"]
    if "min_exposure_steps" in cls:
        e = rec["exposure_steps"]
        v["min_exposure_steps"] = e is not None and e >= cls["min_exposure_steps"]
    return v


def band_verdicts(rec: dict[str, Any], bands: dict[str, Any]) -> dict[str, bool | None]:
    """Verdict per frozen band (inclusive); None when the band is null (not checked).

    Returns:
        Mapping measure to True/False/None.
    """
    out: dict[str, bool | None] = {}
    for measure in MEASURES:
        band = bands.get(measure)
        if band is None:
            out[measure] = None
            continue
        value = rec[measure]
        out[measure] = value is not None and float(band[0]) <= value <= float(band[1])
    return out


def seed_accepted(per_arm: dict[str, dict[str, Any]], claim: dict[str, Any]) -> bool:
    """rules.acceptance: every arm matches its class; typical claims also every band.

    Returns:
        True when the seed is accepted.
    """
    typical = claim["kind"] == "typical"
    for arm in claim["arms"]:
        v = per_arm[arm["arm"]]
        if not all(v["class"].values()):
            return False
        if typical and any(b is False for b in v["bands"].values()):
            return False
    return True


def evaluate(rows: dict[str, dict[str, Any]], claim: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one screened seed from its raw rows keyed by arm.

    Returns:
        Per-arm records with verdicts and the accepted flag.
    """
    per_arm = {}
    for arm in claim["arms"]:
        raw = rows[arm["arm"]]
        rec = project(raw)
        per_arm[arm["arm"]] = {
            "record": rec,
            "class": class_verdicts(rec, arm["class"]),
            "bands": band_verdicts(rec, claim["frozen"]["bands"][arm["arm"]]),
            "mode": mode_of(raw),
            "mode_evidence": mode_evidence(raw),
        }
    return {"per_arm": per_arm, "accepted": seed_accepted(per_arm, claim)}


def row_sha256(line: str) -> str:
    """SHA-256 of one serialized episode row (the JSONL line without its newline).

    Returns:
        Hex digest.
    """
    return hashlib.sha256(line.rstrip("\n").encode("utf-8")).hexdigest()


def read_rows(path: str) -> list[tuple[str, dict[str, Any]]]:
    """Read (line, parsed row) pairs from an episodes.jsonl.

    Returns:
        List of pairs in file order.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append((line.rstrip("\n"), json.loads(line)))
    return out
