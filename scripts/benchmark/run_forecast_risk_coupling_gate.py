#!/usr/bin/env python3
"""Forecast-risk closed-loop coupling gate (issue #2916, evidence_tier: stress).

Tests whether forecast-DERIVED risk improves navigation OUTCOMES rather than only
prediction metrics.  Four rows -- ``no_forecast``, ``cv_risk``, ``semantic_risk``,
``interaction_risk`` -- run against the SAME fixture frames, SAME seed, and SAME
scenario.  Only the forecast risk SOURCE feeding a bounded deterministic risk gate
differs across rows.

This follows the bounded fixture-driven proxy pattern of
``scripts/benchmark/run_observation_noise_envelope.py``.  It does NOT stand up the
heavy ``robot_sf.benchmark.runner`` simulator and does NOT promote any learned
predictor.  All risk sources are deterministic CV-family baselines.

Closed-loop proxy: the robot follows the fixture trajectory, but the risk gate may
hold the robot back (STOP / YIELD) when per-step forecast risk is high.  Metrics
are computed on the gate-modified trajectory against the ground-truth pedestrian
trajectory from the fixture.

Fail-closed: any row whose forecast risk signal is unavailable (degraded,
fallback, oracle, or empty) for the conflict window is marked ``blocked`` and
never counts as a success row.

Usage::

    uv run python scripts/benchmark/run_forecast_risk_coupling_gate.py
    uv run python scripts/benchmark/run_forecast_risk_coupling_gate.py --help
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.forecast_risk_adapter import compute_forecast_risk
from robot_sf.benchmark.metrics import snqi
from robot_sf.benchmark.pedestrian_forecast import (
    NeighborContext,
    PedestrianState,
    constant_velocity_gaussian_baseline,
    interaction_aware_cv_baseline,
    semantic_cv_baseline,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/research/forecast_risk_coupling_issue_2916.yaml"
DEFAULT_OUTPUT_DIR = "output/issue_2916_coupling_gate"

# Near-miss / collision thresholds (meters) for the robot-pedestrian pair.
COLLISION_DISTANCE_M = 0.3
NEAR_MISS_DISTANCE_M = 0.7

# Forecast horizons used to build per-step batches (short, bounded).
FORECAST_HORIZONS_S = (0.5, 1.0)

# Minimal feature schema required by ForecastBatch.v1 provenance.
_FEATURE_SCHEMA = {"position": "xy_m", "velocity": "xy_m_s"}
_EXPECTED_ROW_RISK_SOURCES = {
    "no_forecast": "none",
    "cv_risk": "constant_velocity",
    "semantic_risk": "semantic_cv",
    "interaction_risk": "interaction_aware_cv",
}


def _git_head() -> str:
    """Return the short git HEAD, or empty string on failure.

    Returns:
        Short commit SHA or an empty string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _load_fixture(path: pathlib.Path) -> dict[str, Any]:
    """Load the trace fixture JSON.

    Returns:
        Parsed fixture mapping.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _finite_xy(value: Any) -> np.ndarray | None:
    """Return a finite xy vector, or None when the payload is malformed."""
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _repo_relative_or_absolute(path: pathlib.Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute/path string."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ped_state_from_frame(frame: dict[str, Any]) -> PedestrianState | None:
    """Build a PedestrianState from the first observed pedestrian in a frame.

    Uses the deployable ``observed_pedestrians`` tier so the risk source reflects
    what a deployable consumer could see, not oracle ground truth.

    Returns:
        A pedestrian state, or None when no pedestrian is observable in the frame.
    """
    observed = frame.get("observed_pedestrians") or []
    if not observed:
        return None
    payload = dict(observed[0])
    if _finite_xy(payload.get("position")) is None:
        return None
    velocity = payload.get("velocity")
    if velocity is not None and _finite_xy(velocity) is None:
        return None
    payload.setdefault("id", 0)
    return PedestrianState.from_trace(payload)


def _build_forecast_batch(
    risk_source: str,
    ped_state: PedestrianState,
    *,
    scenario_id: str,
    seed: int,
    dt_s: float,
):
    """Build a deterministic ForecastBatch for one risk source from a ped state.

    Returns:
        A validated ForecastBatch carrying a deterministic forecast for the actor.
    """
    from robot_sf.benchmark.forecast_batch import (
        ActorForecast,
        CoordinateFrame,
        ForecastBatch,
        ForecastBatchProvenance,
    )

    if risk_source == "constant_velocity":
        forecast = constant_velocity_gaussian_baseline(ped_state, FORECAST_HORIZONS_S)
        family = "constant_velocity"
        predictor_id = "cv-gaussian-issue2916"
    elif risk_source == "semantic_cv":
        forecast = semantic_cv_baseline(ped_state, FORECAST_HORIZONS_S)
        family = "semantic_cv"
        predictor_id = "semantic-cv-issue2916"
    elif risk_source == "interaction_aware_cv":
        # Single observed pedestrian: no neighbor context available, so this
        # degrades to plain CV mean. Deterministic and labeled.
        neighbors: list[NeighborContext] = []
        forecast = interaction_aware_cv_baseline(
            ped_state, FORECAST_HORIZONS_S, neighbors=neighbors
        )
        family = "interaction_aware_cv"
        predictor_id = "interaction-cv-issue2916"
    else:
        raise ValueError(f"unknown risk_source: {risk_source}")

    actor_id = str(ped_state.id)
    provenance = ForecastBatchProvenance(
        predictor_id=predictor_id,
        predictor_family=family,
        observation_tier="tracked_agents",
        frame=CoordinateFrame(name="world", units="m", axes=("x", "y")),
        dt_s=dt_s,
        horizons_s=list(FORECAST_HORIZONS_S),
        scenario_id=scenario_id,
        seed=seed,
        fallback_status="native",
        degraded_status="none",
        actor_ids=[actor_id],
        actor_mask=[True],
        actor_mask_metadata={"semantics": "true means available in tracked tier"},
        feature_schema=dict(_FEATURE_SCHEMA),
        timestamp="1970-01-01T00:00:00+00:00",
        oracle_state=False,
    )
    deterministic = np.asarray(
        [prediction.mean for prediction in forecast.predictions], dtype=float
    )
    batch = ForecastBatch(
        provenance=provenance,
        forecasts=[ActorForecast(actor_id=actor_id, deterministic=deterministic)],
        metadata={"artifact_role": "issue_2916_coupling_gate"},
    )
    return batch


def _gate_decision(
    risk: float,
    *,
    stop_threshold: float,
    yield_threshold: float,
) -> str:
    """Map a scalar risk to a gate action.

    Returns:
        ``"stop"``, ``"yield"``, or ``"go"``.
    """
    if risk >= stop_threshold:
        return "stop"
    if risk >= yield_threshold:
        return "yield"
    return "go"


@dataclass
class _RowState:
    """Mutable accumulator for one row's closed-loop replay."""

    robot_pos: np.ndarray
    min_distance_m: float = float("inf")
    collision: bool = False
    near_miss: bool = False
    progress_m: float = 0.0
    false_positive_stops: int = 0
    true_positive_stops: int = 0
    first_stop_step: int | None = None
    first_yield_step: int | None = None
    gate_actions: list[str] = field(default_factory=list)
    risk_available_steps: int = 0
    risk_unavailable_in_conflict: int = 0
    conflict_steps: int = 0
    risk_unavailable_this_step: bool = False


def _gate_action_for_step(
    frame: dict[str, Any],
    robot_pos: np.ndarray,
    row: dict[str, Any],
    cfg: dict[str, Any],
    state: _RowState,
) -> str:
    """Compute the gate action for one step from the row's forecast risk source.

    Increments ``state.risk_available_steps`` as a side effect when a usable
    signal is produced.

    Returns:
        ``"stop"``, ``"yield"``, or ``"go"``.
    """
    risk_source = row["risk_source"]
    state.risk_unavailable_this_step = False
    if risk_source == "none":
        return "go"
    ped_state = _ped_state_from_frame(frame)
    if ped_state is None:
        state.risk_unavailable_this_step = True
        return "go"
    gate_cfg = cfg["risk_gate"]
    batch = _build_forecast_batch(
        risk_source,
        ped_state,
        scenario_id=cfg["fixture"]["scenario_id"],
        seed=int(cfg["fixture"]["seed"]),
        dt_s=float(cfg["fixture"]["dt_s"]),
    )
    signal = compute_forecast_risk(
        batch, robot_pos, influence_radius_m=float(gate_cfg["influence_radius_m"])
    )
    if not signal.available:
        state.risk_unavailable_this_step = True
        return "go"
    state.risk_available_steps += 1
    return _gate_decision(
        signal.risk,
        stop_threshold=float(gate_cfg["stop_risk_threshold"]),
        yield_threshold=float(gate_cfg["yield_risk_threshold"]),
    )


def _apply_gate_step(action: str, intended: np.ndarray, index: int, state: _RowState) -> np.ndarray:
    """Apply the gate action to the intended forward step.

    Returns:
        The realized displacement vector after the gate hold.
    """
    if action == "stop":
        if state.first_stop_step is None:
            state.first_stop_step = index
        return np.zeros(2, dtype=float)
    if action == "yield":
        if state.first_yield_step is None:
            state.first_yield_step = index
        return intended * 0.5
    return intended


def _score_outcome_step(
    frame: dict[str, Any],
    action: str,
    risk_source: str,
    conflict_distance_m: float,
    state: _RowState,
) -> None:
    """Score one step against the ground-truth pedestrian (outcome only)."""
    peds = frame.get("pedestrians") or []
    if not peds:
        return
    ped_pos = _finite_xy(peds[0].get("position"))
    if ped_pos is None:
        raise ValueError("ground-truth pedestrian position must be a finite xy vector")
    distance = float(np.linalg.norm(state.robot_pos - ped_pos))
    state.min_distance_m = min(state.min_distance_m, distance)
    if distance <= COLLISION_DISTANCE_M:
        state.collision = True
    elif distance <= NEAR_MISS_DISTANCE_M:
        state.near_miss = True
    in_conflict = distance <= conflict_distance_m
    if in_conflict:
        state.conflict_steps += 1
    if action == "stop":
        if in_conflict:
            state.true_positive_stops += 1
        else:
            state.false_positive_stops += 1
    if in_conflict and risk_source != "none" and state.risk_unavailable_this_step:
        state.risk_unavailable_in_conflict += 1


def _row_snqi(state: _RowState) -> float:
    """Compute the bounded SNQI proxy for a row's outcome.

    Returns:
        The SNQI scalar (higher is better).
    """
    success = 1.0 if (state.progress_m > 0.3 and not state.collision) else 0.0
    time_to_goal_norm = float(np.clip(1.0 - state.progress_m / 1.5, 0.0, 1.0))
    return float(
        snqi(
            {
                "success": success,
                "time_to_goal_norm": time_to_goal_norm,
                "collisions": float(int(state.collision)),
                "near_misses": float(int(state.near_miss)),
            },
            weights={"w_success": 1.0, "w_time": 0.5, "w_collisions": 1.0, "w_near": 0.5},
            baseline_stats={
                "collisions": {"med": 0.0, "p95": 1.0},
                "near_misses": {"med": 0.0, "p95": 1.0},
            },
        )
    )


def evaluate_row(
    row: dict[str, Any],
    frames: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one row's closed-loop outcome against the shared fixture frames.

    The robot replays the fixture trajectory, but the gate may hold it back
    (yield = half forward step, stop = no forward step) when forecast risk is
    high.  Metrics are computed on the gate-modified trajectory.

    Returns:
        A compact per-row result with metrics, fail-closed classification, and
        the SNQI proxy.
    """
    start = time.perf_counter()
    risk_source = row["risk_source"]
    conflict_distance_m = float(cfg["risk_gate"]["conflict_distance_m"])
    state = _RowState(robot_pos=np.asarray(frames[0]["robot"]["position"], dtype=float).copy())

    for index in range(1, len(frames)):
        frame = frames[index]
        intended = np.asarray(frame["robot"]["position"], dtype=float) - np.asarray(
            frames[index - 1]["robot"]["position"], dtype=float
        )
        action = _gate_action_for_step(frame, state.robot_pos, row, cfg, state)
        state.gate_actions.append(action)
        step_vec = _apply_gate_step(action, intended, index, state)
        state.robot_pos = state.robot_pos + step_vec
        state.progress_m += float(np.linalg.norm(step_vec))
        _score_outcome_step(frame, action, risk_source, conflict_distance_m, state)

    runtime_s = time.perf_counter() - start
    safety_events = int(state.collision) + int(state.near_miss)
    classification, reason = _classify_row(
        risk_source=risk_source,
        conflict_steps=state.conflict_steps,
        risk_available_steps=state.risk_available_steps,
        risk_unavailable_in_conflict=state.risk_unavailable_in_conflict,
    )

    return {
        "row": row["name"],
        "risk_source": risk_source,
        "description": row.get("description", ""),
        "classification": classification,
        "classification_reason": reason,
        "metrics": {
            "collision": state.collision,
            "near_miss": state.near_miss,
            "safety_events": safety_events,
            "min_distance_m": round(state.min_distance_m, 4)
            if np.isfinite(state.min_distance_m)
            else None,
            "progress_m": round(state.progress_m, 4),
            "first_stop_step": state.first_stop_step,
            "first_yield_step": state.first_yield_step,
            "stop_yield_timing_steps": state.first_stop_step
            if state.first_stop_step is not None
            else state.first_yield_step,
            "false_positive_stops": state.false_positive_stops,
            "true_positive_stops": state.true_positive_stops,
            "runtime_s": round(runtime_s, 6),
            "snqi": round(_row_snqi(state), 6),
        },
        "diagnostics": {
            "conflict_steps": state.conflict_steps,
            "risk_available_steps": state.risk_available_steps,
            "risk_unavailable_in_conflict": state.risk_unavailable_in_conflict,
            "gate_actions": state.gate_actions,
        },
    }


def _classify_row(
    *,
    risk_source: str,
    conflict_steps: int,
    risk_available_steps: int,
    risk_unavailable_in_conflict: int,
) -> tuple[str, str]:
    """Classify a row under the fail-closed policy.

    Returns:
        ``(classification, reason)`` where classification is ``"ok"`` or
        ``"blocked"``.
    """
    if risk_source == "none":
        return "ok", "control row; no forecast risk required"
    if risk_available_steps == 0:
        return (
            "blocked",
            "forecast risk signal never available (fail-closed); row cannot test coupling",
        )
    if conflict_steps > 0 and risk_unavailable_in_conflict >= conflict_steps:
        return (
            "blocked",
            "forecast risk unavailable across the entire conflict window (fail-closed)",
        )
    return "ok", "forecast risk available in the conflict window"


def emit_verdict(
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Emit a continue | revise | stop verdict from the per-row results.

    Makes the false-positive-stopping vs safety-benefit trade-off explicit.

    Returns:
        A verdict mapping with the decision, rationale, and the trade-off table.
    """
    verdict_cfg = cfg["verdict"]
    min_reduction = int(verdict_cfg["min_safety_event_reduction"])
    max_fp_ratio = float(verdict_cfg["max_false_positive_stop_ratio"])
    max_runtime = float(verdict_cfg["max_runtime_s"])

    by_row = {r["row"]: r for r in results}
    control = by_row.get("no_forecast")
    if control is None:
        return {
            "decision": "stop",
            "rationale": "no_forecast control row missing; cannot establish a baseline",
            "tradeoff": [],
        }

    control_events = control["metrics"]["safety_events"]
    control_fp = max(control["metrics"]["false_positive_stops"], 0)

    tradeoff: list[dict[str, Any]] = []
    any_safety_benefit = False
    any_regression = False
    blocked_rows: list[str] = []

    for name in ("cv_risk", "semantic_risk", "interaction_risk"):
        row = by_row.get(name)
        if row is None:
            continue
        if row["classification"] == "blocked":
            blocked_rows.append(name)
            tradeoff.append(
                {
                    "row": name,
                    "blocked": True,
                    "reason": row["classification_reason"],
                }
            )
            continue
        events = row["metrics"]["safety_events"]
        fp = row["metrics"]["false_positive_stops"]
        runtime = row["metrics"]["runtime_s"]
        safety_reduction = control_events - events
        is_safety_benefit = safety_reduction >= min_reduction
        # False-positive regression relative to control (guard divide-by-zero).
        fp_ratio = (fp / control_fp) if control_fp > 0 else (float("inf") if fp > 0 else 1.0)
        fp_regression = fp_ratio > max_fp_ratio
        runtime_regression = runtime > max_runtime
        outcome_regression = events > control_events
        if is_safety_benefit and not fp_regression and not runtime_regression:
            any_safety_benefit = True
        if outcome_regression or fp_regression or runtime_regression:
            any_regression = True
        tradeoff.append(
            {
                "row": name,
                "blocked": False,
                "safety_events": events,
                "safety_event_reduction_vs_control": safety_reduction,
                "false_positive_stops": fp,
                "false_positive_stop_ratio_vs_control": (
                    None if fp_ratio == float("inf") else round(fp_ratio, 4)
                ),
                "runtime_s": runtime,
                "is_safety_benefit": is_safety_benefit,
                "fp_regression": fp_regression,
                "runtime_regression": runtime_regression,
                "outcome_regression": outcome_regression,
            }
        )

    if any_safety_benefit and not any_regression:
        decision = "continue"
        rationale = (
            "At least one forecast-risk row reduced safety events vs the control "
            "without a false-positive-stop or runtime regression. Forecast-risk "
            "coupling shows a navigation-outcome benefit worth pursuing."
        )
    elif any_regression and not any_safety_benefit:
        decision = "stop"
        rationale = (
            "Forecast-risk rows showed regressions (worse safety events, "
            "false-positive-stop explosion, or runtime) with no offsetting safety "
            "benefit. Forecast-risk coupling is not justified on this evidence."
        )
    else:
        decision = "revise"
        rationale = (
            "No clear navigation-outcome benefit and no clear regression. "
            "Forecast-risk coupling is inconclusive on this bounded fixture; revise "
            "the gate / scenario before investing further."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "control_safety_events": control_events,
        "control_false_positive_stops": control_fp,
        "blocked_rows": blocked_rows,
        "tradeoff": tradeoff,
    }


def build_report(
    results: list[dict[str, Any]],
    verdict: dict[str, Any],
    cfg: dict[str, Any],
    repro: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full JSON report.

    Returns:
        The complete report mapping.
    """
    fixture = cfg["fixture"]
    rows = [
        {
            **row,
            "seed": int(fixture["seed"]),
            "scenario_id": fixture["scenario_id"],
        }
        for row in results
    ]
    return {
        "schema_version": "forecast_risk_coupling_gate.v1",
        "issue": 2916,
        "evidence_tier": "stress",
        "claim_boundary": "diagnostic_only",
        "paper_grade": False,
        "claim_boundary_text": (
            "Diagnostic-only forecast-risk closed-loop coupling gate on a single "
            "deterministic occluded-emergence fixture. Not paper-facing benchmark "
            "evidence. Tests whether forecast-derived risk changes navigation "
            "outcomes, not whether any predictor is accurate. No learned predictor "
            "is promoted."
        ),
        "reproducibility": repro,
        "config": {
            "config_path": repro["config_path"],
            "fixture": fixture,
            "risk_gate": cfg["risk_gate"],
            "verdict_thresholds": cfg["verdict"],
        },
        "rows": rows,
        "verdict": verdict,
        "caveats": [
            "Single deterministic fixture (seed=111), single occluded-emergence scenario.",
            "Closed-loop proxy: robot replays the fixture trajectory under a gate hold; "
            "no live planner or simulator is re-executed.",
            "interaction_risk degrades to plain CV mean (single observed pedestrian, no "
            "neighbor context); it is reported honestly, not as interaction evidence.",
            "Ground-truth pedestrian distance is used only for OUTCOME scoring, never as a "
            "risk source. Risk sources use the deployable tracked-agents observation tier.",
            "Not statistically powered; diagnostic_only.",
        ],
    }


def generate_markdown(report: dict[str, Any]) -> str:
    """Render the report as Markdown.

    Returns:
        Markdown document text.
    """
    repro = report["reproducibility"]
    verdict = report["verdict"]
    lines: list[str] = [
        "# Forecast-Risk Closed-Loop Coupling Gate (issue #2916)",
        "",
        "## Claim boundary",
        "",
        f"- evidence_tier: `{report['evidence_tier']}`",
        f"- claim_boundary: `{report['claim_boundary']}`",
        f"- paper_grade: `{report['paper_grade']}`",
        "",
        report["claim_boundary_text"],
        "",
        "## Verdict",
        "",
        f"**Decision: `{verdict['decision']}`**",
        "",
        verdict["rationale"],
        "",
        "## Reproducibility",
        "",
        f"- Issue: #{report['issue']}",
        f"- Generated (UTC): {repro['generated_at_utc']}",
        f"- Command: `{repro['command']}`",
        f"- Repo HEAD: `{repro['repo_head']}`",
        f"- Config: `{repro['config_path']}`",
        f"- Fixture: `{report['config']['fixture']['trace_path']}`",
        f"- Seed: {report['config']['fixture']['seed']}",
        "",
        "## Per-row outcomes",
        "",
        "| row | class | collision | near_miss | safety_events | progress_m | "
        "stop_step | FP_stops | runtime_s | SNQI |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        metrics = row["metrics"]
        lines.append(
            f"| {row['row']} | {row['classification']} | {metrics['collision']} | "
            f"{metrics['near_miss']} | {metrics['safety_events']} | "
            f"{metrics['progress_m']} | {metrics['stop_yield_timing_steps']} | "
            f"{metrics['false_positive_stops']} | {metrics['runtime_s']} | "
            f"{metrics['snqi']} |"
        )
    lines.extend(["", "## Safety-benefit vs false-positive-stopping trade-off", ""])
    for entry in verdict["tradeoff"]:
        if entry.get("blocked"):
            lines.append(f"- **{entry['row']}**: BLOCKED -- {entry['reason']}")
        else:
            lines.append(
                f"- **{entry['row']}**: safety_event_reduction="
                f"{entry['safety_event_reduction_vs_control']}, "
                f"FP_stops={entry['false_positive_stops']}, "
                f"benefit={entry['is_safety_benefit']}, "
                f"regression={entry['outcome_regression'] or entry['fp_regression'] or entry['runtime_regression']}"
            )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.append("")
    return "\n".join(lines)


def _verify_seed_identity(cfg: dict[str, Any]) -> None:
    """Assert all rows share the same seed/scenario denominator.

    Raises:
        ValueError: when the config does not pin a single shared seed/scenario.
    """
    fixture = cfg.get("fixture", {})
    if "seed" not in fixture or "scenario_id" not in fixture:
        raise ValueError("config fixture must pin a single shared seed and scenario_id")
    rows = cfg.get("rows")
    if not rows:
        raise ValueError("config must declare rows")
    observed: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("config rows must be mappings")
        name = str(row.get("name", "")).strip()
        risk_source = str(row.get("risk_source", "")).strip()
        if name in observed:
            raise ValueError(f"duplicate config row: {name}")
        observed[name] = risk_source
    if observed != _EXPECTED_ROW_RISK_SOURCES:
        raise ValueError(
            "config must declare exactly the four forecast-risk rows with expected risk_source "
            f"mapping: {_EXPECTED_ROW_RISK_SOURCES}"
        )


def run(config_path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    """Execute all rows and write the evidence bundle.

    Returns:
        The full report mapping.
    """
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _verify_seed_identity(cfg)

    fixture_path = REPO_ROOT / cfg["fixture"]["trace_path"]
    trace = _load_fixture(fixture_path)
    frames = trace["frames"]

    results = [evaluate_row(row, frames, cfg) for row in cfg["rows"]]
    verdict = emit_verdict(results, cfg)

    repro = {
        "issue": 2916,
        "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": (
            "uv run python scripts/benchmark/run_forecast_risk_coupling_gate.py "
            f"--config {_repo_relative_or_absolute(config_path)} "
            f"--output-dir {_repo_relative_or_absolute(output_dir)}"
        ),
        "repo_head": _git_head(),
        "config_path": _repo_relative_or_absolute(config_path),
    }

    report = build_report(results, verdict, cfg, repro)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "forecast_risk_coupling_gate_report.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    md_path = output_dir / "README.md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(generate_markdown(report))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Verdict: {verdict['decision']}")
    for row in results:
        metrics = row["metrics"]
        print(
            f"  [{row['classification']:>7s}] {row['row']:>16s}: "
            f"collision={metrics['collision']} near_miss={metrics['near_miss']} "
            f"progress_m={metrics['progress_m']} FP_stops={metrics['false_positive_stops']} "
            f"snqi={metrics['snqi']}"
        )
    return report


def main() -> None:
    """Parse arguments and run the coupling gate."""
    parser = argparse.ArgumentParser(
        description=(
            "Forecast-risk closed-loop coupling gate (issue #2916, stress tier). "
            "Runs 4 same-seed rows and emits a continue|revise|stop verdict."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the coupling-gate config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON + Markdown evidence bundle.",
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    run(config_path, output_dir)


if __name__ == "__main__":
    main()
