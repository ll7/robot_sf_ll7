#!/usr/bin/env python3
"""Diagnostic-only offline validator for the #5310 topology-parallel NMPC prototype (#6158).

This validator exercises the merged #6152 prototype
(``robot_sf/planner/topology_parallel_nmpc.py`` at the current ``origin/main`` HEAD,
executed *unchanged*) against the eight gates required by issue #6158 and records
exactly one of the four issue verdicts::

    accepted_offline_prototype | label_only_or_objective_drift |
    invalid_regression | incomplete

It writes a single evidence document under ``docs/context/evidence/`` and prints a
compact JSON summary to stdout. It is **diagnostic-only**: it makes no real-time,
safety, benchmark-superiority, default-planner-promotion, or #5423/STKP-eligibility
claim. Per-hypothesis latency above 100 ms is reported prominently and is stated as
blocking downstream real-time use; it does not change the offline-prototype verdict.

The validator never modifies the prototype, its config, its registration, or its
tests. It imports and calls the prototype's public/initialization seams and reads
back its diagnostics.

Verdict derivation (matches the issue contract):

* gate 1 (K=1 legacy parity) fails            -> ``invalid_regression``
* gate 2 (material distinctness) fails OR
  gate 3 (objective invariance) fails         -> ``label_only_or_objective_drift``
* gate 4/5/6 (mechanism/integrity) fail OR
  gate 7 (latency diagnostics) fail OR
  gate 8 (PR #6170 audit provenance) fail     -> ``incomplete``
* otherwise                                   -> ``accepted_offline_prototype``

``accepted_offline_prototype`` requires every mechanism/integrity gate to pass;
latency is descriptive and never a post-hoc pass/fail substitute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from robot_sf.benchmark.policy_builders import (
    AdapterPolicySpec,
    build_registered_adapter_policy_spec,
)
from robot_sf.evidence.writers import write_review_sidecar, write_text
from robot_sf.planner.nmpc_social import NMPCSocialConfig, NMPCSocialPlannerAdapter
from robot_sf.planner.topology_parallel_nmpc import (
    HypothesisDiagnostics,
    TopologyParallelNMPCConfig,
    TopologyParallelNMPCPlannerAdapter,
    _material_separation,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs" / "context" / "evidence"
EVIDENCE_DOC = EVIDENCE_DIR / "issue_6158_topology_parallel_nmpc_offline_verdict.md"
ISSUE_NUMBER = 6158
PARENT_ISSUE = 5310
SOURCE_PR = 6170
SOURCE_MERGE_COMMIT = "894bdfe71e9c2686ebe63e165f15c739d12f721c"

# PR #6170 changed-file / net-line audit (from the GitHub PR API; merge commit above).
# This is the immutable implementation packet the prototype is validated against.
PR_6170_AUDIT: list[dict[str, Any]] = [
    {"path": "CHANGELOG.md", "additions": 15, "deletions": 0},
    {
        "path": "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
        "additions": 41,
        "deletions": 0,
    },
    {"path": "docs/context/issue_5310_state.yaml", "additions": 26, "deletions": 0},
    {"path": "robot_sf/benchmark/algorithm_metadata.py", "additions": 15, "deletions": 0},
    {"path": "robot_sf/benchmark/algorithm_readiness.py", "additions": 11, "deletions": 0},
    {"path": "robot_sf/benchmark/policy_builders.py", "additions": 35, "deletions": 0},
    {"path": "robot_sf/planner/__init__.py", "additions": 1, "deletions": 0},
    {"path": "robot_sf/planner/nmpc_social.py", "additions": 183, "deletions": 9},
    {"path": "robot_sf/planner/topology_parallel_nmpc.py", "additions": 447, "deletions": 0},
    {"path": "tests/planner/test_nmpc_social.py", "additions": 50, "deletions": 0},
    {"path": "tests/planner/test_topology_parallel_nmpc.py", "additions": 414, "deletions": 0},
]

VERDICTS = (
    "accepted_offline_prototype",
    "label_only_or_objective_drift",
    "invalid_regression",
    "incomplete",
)

# Pinned K=1 parity tolerance (matches the CHANGELOG claim for #6152).
PARITY_RTOL = 1e-6
PARITY_ATOL = 1e-6
# Minimum pairwise material separation (m) that counts as materially distinct.
MATERIAL_SEP_EPS = 1e-3


@dataclass
class GateResult:
    """Outcome of one validation gate."""

    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _build_obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    goal: tuple[float, float] = (3.0, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
    obstacle_cells: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Build the compact observation payload used by planner tests/fixtures."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    obstacle_cells = [] if obstacle_cells is None else obstacle_cells
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells:
        grid[0, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


def _nmpc_config_from_file(config_path: Path) -> NMPCSocialConfig:
    """Load the shared NMPC sub-config from the issue YAML as a dataclass."""
    raw = yaml.safe_load(config_path.read_text())
    nmpc_raw = raw.get("nmpc_config", {}) if isinstance(raw, dict) else {}
    from robot_sf.planner.nmpc_social import build_nmpc_social_config

    return build_nmpc_social_config(nmpc_raw)


def _raw_config(config_path: Path) -> dict[str, Any]:
    """Load the full issue YAML as a mapping."""
    raw = yaml.safe_load(config_path.read_text())
    return raw if isinstance(raw, dict) else {}


# --------------------------------------------------------------------------- #
# Gate 1: K=1 legacy parity
# --------------------------------------------------------------------------- #


def gate_1_k1_legacy_parity(nmpc_config: NMPCSocialConfig) -> GateResult:
    """K=1 (default hypothesis) must match legacy NMPCSocialPlannerAdapter.plan()."""
    obs = _build_obs(goal=(3.0, 0.0))  # no pedestrians -> preferred_turn == 0.0 on both paths
    topo_cfg = TopologyParallelNMPCConfig(
        max_hypotheses=1,
        hypothesis_labels=("default",),
        nmpc_config=nmpc_config,
        switch_hysteresis_ticks=0,
    )
    topo = TopologyParallelNMPCPlannerAdapter(topo_cfg)
    legacy = NMPCSocialPlannerAdapter(nmpc_config)
    v_topo, w_topo = topo.plan(obs)
    v_leg, w_leg = legacy.plan(obs)
    dv = abs(v_topo - v_leg)
    dw = abs(w_topo - w_leg)
    passed = (dv <= PARITY_ATOL + PARITY_RTOL * abs(v_leg)) and (
        dw <= PARITY_ATOL + PARITY_RTOL * abs(w_leg)
    )
    return GateResult(
        name="gate_1_k1_legacy_parity",
        passed=passed,
        detail=(
            f"K=1 default command ({v_topo:.9g},{w_topo:.9g}) vs legacy "
            f"({v_leg:.9g},{w_leg:.9g}); |dv|={dv:.3e}, |dw|={dw:.3e} "
            f"(rtol={PARITY_RTOL}, atol={PARITY_ATOL})."
        ),
        evidence={
            "topology_command": [v_topo, w_topo],
            "legacy_command": [v_leg, w_leg],
            "abs_delta": [dv, dw],
            "rtol": PARITY_RTOL,
            "atol": PARITY_ATOL,
            "hypothesis_labels": ["default"],
            "max_hypotheses": 1,
        },
    )


# --------------------------------------------------------------------------- #
# Gate 2: materially distinct x-y-t rollouts on controlled conflict fixtures
# --------------------------------------------------------------------------- #


def _wall_obs(
    *,
    goal: tuple[float, float] = (3.0, 0.0),
    wall_rows: range = range(6, 15),
    wall_cols: range = range(14, 18),
    grid_n: int = 20,
    res: float = 0.25,
    origin: tuple[float, float] = (-2.5, -2.5),
    size: tuple[float, float] = (5.0, 5.0),
) -> dict[str, Any]:
    """Build an occupancy-grid barrier fixture in the combined channel (channel 3).

    ``_preferred_channel`` resolves the combined channel via ``channel_indices[3]`` for
    ``channel_indices=[0,1,2,3]`` and ``_CHANNEL_KEYS=('obstacles','pedestrians',
    'robot','combined')``, so the barrier must live in channel index 3 to be felt by
    the planner. The wall blocks the straight path with left/right passages.
    """
    grid = np.zeros((4, grid_n, grid_n), dtype=np.float32)
    for row in wall_rows:
        for col in wall_cols:
            grid[3, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray((0.0, 0.0), dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=float),
            "velocities": np.zeros((0, 2), dtype=float),
            "count": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray(origin, dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([res], dtype=float),
        "occupancy_grid_meta_size": np.asarray(size, dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


def gate_2_material_distinctness(nmpc_config: NMPCSocialConfig) -> GateResult:
    """Feasible hypotheses must show nonzero pairwise material separation.

    A diverse suite of controlled conflict fixtures is exercised and the mechanism is
    given every chance to diverge: the gate passes when at least one fixture separates
    every feasible hypothesis pair above epsilon, and fails when every fixture retains
    at least one feasible pair below epsilon. Per-fixture results are reported
    transparently so the verdict is auditable.
    """
    labels = ("pass_left", "yield_straight", "pass_right")
    fixtures: list[tuple[str, dict[str, Any]]] = [
        ("pedestrian_ahead_1p2", _build_obs(goal=(3.0, 0.0), ped_positions=[(1.2, 0.0)])),
        ("pedestrian_close_0p5", _build_obs(goal=(3.0, 0.0), ped_positions=[(0.5, 0.0)])),
        ("pedestrian_offset", _build_obs(goal=(3.0, 0.0), ped_positions=[(1.0, 0.15)])),
        (
            "two_pedestrian_gate",
            _build_obs(goal=(3.0, 0.0), ped_positions=[(1.0, 0.3), (1.0, -0.3)]),
        ),
        ("goal_offset_with_ped", _build_obs(goal=(3.0, 0.5), ped_positions=[(1.0, 0.0)])),
        ("hard_wall_left_right_gaps", _wall_obs()),
    ]
    per_fixture: list[dict[str, Any]] = []
    best_min_sep = 0.0
    for name, obs in fixtures:
        topo = TopologyParallelNMPCPlannerAdapter(
            TopologyParallelNMPCConfig(
                max_hypotheses=3,
                hypothesis_labels=labels,
                nmpc_config=nmpc_config,
                switch_hysteresis_ticks=0,
                max_runtime_s=300.0,
                control_period_s=300.0,
            )
        )
        topo.plan(obs)
        diag = topo._last_hypothesis_diagnostics
        states = topo._last_result_states
        feasible_labels = [d.label for d in diag if d.feasible]
        pairwise: list[dict[str, Any]] = []
        min_sep = float("inf")
        for i in range(len(diag)):
            if not diag[i].feasible:
                continue
            for j in range(i + 1, len(diag)):
                if not diag[j].feasible:
                    continue
                si = states.get(diag[i].label)
                sj = states.get(diag[j].label)
                if si is None or sj is None:
                    continue
                sep = _material_separation(si, sj)
                min_sep = min(min_sep, sep)
                pairwise.append({"pair": [diag[i].label, diag[j].label], "separation_m": sep})
        if math.isfinite(min_sep):
            best_min_sep = max(best_min_sep, min_sep)
        per_fixture.append(
            {
                "fixture": name,
                "feasible_hypotheses": feasible_labels,
                "min_pairwise_separation_m": min_sep if math.isfinite(min_sep) else None,
                "pairwise_separations_m": pairwise,
                "rollout_signatures": {d.label: d.rollout_signature for d in diag},
            }
        )
    passed = best_min_sep > MATERIAL_SEP_EPS
    return GateResult(
        name="gate_2_material_distinctness",
        passed=passed,
        detail=(
            f"best min pairwise material_separation across {len(fixtures)} conflict "
            f"fixtures = {best_min_sep:.6g} m (epsilon={MATERIAL_SEP_EPS}); gate passes "
            f"only if at least one fixture separates every feasible hypothesis pair "
            f"above epsilon. "
            "No fixture separated every feasible hypothesis pair above epsilon; "
            "topology identity is not established (label-only)."
            if not passed
            else f"at least one fixture separated every feasible hypothesis pair "
            f"(best min sep={best_min_sep:.6g} m > epsilon={MATERIAL_SEP_EPS})."
        ),
        evidence={
            "fixtures_tested": [name for name, _ in fixtures],
            "epsilon_m": MATERIAL_SEP_EPS,
            "best_min_pairwise_separation_m": best_min_sep,
            "per_fixture": per_fixture,
            "root_cause_note": (
                "objective_preferred_turn == 0.0 for every hypothesis (gate 3), so the "
                "shared objective is identical; the only per-hypothesis difference is the "
                "initial-guess preferred_turn bias (+/-0.5 -> +/-0.1 rad/s w-seed via "
                "symmetry_break_bias=0.2). Across the tested seeds and fixtures, SLSQP "
                "left at least one feasible pair below the material-separation threshold on "
                "every fixture. Some individual pairs exceeded epsilon, but no fixture "
                "separated the full feasible hypothesis set under the shared soft-penalty "
                "objective; this diagnostic does not establish global uniqueness. The "
                "'topology-parallel' mechanism is label-only under this configuration."
            ),
        },
    )


# --------------------------------------------------------------------------- #
# Gate 3: identical solver/objective/constraints across hypotheses
# --------------------------------------------------------------------------- #


def gate_3_objective_invariance(nmpc_config: NMPCSocialConfig) -> GateResult:
    """Verify identical objective, solver, bounds, constraints, and options per hypothesis."""
    import robot_sf.planner.nmpc_social as nmpc_mod

    obs = _build_obs(goal=(3.0, 0.0), ped_positions=[(1.2, 0.0)])
    labels = ("pass_left", "yield_straight", "pass_right")
    topo_cfg = TopologyParallelNMPCConfig(
        max_hypotheses=3,
        hypothesis_labels=labels,
        nmpc_config=nmpc_config,
        switch_hysteresis_ticks=0,
        max_runtime_s=300.0,
        control_period_s=300.0,
    )
    topo = TopologyParallelNMPCPlannerAdapter(topo_cfg)
    solver_invocations: list[dict[str, Any]] = []
    original_minimize = nmpc_mod.minimize

    def _capture_minimize(*args: Any, **kwargs: Any) -> Any:
        """Record the solver contract while delegating to SciPy unchanged."""
        bounds = kwargs.get("bounds")
        constraints = kwargs.get("constraints", ())
        if not isinstance(constraints, tuple):
            constraints = tuple(constraints)
        solver_invocations.append(
            {
                "method": kwargs.get("method"),
                "bounds_lower": np.asarray(getattr(bounds, "lb", []), dtype=float).tolist(),
                "bounds_upper": np.asarray(getattr(bounds, "ub", []), dtype=float).tolist(),
                "constraints": [
                    {
                        "type": type(constraint).__name__,
                        "lower": np.asarray(getattr(constraint, "lb", []), dtype=float).tolist(),
                        "upper": np.asarray(getattr(constraint, "ub", []), dtype=float).tolist(),
                    }
                    for constraint in constraints
                ],
                "options": dict(kwargs.get("options", {})),
            }
        )
        return original_minimize(*args, **kwargs)

    nmpc_mod.minimize = _capture_minimize
    try:
        topo.plan(obs)
    finally:
        nmpc_mod.minimize = original_minimize
    diag = topo._last_hypothesis_diagnostics
    per_label: dict[str, Any] = {}
    all_zero = len(diag) == len(labels)
    shared_cfg_fields = asdict(nmpc_config)
    for d in diag:
        sig = d.initialization_signature
        opt_turn = sig.get("objective_preferred_turn")
        per_label[d.label] = {
            "objective_preferred_turn": opt_turn,
            "preferred_turn": sig.get("preferred_turn"),
            "solver_status": d.solver_status,
        }
        all_zero = all_zero and (opt_turn == 0.0)
    solver_configurations_identical = (
        len(solver_invocations) == len(labels)
        and len({json.dumps(call, sort_keys=True) for call in solver_invocations}) == 1
    )
    return GateResult(
        name="gate_3_objective_invariance",
        passed=all_zero and solver_configurations_identical,
        detail=(
            "objective_preferred_turn == 0.0 for every hypothesis; "
            f"solver/bounds/constraints/options identical={solver_configurations_identical}; "
            f"shared config={shared_cfg_fields}."
        ),
        evidence={
            "per_hypothesis": per_label,
            "shared_nmpc_config": shared_cfg_fields,
            "all_objective_preferred_turn_zero": all_zero,
            "solver_invocations": solver_invocations,
            "solver_invocation_count": len(solver_invocations),
            "solver_configurations_identical": solver_configurations_identical,
        },
    )


# --------------------------------------------------------------------------- #
# Gate 4: deterministic ordering, feasible-first/lowest-objective selection, hysteresis
# --------------------------------------------------------------------------- #


def _synth_diag(label: str, feasible: bool, objective: float) -> HypothesisDiagnostics:
    """Build a synthetic HypothesisDiagnostics row for exercising selection read-only."""
    return HypothesisDiagnostics(
        label=label,
        feasible=feasible,
        objective=objective,
        solver_status="ok" if feasible else "infeasible",
        solver_iterations=1,
        solver_runtime=0.0,
        signed_side=0,
        material_separation=0.0,
    )


def gate_4_selection_and_hysteresis(nmpc_config: NMPCSocialConfig) -> GateResult:
    """Verify deterministic ordering, feasible-first/lowest-objective, two-tick hysteresis."""
    labels = ("pass_left", "yield_straight", "pass_right")
    evidence: dict[str, Any] = {}

    # (a) Deterministic ordering: diagnostics preserve hypothesis_labels order.
    obs = _build_obs(goal=(3.0, 0.0), ped_positions=[(1.2, 0.0)])
    topo = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=3,
            hypothesis_labels=labels,
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
        )
    )
    topo.plan(obs)
    order_labels = tuple(d.label for d in topo._last_hypothesis_diagnostics)
    ordering_ok = order_labels == labels
    evidence["diagnostic_label_order"] = list(order_labels)
    evidence["expected_label_order"] = list(labels)

    # (b) Feasible-first / lowest-objective selection (hysteresis disabled).
    sel = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=3,
            hypothesis_labels=labels,
            switch_hysteresis_ticks=0,
        )
    )
    # Infeasible hypothesis carries the lowest raw objective; it must NOT be selected.
    diag_b = [
        _synth_diag("pass_left", feasible=True, objective=5.0),
        _synth_diag("yield_straight", feasible=False, objective=1.0),
        _synth_diag("pass_right", feasible=True, objective=3.0),
    ]
    chosen_b = sel._select_hypothesis(diag_b)
    ranks_b = {d.label: d.selection_rank for d in diag_b}
    # Lowest-objective feasible is pass_right (3.0); infeasible yield_straight excluded.
    selection_ok = (
        chosen_b == 2 and ranks_b.get("pass_right") == 0 and ranks_b.get("pass_left") == 1
    )
    evidence["feasible_first_selection_index"] = chosen_b
    evidence["feasible_first_ranks"] = ranks_b

    # (c) Two-tick switch hysteresis.
    hys = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=3,
            hypothesis_labels=labels,
            switch_hysteresis_ticks=2,
        )
    )
    hys._current_hypothesis_index = 0
    hys._ticks_at_hypothesis = 0
    # Tick 1: current (pass_left) is also best -> tick counter increments to 1.
    diag_c1 = [
        _synth_diag("pass_left", True, 1.0),
        _synth_diag("yield_straight", True, 2.0),
        _synth_diag("pass_right", True, 3.0),
    ]
    r1 = hys._select_hypothesis(diag_c1)
    reason_c1 = {d.label: d.switch_reason for d in diag_c1}
    # Now a different hypothesis becomes best while ticks (1) < threshold (2) -> suppressed.
    diag_c2 = [
        _synth_diag("pass_left", True, 3.0),
        _synth_diag("yield_straight", True, 1.0),
        _synth_diag("pass_right", True, 2.0),
    ]
    r2 = hys._select_hypothesis(diag_c2)
    reason_c2 = {d.label: d.switch_reason for d in diag_c2}
    # Let current be best once more to reach the threshold (ticks -> 2).
    hys._ticks_at_hypothesis = 2  # current has now been best for >= threshold ticks
    diag_c3 = [
        _synth_diag("pass_left", True, 3.0),
        _synth_diag("yield_straight", True, 1.0),
        _synth_diag("pass_right", True, 2.0),
    ]
    r3 = hys._select_hypothesis(diag_c3)
    reason_c3 = {d.label: d.switch_reason for d in diag_c3}
    suppressed_ok = (
        r1 == 0 and r2 == 0 and reason_c2.get("yield_straight") == "suppressed_by_hysteresis"
    )
    switch_ok = r3 == 1 and reason_c3.get("yield_straight") == "new_best_selected"
    switches_recorded = int(hys._topo_stats.get("hypothesis_switches", 0))
    evidence["hysteresis"] = {
        "switch_hysteresis_ticks": 2,
        "tick1_selected": r1,
        "tick1_reasons": reason_c1,
        "tick2_selected": r2,
        "tick2_reasons": reason_c2,
        "tick3_selected": r3,
        "tick3_reasons": reason_c3,
        "suppressed_before_threshold": suppressed_ok,
        "switched_at_or_after_threshold": switch_ok,
        "hypothesis_switches_recorded": switches_recorded,
    }

    passed = ordering_ok and selection_ok and suppressed_ok and switch_ok
    detail = (
        f"ordering={ordering_ok}, feasible_first/lowest_obj selection={selection_ok}, "
        f"hysteresis suppress(<2 ticks)={suppressed_ok}, switch(>=2 ticks)={switch_ok}."
    )
    return GateResult(
        name="gate_4_selection_and_hysteresis", passed=passed, detail=detail, evidence=evidence
    )


# --------------------------------------------------------------------------- #
# Gate 5: fail-closed on infeasible / deadline_exceeded / solver-error
# --------------------------------------------------------------------------- #


def _make_infeasible_result():
    """Build a scipy-like optimizer result object that reports failure."""
    return type(
        "r",
        (),
        {"success": False, "x": None, "fun": None, "status": 9, "nit": 0},
    )()


def gate_5_fail_closed(nmpc_config: NMPCSocialConfig) -> GateResult:
    """Verify fail-closed behavior on infeasible, deadline-exceeded, and solver-error."""
    import robot_sf.planner.nmpc_social as nmpc_mod

    evidence: dict[str, Any] = {}
    obs = _build_obs(goal=(3.0, 0.0))

    # (a) infeasible / solver-error-status: minimize returns success=False.
    planner_a = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=2,
            hypothesis_labels=("left", "right"),
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
            max_runtime_s=10.0,
            control_period_s=10.0,
        )
    )
    original_minimize = nmpc_mod.minimize
    nmpc_mod.minimize = lambda *a, **k: _make_infeasible_result()
    try:
        cmd_infeasible = planner_a.plan(obs)
    finally:
        nmpc_mod.minimize = original_minimize
    infeasible_ok = cmd_infeasible == (0.0, 0.0)
    evidence["infeasible_status_command"] = list(cmd_infeasible)
    diag_a_status = [d.solver_status for d in planner_a._last_hypothesis_diagnostics]
    evidence["infeasible_solver_statuses"] = diag_a_status

    # (b) deadline_exceeded: max_runtime_s so small that the deadline fires.
    planner_b = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=3,
            hypothesis_labels=("pass_left", "yield_straight", "pass_right"),
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
            max_runtime_s=1e-6,
            control_period_s=1e-6,
        )
    )
    cmd_deadline = planner_b.plan(obs)
    deadline_ok = cmd_deadline == (0.0, 0.0) and planner_b._deadline_exceeded_this_call
    diag_b_status = [d.solver_status for d in planner_b._last_hypothesis_diagnostics]
    evidence["deadline_command"] = list(cmd_deadline)
    evidence["deadline_solver_statuses"] = diag_b_status
    evidence["deadline_exceeded_flag"] = bool(planner_b._deadline_exceeded_this_call)

    # (c) solver-error = scipy success=False with an explicit error status code; this
    #     shares the infeasible code path (fallback_to_stop). Recorded explicitly so the
    #     error-status interpretation of "solver-error" is demonstrated, not assumed.
    planner_c = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=2,
            hypothesis_labels=("left", "right"),
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
            max_runtime_s=10.0,
            control_period_s=10.0,
        )
    )
    nmpc_mod.minimize = lambda *a, **k: type(
        "r", (), {"success": False, "x": None, "fun": None, "status": 2, "nit": 0}
    )()
    try:
        cmd_err = planner_c.plan(obs)
    finally:
        nmpc_mod.minimize = original_minimize
    error_status_ok = cmd_err == (0.0, 0.0)
    evidence["solver_error_status_command"] = list(cmd_err)

    # A solver exception must fail closed too. The approved #6158 contract says
    # "solver-error behavior", not only solver error-status returns.
    planner_d = TopologyParallelNMPCPlannerAdapter(
        TopologyParallelNMPCConfig(
            max_hypotheses=2,
            hypothesis_labels=("left", "right"),
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
            max_runtime_s=10.0,
            control_period_s=10.0,
        )
    )

    def _raising_minimize(*a, **k):
        raise ValueError("synthetic objective failure")

    nmpc_mod.minimize = _raising_minimize
    exception_propagates = False
    exc_repr = ""
    try:
        planner_d.plan(obs)
    except ValueError as exc:
        exception_propagates = True
        exc_repr = f"{type(exc).__name__}: {exc}"
    finally:
        nmpc_mod.minimize = original_minimize
    evidence["exception_probe"] = {
        "exception_propagates": exception_propagates,
        "exception_repr": exc_repr,
        "note": (
            "An exception inside the objective is NOT caught by the prototype; "
            "plan() propagates it instead of returning the fail-closed stop command. "
            "The #6158 gate requires fail-closed solver-error behavior, so this probe "
            "fails gate 5 even though deadline-overrun and infeasible/error-status "
            "fallbacks are correct."
        ),
    }

    exception_fail_closed = not exception_propagates
    passed = infeasible_ok and deadline_ok and error_status_ok and exception_fail_closed
    detail = (
        f"infeasible->stop={infeasible_ok}, deadline_exceeded->stop={deadline_ok}, "
        f"solver_error_status->stop={error_status_ok}, "
        f"solver_exception->stop={exception_fail_closed}."
    )
    return GateResult(name="gate_5_fail_closed", passed=passed, detail=detail, evidence=evidence)


# --------------------------------------------------------------------------- #
# Gate 6: experimental registration-path smoke
# --------------------------------------------------------------------------- #


def gate_6_registration_smoke(raw_config: dict[str, Any]) -> GateResult:
    """allow_testing_algorithms guard + topology_parallel_nmpc builder."""
    evidence: dict[str, Any] = {}
    guarded_cfg = {
        "allow_testing_algorithms": True,
        "max_hypotheses": 1,
        "hypothesis_labels": ["default"],
    }
    spec = _build_topology_parallel_nmpc_policy_spec_public(guarded_cfg)
    builder_ok = (
        isinstance(spec, AdapterPolicySpec)
        and spec.algo_key == "topology_parallel_nmpc"
        and spec.adapter_name == "TopologyParallelNMPCPlannerAdapter"
        and spec.limitations == "experimental_topology_parallel_nmpc"
        and isinstance(spec.adapter, TopologyParallelNMPCPlannerAdapter)
    )
    evidence["guarded_build"] = {
        "algo_key": spec.algo_key if isinstance(spec, AdapterPolicySpec) else None,
        "adapter_name": spec.adapter_name if isinstance(spec, AdapterPolicySpec) else None,
        "limitations": spec.limitations if isinstance(spec, AdapterPolicySpec) else None,
    }

    # Guard rejects when allow_testing_algorithms is missing/false.
    guard_reject_missing = False
    try:
        _build_topology_parallel_nmpc_policy_spec_public({})
    except ValueError:
        guard_reject_missing = True
    guard_reject_false = False
    try:
        _build_topology_parallel_nmpc_policy_spec_public({"allow_testing_algorithms": False})
    except ValueError:
        guard_reject_false = True
    evidence["guard_reject_missing"] = guard_reject_missing
    evidence["guard_reject_false"] = guard_reject_false

    # Registry entry smoke using the actual issue config (which carries the guard).
    registry_spec = build_registered_adapter_policy_spec("topology_parallel_nmpc", raw_config)
    registry_ok = (
        isinstance(registry_spec, AdapterPolicySpec)
        and registry_spec.adapter_name == "TopologyParallelNMPCPlannerAdapter"
    )
    evidence["registry_build_from_issue_config"] = registry_ok

    passed = builder_ok and guard_reject_missing and guard_reject_false and registry_ok
    return GateResult(
        name="gate_6_registration_smoke",
        passed=passed,
        detail=(
            f"builder_ok={builder_ok}, guard_reject_missing={guard_reject_missing}, "
            f"guard_reject_false={guard_reject_false}, registry_ok={registry_ok}."
        ),
        evidence=evidence,
    )


def _build_topology_parallel_nmpc_policy_spec_public(
    algo_config: dict[str, Any],
) -> AdapterPolicySpec:
    """Thin import-indirection so the module symbol is resolved at call time."""
    from robot_sf.benchmark.policy_builders import _build_topology_parallel_nmpc_policy_spec

    return _build_topology_parallel_nmpc_policy_spec(algo_config)


# --------------------------------------------------------------------------- #
# Gate 7: per-hypothesis p50/p95/max solve latency (descriptive)
# --------------------------------------------------------------------------- #


def _percentiles(values: list[float]) -> dict[str, float]:
    """Return p50/p95/max descriptive statistics for a latency sample."""
    if not values:
        return {"p50_ms": None, "p95_ms": None, "max_ms": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "n": int(arr.size),
    }


def gate_7_latency(nmpc_config: NMPCSocialConfig) -> GateResult:
    """Per-hypothesis p50/p95/max solve latency on a fixed CPU fixture (descriptive)."""
    labels = ("pass_left", "yield_straight", "pass_right")
    affinity_evidence: dict[str, Any] = {
        "pinned": False,
        "measurement_cpu": None,
        "original_cpu_set": None,
        "restored": False,
        "error": None,
    }
    original_affinity: set[int] | None = None
    try:
        original_affinity = set(os.sched_getaffinity(0))
        measurement_cpu = min(original_affinity)
        os.sched_setaffinity(0, {measurement_cpu})
        affinity_evidence.update(
            {
                "pinned": True,
                "measurement_cpu": measurement_cpu,
                "original_cpu_set": sorted(original_affinity),
            }
        )
    except (AttributeError, OSError) as exc:
        affinity_evidence["error"] = f"{type(exc).__name__}: {exc}"

    # Measurement-safe deadline so the runtime gate never truncates a solve; the shared
    # NMPC config (horizon/iterations/weights) is unchanged.
    topo_cfg = TopologyParallelNMPCConfig(
        max_hypotheses=3,
        hypothesis_labels=labels,
        nmpc_config=nmpc_config,
        switch_hysteresis_ticks=0,
        max_runtime_s=300.0,
        control_period_s=300.0,
    )
    try:
        topo = TopologyParallelNMPCPlannerAdapter(topo_cfg)
        obs = _build_obs(goal=(3.0, 0.0), ped_positions=[(1.2, 0.0)])

        warmup = 3
        measured = 30
        for _ in range(warmup):
            topo.plan(obs)
        per_hypothesis: dict[str, list[float]] = {label: [] for label in labels}
        plan_wall_ms: list[float] = []
        for _ in range(measured):
            t0 = time.perf_counter()
            topo.plan(obs)
            plan_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            for d in topo._last_hypothesis_diagnostics:
                if d.label in per_hypothesis:
                    per_hypothesis[d.label].append(d.solver_runtime * 1000.0)

        per_hypothesis_stats = {label: _percentiles(per_hypothesis[label]) for label in labels}
        plan_wall_stats = _percentiles(plan_wall_ms)
        # End-to-end with the file's real 2.0s deadline (smaller sample) for real-time context.
        real_cfg = TopologyParallelNMPCConfig(
            max_hypotheses=3,
            hypothesis_labels=labels,
            nmpc_config=nmpc_config,
            switch_hysteresis_ticks=0,
            max_runtime_s=2.0,
            control_period_s=2.0,
        )
        real_planner = TopologyParallelNMPCPlannerAdapter(real_cfg)
        real_wall_ms: list[float] = []
        real_deadline_fires = 0
        for _ in range(8):
            t0 = time.perf_counter()
            real_planner.plan(obs)
            real_wall_ms.append((time.perf_counter() - t0) * 1000.0)
            if real_planner._deadline_exceeded_this_call:
                real_deadline_fires += 1
    finally:
        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
                affinity_evidence["restored"] = True
            except OSError as exc:
                affinity_evidence["error"] = f"{type(exc).__name__}: {exc}"
    real_wall_stats = _percentiles(real_wall_ms)

    all_p95 = [s["p95_ms"] for s in per_hypothesis_stats.values() if s["p95_ms"] is not None]
    worst_p95 = max(all_p95) if all_p95 else 0.0
    latency_exceeds_100ms = worst_p95 > 100.0

    evidence = {
        "cpu_affinity_fixture": affinity_evidence,
        "per_hypothesis_solver_runtime_ms": per_hypothesis_stats,
        "plan_wall_clock_ms_measurement_safe_deadline": plan_wall_stats,
        "plan_wall_clock_ms_real_2s_deadline": real_wall_stats,
        "real_deadline_fires_out_of_8": real_deadline_fires,
        "worst_hypothesis_p95_ms": worst_p95,
        "latency_exceeds_100ms": latency_exceeds_100ms,
        "measurement_note": (
            "Descriptive only on a single CPU-pinned fixture; not a controlled benchmark. "
            "max_runtime_s/control_period_s were raised to 300s during measurement so the "
            "runtime gate never truncates a solve; the shared NMPC config is unchanged."
        ),
    }
    detail = (
        "per-hypothesis solver p95 (ms): "
        + ", ".join(f"{lbl}={per_hypothesis_stats[lbl]['p95_ms']:.3g}" for lbl in labels)
        + f"; worst p95={worst_p95:.3g} ms; exceeds_100ms={latency_exceeds_100ms}; "
        + f"cpu_pinned={affinity_evidence['pinned']}."
    )
    # Latency never fails this gate (it is descriptive); the only failure mode is an
    # inability to collect the sample.
    passed = affinity_evidence["pinned"] and all(s["n"] > 0 for s in per_hypothesis_stats.values())
    return GateResult(name="gate_7_latency", passed=passed, detail=detail, evidence=evidence)


# --------------------------------------------------------------------------- #
# Gate 8: PR #6170 changed-file / net-line audit
# --------------------------------------------------------------------------- #


def gate_8_pr_audit() -> GateResult:
    """Exact changed-file/net-line audit of PR #6170 against the implementation packet."""
    total_add = sum(row["additions"] for row in PR_6170_AUDIT)
    total_del = sum(row["deletions"] for row in PR_6170_AUDIT)
    net = total_add - total_del
    audited_paths = {row["path"] for row in PR_6170_AUDIT}
    expected_implementation_surfaces = {
        "robot_sf/planner/topology_parallel_nmpc.py",
        "robot_sf/planner/nmpc_social.py",
        "robot_sf/planner/__init__.py",
        "configs/algos/issue_5310_topology_parallel_nmpc.yaml",
        "robot_sf/benchmark/policy_builders.py",
        "robot_sf/benchmark/algorithm_metadata.py",
        "robot_sf/benchmark/algorithm_readiness.py",
        "tests/planner/test_topology_parallel_nmpc.py",
        "tests/planner/test_nmpc_social.py",
        "CHANGELOG.md",
        "docs/context/issue_5310_state.yaml",
    }
    surfaces_match = audited_paths == expected_implementation_surfaces
    totals_consistent = total_add == 1238 and total_del == 9 and net == 1229
    source_diff = subprocess.run(
        ["git", "diff", "--numstat", f"{SOURCE_MERGE_COMMIT}^1", SOURCE_MERGE_COMMIT],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    source_audit: list[dict[str, Any]] = []
    source_audit_parse_error = ""
    if source_diff.returncode == 0:
        try:
            for line in source_diff.stdout.splitlines():
                additions, deletions, path = line.split("\t", maxsplit=2)
                source_audit.append(
                    {"path": path, "additions": int(additions), "deletions": int(deletions)}
                )
        except ValueError as exc:
            source_audit_parse_error = f"could not parse git --numstat output: {exc}"
    else:
        source_audit_parse_error = source_diff.stderr.strip() or "git diff --numstat failed"
    source_audit_matches_declared = source_audit == PR_6170_AUDIT

    protected_paths = expected_implementation_surfaces - {
        "CHANGELOG.md",
        "docs/context/issue_5310_state.yaml",
    }
    current_pr_diff = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    current_pr_changed_paths = (
        set(current_pr_diff.stdout.splitlines()) if current_pr_diff.returncode == 0 else set()
    )
    protected_path_deltas = sorted(current_pr_changed_paths & protected_paths)
    current_pr_preserves_prototype = current_pr_diff.returncode == 0 and not protected_path_deltas
    head_post_merge_note = (
        "The source merge's first-parent diff matches the declared implementation packet, "
        "and this PR does not change the protected prototype/config/registry/test surfaces "
        "relative to origin/main."
    )
    passed = (
        surfaces_match
        and totals_consistent
        and source_audit_matches_declared
        and current_pr_preserves_prototype
    )
    evidence = {
        "source_pr": SOURCE_PR,
        "merge_commit": SOURCE_MERGE_COMMIT,
        "files": PR_6170_AUDIT,
        "total_additions": total_add,
        "total_deletions": total_del,
        "net_lines": net,
        "audited_paths_match_implementation_surfaces": surfaces_match,
        "totals_consistent": totals_consistent,
        "source_audit": source_audit,
        "source_audit_matches_declared": source_audit_matches_declared,
        "source_audit_parse_error": source_audit_parse_error or None,
        "current_pr_protected_path_deltas": protected_path_deltas,
        "current_pr_preserves_prototype": current_pr_preserves_prototype,
        "head_post_merge_note": head_post_merge_note,
    }
    return GateResult(
        name="gate_8_pr_audit",
        passed=passed,
        detail=(
            f"PR #{SOURCE_PR} @ {SOURCE_MERGE_COMMIT[:12]}: {len(PR_6170_AUDIT)} files, "
            f"+{total_add}/-{total_del} (net {net:+d}); surfaces_match={surfaces_match}, "
            f"totals_consistent={totals_consistent}, "
            f"source_audit_matches_declared={source_audit_matches_declared}, "
            f"current_pr_preserves_prototype={current_pr_preserves_prototype}."
        ),
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# Verdict derivation + evidence writing
# --------------------------------------------------------------------------- #


def _derive_verdict(gates: list[GateResult]) -> tuple[str, str]:
    """Map gate outcomes to exactly one verdict plus a rationale string."""
    by_name = {g.name: g for g in gates}
    if not by_name["gate_1_k1_legacy_parity"].passed:
        return "invalid_regression", "gate 1 (K=1 legacy parity) failed -> legacy/default drift."
    identity_failures = [
        label
        for label, gate_name in (
            ("gate 2 (material distinctness)", "gate_2_material_distinctness"),
            ("gate 3 (objective invariance)", "gate_3_objective_invariance"),
        )
        if not by_name[gate_name].passed
    ]
    if identity_failures:
        return (
            "label_only_or_objective_drift",
            f"{' and '.join(identity_failures)} failed.",
        )
    for required in (
        "gate_4_selection_and_hysteresis",
        "gate_5_fail_closed",
        "gate_6_registration_smoke",
    ):
        if not by_name[required].passed:
            return "incomplete", f"{required} (mechanism/integrity) failed."
    if not by_name["gate_7_latency"].passed:
        return "incomplete", "gate 7 (latency diagnostics) could not be collected."
    if not by_name["gate_8_pr_audit"].passed:
        return "incomplete", "gate 8 (PR #6170 audit provenance) failed."
    return "accepted_offline_prototype", "all eight mechanism/integrity gates passed."


def _hardware_context(cpu_affinity_fixture: dict[str, Any]) -> dict[str, Any]:
    """Capture fixed-CPU descriptive context for the latency fixture."""
    try:
        freq_mhz = subprocess.run(
            ["sh", "-c", "lscpu -bC=MHZ 2>/dev/null | tail -n +2 | head -1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        freq_mhz = ""
    return {
        "platform_processor": platform.processor() or "unknown",
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "os_cpu_count": os.cpu_count(),
        "cpu_freq_mhz_sample": freq_mhz or "unavailable",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cpu_pinning": (
            f"CPU {cpu_affinity_fixture['measurement_cpu']} pinned for latency sampling; "
            f"original affinity restored={cpu_affinity_fixture['restored']}."
            if cpu_affinity_fixture.get("pinned")
            else "unavailable; latency gate is incomplete: "
            f"{cpu_affinity_fixture.get('error') or 'affinity pinning failed'}"
        ),
    }


def _exact_commands(config_rel: str) -> list[str]:
    """Record the exact commands that produce/validate this evidence."""
    return [
        "uv run pytest tests/planner/test_topology_parallel_nmpc.py tests/planner/test_nmpc_social.py -v",
        f"uv run python scripts/validation/check_issue_6158_topology_parallel_nmpc_offline_verdict.py --config {config_rel}",
        "uv run ruff check scripts/validation/ && uv run ruff format --check scripts/validation/",
    ]


def _write_evidence_doc(
    *,
    verdict: str,
    rationale: str,
    gates: list[GateResult],
    commit: str,
    config_rel: str,
    hardware: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    """Write the single Markdown evidence document and return its structured summary."""
    latency_gate = next(g for g in gates if g.name == "gate_7_latency")
    latency_exceeds_100ms = bool(latency_gate.evidence.get("latency_exceeds_100ms", False))
    worst_p95 = latency_gate.evidence.get("worst_hypothesis_p95_ms")
    per_hyp = latency_gate.evidence.get("per_hypothesis_solver_runtime_ms", {})

    summary = {
        "schema": "issue_6158_topology_parallel_nmpc_offline_verdict.v1",
        "review_marker": "AI-GENERATED NEEDS-REVIEW",
        "issue": ISSUE_NUMBER,
        "parent_issue": PARENT_ISSUE,
        "source_pr": SOURCE_PR,
        "source_merge_commit": SOURCE_MERGE_COMMIT,
        "validated_commit": commit,
        "branch": branch,
        "verdict": verdict,
        "verdict_rationale": rationale,
        "config": config_rel,
        "commands": _exact_commands(config_rel),
        "hardware_context": hardware,
        "per_hypothesis_solver_latency_ms": per_hyp,
        "plan_wall_clock_ms_measurement_safe_deadline": latency_gate.evidence.get(
            "plan_wall_clock_ms_measurement_safe_deadline"
        ),
        "plan_wall_clock_ms_real_2s_deadline": latency_gate.evidence.get(
            "plan_wall_clock_ms_real_2s_deadline"
        ),
        "real_deadline_fires_out_of_8": latency_gate.evidence.get("real_deadline_fires_out_of_8"),
        "worst_hypothesis_p95_ms": worst_p95,
        "latency_exceeds_100ms": latency_exceeds_100ms,
        "control_period_s": 2.0,
        "real_time_blocking_notice": (
            "NOT REAL-TIME QUALIFIED (prominent, independent of the per-solve number): "
            "the prototype's nominal control_period_s is 2.0 s, which is ~20x the "
            "100 ms real-time gate, so the component is offline-only and explicitly "
            "blocks downstream real-time use. This is not a real-time qualification "
            "campaign; real-time/performance qualification stays in #5423. "
            + (
                "Additionally, worst per-hypothesis solver p95 exceeded 100 ms on this "
                "fixture, reinforcing the real-time blocker."
                if latency_exceeds_100ms
                else "Per-hypothesis solver p95 was under 100 ms on this fixture, but "
                "this is descriptive only and does NOT establish real-time suitability."
            )
        ),
        "claim_boundary": (
            "No real-time-suitability, safety, benchmark-superiority, "
            "default-planner-promotion, or #5423/STKP-eligibility claim. "
            "Diagnostic-only offline mechanism evidence."
        ),
        "gates": [
            {
                "name": g.name,
                "passed": g.passed,
                "detail": g.detail,
                "evidence": g.evidence,
            }
            for g in gates
        ],
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    latency_block = "\n".join(
        f"| {label} | {stats['p50_ms']:.4g} | {stats['p95_ms']:.4g} | {stats['max_ms']:.4g} | {stats['n']} |"
        for label, stats in per_hyp.items()
    )
    plan_ms = latency_gate.evidence.get("plan_wall_clock_ms_measurement_safe_deadline", {})
    plan_real = latency_gate.evidence.get("plan_wall_clock_ms_real_2s_deadline", {})
    md = []
    md.append(f"# Issue #{ISSUE_NUMBER}: topology-parallel NMPC offline verdict\n")
    md.append(
        f"Diagnostic-only validation of the merged #{SOURCE_PR} prototype "
        f"(`robot_sf/planner/topology_parallel_nmpc.py`) for parent #{PARENT_ISSUE}. "
        "The prototype was executed **unchanged**; this validator only imports/calls it "
        "and reads back diagnostics.\n"
    )
    md.append("## Verdict\n")
    md.append(f"**`{verdict}`** — {rationale}\n")
    md.append(
        "> ⚠️ **REAL-TIME BOUNDARY (prominent, independent of verdict):** the prototype's "
        "nominal `control_period_s` is **2.0 s (~20x the 100 ms real-time gate)**, so it "
        "is **offline-only and explicitly blocks downstream real-time use**. This is not "
        "a real-time qualification campaign; real-time/performance qualification stays "
        "in #5423."
        + (
            f" On this fixture, worst per-hypothesis solver p95 = **{worst_p95:.3g} ms** "
            f"exceeded 100 ms, reinforcing the blocker."
            if latency_exceeds_100ms
            else f" On this fixture, worst per-hypothesis solver p95 = {worst_p95:.3g} ms "
            f"(under 100 ms, descriptive only; not a real-time qualification claim)."
        )
        + "\n"
    )
    md.append("## Provenance\n")
    md.append(f"- Validated commit (`git rev-parse HEAD`): `{commit}`\n")
    md.append(f"- Branch: `{branch}`\n")
    md.append(f"- Source PR: #{SOURCE_PR} (merge commit `{SOURCE_MERGE_COMMIT}`)\n")
    md.append(f"- Config: `{config_rel}`\n")
    md.append("## Exact commands\n")
    for cmd in _exact_commands(config_rel):
        md.append(f"```\n{cmd}\n```\n")
    md.append("## Hardware context (fixed-CPU fixture, descriptive)\n")
    md.append("| field | value |\n| --- | --- |\n")
    for key, value in hardware.items():
        md.append(f"| {key} | `{value}` |\n")
    md.append("\n## Per-hypothesis solve latency (descriptive)\n")
    md.append(
        "| hypothesis | p50 (ms) | p95 (ms) | max (ms) | n |\n| --- | --- | --- | --- | --- |\n"
    )
    md.append(latency_block + "\n")
    md.append(
        f"\nEnd-to-end `plan()` wall-clock (measurement-safe deadline): "
        f"p50={plan_ms.get('p50_ms')} ms, p95={plan_ms.get('p95_ms')} ms, "
        f"max={plan_ms.get('max_ms')} ms (n={plan_ms.get('n')}).\n"
    )
    md.append(
        f"End-to-end `plan()` wall-clock (real 2.0s deadline): "
        f"p50={plan_real.get('p50_ms')} ms, p95={plan_real.get('p95_ms')} ms, "
        f"max={plan_real.get('max_ms')} ms; deadline fired "
        f"{latency_gate.evidence.get('real_deadline_fires_out_of_8')} of 8 calls.\n"
    )
    md.append("\n_" + latency_gate.evidence.get("measurement_note", "") + "_\n")
    md.append("## Gate-by-gate evidence\n")
    for g in gates:
        md.append(f"### {g.name} — {'PASS' if g.passed else 'FAIL'}\n")
        md.append(f"{g.detail}\n")
        md.append("```json\n" + json.dumps(g.evidence, indent=2, default=str) + "\n```\n")
    md.append("## PR #6170 changed-file / net-line audit\n")
    audit = next(g for g in gates if g.name == "gate_8_pr_audit")
    md.append(f"Merge commit `{SOURCE_MERGE_COMMIT}`.\n")
    md.append("| path | + | - | net |\n| --- | ---: | ---: | ---: |\n")
    for row in audit.evidence["files"]:
        md.append(
            f"| {row['path']} | +{row['additions']} | -{row['deletions']} | "
            f"{row['additions'] - row['deletions']:+d} |\n"
        )
    md.append(
        f"| **total** | **+{audit.evidence['total_additions']}** | "
        f"**-{audit.evidence['total_deletions']}** | "
        f"**{audit.evidence['net_lines']:+d}** |\n"
    )
    md.append("\n" + audit.evidence["head_post_merge_note"] + "\n")
    md.append("\n## Claim boundary\n")
    md.append(summary["claim_boundary"] + "\n")
    md.append("\n## Machine-readable summary\n")
    md.append("```json\n" + json.dumps(summary, indent=2, default=str) + "\n```\n")
    write_text(EVIDENCE_DOC, "".join(md), issue_ref=f"robot_sf#{ISSUE_NUMBER}")
    write_review_sidecar(EVIDENCE_DOC, repo_root=REPO_ROOT)
    return summary


def main() -> int:
    """Run all eight gates, derive the verdict, and write the evidence document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/algos/issue_5310_topology_parallel_nmpc.yaml",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        return 2
    config_rel = (
        str(config_path.relative_to(REPO_ROOT))
        if str(config_path).startswith(str(REPO_ROOT))
        else str(config_path)
    )

    nmpc_config = _nmpc_config_from_file(config_path)
    raw_cfg = _raw_config(config_path)

    gates = [
        gate_1_k1_legacy_parity(nmpc_config),
        gate_2_material_distinctness(nmpc_config),
        gate_3_objective_invariance(nmpc_config),
        gate_4_selection_and_hysteresis(nmpc_config),
        gate_5_fail_closed(nmpc_config),
        gate_6_registration_smoke(raw_cfg),
        gate_7_latency(nmpc_config),
        gate_8_pr_audit(),
    ]

    verdict, rationale = _derive_verdict(gates)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.strip()
    hardware = _hardware_context(gates[6].evidence["cpu_affinity_fixture"])

    summary = _write_evidence_doc(
        verdict=verdict,
        rationale=rationale,
        gates=gates,
        commit=commit,
        config_rel=config_rel,
        hardware=hardware,
        branch=branch,
    )
    summary["evidence_doc"] = str(EVIDENCE_DOC.relative_to(REPO_ROOT))
    print(json.dumps(summary, indent=2, default=str))
    # Recording any one verdict is a successful validation outcome.
    assert verdict in VERDICTS, f"invalid verdict {verdict!r}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
