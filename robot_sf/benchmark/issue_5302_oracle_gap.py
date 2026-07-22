"""Issue #5302 selection ceilings and hierarchical uncertainty analysis module.

Computes the four selection ceilings (best-fixed, family-best, cell-best,
hindsight-oracle), paired gaps, hierarchical bootstrap uncertainty, Pareto
dominance probability, normalized regret, and pre-registered claim gate from
complete native six-arm benchmark rows.

Fail-closed contract enforcement:
- Incomplete six-arm episodes, non-native rows, invalid row_status, duplicate keys,
  or split leakage block output.
- Safety constraint ordering prevents completion gains from compensating safety violations.
- Never emit 'universally best'.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PACKET_PATH = Path("configs/analysis/issue_5302_oracle_gap_packet.yaml")
SCHEMA_VERSION = "issue_5302_oracle_gap_analysis_packet.v1"
ISSUE_NUMBER = 5302

EXPECTED_PLANNERS: tuple[str, ...] = (
    "orca",
    "ppo",
    "prediction_planner",
    "scenario_adaptive_hybrid_orca_v1",
    "prediction_mpc",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
)

REQUIRED_ROW_FIELDS: tuple[str, ...] = (
    "episode_id",
    "scenario_id",
    "scenario_family",
    "scenario_cell",
    "split",
    "seed",
    "planner_id",
    "row_status",
    "execution_mode",
    "config_hash",
    "repo_commit",
    "selection_score",
    "collision_rate",
    "severe_intrusion_rate",
    "completion_rate",
    "timeout_rate",
    "tail_clearance",
    "jerk",
    "pedestrian_disturbance",
    "compute_time_ms",
)

REQUIRED_METRICS: tuple[str, ...] = (
    "collision_rate",
    "severe_intrusion_rate",
    "completion_rate",
    "timeout_rate",
    "tail_clearance",
    "worst_family_performance",
    "jerk",
    "pedestrian_disturbance",
    "compute_time_ms_p50",
    "compute_time_ms_p95",
    "compute_time_ms_p99",
)

CEILING_IDS: tuple[str, ...] = (
    "best_fixed_planner",
    "best_planner_per_scenario_family",
    "best_planner_per_scenario_cell",
    "hindsight_per_episode_oracle",
)

PRACTICAL_EQUIVALENCE_THRESHOLD = 0.02


class OracleGapAnalysisError(ValueError):
    """Base error for issue #5302 oracle gap analysis failures."""


class NonNativeRowError(OracleGapAnalysisError):
    """Raised when non-native execution mode rows are encountered."""


class InvalidRowStatusError(OracleGapAnalysisError):
    """Raised when row_status is not 'successful_evidence'."""


class SplitLeakageError(OracleGapAnalysisError):
    """Raised when selection and evaluation scenario families overlap."""


class IncompleteEpisodeError(OracleGapAnalysisError):
    """Raised when an episode does not contain complete 6-arm planner roster."""


class ProvenanceGapError(OracleGapAnalysisError):
    """Raised when required fields or metadata are missing or invalid."""


def planner_roster_index(planner_id: str) -> int:
    """Return 0-indexed position in EXPECTED_PLANNERS for tie-breaking.

    Returns:
        int: 0-based index of planner in EXPECTED_PLANNERS roster.
    """
    try:
        return EXPECTED_PLANNERS.index(planner_id)
    except ValueError:
        return 999


def safety_ordering_key(
    collision_rate: float,
    severe_intrusion_rate: float,
    selection_score: float,
    planner_id: str,
) -> tuple[float, float, float, int]:
    """Return a tuple key for lexicographical constraint-first comparison.

    Ordering:
      1. collision_rate (lower is better -> -collision_rate higher is better)
      2. severe_intrusion_rate (lower is better -> -severe_intrusion_rate higher is better)
      3. selection_score (higher is better)
      4. tie-breaker: roster index (earlier in roster -> -index higher is better)

    Returns:
        tuple[float, float, float, int]: Safety-ordered lexicographical key.
    """
    return (
        -float(collision_rate),
        -float(severe_intrusion_rate),
        float(selection_score),
        -planner_roster_index(planner_id),
    )


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Return percentile value from a sorted list.

    Returns:
        float: Computed percentile value.
    """
    if not sorted_vals:
        return math.nan
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _percentile_ci(sorted_vals: list[float], conf: float = 0.95) -> tuple[float, float]:
    """Return central confidence interval from sorted bootstrap values.

    Returns:
        tuple[float, float]: (low, high) confidence interval.
    """
    n = len(sorted_vals)
    if n == 0:
        return (math.nan, math.nan)
    alpha = 1.0 - conf
    low_idx = int(alpha / 2 * n)
    high_idx = min(n - 1, int((1 - alpha / 2) * n) - 1)
    high_idx = max(high_idx, low_idx)
    return (sorted_vals[low_idx], sorted_vals[high_idx])


def validate_rows_fail_closed(rows: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: C901
    """Validate input rows under fail-closed contract rules.

    Returns:
        dict[str, Any]: Preflight summary mapping if all checks pass.

    Raises:
        ProvenanceGapError: If required fields or rows are missing.
        NonNativeRowError: If non-native execution mode rows exist.
        InvalidRowStatusError: If invalid row_status rows exist.
        SplitLeakageError: If selection and evaluation families overlap.
        IncompleteEpisodeError: If episodes do not contain all 6 planners.
    """
    if not rows:
        raise ProvenanceGapError("Input row list is empty")

    required_set = set(REQUIRED_ROW_FIELDS)
    roster_set = set(EXPECTED_PLANNERS)

    non_native_rows = 0
    invalid_status_rows = 0

    for i, row in enumerate(rows):
        missing = required_set - set(row.keys())
        if missing:
            raise ProvenanceGapError(f"Row {i} is missing required fields: {sorted(missing)}")
        if str(row["execution_mode"]).strip().lower() != "native":
            non_native_rows += 1
        if str(row["row_status"]).strip().lower() != "successful_evidence":
            invalid_status_rows += 1

    if non_native_rows > 0:
        raise NonNativeRowError(
            f"Encountered {non_native_rows} non-native execution mode rows; "
            "only native rows are eligible evidence"
        )
    if invalid_status_rows > 0:
        raise InvalidRowStatusError(
            f"Encountered {invalid_status_rows} invalid row_status rows; "
            "only successful_evidence rows are eligible evidence"
        )

    # Check split leakage (disjoint scenario_family sets between selection and evaluation)
    selection_families = {
        str(r["scenario_family"]) for r in rows if str(r["split"]).strip().lower() == "selection"
    }
    evaluation_families = {
        str(r["scenario_family"]) for r in rows if str(r["split"]).strip().lower() == "evaluation"
    }
    overlap = selection_families & evaluation_families
    if overlap:
        raise SplitLeakageError(
            f"Split leakage detected: scenario families overlap between selection "
            f"and evaluation splits: {sorted(overlap)}"
        )

    # Group by episode_id and check 6-planner completeness
    episodes: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        ep_id = str(r["episode_id"])
        episodes.setdefault(ep_id, []).append(r)

    incomplete_episodes = 0
    for ep_id, ep_rows in episodes.items():
        planner_ids = [str(r["planner_id"]) for r in ep_rows]
        if len(planner_ids) != 6 or set(planner_ids) != roster_set:
            incomplete_episodes += 1
            raise IncompleteEpisodeError(
                f"Episode {ep_id} does not contain complete 6-arm roster. "
                f"Found planners: {sorted(planner_ids)}, expected: {EXPECTED_PLANNERS}"
            )

    found_planners = sorted({str(r["planner_id"]) for r in rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_NUMBER,
        "status": "ok",
        "total_input_rows": len(rows),
        "eligible_rows": len(rows),
        "excluded_rows": 0,
        "planner_roster": list(EXPECTED_PLANNERS),
        "planners_present": found_planners,
        "total_episodes": len(episodes),
        "complete_episodes": len(episodes),
        "incomplete_episodes": 0,
        "selection_families": sorted(selection_families),
        "evaluation_families": sorted(evaluation_families),
        "split_disjoint": True,
    }


def _select_fixed_planner(
    selection_rows: list[dict[str, Any]],
) -> str:
    """Select single best fixed planner based on selection_rows under safety ordering key.

    Returns:
        str: Planner ID of the selected fixed planner.
    """
    if not selection_rows:
        return EXPECTED_PLANNERS[0]
    roster_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in EXPECTED_PLANNERS}
    for r in selection_rows:
        pid = str(r["planner_id"])
        if pid in roster_rows:
            roster_rows[pid].append(r)

    best_planner = EXPECTED_PLANNERS[0]
    best_key = (-999.0, -999.0, -999.0, -999)

    for p in EXPECTED_PLANNERS:
        p_rows = roster_rows[p]
        if not p_rows:
            continue
        avg_col = float(sum(r["collision_rate"] for r in p_rows) / len(p_rows))
        avg_sev = float(sum(r["severe_intrusion_rate"] for r in p_rows) / len(p_rows))
        avg_score = float(sum(r["selection_score"] for r in p_rows) / len(p_rows))
        key = safety_ordering_key(avg_col, avg_sev, avg_score, p)
        if key > best_key:
            best_key = key
            best_planner = p

    return best_planner


def _select_family_planners(
    eval_episodes: list[list[dict[str, Any]]],
) -> dict[str, str]:
    """Select best planner per scenario family across evaluation episodes.

    Returns:
        dict[str, str]: Mapping from scenario_family to selected planner ID.
    """
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for ep in eval_episodes:
        for r in ep:
            fam = str(r["scenario_family"])
            family_rows.setdefault(fam, []).append(r)

    return {fam: _select_fixed_planner(f_rows) for fam, f_rows in family_rows.items()}


def _select_cell_planners(
    eval_episodes: list[list[dict[str, Any]]],
) -> dict[str, str]:
    """Select best planner per scenario cell across evaluation episodes.

    Returns:
        dict[str, str]: Mapping from scenario_cell to selected planner ID.
    """
    cell_rows: dict[str, list[dict[str, Any]]] = {}
    for ep in eval_episodes:
        for r in ep:
            cell = str(r["scenario_cell"])
            cell_rows.setdefault(cell, []).append(r)

    return {cell: _select_fixed_planner(c_rows) for cell, c_rows in cell_rows.items()}


@dataclass
class CeilingEpisodeSelection:
    """Selected planner rows for a single episode under all four ceilings."""

    episode_id: str
    scenario_family: str
    scenario_cell: str
    fixed_row: dict[str, Any]
    family_row: dict[str, Any]
    cell_row: dict[str, Any]
    oracle_row: dict[str, Any]


def evaluate_ceilings_on_episodes(
    eval_episodes: list[list[dict[str, Any]]],
    fixed_planner: str,
    family_planners: dict[str, str],
    cell_planners: dict[str, str],
) -> list[CeilingEpisodeSelection]:
    """Given evaluation episodes, compute ceiling row selections for each episode.

    Returns:
        list[CeilingEpisodeSelection]: List of episode ceiling selections.
    """
    results: list[CeilingEpisodeSelection] = []
    for ep_rows in eval_episodes:
        if not ep_rows:
            continue
        ep_id = str(ep_rows[0]["episode_id"])
        fam = str(ep_rows[0]["scenario_family"])
        cell = str(ep_rows[0]["scenario_cell"])

        by_planner = {str(r["planner_id"]): r for r in ep_rows}

        # 1. Best fixed planner
        fixed_p = fixed_planner if fixed_planner in by_planner else EXPECTED_PLANNERS[0]
        fixed_r = by_planner.get(fixed_p, ep_rows[0])

        # 2. Best planner per family
        fam_p = family_planners.get(fam, fixed_p)
        if fam_p not in by_planner:
            fam_p = _select_fixed_planner(ep_rows)
        fam_r = by_planner.get(fam_p, ep_rows[0])

        # 3. Best planner per cell
        cell_p = cell_planners.get(cell, fam_p)
        if cell_p not in by_planner:
            cell_p = _select_fixed_planner(ep_rows)
        cell_r = by_planner.get(cell_p, ep_rows[0])

        # 4. Hindsight per-episode oracle
        oracle_r = ep_rows[0]
        oracle_key = (-999.0, -999.0, -999.0, -999)
        for r in ep_rows:
            key = safety_ordering_key(
                float(r["collision_rate"]),
                float(r["severe_intrusion_rate"]),
                float(r["selection_score"]),
                str(r["planner_id"]),
            )
            if key > oracle_key:
                oracle_key = key
                oracle_r = r

        results.append(
            CeilingEpisodeSelection(
                episode_id=ep_id,
                scenario_family=fam,
                scenario_cell=cell,
                fixed_row=fixed_r,
                family_row=fam_r,
                cell_row=cell_r,
                oracle_row=oracle_r,
            )
        )

    return results


def summarize_ceiling_metrics(
    selections: list[CeilingEpisodeSelection],
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics for all four ceilings across episode selections.

    Returns:
        dict[str, dict[str, float]]: Mapping from ceiling ID to metric dictionary.
    """
    ceilings_data: dict[str, list[dict[str, Any]]] = {
        "best_fixed_planner": [s.fixed_row for s in selections],
        "best_planner_per_scenario_family": [s.family_row for s in selections],
        "best_planner_per_scenario_cell": [s.cell_row for s in selections],
        "hindsight_per_episode_oracle": [s.oracle_row for s in selections],
    }

    summary: dict[str, dict[str, float]] = {}

    for cid in CEILING_IDS:
        rows = ceilings_data[cid]
        if not rows:
            continue
        n = len(rows)
        col_rate = float(sum(r["collision_rate"] for r in rows) / n)
        sev_rate = float(sum(r["severe_intrusion_rate"] for r in rows) / n)
        comp_rate = float(sum(r["completion_rate"] for r in rows) / n)
        to_rate = float(sum(r["timeout_rate"] for r in rows) / n)
        score = float(sum(r["selection_score"] for r in rows) / n)
        tail_clr = float(sum(r["tail_clearance"] for r in rows) / n)
        jerk_val = float(sum(r["jerk"] for r in rows) / n)
        ped_dist = float(sum(r["pedestrian_disturbance"] for r in rows) / n)

        # compute worst_family_performance (min family-average selection_score)
        fam_scores: dict[str, list[float]] = {}
        for r in rows:
            fam_scores.setdefault(str(r["scenario_family"]), []).append(float(r["selection_score"]))
        fam_means = [sum(scores) / len(scores) for scores in fam_scores.values()]
        worst_fam = float(min(fam_means)) if fam_means else score

        # compute_time percentiles
        times = sorted([float(r["compute_time_ms"]) for r in rows])
        p50 = _percentile(times, 50)
        p95 = _percentile(times, 95)
        p99 = _percentile(times, 99)

        summary[cid] = {
            "selection_score": score,
            "collision_rate": col_rate,
            "severe_intrusion_rate": sev_rate,
            "completion_rate": comp_rate,
            "timeout_rate": to_rate,
            "tail_clearance": tail_clr,
            "worst_family_performance": worst_fam,
            "jerk": jerk_val,
            "pedestrian_disturbance": ped_dist,
            "compute_time_ms_p50": p50,
            "compute_time_ms_p95": p95,
            "compute_time_ms_p99": p99,
        }

    return summary


def run_hierarchical_bootstrap(  # noqa: C901
    eval_episodes_by_hierarchy: dict[str, dict[str, list[list[dict[str, Any]]]]],
    selection_rows: list[dict[str, Any]],
    n_samples: int = 1000,
    seed: int = 5302,
) -> dict[str, Any]:
    """Run 3-level (family -> cell -> episode) hierarchical bootstrap resampling.

    Returns:
        dict[str, Any]: Bootstrap distributions and 95% CIs for metrics and paired gaps.
    """
    rng = random.Random(seed)

    families = sorted(eval_episodes_by_hierarchy.keys())
    if not families:
        return {}

    # Identify pre-selected fixed/family/cell planners on base dataset
    eval_rows_flat = [
        r
        for fam_cells in eval_episodes_by_hierarchy.values()
        for cell_eps in fam_cells.values()
        for ep in cell_eps
        for r in ep
    ]
    sel_rows = selection_rows if selection_rows else eval_rows_flat

    base_fixed = _select_fixed_planner(sel_rows)
    base_family_map = _select_family_planners(
        [
            ep
            for fam_cells in eval_episodes_by_hierarchy.values()
            for cell_eps in fam_cells.values()
            for ep in cell_eps
        ]
    )
    base_cell_map = _select_cell_planners(
        [
            ep
            for fam_cells in eval_episodes_by_hierarchy.values()
            for cell_eps in fam_cells.values()
            for ep in cell_eps
        ]
    )

    # Storage for bootstrap replicates
    bootstrap_results: dict[str, list[float]] = {}

    for _ in range(n_samples):
        # 1. Resample families
        n_fam = len(families)
        resampled_fams = [families[rng.randrange(n_fam)] for _ in range(n_fam)]

        # 2. Resample cells & episodes
        boot_episodes: list[list[dict[str, Any]]] = []
        for fam in resampled_fams:
            cells_map = eval_episodes_by_hierarchy[fam]
            cells = sorted(cells_map.keys())
            n_cells = len(cells)
            resampled_cells = [cells[rng.randrange(n_cells)] for _ in range(n_cells)]
            for cell in resampled_cells:
                eps = cells_map[cell]
                n_eps = len(eps)
                resampled_eps = [eps[rng.randrange(n_eps)] for _ in range(n_eps)]
                boot_episodes.extend(resampled_eps)

        # Evaluate selections on boot_episodes
        selections = evaluate_ceilings_on_episodes(
            boot_episodes, base_fixed, base_family_map, base_cell_map
        )
        summary = summarize_ceiling_metrics(selections)

        # Store ceiling metrics
        for cid in CEILING_IDS:
            if cid in summary:
                for m_name, val in summary[cid].items():
                    key = f"ceiling.{cid}.{m_name}"
                    bootstrap_results.setdefault(key, []).append(val)

        # Compute paired gaps for selection_score & completion_rate
        fixed_summary = summary.get("best_fixed_planner", {})
        fixed_score = fixed_summary.get("selection_score", 0.0)
        fixed_comp = fixed_summary.get("completion_rate", 0.0)

        for cid, gap_name in (
            ("best_planner_per_scenario_family", "family_gap"),
            ("best_planner_per_scenario_cell", "cell_gap"),
            ("hindsight_per_episode_oracle", "oracle_gap"),
        ):
            c_sum = summary.get(cid, {})
            score_gap = c_sum.get("selection_score", 0.0) - fixed_score
            comp_gap = c_sum.get("completion_rate", 0.0) - fixed_comp
            bootstrap_results.setdefault(f"gap.{gap_name}.selection_score", []).append(score_gap)
            bootstrap_results.setdefault(f"gap.{gap_name}.completion_rate", []).append(comp_gap)

        # Store individual planner metrics
        planner_ep_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in EXPECTED_PLANNERS}
        for ep in boot_episodes:
            for r in ep:
                pid = str(r["planner_id"])
                if pid in planner_ep_rows:
                    planner_ep_rows[pid].append(r)

        for pid, p_rows in planner_ep_rows.items():
            if not p_rows:
                continue
            n_p = len(p_rows)
            p_comp = float(sum(r["completion_rate"] for r in p_rows) / n_p)
            p_col = float(sum(r["collision_rate"] for r in p_rows) / n_p)
            p_score = float(sum(r["selection_score"] for r in p_rows) / n_p)
            bootstrap_results.setdefault(f"planner.{pid}.completion_rate", []).append(p_comp)
            bootstrap_results.setdefault(f"planner.{pid}.collision_rate", []).append(p_col)
            bootstrap_results.setdefault(f"planner.{pid}.selection_score", []).append(p_score)

    # Compute summary intervals
    cis: dict[str, dict[str, Any]] = {}
    for key, vals in bootstrap_results.items():
        s_vals = sorted(vals)
        low, high = _percentile_ci(s_vals, 0.95)
        mean_val = float(sum(s_vals) / len(s_vals))
        cis[key] = {
            "point_estimate": mean_val,
            "mean": mean_val,
            "ci_95": [float(low), float(high)],
            "ci_low": float(low),
            "ci_high": float(high),
        }

    return cis


def compute_pareto_dominance(
    planner_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute Pareto dominance relation matrix among planners and ceilings.

    Returns:
        dict[str, Any]: Mapping containing entities and pareto dominance matrix.
    """
    entities = list(planner_metrics.keys())
    matrix: dict[str, dict[str, bool]] = {}

    for e1 in entities:
        matrix[e1] = {}
        m1 = planner_metrics[e1]
        v1 = (
            m1.get("completion_rate", 0.0),
            -m1.get("collision_rate", 1.0),
            -m1.get("severe_intrusion_rate", 1.0),
            m1.get("tail_clearance", 0.0),
        )
        for e2 in entities:
            if e1 == e2:
                matrix[e1][e2] = False
                continue
            m2 = planner_metrics[e2]
            v2 = (
                m2.get("completion_rate", 0.0),
                -m2.get("collision_rate", 1.0),
                -m2.get("severe_intrusion_rate", 1.0),
                m2.get("tail_clearance", 0.0),
            )
            # e1 dominates e2 iff v1 >= v2 elementwise and v1 > v2 in at least one element
            ge = all(v1[k] >= v2[k] for k in range(4))
            gt = any(v1[k] > v2[k] for k in range(4))
            matrix[e1][e2] = ge and gt

    return {
        "schema_version": SCHEMA_VERSION,
        "entities": entities,
        "pareto_dominance_matrix": matrix,
    }


def compute_normalized_regret(
    eval_episodes: list[list[dict[str, Any]]],
    selections: list[CeilingEpisodeSelection],
) -> list[dict[str, Any]]:
    """Compute normalized regret for each planner relative to hindsight oracle.

    Returns:
        list[dict[str, Any]]: Normalized regret rows per planner.
    """
    roster_regret: dict[str, list[float]] = {p: [] for p in EXPECTED_PLANNERS}

    for s in selections:
        oracle_score = float(s.oracle_row["selection_score"])
        ep_id = s.episode_id
        # find episode in eval_episodes
        ep_rows = [
            r for ep in eval_episodes if ep and str(ep[0]["episode_id"]) == ep_id for r in ep
        ]
        for r in ep_rows:
            pid = str(r["planner_id"])
            if pid in roster_regret:
                score = float(r["selection_score"])
                regret = max(0.0, oracle_score - score)
                roster_regret[pid].append(regret)

    results: list[dict[str, Any]] = []
    for pid in EXPECTED_PLANNERS:
        regs = sorted(roster_regret[pid])
        if not regs:
            continue
        n = len(regs)
        results.append(
            {
                "planner_id": pid,
                "count": n,
                "mean_normalized_regret": float(sum(regs) / n),
                "p50_normalized_regret": _percentile(regs, 50),
                "p95_normalized_regret": _percentile(regs, 95),
                "max_normalized_regret": float(max(regs)),
            }
        )

    return results


def compute_claim_gate(
    bootstrap_cis: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate pre-registered claim gate based on held-out hierarchical intervals.

    Returns:
        dict[str, Any]: Claim gate decision rule status, thresholds, and rationale.
    """
    family_gap_ci = bootstrap_cis.get(
        "gap.family_gap.selection_score", {"ci_low": 0.0, "ci_high": 0.0}
    )
    cell_gap_ci = bootstrap_cis.get("gap.cell_gap.selection_score", {"ci_low": 0.0, "ci_high": 0.0})

    fam_low = float(family_gap_ci.get("ci_low", 0.0))
    cell_low = float(cell_gap_ci.get("ci_low", 0.0))

    if fam_low <= PRACTICAL_EQUIVALENCE_THRESHOLD or cell_low <= PRACTICAL_EQUIVALENCE_THRESHOLD:
        status = "STOP_SELECTOR"
        rationale = (
            f"Family gap 95% CI lower bound ({fam_low:.4f}) or cell gap lower bound "
            f"({cell_low:.4f}) is within the practical-equivalence band "
            f"([0.0, {PRACTICAL_EQUIVALENCE_THRESHOLD:.2f}]). Stop selector work "
            "and invest in the fixed planner."
        )
    else:
        status = "PROCEED_TO_SELECTOR_ISSUE"
        rationale = (
            f"Family gap 95% CI lower bound ({fam_low:.4f}) and cell gap lower bound "
            f"({cell_low:.4f}) exceed the practical-equivalence band "
            f"({PRACTICAL_EQUIVALENCE_THRESHOLD:.2f}). Open a separate issue for a "
            "scenario-conditioned selector."
        )

    return {
        "status": status,
        "practical_equivalence_threshold": PRACTICAL_EQUIVALENCE_THRESHOLD,
        "family_gap_ci_lower": fam_low,
        "cell_gap_ci_lower": cell_low,
        "universally_best_emitted": False,
        "rationale": rationale,
    }


def run_full_oracle_gap_analysis(
    rows: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 5302,
) -> dict[str, Any]:
    """Execute complete issue #5302 oracle gap analysis and return report structures.

    Returns:
        dict[str, Any]: Complete analysis outputs including preflight, ceilings, CIs, Pareto, regret, and claim gate.
    """
    preflight = validate_rows_fail_closed(rows)

    # Split rows into selection vs evaluation
    selection_rows = [r for r in rows if str(r["split"]).strip().lower() == "selection"]
    evaluation_rows = [r for r in rows if str(r["split"]).strip().lower() == "evaluation"]

    # If no evaluation split present, use all rows as evaluation
    if not evaluation_rows:
        evaluation_rows = list(rows)

    # Build evaluation episode hierarchy: family -> cell -> list of episodes (each 6 rows)
    eval_episodes_map: dict[str, list[dict[str, Any]]] = {}
    for r in evaluation_rows:
        ep_id = str(r["episode_id"])
        eval_episodes_map.setdefault(ep_id, []).append(r)

    eval_episodes = list(eval_episodes_map.values())

    eval_hierarchy: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}
    for ep in eval_episodes:
        fam = str(ep[0]["scenario_family"])
        cell = str(ep[0]["scenario_cell"])
        eval_hierarchy.setdefault(fam, {}).setdefault(cell, []).append(ep)

    # Compute base ceiling selections
    sel_input = selection_rows if selection_rows else evaluation_rows
    fixed_planner = _select_fixed_planner(sel_input)
    family_planners = _select_family_planners(eval_episodes)
    cell_planners = _select_cell_planners(eval_episodes)

    selections = evaluate_ceilings_on_episodes(
        eval_episodes, fixed_planner, family_planners, cell_planners
    )
    base_ceiling_summary = summarize_ceiling_metrics(selections)

    # Hierarchical bootstrap
    bootstrap_cis = run_hierarchical_bootstrap(
        eval_hierarchy, selection_rows, n_samples=n_bootstrap, seed=seed
    )

    # Per-planner metrics on evaluation set
    planner_eval_rows: dict[str, list[dict[str, Any]]] = {p: [] for p in EXPECTED_PLANNERS}
    for r in evaluation_rows:
        pid = str(r["planner_id"])
        if pid in planner_eval_rows:
            planner_eval_rows[pid].append(r)

    planner_metrics: dict[str, dict[str, float]] = {}
    for pid in EXPECTED_PLANNERS:
        p_rows = planner_eval_rows[pid]
        if not p_rows:
            continue
        n_p = len(p_rows)
        planner_metrics[pid] = {
            "selection_score": float(sum(r["selection_score"] for r in p_rows) / n_p),
            "collision_rate": float(sum(r["collision_rate"] for r in p_rows) / n_p),
            "severe_intrusion_rate": float(sum(r["severe_intrusion_rate"] for r in p_rows) / n_p),
            "completion_rate": float(sum(r["completion_rate"] for r in p_rows) / n_p),
            "timeout_rate": float(sum(r["timeout_rate"] for r in p_rows) / n_p),
            "tail_clearance": float(sum(r["tail_clearance"] for r in p_rows) / n_p),
        }

    # Include ceilings in planner_metrics for Pareto analysis
    entity_metrics = dict(planner_metrics)
    entity_metrics.update(base_ceiling_summary)

    pareto_data = compute_pareto_dominance(entity_metrics)
    normalized_regret_rows = compute_normalized_regret(eval_episodes, selections)
    claim_gate = compute_claim_gate(bootstrap_cis)

    return {
        "preflight": preflight,
        "best_fixed_planner": fixed_planner,
        "ceiling_summary": base_ceiling_summary,
        "bootstrap_intervals": bootstrap_cis,
        "pareto_dominance": pareto_data,
        "normalized_regret": normalized_regret_rows,
        "claim_gate": claim_gate,
        "selections": selections,
        "eval_episodes": eval_episodes,
        "evaluation_rows": evaluation_rows,
    }


def write_report_artifacts(  # noqa: C901, PLR0915, PLR0912
    analysis_result: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Write all 10 required report files to output_dir and return list of written paths.

    Returns:
        list[Path]: List of absolute or relative paths to written report files.
    """
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    # 1. reports/preflight.json
    preflight_path = reports_dir / "preflight.json"
    preflight_path.write_text(json.dumps(analysis_result["preflight"], indent=2), encoding="utf-8")
    written.append(preflight_path)

    # 2. reports/ceiling_summary.json
    ceiling_json_path = reports_dir / "ceiling_summary.json"
    bs_cis = analysis_result["bootstrap_intervals"]
    ceiling_summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_NUMBER,
        "best_fixed_planner_id": analysis_result["best_fixed_planner"],
        "ceilings": {},
        "gaps": {
            "family_gap": {
                "selection_score_delta": bs_cis.get("gap.family_gap.selection_score", {}).get(
                    "point_estimate", 0.0
                ),
                "selection_score_delta_ci": bs_cis.get("gap.family_gap.selection_score", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
                "completion_rate_delta": bs_cis.get("gap.family_gap.completion_rate", {}).get(
                    "point_estimate", 0.0
                ),
                "completion_rate_delta_ci": bs_cis.get("gap.family_gap.completion_rate", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
            },
            "cell_gap": {
                "selection_score_delta": bs_cis.get("gap.cell_gap.selection_score", {}).get(
                    "point_estimate", 0.0
                ),
                "selection_score_delta_ci": bs_cis.get("gap.cell_gap.selection_score", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
                "completion_rate_delta": bs_cis.get("gap.cell_gap.completion_rate", {}).get(
                    "point_estimate", 0.0
                ),
                "completion_rate_delta_ci": bs_cis.get("gap.cell_gap.completion_rate", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
            },
            "oracle_gap": {
                "selection_score_delta": bs_cis.get("gap.oracle_gap.selection_score", {}).get(
                    "point_estimate", 0.0
                ),
                "selection_score_delta_ci": bs_cis.get("gap.oracle_gap.selection_score", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
                "completion_rate_delta": bs_cis.get("gap.oracle_gap.completion_rate", {}).get(
                    "point_estimate", 0.0
                ),
                "completion_rate_delta_ci": bs_cis.get("gap.oracle_gap.completion_rate", {}).get(
                    "ci_95", [0.0, 0.0]
                ),
            },
        },
        "claim_gate": analysis_result["claim_gate"],
    }
    for cid, metrics in analysis_result["ceiling_summary"].items():
        ceiling_summary_payload["ceilings"][cid] = {
            **metrics,
            "selection_score_ci": bs_cis.get(f"ceiling.{cid}.selection_score", {}).get(
                "ci_95", [0.0, 0.0]
            ),
            "collision_rate_ci": bs_cis.get(f"ceiling.{cid}.collision_rate", {}).get(
                "ci_95", [0.0, 0.0]
            ),
        }
    ceiling_json_path.write_text(json.dumps(ceiling_summary_payload, indent=2), encoding="utf-8")
    written.append(ceiling_json_path)

    # 3. reports/ceiling_summary.csv
    ceiling_csv_path = reports_dir / "ceiling_summary.csv"
    with ceiling_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "estimand",
                "planner_id",
                "selection_score",
                "selection_score_ci_low",
                "selection_score_ci_high",
                "collision_rate",
                "severe_intrusion_rate",
                "completion_rate",
                "timeout_rate",
                "tail_clearance",
                "worst_family_performance",
                "jerk",
                "pedestrian_disturbance",
                "compute_time_ms_p50",
                "compute_time_ms_p95",
                "compute_time_ms_p99",
            ]
        )
        for cid in CEILING_IDS:
            m = analysis_result["ceiling_summary"].get(cid, {})
            score_ci = bs_cis.get(f"ceiling.{cid}.selection_score", {}).get("ci_95", [0.0, 0.0])
            pid = analysis_result["best_fixed_planner"] if cid == "best_fixed_planner" else cid
            writer.writerow(
                [
                    cid,
                    pid,
                    m.get("selection_score", 0.0),
                    score_ci[0],
                    score_ci[1],
                    m.get("collision_rate", 0.0),
                    m.get("severe_intrusion_rate", 0.0),
                    m.get("completion_rate", 0.0),
                    m.get("timeout_rate", 0.0),
                    m.get("tail_clearance", 0.0),
                    m.get("worst_family_performance", 0.0),
                    m.get("jerk", 0.0),
                    m.get("pedestrian_disturbance", 0.0),
                    m.get("compute_time_ms_p50", 0.0),
                    m.get("compute_time_ms_p95", 0.0),
                    m.get("compute_time_ms_p99", 0.0),
                ]
            )
    written.append(ceiling_csv_path)

    # 4. reports/family_breakdown.csv
    family_csv_path = reports_dir / "family_breakdown.csv"
    eval_rows = analysis_result["evaluation_rows"]
    fams = sorted({str(r["scenario_family"]) for r in eval_rows})
    with family_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario_family",
                "entity_type",
                "entity_id",
                "episodes",
                "selection_score",
                "collision_rate",
                "severe_intrusion_rate",
                "completion_rate",
                "timeout_rate",
                "tail_clearance",
                "jerk",
                "pedestrian_disturbance",
                "compute_time_ms_p50",
                "compute_time_ms_p95",
                "compute_time_ms_p99",
            ]
        )

        # Per planner per family
        for fam in fams:
            fam_rows = [r for r in eval_rows if str(r["scenario_family"]) == fam]
            for pid in EXPECTED_PLANNERS:
                p_fam_rows = [r for r in fam_rows if str(r["planner_id"]) == pid]
                if not p_fam_rows:
                    continue
                n_ep = len(p_fam_rows)
                times = sorted([float(r["compute_time_ms"]) for r in p_fam_rows])
                writer.writerow(
                    [
                        fam,
                        "planner",
                        pid,
                        n_ep,
                        sum(float(r["selection_score"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["collision_rate"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["severe_intrusion_rate"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["completion_rate"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["timeout_rate"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["tail_clearance"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["jerk"]) for r in p_fam_rows) / n_ep,
                        sum(float(r["pedestrian_disturbance"]) for r in p_fam_rows) / n_ep,
                        _percentile(times, 50),
                        _percentile(times, 95),
                        _percentile(times, 99),
                    ]
                )

        # Ceilings per family
        selections: list[CeilingEpisodeSelection] = analysis_result["selections"]
        for fam in fams:
            fam_sels = [s for s in selections if s.scenario_family == fam]
            if not fam_sels:
                continue
            n_ep = len(fam_sels)
            for cid in CEILING_IDS:
                attr = (
                    "fixed_row"
                    if cid == "best_fixed_planner"
                    else (
                        "family_row"
                        if cid == "best_planner_per_scenario_family"
                        else "cell_row"
                        if cid == "best_planner_per_scenario_cell"
                        else "oracle_row"
                    )
                )
                c_rows = [getattr(s, attr) for s in fam_sels]
                times = sorted([float(r["compute_time_ms"]) for r in c_rows])
                writer.writerow(
                    [
                        fam,
                        "ceiling",
                        cid,
                        n_ep,
                        sum(float(r["selection_score"]) for r in c_rows) / n_ep,
                        sum(float(r["collision_rate"]) for r in c_rows) / n_ep,
                        sum(float(r["severe_intrusion_rate"]) for r in c_rows) / n_ep,
                        sum(float(r["completion_rate"]) for r in c_rows) / n_ep,
                        sum(float(r["timeout_rate"]) for r in c_rows) / n_ep,
                        sum(float(r["tail_clearance"]) for r in c_rows) / n_ep,
                        sum(float(r["jerk"]) for r in c_rows) / n_ep,
                        sum(float(r["pedestrian_disturbance"]) for r in c_rows) / n_ep,
                        _percentile(times, 50),
                        _percentile(times, 95),
                        _percentile(times, 99),
                    ]
                )
    written.append(family_csv_path)

    # 5. reports/cell_breakdown.csv
    cell_csv_path = reports_dir / "cell_breakdown.csv"
    cells = sorted({(str(r["scenario_family"]), str(r["scenario_cell"])) for r in eval_rows})
    with cell_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario_family",
                "scenario_cell",
                "entity_type",
                "entity_id",
                "episodes",
                "selection_score",
                "collision_rate",
                "severe_intrusion_rate",
                "completion_rate",
                "timeout_rate",
                "tail_clearance",
                "jerk",
                "pedestrian_disturbance",
                "compute_time_ms_p50",
                "compute_time_ms_p95",
                "compute_time_ms_p99",
            ]
        )
        for fam, cell in cells:
            cell_rows = [
                r
                for r in eval_rows
                if str(r["scenario_family"]) == fam and str(r["scenario_cell"]) == cell
            ]
            for pid in EXPECTED_PLANNERS:
                p_cell_rows = [r for r in cell_rows if str(r["planner_id"]) == pid]
                if not p_cell_rows:
                    continue
                n_ep = len(p_cell_rows)
                times = sorted([float(r["compute_time_ms"]) for r in p_cell_rows])
                writer.writerow(
                    [
                        fam,
                        cell,
                        "planner",
                        pid,
                        n_ep,
                        sum(float(r["selection_score"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["collision_rate"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["severe_intrusion_rate"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["completion_rate"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["timeout_rate"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["tail_clearance"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["jerk"]) for r in p_cell_rows) / n_ep,
                        sum(float(r["pedestrian_disturbance"]) for r in p_cell_rows) / n_ep,
                        _percentile(times, 50),
                        _percentile(times, 95),
                        _percentile(times, 99),
                    ]
                )
    written.append(cell_csv_path)

    # 6. reports/failure_mechanism_map.csv
    fail_csv_path = reports_dir / "failure_mechanism_map.csv"
    with fail_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario_family",
                "entity_id",
                "total_episodes",
                "success_count",
                "success_rate",
                "collision_count",
                "collision_rate",
                "severe_intrusion_count",
                "severe_intrusion_rate",
                "timeout_count",
                "timeout_rate",
            ]
        )
        for fam in fams:
            fam_rows = [r for r in eval_rows if str(r["scenario_family"]) == fam]
            for pid in EXPECTED_PLANNERS:
                p_rows = [r for r in fam_rows if str(r["planner_id"]) == pid]
                if not p_rows:
                    continue
                n_ep = len(p_rows)
                col_cnt = sum(1 for r in p_rows if float(r["collision_rate"]) > 0)
                sev_cnt = sum(
                    1
                    for r in p_rows
                    if float(r["severe_intrusion_rate"]) > 0 and float(r["collision_rate"]) == 0
                )
                to_cnt = sum(
                    1
                    for r in p_rows
                    if float(r["timeout_rate"]) > 0
                    and float(r["collision_rate"]) == 0
                    and float(r["severe_intrusion_rate"]) == 0
                )
                succ_cnt = sum(
                    1
                    for r in p_rows
                    if float(r["completion_rate"]) == 1.0
                    and float(r["collision_rate"]) == 0
                    and float(r["severe_intrusion_rate"]) == 0
                )
                writer.writerow(
                    [
                        fam,
                        pid,
                        n_ep,
                        succ_cnt,
                        succ_cnt / n_ep,
                        col_cnt,
                        col_cnt / n_ep,
                        sev_cnt,
                        sev_cnt / n_ep,
                        to_cnt,
                        to_cnt / n_ep,
                    ]
                )
    written.append(fail_csv_path)

    # 7. reports/runtime_tail.csv
    runtime_csv_path = reports_dir / "runtime_tail.csv"
    with runtime_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "entity_id",
                "count",
                "p50_ms",
                "p90_ms",
                "p95_ms",
                "p99_ms",
                "max_ms",
            ]
        )
        for pid in EXPECTED_PLANNERS:
            p_rows = [r for r in eval_rows if str(r["planner_id"]) == pid]
            if not p_rows:
                continue
            times = sorted([float(r["compute_time_ms"]) for r in p_rows])
            n = len(times)
            writer.writerow(
                [
                    pid,
                    n,
                    _percentile(times, 50),
                    _percentile(times, 90),
                    _percentile(times, 95),
                    _percentile(times, 99),
                    max(times),
                ]
            )
        for cid in CEILING_IDS:
            attr = (
                "fixed_row"
                if cid == "best_fixed_planner"
                else (
                    "family_row"
                    if cid == "best_planner_per_scenario_family"
                    else "cell_row"
                    if cid == "best_planner_per_scenario_cell"
                    else "oracle_row"
                )
            )
            c_rows = [getattr(s, attr) for s in analysis_result["selections"]]
            times = sorted([float(r["compute_time_ms"]) for r in c_rows])
            n = len(times)
            writer.writerow(
                [
                    cid,
                    n,
                    _percentile(times, 50),
                    _percentile(times, 90),
                    _percentile(times, 95),
                    _percentile(times, 99),
                    max(times),
                ]
            )
    written.append(runtime_csv_path)

    # 8. reports/pareto_dominance.json
    pareto_json_path = reports_dir / "pareto_dominance.json"
    pareto_json_path.write_text(
        json.dumps(analysis_result["pareto_dominance"], indent=2), encoding="utf-8"
    )
    written.append(pareto_json_path)

    # 9. reports/normalized_regret.csv
    regret_csv_path = reports_dir / "normalized_regret.csv"
    with regret_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "planner_id",
                "episodes",
                "mean_normalized_regret",
                "p50_normalized_regret",
                "p95_normalized_regret",
                "max_normalized_regret",
            ]
        )
        for row in analysis_result["normalized_regret"]:
            writer.writerow(
                [
                    row["planner_id"],
                    row["count"],
                    row["mean_normalized_regret"],
                    row["p50_normalized_regret"],
                    row["p95_normalized_regret"],
                    row["max_normalized_regret"],
                ]
            )
    written.append(regret_csv_path)

    # 10. reports/bootstrap_intervals.json
    bs_json_path = reports_dir / "bootstrap_intervals.json"
    bs_json_path.write_text(
        json.dumps(analysis_result["bootstrap_intervals"], indent=2), encoding="utf-8"
    )
    written.append(bs_json_path)

    return written
