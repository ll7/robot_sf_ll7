"""Tests for issue #5302 selection ceilings and hierarchical uncertainty analysis."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path  # noqa: TC003
from typing import Any

import pytest

from robot_sf.benchmark.issue_5302_oracle_gap import (
    EXPECTED_PLANNERS,
    IncompleteEpisodeError,
    InvalidRowStatusError,
    NonNativeRowError,
    SplitLeakageError,
    compute_claim_gate,
    run_full_oracle_gap_analysis,
    safety_ordering_key,
    validate_rows_fail_closed,
)


def _make_synthetic_six_arm_dataset(
    n_families: int = 2,
    cells_per_family: int = 2,
    episodes_per_cell: int = 3,
    with_selection_split: bool = True,
) -> list[dict[str, Any]]:
    """Helper to generate a synthetic six-arm benchmark dataset for testing."""
    rows: list[dict[str, Any]] = []

    families = [f"family_{i}" for i in range(n_families)]
    if with_selection_split:
        # First half of families in selection split, second half in evaluation split
        mid = max(1, n_families // 2)
        sel_families = families[:mid]
        eval_families = families[mid:]
        if not eval_families:
            eval_families = [f"eval_family_{i}" for i in range(n_families)]
    else:
        sel_families = []
        eval_families = families

    all_specs = []
    for f in sel_families:
        all_specs.append((f, "selection"))
    for f in eval_families:
        all_specs.append((f, "evaluation"))

    ep_counter = 0

    for fam, split in all_specs:
        for c in range(cells_per_family):
            cell = f"{fam}_cell_{c}"
            for e in range(episodes_per_cell):
                ep_id = f"ep_{ep_counter}"
                ep_counter += 1
                seed = 1000 + ep_counter
                for p_idx, planner_id in enumerate(EXPECTED_PLANNERS):
                    # Introduce deterministic metric differences per planner
                    base_score = 0.70 + 0.04 * p_idx
                    col_rate = 0.0
                    sev_rate = 0.0
                    comp_rate = 0.90 + 0.01 * p_idx
                    to_rate = 0.10 - 0.01 * p_idx

                    # Add minor collision/intrusion to planner 0 for testing safety keys
                    if planner_id == "orca" and ep_counter % 3 == 0:
                        col_rate = 1.0

                    rows.append(
                        {
                            "episode_id": ep_id,
                            "scenario_id": f"{cell}_scen_{e}",
                            "scenario_family": fam,
                            "scenario_cell": cell,
                            "split": split,
                            "seed": seed,
                            "planner_id": planner_id,
                            "row_status": "successful_evidence",
                            "execution_mode": "native",
                            "config_hash": f"hash_{planner_id}",
                            "repo_commit": "abcdef1234567890abcdef1234567890abcdef12",
                            "selection_score": base_score,
                            "collision_rate": col_rate,
                            "severe_intrusion_rate": sev_rate,
                            "completion_rate": comp_rate,
                            "timeout_rate": to_rate,
                            "tail_clearance": 0.85,
                            "jerk": 1.2,
                            "pedestrian_disturbance": 0.05,
                            "compute_time_ms": 10.0 + 5.0 * p_idx,
                        }
                    )

    return rows


def test_estimator_ceilings_and_nesting() -> None:
    """Verify that selection ceilings satisfy nesting: fixed <= family <= cell <= oracle."""
    rows = _make_synthetic_six_arm_dataset(
        n_families=4, cells_per_family=2, episodes_per_cell=3, with_selection_split=True
    )
    res = run_full_oracle_gap_analysis(rows, n_bootstrap=50, seed=5302)

    ceilings = res["ceiling_summary"]
    fixed_score = ceilings["best_fixed_planner"]["selection_score"]
    family_score = ceilings["best_planner_per_scenario_family"]["selection_score"]
    cell_score = ceilings["best_planner_per_scenario_cell"]["selection_score"]
    oracle_score = ceilings["hindsight_per_episode_oracle"]["selection_score"]

    assert fixed_score <= family_score + 1e-9
    assert family_score <= cell_score + 1e-9
    assert cell_score <= oracle_score + 1e-9


def test_bootstrap_determinism_and_seed() -> None:
    """Verify that hierarchical bootstrap returns identical results given the same seed."""
    rows = _make_synthetic_six_arm_dataset(
        n_families=4, cells_per_family=2, episodes_per_cell=3, with_selection_split=True
    )

    res1 = run_full_oracle_gap_analysis(rows, n_bootstrap=100, seed=5302)
    res2 = run_full_oracle_gap_analysis(rows, n_bootstrap=100, seed=5302)

    cis1 = res1["bootstrap_intervals"]
    cis2 = res2["bootstrap_intervals"]

    assert cis1.keys() == cis2.keys()
    for k in cis1:
        assert cis1[k]["ci_95"] == cis2[k]["ci_95"]
        assert pytest.approx(cis1[k]["mean"]) == cis2[k]["mean"]


def test_missing_arm_fails_closed() -> None:
    """Verify that incomplete episodes (< 6 arms) raise IncompleteEpisodeError."""
    rows = _make_synthetic_six_arm_dataset(n_families=2, with_selection_split=False)

    # Omit one planner row from the first episode
    ep0_id = rows[0]["episode_id"]
    rows = [r for r in rows if not (r["episode_id"] == ep0_id and r["planner_id"] == "ppo")]

    with pytest.raises(IncompleteEpisodeError, match="does not contain complete 6-arm roster"):
        validate_rows_fail_closed(rows)


def test_split_leakage_fails_closed() -> None:
    """Verify that overlapping scenario families between selection and evaluation fail closed."""
    rows = _make_synthetic_six_arm_dataset(n_families=2, with_selection_split=True)

    # Force a scenario family to be in both selection and evaluation split
    fam0 = rows[0]["scenario_family"]
    for r in rows:
        if r["split"] == "evaluation":
            r["scenario_family"] = fam0

    with pytest.raises(SplitLeakageError, match="Split leakage detected"):
        validate_rows_fail_closed(rows)


def test_non_native_row_fails_closed() -> None:
    """Verify that non-native execution mode rows raise NonNativeRowError."""
    rows = _make_synthetic_six_arm_dataset(n_families=2, with_selection_split=False)
    rows[0]["execution_mode"] = "adapter"

    with pytest.raises(NonNativeRowError, match="non-native execution mode rows"):
        validate_rows_fail_closed(rows)


def test_invalid_row_status_fails_closed() -> None:
    """Verify that invalid row_status raises InvalidRowStatusError."""
    rows = _make_synthetic_six_arm_dataset(n_families=2, with_selection_split=False)
    rows[0]["row_status"] = "diagnostic_only"

    with pytest.raises(InvalidRowStatusError, match="invalid row_status rows"):
        validate_rows_fail_closed(rows)


def test_safety_constraint_ordering() -> None:
    """Verify that a planner with collisions is ranked below a collision-free planner regardless of score."""
    # Key for Planner A: collision=1.0, score=0.99
    key_a = safety_ordering_key(
        collision_rate=1.0, severe_intrusion_rate=0.0, selection_score=0.99, planner_id="ppo"
    )
    # Key for Planner B: collision=0.0, score=0.70
    key_b = safety_ordering_key(
        collision_rate=0.0, severe_intrusion_rate=0.0, selection_score=0.70, planner_id="orca"
    )

    assert key_b > key_a, "Collision-free planner B must rank higher than colliding planner A"


def test_claim_gate_practical_equivalence_and_universally_best() -> None:
    """Verify practical equivalence band thresholding and universally_best_emitted guard."""
    # Case A: Gaps small (low CI <= 0.02) -> STOP_SELECTOR
    cis_small = {
        "gap.family_gap.selection_score": {"ci_low": 0.01, "ci_high": 0.03},
        "gap.cell_gap.selection_score": {"ci_low": 0.015, "ci_high": 0.04},
    }
    gate_small = compute_claim_gate(cis_small)
    assert gate_small["status"] == "STOP_SELECTOR"
    assert gate_small["universally_best_emitted"] is False

    # Case B: Gaps large (low CI > 0.02) -> PROCEED_TO_SELECTOR_ISSUE
    cis_large = {
        "gap.family_gap.selection_score": {"ci_low": 0.03, "ci_high": 0.06},
        "gap.cell_gap.selection_score": {"ci_low": 0.04, "ci_high": 0.08},
    }
    gate_large = compute_claim_gate(cis_large)
    assert gate_large["status"] == "PROCEED_TO_SELECTOR_ISSUE"
    assert gate_large["universally_best_emitted"] is False


def test_cli_end_to_end_execution(tmp_path: Path) -> None:
    """Verify CLI compute_issue_5302_oracle_gap.py end-to-end execution and report output."""
    rows = _make_synthetic_six_arm_dataset(
        n_families=4, cells_per_family=2, episodes_per_cell=2, with_selection_split=True
    )
    rows_path = tmp_path / "rows.csv"

    # Write CSV
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "output_oracle_gap"

    cmd = [
        sys.executable,
        "scripts/analysis/compute_issue_5302_oracle_gap.py",
        "--input-rows",
        str(rows_path),
        "--output-dir",
        str(out_dir),
        "--n-bootstrap",
        "50",
        "--seed",
        "5302",
        "--json",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out_payload = json.loads(proc.stdout)

    assert out_payload["status"] == "ok"
    assert out_payload["issue"] == 5302
    assert "best_fixed_planner" in out_payload

    # Verify all 10 required report files exist
    reports_dir = out_dir / "reports"
    expected_files = [
        "preflight.json",
        "ceiling_summary.json",
        "ceiling_summary.csv",
        "family_breakdown.csv",
        "cell_breakdown.csv",
        "failure_mechanism_map.csv",
        "runtime_tail.csv",
        "pareto_dominance.json",
        "normalized_regret.csv",
        "bootstrap_intervals.json",
    ]

    for fname in expected_files:
        fpath = reports_dir / fname
        assert fpath.is_file(), f"Expected report file missing: {fname}"
