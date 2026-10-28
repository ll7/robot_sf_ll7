"""Exercise preregistered O2 ordering semantics on independently specified outcomes."""

import csv
import importlib.util
import json
from pathlib import Path

import yaml

PACKET = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bounded_analyze", PACKET / "analyze.py")
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _synthetic_summary(tmp_path, *, resolved):
    """Build fixed-roster rows with a resolved or unresolved point-estimate reversal."""
    contract = yaml.safe_load((PACKET / "contract.yaml").read_text())
    cells = [cell["id"] for cell in contract["cells"]]
    rows = []
    for seed in range(1001, 1031):
        for planner in contract["planners"]:
            for plant in ("DD", "BI"):
                if planner == "goal":
                    successful_cells = 4
                elif planner == "orca":
                    successful_cells = 6
                else:
                    # DD beats goal and BI loses to goal in the resolved case.
                    # The other case reverses only the point estimate: its
                    # seed-block confidence intervals include zero.
                    dd_wins = resolved or seed < 1017
                    successful_cells = 3 if (plant == "DD") == dd_wins else 5
                for index, cell in enumerate(cells):
                    success = int(index < successful_cells)
                    rows.append(
                        {
                            "probe": 2,
                            "planner": planner,
                            "plant": plant,
                            "scenario": cell,
                            "seed": seed,
                            "mode": "native",
                            "success": success,
                            "collision": 0,
                            "timeout": 1 - success,
                            "time_s": 10,
                            "stuck": 0,
                            "creep_steps": 0,
                            "source_sha": "synthetic-only",
                        }
                    )
    for planner in contract["planners"]:
        for plant in ("DD", "BI"):
            for bearing in contract["empty_probe"]["bearings"]:
                rows.append(
                    {
                        "probe": 1,
                        "planner": planner,
                        "plant": plant,
                        "scenario": bearing,
                        "seed": 1001,
                        "mode": "native",
                        "success": 1,
                        "collision": 0,
                        "timeout": 0,
                        "time_s": 10,
                        "stuck": 0,
                        "creep_steps": 0,
                        "source_sha": "synthetic-only",
                    }
                )
    episodes = tmp_path / "synthetic.csv"
    with episodes.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ANALYZER.analyze(episodes, PACKET / "contract.yaml", tmp_path / "analysis")
    return json.loads((tmp_path / "analysis" / "summary.json").read_text())


def test_resolved_pair_reversal_triggers_o2_for_both_planners(tmp_path):
    """O2 includes both participants in a resolved reversal, even without a margin trigger."""
    summary = _synthetic_summary(tmp_path, resolved=True)
    pair = next(row for row in summary["ordering"] if row["pair"] == "goal-minus-social_force")
    assert pair["DD"]["ci_pp"][0] > 0
    assert pair["BI"]["ci_pp"][1] < 0
    assert pair["reversal"] is True
    outcomes = summary["per_planner_outcome"]
    assert outcomes["goal"]["O2_no_transfer"] is True
    assert outcomes["social_force"]["O2_no_transfer"] is True
    assert outcomes["social_force"]["O5_open"] is False
    assert outcomes["orca"]["O2_no_transfer"] is False


def test_unresolved_point_reversal_does_not_trigger_o2(tmp_path):
    """A descriptive reversal without intervals excluding zero is insufficient for O2."""
    summary = _synthetic_summary(tmp_path, resolved=False)
    pair = next(row for row in summary["ordering"] if row["pair"] == "goal-minus-social_force")
    assert pair["DD"]["pp"] > 0 > pair["BI"]["pp"]
    assert pair["DD"]["ci_pp"][0] <= 0 <= pair["DD"]["ci_pp"][1]
    assert pair["BI"]["ci_pp"][0] <= 0 <= pair["BI"]["ci_pp"][1]
    assert pair["reversal"] is False
    assert all(not outcome["O2_no_transfer"] for outcome in summary["per_planner_outcome"].values())
