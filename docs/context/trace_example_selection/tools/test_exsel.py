"""One test per frozen protocol rule applied by exsel.py (run: pytest test_exsel.py)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exsel  # noqa: E402

CLAIMS = {
    p.parent.name: exsel.load_claim(yaml.safe_load(p.read_text(encoding="utf-8")))
    for p in sorted(HERE.parent.glob("exsel-*/contract.yaml"))
}
WALK = CLAIMS["exsel-6.4-measure-walkthrough"]
DOOR = CLAIMS["exsel-7.2-narrow-doorway"]


def row(  # noqa: PLR0913
    success=False,
    collision=False,
    timeout=False,
    ped=0,
    obst=0,
    wall=0.0,
    near=0,
    sep=3.0,
    steps=400,
    dt=0.1,
    status="ok",
    ped_model="native",
    degraded=False,
):
    """A minimal serialized episode row with the fields the protocol reads."""
    return {
        "scenario_id": "cell",
        "seed": 1161,
        "steps": steps,
        "termination_reason": "x",
        "metrics": {
            "success": success,
            "ped_collision_count": ped,
            "obstacle_collision_count": obst,
            "wall_collisions": wall,
            "near_misses": near,
        },
        "outcome": {"collision_event": collision, "timeout_event": timeout},
        "scenario_params": {"run_dt": dt},
        "safety_predicates": {
            "occlusion_near_miss_predicate": {"fields": {"actual_minimum_separation_m": sep}}
        },
        "algorithm_metadata": {"status": status},
        "pedestrian_model": {"fallback_degraded_status": ped_model},
        "integrity": {"effective_view": {"degraded": degraded}},
    }


def test_outcome_precedence_success_then_collision_then_timeout():
    """Outcome classes S, C, T are checked in that order; none gives "?"."""
    assert exsel.project(row(success=True, collision=True, timeout=True))["outcome"] == "S"
    assert exsel.project(row(collision=True, timeout=True))["outcome"] == "C"
    assert exsel.project(row(timeout=True))["outcome"] == "T"
    assert exsel.project(row())["outcome"] == "?"


def test_collision_kind_pedestrian_before_obstacle_and_wall_counts_as_obstacle():
    """A pedestrian collision wins; obstacle or wall counts give obstacle."""
    assert exsel.project(row(collision=True, ped=1, obst=1))["collision_kind"] == "pedestrian"
    assert exsel.project(row(collision=True, obst=1))["collision_kind"] == "obstacle"
    assert exsel.project(row(collision=True, wall=1.0))["collision_kind"] == "obstacle"
    assert exsel.project(row(collision=True))["collision_kind"] is None


def test_min_exposure_steps_needs_recorded_near_misses_at_least_k():
    """min_exposure_steps needs near_misses >= k and fails when absent."""
    cls = {"outcome": "S", "min_exposure_steps": 1}
    assert exsel.class_verdicts(exsel.project(row(success=True, near=1)), cls)["min_exposure_steps"]
    assert not exsel.class_verdicts(exsel.project(row(success=True, near=0)), cls)[
        "min_exposure_steps"
    ]
    r = row(success=True)
    del r["metrics"]["near_misses"]
    assert not exsel.class_verdicts(exsel.project(r), cls)["min_exposure_steps"]


def test_clearance_is_separation_minus_envelope_and_sentinel_is_absent():
    """Clearance subtracts 1.4 m; the no-pedestrian sentinel is absent."""
    assert exsel.project(row(sep=1.8))["min_surface_clearance_m"] == pytest.approx(0.4)
    assert exsel.project(row(sep=1e6))["min_surface_clearance_m"] is None


def test_time_to_goal_successes_only():
    """Time to goal is steps times dt for successes only."""
    assert exsel.project(row(success=True, steps=350))["time_to_goal_s"] == 35.0
    assert exsel.project(row(timeout=True, steps=400))["time_to_goal_s"] is None


def test_bands_inclusive_absent_outside_null_unchecked():
    """Bands are inclusive, absent values fail and null bands are unchecked."""
    bands = WALK["frozen"]["bands"]["social_force"]
    lo_sep = bands["min_surface_clearance_m"][0] + exsel.ENVELOPE_RADIUS_SUM_M
    rec = exsel.project(row(success=True, near=8, sep=lo_sep, steps=int(33.25 / 0.1 + 0.5)))
    rec["min_surface_clearance_m"] = bands["min_surface_clearance_m"][0]  # exact lower bound
    v = exsel.band_verdicts(rec, bands)
    assert v == {"min_surface_clearance_m": True, "exposure_steps": True, "time_to_goal_s": True}
    rec["exposure_steps"] = 13.5 + 1e-9
    assert exsel.band_verdicts(rec, bands)["exposure_steps"] is False
    rec["min_surface_clearance_m"] = None
    assert exsel.band_verdicts(rec, bands)["min_surface_clearance_m"] is False
    assert exsel.band_verdicts(rec, DOOR["frozen"]["bands"]["orca"])["time_to_goal_s"] is None


def test_acceptance_walkthrough_needs_class_and_all_bands():
    """The walkthrough needs its class and every band."""
    good = row(success=True, near=10, sep=0.4 + 1.4, steps=360)
    assert exsel.evaluate({"social_force": good}, WALK)["accepted"]
    assert not exsel.evaluate(
        {"social_force": row(success=True, near=0, sep=1.8, steps=360)}, WALK
    )["accepted"]
    assert not exsel.evaluate(
        {"social_force": row(success=True, near=10, sep=1.8, steps=500)}, WALK
    )["accepted"]


def test_acceptance_doorway_needs_both_arms_on_the_same_seed():
    """The doorway claim needs both arms classes and bands on one seed."""
    orca = row(timeout=True, near=0, sep=2.99 + 1.4)
    goal = row(collision=True, wall=1.0, near=0, sep=4.0 + 1.4)
    assert exsel.evaluate({"orca": orca, "goal": goal}, DOOR)["accepted"]
    goal_ped = row(collision=True, ped=1, near=0, sep=4.0 + 1.4)
    assert not exsel.evaluate({"orca": orca, "goal": goal_ped}, DOOR)["accepted"]
    orca_out = row(timeout=True, near=0, sep=3.1 + 1.4)
    assert not exsel.evaluate({"orca": orca_out, "goal": goal}, DOOR)["accepted"]


def test_boundary_claim_checks_class_only():
    """A boundary claim ignores the bands."""
    boundary = copy.deepcopy(WALK)
    boundary["kind"] = "boundary"
    assert exsel.evaluate(
        {"social_force": row(success=True, near=10, sep=1.8, steps=500)}, boundary
    )["accepted"]


def test_screening_order_is_sha256_sorted_band_and_refuses_bad_seeds():
    """The order is the SHA-256 sort; retired and non-development seeds refuse."""
    scr = WALK["screening"]
    order = exsel.screening_order(scr["seed_band"], scr["selection_rng_seed"], scr["budget"])
    assert sorted(order) == list(range(1161, 1201))
    assert order[:3] == [1197, 1199, 1194]
    with pytest.raises(ValueError):
        exsel.screening_order([111, 140], "x", 5)
    with pytest.raises(ValueError):
        exsel.screening_order([1195, 1205], "x", 5)


def test_mode_markers():
    """Mode is fallback, degraded or native from the row markers."""
    assert exsel.mode_of(row()) == "native"
    assert exsel.mode_of(row(status="fallback")) == "fallback"
    assert exsel.mode_of(row(ped_model="fallback")) == "fallback"
    assert exsel.mode_of(row(degraded=True)) == "degraded"
    r = row()
    r["algorithm_metadata"]["guard_stats"] = {"fallback_safe": 2}
    assert exsel.mode_of(r) == "fallback"


def _write_run(runs, prefix, idx, seed, arm, raw):
    d = runs / f"{prefix}__screen__{idx:02d}__{seed}"
    ep = d / "out" / "c" / "runs" / f"{arm}__differential_drive"
    ep.mkdir(parents=True)
    (ep / "episodes.jsonl").write_text(__import__("json").dumps(raw) + "\n")
    (d / "receipt.json").write_text('{"rc": 0}')


def test_ledger_stops_at_first_accepted_and_skips_unrun_seeds(tmp_path):
    """The ledger stops at the first accepted seed and at the first unrun seed."""
    import build_ledger

    contract = yaml.safe_load(
        (HERE.parent / "exsel-6.4-measure-walkthrough" / "contract.yaml").read_text(
            encoding="utf-8"
        )
    )
    order = contract["screening"]["order"]
    prefix = contract["claim_id"].lower()
    reject = row(success=True, near=0, sep=1.8, steps=360)
    accept = row(success=True, near=10, sep=1.8, steps=360)
    for idx, (seed, raw) in enumerate(
        zip(order[:3], [reject, accept, accept], strict=True), start=1
    ):
        raw = copy.deepcopy(raw)
        raw.update({"seed": seed, "scenario_id": "classic_doorway_high"})
        _write_run(tmp_path, prefix, idx, seed, "social_force", raw)
    text, summary = build_ledger.build(contract, tmp_path)
    assert summary["accepted_seed"] == order[1] and summary["accepted_screening_index"] == 2
    assert summary["seeds_screened"] == 2
    assert len(text.strip().splitlines()) == 1 + 2
    # A seed with no successful run ends the ledger: nothing after it is screened.
    empty = tmp_path / "gap"
    raw = copy.deepcopy(accept)
    raw.update({"seed": order[1], "scenario_id": "classic_doorway_high"})
    _write_run(empty, prefix, 2, order[1], "social_force", raw)
    _, gap = build_ledger.build(contract, empty)
    assert gap["seeds_screened"] == 0 and gap["accepted_seed"] is None
