"""One test per frozen rule as applied (run: python -m pytest tools/test_rules.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exsel_screen as ex


def row(success=False, coll=False, timeout=False, ped=0, obs=0, steps=100):
    m = {"success": success, "ped_collision_count": ped, "obstacle_collision_count": obs}
    o = {"collision_event": coll, "timeout_event": timeout}
    outcome, kind = ex.classify(m, o)
    return {"outcome_class": outcome, "collision_kind": kind, "near_misses": 3,
            "min_surface_clearance_m": 0.5, "steps": steps,
            "time_to_goal_s": round(steps * ex.DT_S, 6) if outcome == "S" else None}


def seed(g0, g1):
    pair = {"G0E1": g0, "G1E1": g1}
    return ex.verdicts(pair), pair


def test_outcome_precedence_success_over_collision():
    assert ex.classify({"success": True}, {"collision_event": True})[0] == "S"


def test_outcome_collision_over_timeout():
    assert ex.classify({"success": False}, {"collision_event": True, "timeout_event": True})[0] == "C"


def test_collision_kind_pedestrian_wins():
    assert ex.classify({"ped_collision_count": 1, "obstacle_collision_count": 1}, {"collision_event": True}) == ("C", "pedestrian")


def test_collision_kind_obstacle():
    assert ex.classify({"ped_collision_count": 0, "obstacle_collision_count": 2}, {"collision_event": True}) == ("C", "obstacle")


def test_collision_without_counts_has_no_kind():
    assert ex.classify({}, {"collision_event": True}) == ("C", None)


def test_accept_obstacle_collision_and_success_in_band():
    ok, _ = seed(row(coll=True, obs=1), row(success=True, steps=379))
    assert ok


def test_band_inclusive_lower_and_upper():
    assert seed(row(coll=True, obs=1), row(success=True, steps=332))[0]
    assert seed(row(coll=True, obs=1), row(success=True, steps=416))[0]


def test_band_outside():
    assert not seed(row(coll=True, obs=1), row(success=True, steps=331))[0]
    assert not seed(row(coll=True, obs=1), row(success=True, steps=417))[0]


def test_reject_pedestrian_collision_arm():
    ok, pair = seed(row(coll=True, ped=1, obs=1), row(success=True, steps=379))
    assert not ok and pair["G0E1"]["class_verdict"] == "no_match"


def test_reject_g0e1_success():
    assert not seed(row(success=True, steps=200), row(success=True, steps=379))[0]


def test_reject_g1e1_timeout():
    assert not seed(row(coll=True, obs=1), row(timeout=True, steps=600))[0]


def test_null_bands_not_checked():
    _, pair = seed(row(coll=True, obs=1), row(success=True, steps=379))
    assert pair["G0E1"]["band_time_to_goal_s"] == "not_checked"
    assert pair["G1E1"]["band_exposure_steps"] == "not_checked"
    assert pair["G1E1"]["band_min_surface_clearance_m"] == "not_checked"


def test_screening_order():
    import hashlib
    order = [int(x) for x in (Path(__file__).parent.parent / "screening_order.txt").read_text().split()]
    assert order == sorted(range(1161, 1201), key=lambda s: hashlib.sha256(
        f"exsel-7.5-guard-pair-2026-10-28:{s}".encode()).hexdigest())
