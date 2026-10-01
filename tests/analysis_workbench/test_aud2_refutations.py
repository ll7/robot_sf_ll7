"""Four reviewer regression nodes imported from RV10 test_refutations.py."""

import pytest

from robot_sf.analysis_workbench.audit_scan import scan_campaign


@pytest.mark.parametrize(
    "metric", ["clearance_m", "time_to_collision_min", "ped_force_mean", "total_collision_count"]
)
def test_malformed_physical_scalar_does_not_report_clear(metric):
    row = {"episode_id": "malformed", "metrics": {"valid_distance": 1.0, metric: {"value": -1.0}}}
    scan = scan_campaign([row], detector_ids=["extreme_measurements"])
    assert scan.signals[0].status == "error"
    assert scan.signals[0].reason_code == "nonfinite_or_malformed_measurement"
