"""A historical exception cannot conceal an unpinned production caller."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


def test_historical_exemptions_require_matching_immutable_packet_entries():
    root = Path(__file__).resolve().parents[2]
    audit = runpy.run_path(str(root / "tests/research/test_reference_guarded_imports.py"))
    assert "HISTORICAL_INPUTS" in audit, "missing source-bound historical exemption list"
    assert "_validate_historical_inputs" in audit, "missing historical exemption validation"
    entries = audit["HISTORICAL_INPUTS"]
    validate = audit["_validate_historical_inputs"]
    validate(root, entries)
    extra = {
        "path": "robot_sf/research/emergent_phenomena.py",
        "packet": "configs/benchmarks/issue_6969_lane_formation_stage_b_preregistration.yaml",
        "reason": "attempted unpinned exemption",
    }
    with pytest.raises(ValueError, match="not pinned by"):
        validate(root, (*entries, extra))
    for path in ("robot_sf/research/*", "robot_sf/research"):
        with pytest.raises(ValueError, match="not pinned by"):
            validate(root, (*entries, {**extra, "path": path}))
