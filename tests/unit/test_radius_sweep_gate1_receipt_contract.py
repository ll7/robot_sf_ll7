"""Fast contract coverage for the frozen Gate 1 receipt admission rule."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.radius_sweep_manifest import (
    GATE1_CANARY_ISSUE,
    ISSUE_6642,
    PARENT_ISSUE_6600,
    RADIUS_SWEEP_MANIFEST_SCHEMA,
    RUNTIME_BINDING_BOUND_RUNTIME,
    RUNTIME_BINDING_CONTRACT_VERSION,
    RadiusSweepManifestError,
    validate_arm_campaign_payload,
)


def test_arm_campaign_payload_rejects_replacement_gate1_receipt() -> None:
    """A syntactically valid replacement cannot impersonate the frozen receipt."""
    payload = {
        "schema_version": RADIUS_SWEEP_MANIFEST_SCHEMA,
        "radius_sweep": {
            "issue": ISSUE_6642,
            "parent_issue": PARENT_ISSUE_6600,
            "arm_key": "r0p5",
            "radius_m": 0.5,
            "baseline_arm": False,
            "runtime_binding_status": RUNTIME_BINDING_BOUND_RUNTIME,
            "binding_contract_version": RUNTIME_BINDING_CONTRACT_VERSION,
            "gate1_canary_issue": GATE1_CANARY_ISSUE,
            "gate1_receipt_sha256": "c" * 64,
            "gate1_source_commit": "a" * 40,
        },
    }

    with pytest.raises(RadiusSweepManifestError, match="frozen Gate 1 receipt digest"):
        validate_arm_campaign_payload(payload, arm_key="r0p5", radius_m=0.5, baseline=False)
