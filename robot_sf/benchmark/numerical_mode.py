"""Validate campaign numerical claims against retained learned-arm manifests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf._numerical_mode import validate_numerical_mode
from robot_sf.benchmark.result_provenance import validate_result_provenance_manifest


def validate_campaign_numerical_manifest(payload: dict, campaign_root: Path) -> None:
    """Reject campaign claims inconsistent with actual learned arm execution.

    Preflight records a requested policy and kernel context. A completed run may
    claim that policy only when every enabled PPO/guarded-PPO arm retained a
    matching, valid inference manifest. Historical manifests have no new field.
    """
    claim = payload.get("numerical_mode")
    if claim is None:
        return
    expected = sum(
        planner.get("algo") in {"ppo", "guarded_ppo"} for planner in payload.get("planners", [])
    ) * len(payload.get("kinematics_matrix", ["differential_drive"]))
    matched = 0
    for path in (campaign_root / "runs").rglob("*.provenance.json"):
        arm = json.loads(path.read_text())
        if arm.get("campaign_identity", {}).get("algorithm") not in {"ppo", "guarded_ppo"}:
            continue
        if arm.get("run", {}).get("numerical_mode") != claim:
            raise ValueError("Pinned numerical mode does not match retained arm manifest")
        validate_result_provenance_manifest(arm)
        observed = arm["run"]["numerical_kernel_context"]
        validate_numerical_mode(claim, {**observed, "inference_dtype": "float64"})
        matched += 1
    if expected == 0 or matched != expected:
        raise ValueError("Pinned numerical mode requires every learned arm's retained manifest")
