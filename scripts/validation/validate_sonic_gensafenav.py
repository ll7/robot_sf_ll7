"""Validate the checked-out SoNIC / GenSafeNav wrapper integration path.

Usage:
    uv run python scripts/validation/validate_sonic_gensafenav.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from robot_sf.planner.sonic_crowdnav import SonicCrowdNavAdapter, build_sonic_crowdnav_config

ROOT = Path(__file__).resolve().parents[2]


def _sample_observation() -> dict[str, object]:
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "velocity_xy": [0.0, 0.0],
            "radius": [0.3],
        },
        "goal": {"current": [5.0, 0.0]},
        "pedestrians": {
            "positions": [[2.0, 0.0], [3.0, 1.0]],
            "velocities": [[0.0, 0.0], [0.0, 0.0]],
            "count": [2],
        },
    }


def _run_case(*, label: str, repo_root: Path, model_name: str | None, checkpoint_name: str) -> None:
    payload: dict[str, object] = {
        "repo_root": str(repo_root),
        "checkpoint_name": checkpoint_name,
        "max_linear_speed": 1.0,
        "max_angular_speed": 1.0,
    }
    if model_name is not None:
        payload["model_name"] = model_name

    adapter = SonicCrowdNavAdapter(build_sonic_crowdnav_config(payload))
    linear, angular, meta = adapter.act(_sample_observation(), time_step=0.25)

    if not math.isfinite(linear) or not math.isfinite(angular):
        raise RuntimeError(f"{label}: non-finite Robot SF command ({linear}, {angular})")
    if not (-1.0 <= linear <= 1.0 and -1.0 <= angular <= 1.0):
        raise RuntimeError(f"{label}: command outside configured bounds ({linear}, {angular})")
    if meta.get("source_action_kinematics") != "holonomic":
        raise RuntimeError(
            f"{label}: unexpected upstream kinematics {meta.get('source_action_kinematics')!r}"
        )
    if meta.get("upstream_policy") != "rl.networks.model.Policy[selfAttn_merge_srnn]":
        raise RuntimeError(f"{label}: unexpected upstream policy {meta.get('upstream_policy')!r}")
    if not meta.get("parity_gaps"):
        raise RuntimeError(f"{label}: expected parity_gaps to remain documented")

    action_xy = np.asarray(meta.get("upstream_action_xy", []), dtype=float).reshape(-1)
    if action_xy.size < 2 or not np.isfinite(action_xy[:2]).all():
        raise RuntimeError(f"{label}: upstream_action_xy missing or non-finite")

    print(f"PASS {label}: command=({linear:.3f}, {angular:.3f}) action_xy={action_xy[:2].tolist()}")


def main() -> int:
    """Run all SoNIC / GenSafeNav real-asset smoke cases and exit non-zero on failure."""
    sonic_root = ROOT / "output" / "repos" / "SoNIC-Social-Nav"
    gensafe_root = ROOT / "output" / "repos" / "GenSafeNav"

    cases = [
        {
            "label": "SoNIC_GST",
            "repo_root": sonic_root,
            "model_name": None,
            "checkpoint_name": "05207.pt",
            "checkpoint_path": sonic_root
            / "trained_models"
            / "SoNIC_GST"
            / "checkpoints"
            / "05207.pt",
        },
        {
            "label": "GenSafeNav Ours_GST",
            "repo_root": gensafe_root,
            "model_name": "Ours_GST",
            "checkpoint_name": "05207.pt",
            "checkpoint_path": gensafe_root
            / "trained_models"
            / "Ours_GST"
            / "checkpoints"
            / "05207.pt",
        },
        {
            "label": "GenSafeNav GST_predictor_rand",
            "repo_root": gensafe_root,
            "model_name": "GST_predictor_rand",
            "checkpoint_name": "05207.pt",
            "checkpoint_path": gensafe_root
            / "trained_models"
            / "GST_predictor_rand"
            / "checkpoints"
            / "05207.pt",
        },
    ]

    missing = [
        f"{case['label']}: {case['checkpoint_path']}"
        for case in cases
        if not case["repo_root"].exists() or not case["checkpoint_path"].exists()
    ]
    if missing:
        raise SystemExit(
            "Missing required SoNIC / GenSafeNav assets for validation:\n- " + "\n- ".join(missing)
        )

    for case in cases:
        _run_case(
            label=str(case["label"]),
            repo_root=Path(case["repo_root"]),
            model_name=case["model_name"],
            checkpoint_name=str(case["checkpoint_name"]),
        )
    print("PASS validate_sonic_gensafenav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
