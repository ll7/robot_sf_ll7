"""Generate maneuver-authority algo-config variants for the hard-case portfolio (#3213).

Each variant is a full predictive-planner algo-config: the camera-ready base with a small,
documented set of maneuver-authority overrides (turn authority, lattice density, sequence-search
depth, near-field turn budget). These are the planner-side bets in the hard-case breakthrough
portfolio (#3215); the evaluation harness scores each against the hard-seed benchmark so we can
tell whether richer maneuver authority closes the hard-case plateau or trades safety for progress.

Run from the repository root:
    uv run python scripts/research/generate_predictive_authority_variants.py
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

_BASE = Path("configs/algos/prediction_planner_camera_ready.yaml")
_OUT_DIR = Path("configs/algos/hardcase_authority")

# Wider heading lattices reused by several variants.
_WIDE_HEADINGS = [
    -1.047198,
    -0.785398,
    -0.523599,
    -0.261799,
    0.0,
    0.261799,
    0.523599,
    0.785398,
    1.047198,
]
_DENSE_SPEEDS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
_WIDE_NEARFIELD = [
    -1.570796,
    -1.047198,
    -0.785398,
    -0.523599,
    -0.261799,
    0.0,
    0.261799,
    0.523599,
    0.785398,
    1.047198,
    1.570796,
]

# variant_name -> override dict (empty = baseline copy for self-consistent comparison).
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "high_angular": {
        "max_angular_speed": 1.8,
        "predictive_candidate_heading_deltas": _WIDE_HEADINGS,
    },
    "dense_lattice": {
        "predictive_candidate_speeds": _DENSE_SPEEDS,
        "predictive_candidate_heading_deltas": _WIDE_HEADINGS,
    },
    "deep_sequence": {
        "predictive_sequence_segments": 4,
        "predictive_sequence_branch_factor": 6,
        "predictive_sequence_beam_width": 12,
    },
    "nearfield_turn": {
        "predictive_near_field_heading_deltas": _WIDE_NEARFIELD,
        "predictive_horizon_boost_steps": 9,
        "predictive_near_field_speed_cap": 0.8,
    },
    "combined_max_authority": {
        "max_angular_speed": 1.8,
        "predictive_candidate_speeds": _DENSE_SPEEDS,
        "predictive_candidate_heading_deltas": _WIDE_HEADINGS,
        "predictive_sequence_segments": 4,
        "predictive_sequence_branch_factor": 6,
        "predictive_sequence_beam_width": 12,
        "predictive_near_field_heading_deltas": _WIDE_NEARFIELD,
    },
}


def main() -> int:
    """Write one full algo-config per authority variant and return a POSIX exit code."""
    base = yaml.safe_load(_BASE.read_text())
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = copy.deepcopy(base)
        config.update(overrides)
        header = (
            f"# Hard-case portfolio maneuver-authority variant: {name} (#3213).\n"
            f"# Generated from {_BASE} by scripts/research/generate_predictive_authority_variants.py.\n"
            f"# Overrides: {overrides or 'none (baseline copy)'}\n"
        )
        out_path = _OUT_DIR / f"prediction_planner_authority_{name}.yaml"
        out_path.write_text(header + yaml.safe_dump(config, sort_keys=False))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
