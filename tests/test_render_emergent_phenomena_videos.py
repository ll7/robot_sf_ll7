"""Tests for the issue #6960 emergent-phenomena replay-video renderer.

Covers the deterministic representative-seed selection rule (majority verdict
with weaker-label tie-break, then median primary order parameter) and a tiny
smoke render proving the GIF writer produces frames headlessly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.research.emergent_phenomena import (
    RELEASED_DEFAULT_CALIBRATION,
    ScenarioConfig,
    run_scenario,
)
from robot_sf.research.representative_selection import majority_verdict
from scripts.validation.render_issue_5149_emergent_phenomena_videos import (
    render_replay_gif,
    select_representative_record,
)


def _record(scenario: str, calibration: str, seed: int, value: float, verdict: str) -> dict:
    """Build a minimal synthetic campaign run record."""
    param = (
        "lane_segregation_index" if scenario == "bidirectional_corridor" else "oscillation_flips"
    )
    return {
        "scenario": scenario,
        "calibration": calibration,
        "seed": seed,
        "order_parameters": {param: value},
        "phenomenon_verdict": verdict,
    }


def test_majority_verdict_tie_breaks_toward_weaker_label():
    assert majority_verdict(["clearly_present", "absent_or_negligible"]) == "absent_or_negligible"
    assert majority_verdict(["clearly_present", "clearly_present", "weak_partial"]) == (
        "clearly_present"
    )


def test_select_representative_record_median_of_majority_pool():
    records = [
        _record("narrow_doorway", "released_default", 1, 5.0, "clearly_present"),
        _record("narrow_doorway", "released_default", 2, 3.0, "clearly_present"),
        _record("narrow_doorway", "released_default", 3, 4.0, "clearly_present"),
        _record("narrow_doorway", "released_default", 4, 0.0, "absent_or_negligible"),
    ]
    chosen = select_representative_record(records, "narrow_doorway", "released_default")
    # Majority pool is the three clearly_present seeds; median value is 4.0.
    assert chosen["seed"] == 3


def test_select_representative_record_even_pool_takes_lower_median():
    records = [
        _record("bidirectional_corridor", "released_default", 10, 0.1, "weak_partial"),
        _record("bidirectional_corridor", "released_default", 11, 0.2, "weak_partial"),
    ]
    chosen = select_representative_record(records, "bidirectional_corridor", "released_default")
    assert chosen["seed"] == 10


def test_select_representative_record_missing_group_raises():
    with pytest.raises(ValueError, match="no run records"):
        select_representative_record([], "narrow_doorway", "released_default")


# The committed campaign bundle names its rendered replay seeds in the GIF
# filenames. Re-deriving those seeds from the archived runs.jsonl is the
# regression that catches any drift in the shared selection rule, including a
# drift introduced somewhere other than this script.
_BUNDLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs/context/evidence/issue_5149_emergent_phenomena_multiseed_2026-08"
)
_ARCHIVED_REPRESENTATIVE_SEEDS = {
    ("bidirectional_corridor", "released_default"): 5155,
    ("bidirectional_corridor", "literature_typical"): 5157,
    ("narrow_doorway", "released_default"): 5156,
    ("narrow_doorway", "literature_typical"): 5153,
}


@pytest.mark.parametrize(
    ("scenario", "calibration", "expected_seed"),
    [(key[0], key[1], seed) for key, seed in _ARCHIVED_REPRESENTATIVE_SEEDS.items()],
)
def test_select_representative_record_reproduces_archived_bundle(
    scenario: str, calibration: str, expected_seed: int
):
    runs_path = _BUNDLE_DIR / "runs.jsonl"
    if not runs_path.is_file():
        pytest.skip(f"campaign bundle not present at {runs_path}")
    records = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chosen = select_representative_record(records, scenario, calibration)
    assert int(chosen["seed"]) == expected_seed


def test_render_replay_gif_smoke(tmp_path):
    scenario = ScenarioConfig(
        name="bidirectional_corridor",
        length=8.0,
        half_width=2.0,
        n_pedestrians=4,
        seed=5149,
        n_steps=8,
    )
    result = run_scenario(scenario, RELEASED_DEFAULT_CALIBRATION)
    record = _record(
        "bidirectional_corridor", "released_default", 5149, 0.1, "absent_or_negligible"
    )
    out = tmp_path / "smoke.gif"
    n_frames = render_replay_gif([(result, record)], out, frame_stride=4, fps=5, dpi=60)
    assert out.exists()
    assert out.stat().st_size > 0
    assert n_frames == 3  # steps 0..8 -> frames at 0, 4, 8
