"""Tests for the opt-in TTC-aware near-miss diagnostic surface (issue #3700).

These cover synthetic closing vs. opening trajectories, the fail-closed input
contract for missing/invalid timing fields, and the guarantee that the canonical
distance-based ``near_misses`` metric is left untouched (backward compatibility).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.near_miss_ttc import (
    DIAGNOSTIC_TTC_THRESHOLD_S,
    NearMissTtcInputError,
    build_ttc_near_miss_decision_packet,
    compute_ttc_near_miss_diagnostic,
    near_miss_ttc_input_readiness,
    render_ttc_near_miss_decision_packet_markdown,
)


def _make_episode(
    robot_pos: np.ndarray,
    peds_pos: np.ndarray,
    *,
    dt: float = 0.1,
    robot_vel: np.ndarray | None = None,
) -> EpisodeData:
    """Build a minimal EpisodeData from robot/ped positions for diagnostic tests."""
    robot_pos = np.asarray(robot_pos, dtype=float)
    peds_pos = np.asarray(peds_pos, dtype=float)
    if robot_vel is None:
        # Finite-difference robot velocity, padded to T frames (first = second).
        if robot_pos.shape[0] >= 2:
            diffs = np.diff(robot_pos, axis=0) / dt
            robot_vel = np.vstack([diffs[:1], diffs])
        else:
            robot_vel = np.zeros_like(robot_pos)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=np.asarray(robot_vel, dtype=float),
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=np.zeros_like(peds_pos),
        goal=np.array([100.0, 0.0]),
        dt=dt,
    )


def _fast_head_on_episode(dt: float = 0.1) -> EpisodeData:
    """Robot and one pedestrian approaching head-on at high closing speed.

    Robot moves +x at ~5 m/s, pedestrian moves -x at ~5 m/s; they start ~5 m
    apart and converge. High closing speed / small TTC but never within the
    static near-miss distance over the window, so the distance metric stays 0.
    """
    n = 6
    robot_pos = np.zeros((n, 2))
    robot_pos[:, 0] = np.arange(n) * 0.5  # 0.0 .. 2.5 m, v ~5 m/s
    peds_pos = np.zeros((n, 1, 2))
    peds_pos[:, 0, 0] = 5.0 - np.arange(n) * 0.5  # 5.0 .. 2.5 m, v ~ -5 m/s
    return _make_episode(robot_pos, peds_pos, dt=dt)


def _slow_opening_episode(dt: float = 0.1) -> EpisodeData:
    """Robot and pedestrian diverging (opening): no approaching pairs."""
    n = 6
    robot_pos = np.zeros((n, 2))
    robot_pos[:, 0] = -np.arange(n) * 0.2  # moving -x, away from ped
    peds_pos = np.zeros((n, 1, 2))
    peds_pos[:, 0, 0] = 3.0 + np.arange(n) * 0.2  # moving +x, away from robot
    return _make_episode(robot_pos, peds_pos, dt=dt)


def test_fast_converging_encounter_flagged_by_ttc():
    """A fast head-on approach yields TTC near-miss steps and a positive closing speed."""
    data = _fast_head_on_episode()
    result = compute_ttc_near_miss_diagnostic(data, t_thr=1.0)

    assert result["near_miss_ttc__status"] == "ok"
    assert result["near_miss_ttc__count"] > 0
    assert result["near_miss_ttc__approaching_steps"] > 0
    assert np.isfinite(result["near_miss_ttc__min_ttc_s"])
    assert result["near_miss_ttc__min_ttc_s"] < 1.0
    # ~10 m/s combined closing speed.
    assert result["near_miss_ttc__max_closing_speed_mps"] > 5.0


def test_ttc_flags_what_distance_metric_misses():
    """The TTC diagnostic flags the fast approach that the distance metric never sees."""
    from robot_sf.benchmark import metrics as metrics_mod

    data = _fast_head_on_episode()
    distance_near_misses = metrics_mod.near_misses(data)
    # Pairs never get within the static near-miss clearance over the window.
    assert distance_near_misses == 0.0

    ttc_result = compute_ttc_near_miss_diagnostic(data, t_thr=1.0)
    assert ttc_result["near_miss_ttc__count"] > 0


def test_opening_encounter_not_flagged():
    """Diverging trajectories produce no approaching pairs and zero TTC count."""
    data = _slow_opening_episode()
    result = compute_ttc_near_miss_diagnostic(data, t_thr=5.0)

    assert result["near_miss_ttc__status"] == "no-approaching-pairs"
    assert result["near_miss_ttc__count"] == 0.0
    assert result["near_miss_ttc__approaching_steps"] == 0.0
    assert np.isnan(result["near_miss_ttc__min_ttc_s"])
    assert np.isnan(result["near_miss_ttc__max_closing_speed_mps"])


def test_threshold_monotonicity():
    """A larger TTC threshold flags at least as many steps as a smaller one."""
    data = _fast_head_on_episode()
    low = compute_ttc_near_miss_diagnostic(data, t_thr=0.5)
    high = compute_ttc_near_miss_diagnostic(data, t_thr=2.0)
    assert high["near_miss_ttc__count"] >= low["near_miss_ttc__count"]


def test_default_threshold_is_uncalibrated_placeholder():
    """The default threshold matches the documented diagnostic placeholder constant."""
    data = _fast_head_on_episode()
    result = compute_ttc_near_miss_diagnostic(data)
    assert result["near_miss_ttc__threshold_s"] == DIAGNOSTIC_TTC_THRESHOLD_S


def test_no_pedestrians_is_ready_but_empty():
    """K=0 is valid input: ready, with an empty 'no-pedestrians' diagnostic."""
    robot_pos = np.zeros((4, 2))
    robot_pos[:, 0] = np.arange(4) * 0.5
    peds_pos = np.zeros((4, 0, 2))
    data = _make_episode(robot_pos, peds_pos)

    readiness = near_miss_ttc_input_readiness(data)
    assert readiness.ready is True
    assert readiness.n_peds == 0

    result = compute_ttc_near_miss_diagnostic(data)
    assert result["near_miss_ttc__status"] == "no-pedestrians"
    assert result["near_miss_ttc__count"] == 0.0
    assert np.isnan(result["near_miss_ttc__min_ttc_s"])


# --- Fail-closed input contract -------------------------------------------------


@pytest.mark.parametrize("bad_dt", [0.0, -0.1, float("nan"), float("inf")])
def test_readiness_fails_closed_on_invalid_dt(bad_dt):
    """Missing/invalid timing field (dt) makes the inputs not ready."""
    data = _fast_head_on_episode()
    data.dt = bad_dt
    readiness = near_miss_ttc_input_readiness(data)
    assert readiness.ready is False
    assert any("dt" in reason for reason in readiness.reasons)


def test_compute_fails_closed_on_invalid_dt():
    """Diagnostic raises (not silently zeros) when the timing field is invalid."""
    data = _fast_head_on_episode()
    data.dt = 0.0
    with pytest.raises(NearMissTtcInputError) as excinfo:
        compute_ttc_near_miss_diagnostic(data)
    assert excinfo.value.readiness is not None
    assert excinfo.value.readiness.ready is False


def test_readiness_fails_closed_on_single_frame():
    """A single frame cannot yield finite-difference velocities -> not ready."""
    robot_pos = np.zeros((1, 2))
    peds_pos = np.zeros((1, 1, 2))
    data = _make_episode(robot_pos, peds_pos, robot_vel=np.zeros((1, 2)))
    readiness = near_miss_ttc_input_readiness(data)
    assert readiness.ready is False
    assert any("2 frames" in reason for reason in readiness.reasons)


def test_readiness_fails_closed_on_mismatched_velocity_shape():
    """robot_vel that does not match robot_pos frames is rejected."""
    data = _fast_head_on_episode()
    data.robot_vel = data.robot_vel[:-1]  # drop a frame
    readiness = near_miss_ttc_input_readiness(data)
    assert readiness.ready is False
    assert any("robot_vel" in reason for reason in readiness.reasons)


def test_readiness_fails_closed_on_malformed_peds_pos():
    """peds_pos without a (T, K, 2) shape is rejected."""
    data = _fast_head_on_episode()
    data.peds_pos = np.zeros((6, 1))  # wrong rank
    readiness = near_miss_ttc_input_readiness(data)
    assert readiness.ready is False
    assert any("peds_pos" in reason for reason in readiness.reasons)


@pytest.mark.parametrize("bad_thr", [0.0, -1.0, float("nan"), float("inf")])
def test_compute_rejects_invalid_threshold(bad_thr):
    """An invalid t_thr fails closed rather than producing misleading counts."""
    data = _fast_head_on_episode()
    with pytest.raises(NearMissTtcInputError):
        compute_ttc_near_miss_diagnostic(data, t_thr=bad_thr)


def test_canonical_near_misses_metric_unchanged():
    """The diagnostic must not alter the canonical distance-based near_misses output."""
    from robot_sf.benchmark import metrics as metrics_mod

    # Construct an encounter that does cross the static near-miss distance so the
    # canonical metric returns a known nonzero value independent of the diagnostic.
    n = 4
    robot_pos = np.zeros((n, 2))
    peds_pos = np.zeros((n, 1, 2))
    # Pedestrian sits 1.6 m away: clearance 1.6 - (1.0 + 0.4) = 0.2 m, inside
    # [0, D_NEAR) so the canonical distance metric fires.
    peds_pos[:, 0, 0] = 1.6
    data = _make_episode(robot_pos, peds_pos)

    before = metrics_mod.near_misses(data)
    _ = compute_ttc_near_miss_diagnostic(data, t_thr=1.0)
    after = metrics_mod.near_misses(data)

    assert before == after
    assert before > 0.0  # sanity: the canonical metric actually fired here


# --- Issue #3808 read-only decision packet -------------------------------------


def test_decision_packet_summarizes_closing_case():
    """Closing trajectories produce diagnostic-only packet values, not claims."""
    data = _fast_head_on_episode()
    packet = build_ttc_near_miss_decision_packet(data, t_thr=1.0)

    assert packet.evidence_status == "diagnostic-only"
    assert packet.diagnostic_status == "ok"
    assert packet.diagnostic["near_miss_ttc__count"] > 0
    assert any("canonical near-miss metric replacement" in item for item in packet.cannot_claim)
    assert any("robot position" in item for item in packet.available_inputs)


def test_decision_packet_summarizes_opening_case_as_unsupported_for_ttc_counts():
    """Opening trajectories are available inputs but unsupported TTC count evidence."""
    data = _slow_opening_episode()
    packet = build_ttc_near_miss_decision_packet(data, t_thr=5.0)

    assert packet.diagnostic_status == "no-approaching-pairs"
    assert packet.diagnostic["near_miss_ttc__count"] == 0.0
    assert any("opening or non-converging pairs" in item for item in packet.unsupported_cases)


def test_decision_packet_fails_closed_on_missing_timing():
    """Missing/invalid timing stays unsupported instead of becoming zero evidence."""
    data = _fast_head_on_episode()
    data.dt = float("nan")

    packet = build_ttc_near_miss_decision_packet(data)

    assert packet.diagnostic_status == "unsupported-inputs"
    assert packet.diagnostic == {}
    assert any("dt" in item for item in packet.unsupported_cases)


def test_decision_packet_fails_closed_on_unsupported_trajectory_shape():
    """Malformed trajectory arrays are listed as unsupported packet inputs."""
    data = _fast_head_on_episode()
    data.peds_pos = np.zeros((data.robot_pos.shape[0], 2))

    packet = build_ttc_near_miss_decision_packet(data)

    assert packet.diagnostic_status == "unsupported-inputs"
    assert any("peds_pos" in item for item in packet.unsupported_cases)


def test_decision_packet_renders_markdown_and_json_safe_dict():
    """Decision packet can be emitted as generated Markdown/JSON-like payload."""
    packet = build_ttc_near_miss_decision_packet(_slow_opening_episode(), t_thr=1.0)

    payload = packet.to_dict()
    markdown = render_ttc_near_miss_decision_packet_markdown(packet)

    json.dumps(payload, allow_nan=False)
    assert payload["issue"] == "#3808"
    assert payload["evidence_status"] == "diagnostic-only"
    assert payload["diagnostic"]["near_miss_ttc__status"] == "no-approaching-pairs"
    assert payload["diagnostic"]["near_miss_ttc__min_ttc_s"] is None
    assert "# TTC Near-Miss Diagnostic Decision Packet" in markdown
    assert "Cannot Claim Before Canonical Metric Change" in markdown
    assert "no planner comparison, benchmark ranking, or paper/dissertation claim" in markdown


def test_decision_packet_json_safe_dict_handles_numpy_scalars():
    """Decision packet JSON conversion handles NumPy scalar values."""
    packet = build_ttc_near_miss_decision_packet(_slow_opening_episode(), t_thr=1.0)
    packet.diagnostic["numpy_nan"] = np.float32(np.nan)
    packet.diagnostic["numpy_float"] = np.float64(1.25)
    packet.diagnostic["numpy_int"] = np.int64(3)
    packet.diagnostic["bool_flag"] = True

    payload = packet.to_dict()

    json.dumps(payload, allow_nan=False)
    assert payload["diagnostic"]["numpy_nan"] is None
    assert payload["diagnostic"]["numpy_float"] == 1.25
    assert payload["diagnostic"]["numpy_int"] == 3
    assert payload["diagnostic"]["bool_flag"] is True
