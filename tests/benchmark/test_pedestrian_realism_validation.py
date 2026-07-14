"""Tests for issue #4975 empirical pedestrian-model realism validation harness.

These tests prove the metric math on synthetic tracks with known ground truth
(e.g. trajectory RMSE is exactly zero for identical tracks and scales linearly
with a uniform offset), confirm fail-closed behavior when the real reference is
absent, and exercise the scorecard writer and the ETH/UCY trajectory parser on a
synthetic staged layout. They never require license-gated dataset bytes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.benchmark.pedestrian_realism_validation import (
    REALISM_CLAIM_BOUNDARY,
    REALISM_SCORECARD_SCHEMA_VERSION,
    RealismCrowdInputs,
    RealismMetricConfig,
    RealismScorecard,
    RealismStagedDatasetReference,
    RealismTrackPair,
    build_track_reconstruction_plan,
    fundamental_diagram_comparison,
    lane_formation_comparison,
    lane_formation_score_curve,
    match_tracks,
    render_scorecard_markdown,
    resample_track,
    run_realism_validation,
    run_realism_validation_from_staged_dataset,
    run_realism_validation_from_track_set,
    speed_density_points,
    trajectory_rmse,
    write_realism_scorecard,
)
from robot_sf.data.external import eth_ucy
from robot_sf.data.external.eth_ucy_trajectories import (
    ETH_BIWI_OBSMAT_FRAME_PERIOD_S,
    EthUcyTrack,
    EthUcyTrackSet,
    load_split_tracks,
    load_track_set,
    track_set_summary,
)
from scripts.tools import manage_external_data

if TYPE_CHECKING:
    from pathlib import Path

_DT = 0.4  # seconds, matches the canonical ETH BIWI frame period


def _line_track(ped_id: int, start_xy: tuple[float, float], *, steps: int = 10) -> RealismTrackPair:
    """Build a straight-line (sim == real) track pair for ground-truth checks."""

    t = np.arange(steps, dtype=float) * _DT
    pts = np.stack(
        (start_xy[0] + t, np.full(steps, start_xy[1], dtype=float)),
        axis=1,
    )
    return RealismTrackPair(
        sim_time_s=t, sim_positions=pts, real_time_s=t.copy(), real_positions=pts.copy()
    )


# --------------------------------------------------------------------------- #
# Trajectory RMSE
# --------------------------------------------------------------------------- #


def test_trajectory_rmse_is_zero_for_identical_tracks() -> None:
    """Identical sim and real tracks have exactly zero RMSE (ground-truth check)."""

    pair = _line_track(0, (0.0, 1.0))
    result = trajectory_rmse(pair, config=RealismMetricConfig())

    assert result["status"] == "ok"
    assert result["rmse_m"] == pytest.approx(0.0, abs=1e-9)
    assert result["sample_count"] >= 2


def test_trajectory_rmse_scales_with_uniform_offset() -> None:
    """A constant positional offset yields RMSE equal to the offset distance."""

    steps = 8
    t = np.arange(steps, dtype=float) * _DT
    real = np.stack((t, np.zeros(steps)), axis=1)
    offset = 0.5
    sim = real + np.array([offset, 0.0])
    pair = RealismTrackPair(
        sim_time_s=t, sim_positions=sim, real_time_s=t.copy(), real_positions=real
    )
    result = trajectory_rmse(pair, config=RealismMetricConfig())

    assert result["status"] == "ok"
    assert result["rmse_m"] == pytest.approx(offset, abs=1e-6)


def test_trajectory_rmse_empty_when_no_time_overlap() -> None:
    """Tracks sharing no time overlap report empty, not a fabricated value."""

    t1 = np.arange(5, dtype=float) * _DT
    t2 = np.arange(5, dtype=float) * _DT + 100.0
    pair = RealismTrackPair(
        sim_time_s=t1,
        sim_positions=np.zeros((5, 2)),
        real_time_s=t2,
        real_positions=np.zeros((5, 2)),
    )
    result = trajectory_rmse(pair, config=RealismMetricConfig())

    assert result["status"] == "empty"
    assert "rmse_m" not in result


def test_resample_track_requires_monotonic_time() -> None:
    """Non-monotonic sample times raise rather than silently interpolating."""

    t = np.asarray([0.0, 1.0, 0.5])
    pts = np.zeros((3, 2))
    with pytest.raises(ValueError, match="monotonically"):
        resample_track(t, pts, resample_hz=10.0)


def test_match_tracks_filters_degenerate_pairs() -> None:
    """Pairs with fewer than two samples on either side are dropped."""

    good = _line_track(0, (0.0, 0.0))
    tiny = RealismTrackPair(
        sim_time_s=np.asarray([0.0]),
        sim_positions=np.zeros((1, 2)),
        real_time_s=np.asarray([0.0]),
        real_positions=np.zeros((1, 2)),
    )
    assert match_tracks([good, tiny]) == [good]
    assert match_tracks(None) == []


# --------------------------------------------------------------------------- #
# Fundamental diagram + lane formation
# --------------------------------------------------------------------------- #


def _crowd(
    *, steps: int, k: int, speed: float, lateral: np.ndarray, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic crowd moving along ``axis`` at ``speed`` with given lateral coords."""

    t = np.arange(steps, dtype=float) * _DT
    positions = np.zeros((steps, k, 2), dtype=float)
    velocities = np.zeros((steps, k, 2), dtype=float)
    for col in range(k):
        positions[:, col, axis] = speed * t
        positions[:, col, 1 - axis] = lateral[col]
        velocities[:, col, axis] = speed
    return positions, velocities


def test_speed_density_points_shape_and_values() -> None:
    """Density points carry finite (density, speed) values for a moving crowd."""

    pos, vel = _crowd(steps=5, k=3, speed=1.0, lateral=np.asarray([0.0, 0.3, 0.6]))
    points = speed_density_points(pos, vel, neighbor_radius_m=1.0)

    assert points.shape[1] == 2
    assert points.shape[0] == 5 * 3
    assert np.all(np.isfinite(points))
    # All pedestrians move at 1.0 m/s.
    assert np.allclose(points[:, 1], 1.0)


def test_speed_density_ignores_absent_gridded_tracks_without_zeroing_neighbors() -> None:
    """An absent track must not erase density for pedestrians present in a frame."""

    pos, vel = _crowd(steps=2, k=3, speed=1.0, lateral=np.asarray([0.0, 0.2, 0.4]))
    pos[0, 2] = np.nan
    vel[0, 2] = np.nan

    points = speed_density_points(pos, vel, neighbor_radius_m=1.0)

    # The two observed pedestrians have one observed neighbor each.  Before the
    # NaN-aware filtering this frame was incorrectly reported as zero density.
    assert np.allclose(points[:2, 0], 1.0 / np.pi)
    assert np.all(np.isnan(points[2]))


def test_fundamental_diagram_comparison_zero_for_identical_crowds() -> None:
    """Identical sim and real crowds yield zero speed delta and distance."""

    pos, vel = _crowd(steps=5, k=4, speed=1.2, lateral=np.linspace(0.0, 1.0, 4))
    result = fundamental_diagram_comparison(
        speed_density_points(pos, vel, neighbor_radius_m=1.0),
        speed_density_points(pos, vel, neighbor_radius_m=1.0),
    )

    assert result["status"] == "ok"
    assert result["mean_speed_delta_mps"] == pytest.approx(0.0, abs=1e-9)
    assert result["speed_marginal_distance_mps"] == pytest.approx(0.0, abs=1e-9)


def test_fundamental_diagram_comparison_empty_for_missing_real() -> None:
    """An empty real distribution reports empty (fail-closed), not a fake delta."""

    sim = np.asarray([[0.5, 1.0], [0.5, 1.1]])
    result = fundamental_diagram_comparison(sim, np.empty((0, 2)))

    assert result["status"] == "empty"


def test_lane_formation_score_curve_detects_two_lanes() -> None:
    """Two clean counter-flowing lanes score higher than a mixed lateral spread."""

    steps = 6
    lateral_split = np.asarray([0.0, 0.1, 4.0, 4.1])
    pos_a, vel_a = _crowd(steps=steps, k=2, speed=1.0, lateral=lateral_split[:2])
    pos_b, vel_b = _crowd(steps=steps, k=2, speed=-1.0, lateral=lateral_split[2:])
    # Combine into one crowd of 4 with counter-flow lanes.
    pos = np.concatenate((pos_a, pos_b), axis=1)
    vel = np.concatenate((vel_a, vel_b), axis=1)
    score_split = lane_formation_score_curve(
        pos, vel, movement_axis=0, lateral_axis=1, movement_threshold_mps=0.05
    )

    assert score_split.size > 0
    assert np.all(score_split > 0.8)  # strong lateral separation of the two lanes

    lateral_mixed = np.asarray([0.5, 2.0, 2.1, 3.5])
    pos_c, vel_c = _crowd(steps=steps, k=2, speed=1.0, lateral=lateral_mixed[:2])
    pos_d, vel_d = _crowd(steps=steps, k=2, speed=-1.0, lateral=lateral_mixed[2:])
    pos_mix = np.concatenate((pos_c, pos_d), axis=1)
    vel_mix = np.concatenate((vel_c, vel_d), axis=1)
    score_mixed = lane_formation_score_curve(
        pos_mix, vel_mix, movement_axis=0, lateral_axis=1, movement_threshold_mps=0.05
    )
    assert float(np.mean(score_split)) > float(np.mean(score_mixed))


def test_lane_formation_ignores_absent_gridded_tracks() -> None:
    """A missing track does not discard an otherwise scorable counter-flow frame."""

    pos_a, vel_a = _crowd(steps=3, k=2, speed=1.0, lateral=np.asarray([0.0, 0.1]))
    pos_b, vel_b = _crowd(steps=3, k=2, speed=-1.0, lateral=np.asarray([4.0, 4.1]))
    pos = np.concatenate((pos_a, pos_b), axis=1)
    vel = np.concatenate((vel_a, vel_b), axis=1)
    pos[1, 3] = np.nan
    vel[1, 3] = np.nan

    scores = lane_formation_score_curve(
        pos, vel, movement_axis=0, lateral_axis=1, movement_threshold_mps=0.05
    )

    assert scores.shape == (3,)
    assert np.all(scores > 0.8)


def test_lane_formation_comparison_status_ok_and_delta_zero_for_identical() -> None:
    """Identical crowds give an ok lane comparison with a zero score delta."""

    pos, vel = _crowd(steps=6, k=4, speed=1.0, lateral=np.asarray([0.0, 0.2, 3.0, 3.2]))
    # Build counter-flow by flipping half the velocities.
    vel_neg = vel.copy()
    vel_neg[:, 2:, 0] *= -1.0
    result = lane_formation_comparison(pos, vel_neg, pos, vel_neg, config=RealismMetricConfig())

    assert result["status"] == "ok"
    assert result["mean_score_delta"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Orchestrator + scorecard + writer (synthetic end-to-end)
# --------------------------------------------------------------------------- #


def test_run_realism_validation_synthetic_end_to_end() -> None:
    """A full synthetic run computes all three metric families and an ok scorecard."""

    pair = _line_track(0, (0.0, 1.0))
    pos, vel = _crowd(steps=6, k=4, speed=1.0, lateral=np.linspace(0.0, 1.0, 4))
    scorecard = run_realism_validation(
        dataset_id="synthetic/self_consistency",
        crowds=RealismCrowdInputs(
            sim_positions=pos,
            sim_velocities=vel,
            real_positions=pos,
            real_velocities=vel,
        ),
        rmse_pairs=[pair],
        reference_source="synthetic self-consistency crowd (no real data)",
    )

    assert scorecard.status == "ok"
    rmse = scorecard.metrics["trajectory_rmse"]
    assert rmse["pair_count"] == 1
    assert rmse["rmse_m"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert scorecard.metrics["fundamental_diagram_comparison"]["status"] == "ok"
    assert scorecard.metrics["lane_formation_comparison"]["status"] in {"ok", "empty"}


def test_scorecard_fail_closed_when_real_reference_absent() -> None:
    """With no real pairs and no real crowd arrays, the scorecard is not_available."""

    # No rmse_pairs and no (sim/real) crowd arrays => nothing to compare against
    # a real reference, so the harness reports not_available (fail-closed).
    scorecard = run_realism_validation(
        dataset_id="eth-ucy/eth",
        reference_source="not staged",
    )

    assert scorecard.status == "not_available"
    assert scorecard.metrics["fundamental_diagram_comparison"]["status"] == "empty"
    assert scorecard.metrics["lane_formation_comparison"]["status"] == "empty"
    assert scorecard.metrics["trajectory_rmse"]["pair_count"] == 0


def test_scorecard_ok_when_caller_supplies_matched_real_pair() -> None:
    """A caller-supplied (sim, real) pair is itself a real reference for RMSE."""

    # RealismTrackPair carries both a sim and a real track, so a single valid
    # pair legitimately drives the trajectory-RMSE comparison (status ok) even
    # though the crowd-distribution arrays are absent (those report empty).
    pair = _line_track(0, (0.0, 1.0))
    scorecard = run_realism_validation(
        dataset_id="eth-ucy/eth",
        rmse_pairs=[pair],
        reference_source="caller-supplied matched real track",
    )

    assert scorecard.status == "ok"
    assert scorecard.metrics["trajectory_rmse"]["pair_count"] == 1
    assert scorecard.metrics["fundamental_diagram_comparison"]["status"] == "empty"


def test_run_realism_validation_from_track_set_not_staged_fail_closed() -> None:
    """A None track set yields a not_available scorecard with a fail-closed note."""

    scorecard = run_realism_validation_from_track_set(
        dataset_id="eth-ucy/eth",
        track_set=None,
    )

    assert scorecard.status == "not_available"
    assert any(
        "not staged" in note.lower() or "not provided" in note.lower() for note in scorecard.notes
    )


def test_track_reconstruction_plan_is_partial_and_content_light() -> None:
    """Trajectory seeds expose flow directions without implying scene geometry."""

    times = np.asarray([0.0, 0.4, 0.8])
    track_set = EthUcyTrackSet(
        asset_id="eth-ucy",
        group="eth",
        split="eth",
        format="obsmat",
        docs_path="docs/datasets/eth-ucy.md",
        tracks=(
            EthUcyTrack(
                pedestrian_id=1,
                time_s=times,
                positions=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            ),
            EthUcyTrack(
                pedestrian_id=2,
                time_s=times,
                positions=np.asarray([[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]]),
            ),
        ),
        skipped_formats=(),
        frame_period_s=0.4,
    )

    plan = build_track_reconstruction_plan(track_set)

    assert plan.status == "partial"
    assert plan.geometry_status == "trajectory_bounds_only"
    assert plan.scene_bounds_m == ((-1.0, -1.0), (4.0, 2.0))
    assert plan.flow_axis == "x"
    assert plan.flow_direction_counts == {
        "positive": 1,
        "negative": 1,
        "stationary": 0,
        "off_axis": 0,
    }
    assert [ped.id for ped in plan.pedestrians] == ["eth_p1", "eth_p2"]
    assert plan.pedestrians[0].start == (0.0, 0.0)
    assert plan.pedestrians[0].trajectory == [(1.0, 0.0), (2.0, 0.0)]
    assert "static scene geometry" in plan.blockers[0].lower()
    summary = plan.summary_dict()
    assert summary["schema_version"] == "pedestrian_realism_validation.reconstruction.v1"
    assert "positions" not in summary
    assert "trajectory" not in summary


def test_empty_track_reconstruction_plan_is_not_available() -> None:
    """A missing track set yields a fail-closed reconstruction decision packet."""

    plan = build_track_reconstruction_plan(None, dataset_id="eth-ucy/eth")

    assert plan.status == "not_available"
    assert plan.scene_bounds_m is None
    assert plan.pedestrians == ()
    assert any("stage the dataset" in blocker.lower() for blocker in plan.blockers)


def test_scorecard_writer_emits_schema_files(tmp_path: Path) -> None:
    """The writer emits scorecard.json and scorecard.md with the schema version."""

    pair = _line_track(0, (0.0, 1.0))
    scorecard = run_realism_validation(
        dataset_id="synthetic/writer",
        rmse_pairs=[pair],
        reference_source="synthetic",
    )
    paths = write_realism_scorecard(scorecard, tmp_path)

    payload = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == REALISM_SCORECARD_SCHEMA_VERSION
    assert payload["claim_boundary"] == REALISM_CLAIM_BOUNDARY
    md = paths["scorecard_md"].read_text(encoding="utf-8")
    assert "Pedestrian Realism Scorecard" in md
    assert "Trajectory RMSE" in md


def test_scorecard_markdown_leads_with_claim_and_status() -> None:
    """The rendered markdown states the claim boundary and status up front."""

    sc = RealismScorecard(
        dataset_id="synthetic/render",
        status="not_available",
        reference_source="none",
    )
    md = render_scorecard_markdown(sc)

    assert REALISM_CLAIM_BOUNDARY in md
    assert "`not_available`" in md


def test_realism_metric_config_rejects_nonpositive_resample_hz() -> None:
    """A non-positive resample frequency is rejected."""

    with pytest.raises(ValueError):
        RealismMetricConfig(resample_hz=0.0)


# --------------------------------------------------------------------------- #
# ETH/UCY trajectory parsing (synthetic staged layout)
# --------------------------------------------------------------------------- #

_OBSMAT_ROWS = (
    "1 1 2.5 0.0 4.5 0.1 0.0 0.2\n"
    "2 1 2.9 0.0 4.5 0.1 0.0 0.2\n"  # frame 2 at 0.4s: x advances by 0.4 -> ~1 m/s
    "3 1 3.3 0.0 4.5 0.1 0.0 0.2\n"
    "1 2 0.0 0.0 0.0 0.0 0.0 0.0\n"
    "2 2 0.0 0.0 0.5 0.0 0.0 0.0\n"
)
_TXT_ROWS = "1 1 2.0 3.0\n2 1 2.4 3.0\n3 1 2.8 3.0\n1 2 5.0 6.0\n2 2 5.0 6.5\n"


def _stage_dataset(root: Path) -> None:
    """Write a tiny documented ETH/UCY layout with obsmat + txt splits."""

    (root / "eth" / "obsmat.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "eth" / "obsmat.txt").write_text(_OBSMAT_ROWS, encoding="utf-8")
    (root / "hotel" / "obsmat.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "hotel" / "obsmat.txt").write_text(_OBSMAT_ROWS, encoding="utf-8")
    (root / "univ" / "univ.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "univ" / "univ.txt").write_text(_TXT_ROWS, encoding="utf-8")
    (root / "zara01" / "zara01.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "zara01" / "zara01.txt").write_text(_TXT_ROWS, encoding="utf-8")
    (root / "zara02" / "crowds_zara02.vsp").parent.mkdir(parents=True, exist_ok=True)
    (root / "zara02" / "crowds_zara02.vsp").write_text(
        "2\n0 1 2.0 3.0\n10 1 2.1 3.1\n", encoding="utf-8"
    )


def test_parse_obsmat_split_into_per_pedestrian_tracks(tmp_path: Path) -> None:
    """The obsmat parser yields one track per pedestrian with derived times."""

    _stage_dataset(tmp_path / "eth-ucy")
    dataset = eth_ucy.require_available(tmp_path / "eth-ucy")
    eth_split = next(s for s in dataset.splits if s.split == "eth")
    track_set = load_split_tracks(eth_split)

    assert isinstance(track_set, EthUcyTrackSet)
    assert track_set.format == "obsmat"
    assert {track.pedestrian_id for track in track_set.tracks} == {1, 2}
    ped1 = next(track for track in track_set.tracks if track.pedestrian_id == 1)
    assert ped1.time_s.shape[0] == 3
    assert ped1.positions.shape == (3, 2)
    # First frame time is normalized to 0; frame step is the BIWI period.
    assert ped1.time_s[0] == pytest.approx(0.0, abs=1e-9)
    assert ped1.time_s[1] == pytest.approx(ETH_BIWI_OBSMAT_FRAME_PERIOD_S, abs=1e-9)
    # x is column 0 of obsmat (2.5, 2.9, 3.3).
    assert np.allclose(ped1.positions[:, 0], [2.5, 2.9, 3.3])
    # y is obsmat column 4.
    assert np.allclose(ped1.positions[:, 1], 4.5)


def test_parse_txt_split_into_per_pedestrian_tracks(tmp_path: Path) -> None:
    """The 4-col txt parser yields tracks with the documented (frame, id, x, y)."""

    _stage_dataset(tmp_path / "eth-ucy")
    track_set = load_track_set("univ", root=tmp_path / "eth-ucy")

    assert track_set.format == "txt"
    assert len(track_set.tracks) == 2
    ped1 = next(track for track in track_set.tracks if track.pedestrian_id == 1)
    assert np.allclose(ped1.positions[:, 0], [2.0, 2.4, 2.8])
    assert np.allclose(ped1.positions[:, 1], 3.0)


def test_parse_vsp_split_is_skipped_not_failed(tmp_path: Path) -> None:
    """A .vsp split is recorded as skipped, not parsed or failed."""

    _stage_dataset(tmp_path / "eth-ucy")
    track_set = load_track_set("zara02", root=tmp_path / "eth-ucy")

    assert track_set.tracks == ()
    assert track_set.skipped_formats == ("vsp",)


def test_provenance_gated_scorecard_fails_closed_without_manifest(tmp_path: Path) -> None:
    """A staged-looking layout without registry provenance is not scorecard evidence."""

    _stage_dataset(tmp_path / "eth-ucy")
    scorecard = run_realism_validation_from_staged_dataset(
        dataset_id="eth-ucy/eth",
        dataset=RealismStagedDatasetReference(
            split="eth",
            root=tmp_path / "eth-ucy",
            provenance_manifest=tmp_path / "missing.provenance.json",
        ),
    )

    assert scorecard.status == "not_available"
    assert "provenance-gated" in scorecard.reference_source
    assert any("not success evidence" in note.lower() for note in scorecard.notes)


def test_provenance_gated_scorecard_uses_registry_manifest(tmp_path: Path) -> None:
    """A registry-ready synthetic staging reaches the real-reference metric path."""

    source_root = tmp_path / "eth-ucy"
    _stage_dataset(source_root)
    (source_root / "README.md").write_text("Synthetic staging note.\n", encoding="utf-8")
    manifest_path = tmp_path / "eth-ucy.provenance.json"
    manage_external_data.stage_asset(
        "eth-ucy",
        source_path=source_root,
        manifest_out=manifest_path,
    )
    track_set = load_track_set("eth", root=source_root)
    real_positions, real_velocities = _gridded_real(track_set)

    scorecard = run_realism_validation_from_staged_dataset(
        dataset_id="eth-ucy/eth",
        dataset=RealismStagedDatasetReference(
            split="eth",
            root=source_root,
            provenance_manifest=manifest_path,
        ),
        sim_positions=real_positions,
        sim_velocities=real_velocities,
    )

    assert scorecard.status == "ok"
    assert scorecard.metrics["fundamental_diagram_comparison"]["status"] == "ok"
    assert any("provenance manifest passed" in note for note in scorecard.notes)
    assert any("replay seed plan status: partial" in note for note in scorecard.notes)


def test_track_parser_rejects_ragged_rows_fail_closed(tmp_path: Path) -> None:
    """A malformed staged trajectory raises the public data error, never ValueError."""

    path = tmp_path / "obsmat.txt"
    path.write_text("1 1 2.5 0.0 4.5\n2 1 2.9 0.0\n", encoding="utf-8")
    split_path = eth_ucy.EthUcySplitPath("eth", "eth", path, "obsmat")

    with pytest.raises(eth_ucy.EthUcyDataError, match="not rectangular"):
        load_split_tracks(split_path)


def test_track_parser_rejects_nonpositive_frame_period(tmp_path: Path) -> None:
    """A non-positive frame period cannot create a valid time axis."""

    _stage_dataset(tmp_path / "eth-ucy")
    dataset = eth_ucy.require_available(tmp_path / "eth-ucy")
    eth_split = next(split for split in dataset.splits if split.split == "eth")

    with pytest.raises(ValueError, match="frame_period_s"):
        load_split_tracks(eth_split, frame_period_s=0.0)


def test_load_track_set_unknown_split_raises(tmp_path: Path) -> None:
    """An unknown split id raises KeyError naming the documented splits."""

    _stage_dataset(tmp_path / "eth-ucy")
    with pytest.raises(KeyError, match="Unknown ETH/UCY split"):
        load_track_set("nonexistent", root=tmp_path / "eth-ucy")


def test_track_set_summary_is_content_free(tmp_path: Path) -> None:
    """The track set summary reports counts/timing only, not coordinates."""

    _stage_dataset(tmp_path / "eth-ucy")
    track_set = load_track_set("eth", root=tmp_path / "eth-ucy")
    summary = track_set_summary(track_set)

    assert summary["split"] == "eth"
    assert summary["pedestrian_count"] == 2
    assert summary["skipped_formats"] == []
    # No coordinate content should leak into the summary.
    assert "positions" not in summary
    assert "x" not in summary


def test_realism_validation_against_staged_track_set(tmp_path: Path) -> None:
    """A staged track set drives the real distribution path of the harness."""

    _stage_dataset(tmp_path / "eth-ucy")
    track_set = load_track_set("eth", root=tmp_path / "eth-ucy")
    # Build a synthetic sim crowd that mimics the real parsed tracks (so the
    # fundamental-diagram speed delta is small but nonzero due to gridding).
    real_pos, real_vel = _gridded_real(track_set)
    scorecard = run_realism_validation_from_track_set(
        dataset_id="eth-ucy/eth",
        track_set=track_set,
        sim_positions=real_pos,
        sim_velocities=real_vel,
    )

    assert scorecard.status == "ok"
    assert scorecard.metrics["fundamental_diagram_comparison"]["status"] == "ok"
    assert "eth-ucy/eth" in scorecard.reference_source


def _gridded_real(track_set: EthUcyTrackSet) -> tuple[np.ndarray, np.ndarray]:
    """Grid a track set into crowd arrays using the harness internal helper."""

    from robot_sf.benchmark.pedestrian_realism_validation import RealismMetricConfig
    from robot_sf.benchmark.pedestrian_realism_validation import (
        _gridded_crowd_from_tracks as grid,
    )

    return grid(track_set, RealismMetricConfig())
