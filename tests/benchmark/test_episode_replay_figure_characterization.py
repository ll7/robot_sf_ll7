"""Characterization baseline tests for ``robot_sf/benchmark/episode_replay_figure.py``.

These tests pin the *current observable behavior* of the CPU-only episode-replay
bridge on tiny synthetic inputs, focusing on the pure, rendering-free helpers:
finite-float coercion, final-position parsing, episode-row validation, the
determinism-check state machine, and the SHA-256 helpers. They are table-driven
and assert exact golden values, including edge cases (NaN/Inf at the finite
guard, malformed/missing fields, empty steps, the documented
``final_progress``-is-informational regression).

Purpose (issue #4881, wave 2; Refs #4874, #4770): lock a behavioral baseline so
the post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate fix
issue.

These tests are additive: they pin the pure-helper contract and the exact
determinism-check transitions / error messages without rendering matplotlib
figures. They do not duplicate the rendering, provenance-sidecar, or full
re-simulation workflow coverage in ``test_episode_replay_figure.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.episode_replay_figure import (
    OPTIONAL_PROVENANCE_FIELDS,
    REQUIRED_EPISODE_FIELDS,
    EpisodeRow,
    _finite_floats,
    _parse_final_position,
    check_determinism,
    compute_bytes_sha256,
    compute_file_sha256,
)
from robot_sf.benchmark.full_classic.replay import ReplayEpisode, ReplayStep

if TYPE_CHECKING:
    from pathlib import Path


def _replay_episode(steps: list[tuple[float, float, float, float]]) -> ReplayEpisode:
    """Build a minimal ``ReplayEpisode`` from ``(t, x, y, heading)`` tuples."""
    return ReplayEpisode(
        episode_id="ep",
        scenario_id="scn",
        steps=[ReplayStep(t=t, x=x, y=y, heading=h) for (t, x, y, h) in steps],
        dt=1.0,
        map_path=None,
    )


# ---------------------------------------------------------------------------
# Constant tables
# ---------------------------------------------------------------------------


def test_required_and_optional_field_tables_are_pinned() -> None:
    """Pin the exact required-field set and the optional-provenance field set."""
    assert REQUIRED_EPISODE_FIELDS == ["episode_id", "scenario_id", "seed"]
    assert "final_robot_position" in OPTIONAL_PROVENANCE_FIELDS
    assert "final_progress" in OPTIONAL_PROVENANCE_FIELDS
    assert "campaign_id" in OPTIONAL_PROVENANCE_FIELDS
    # Required fields are NOT also listed as optional provenance.
    assert not (set(REQUIRED_EPISODE_FIELDS) & set(OPTIONAL_PROVENANCE_FIELDS))


# ---------------------------------------------------------------------------
# _finite_floats: finite-guard coercion
# ---------------------------------------------------------------------------


def test_finite_floats_coerces_mixed_numeric_types() -> None:
    """Ints, floats, and numeric strings are coerced to finite floats in order."""
    assert _finite_floats(1, 2.5, "3.0") == (1.0, 2.5, 3.0)


def test_finite_floats_rejects_nan_or_inf() -> None:
    """Any non-finite value short-circuits the whole tuple to ``None``."""
    assert _finite_floats(1.0, float("nan"), 3.0) is None
    assert _finite_floats(1.0, float("inf"), 3.0) is None
    assert _finite_floats(1.0, float("-inf"), 3.0) is None


def test_finite_floats_rejects_non_numeric() -> None:
    """A non-numeric value short-circuits to ``None``."""
    assert _finite_floats(1.0, "bad", 3.0) is None
    assert _finite_floats(None) is None


def test_finite_floats_empty_call_returns_empty_tuple() -> None:
    """No arguments -> empty tuple (not ``None``)."""
    assert _finite_floats() == ()


# ---------------------------------------------------------------------------
# _parse_final_position: 2-tuple finite-guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [[1.5, 2.5], (1.5, 2.5), ["1.5", "2.5"]])
def test_parse_final_position_accepts_two_finite_elements(raw: object) -> None:
    """List, tuple, and numeric-string pairs coerce to a finite ``(x, y)``."""
    assert _parse_final_position(raw) == (1.5, 2.5)


@pytest.mark.parametrize(
    "raw",
    [
        [1.5, float("nan")],  # non-finite component
        [1.5, float("inf")],  # non-finite component
        "not_a_list",  # wrong type
        None,  # absent
        [1.5],  # wrong length (too few)
        [1.5, 2.5, 3.5],  # wrong length (too many)
        ["good", "bad"],  # non-numeric component
    ],
)
def test_parse_final_position_rejects_malformed(raw: object) -> None:
    """Malformed positions return ``None`` so downstream checks skip gracefully."""
    assert _parse_final_position(raw) is None


# ---------------------------------------------------------------------------
# EpisodeRow.from_dict: validation + coercion
# ---------------------------------------------------------------------------


def test_episode_row_minimal_fields_and_seed_int_coercion() -> None:
    """Required fields populate; seed is coerced to int; raw is preserved."""
    row = EpisodeRow.from_dict({"episode_id": "e1", "scenario_id": "s1", "seed": "42"})
    assert row.episode_id == "e1"
    assert row.scenario_id == "s1"
    assert row.seed == 42
    assert isinstance(row.seed, int)
    assert row.final_robot_position is None  # absent -> None
    assert row.raw == {"episode_id": "e1", "scenario_id": "s1", "seed": "42"}


def test_episode_row_missing_required_fields_raises_with_field_list() -> None:
    """Missing required fields raise ``ValueError`` naming exactly the missing ones."""
    with pytest.raises(ValueError) as excinfo:
        EpisodeRow.from_dict({"episode_id": "e1"})  # scenario_id + seed missing
    msg = str(excinfo.value)
    assert "missing required fields" in msg
    assert "scenario_id" in msg
    assert "seed" in msg
    assert "episode_id" not in msg  # present field is not in the missing list


def test_episode_row_parses_final_robot_position() -> None:
    """A valid ``final_robot_position`` list is parsed to a finite 2-tuple."""
    row = EpisodeRow.from_dict(
        {"episode_id": "e1", "scenario_id": "s1", "seed": 0, "final_robot_position": [3.0, 1.5]}
    )
    assert row.final_robot_position == (3.0, 1.5)


# ---------------------------------------------------------------------------
# check_determinism: state-machine transitions (exact)
# ---------------------------------------------------------------------------


def _row(**kwargs: object) -> EpisodeRow:
    """Build an EpisodeRow with required fields plus overrides."""
    base: dict[str, object] = {"episode_id": "e1", "scenario_id": "s1", "seed": 0}
    base.update(kwargs)
    return EpisodeRow.from_dict(base)


def test_check_determinism_no_steps_is_not_evaluable() -> None:
    """Empty replay steps -> ``not_evaluable`` with the pinned reason."""
    status, details = check_determinism(_replay_episode([]), _row(), tolerance_m=0.1)
    assert status == "not_evaluable"
    assert details["reason"] == "no replay steps"
    assert details["checks_performed"] == []


def test_check_determinism_position_within_tolerance_is_pass() -> None:
    """Final position within tolerance -> ``pass`` with the check recorded."""
    episode = _replay_episode([(0.0, 0.0, 0.0, 0.0), (1.0, 3.0, 1.5, 0.0)])
    row = _row(final_robot_position=[3.0, 1.5])
    status, details = check_determinism(episode, row, tolerance_m=0.1)
    assert status == "pass"
    assert details["checks_performed"] == ["final_robot_position"]
    assert details["checks_passed"] == ["final_robot_position"]
    assert details["position_error_m"] == pytest.approx(0.0)
    assert details["replay_final_position"] == (3.0, 1.5)
    assert details["original_final_position"] == (3.0, 1.5)


def test_check_determinism_position_outside_tolerance_is_fail() -> None:
    """Final position outside tolerance -> ``fail`` with a formatted failure line."""
    episode = _replay_episode([(0.0, 0.0, 0.0, 0.0), (1.0, 3.0, 1.5, 0.0)])
    row = _row(final_robot_position=[3.0, 2.0])  # 0.5 m off
    status, details = check_determinism(episode, row, tolerance_m=0.1)
    assert status == "fail"
    assert details["checks_failed"] == ["position error 0.500m > tolerance 0.1m"]


def test_check_determinism_progress_only_is_not_evaluable() -> None:
    """``final_progress`` alone is informational and must NOT yield ``pass``.

    Regression guard: previously an episode carrying only ``final_progress``
    reported determinism ``pass`` while verifying nothing.
    """
    episode = _replay_episode([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)])
    row = _row(final_progress=0.85)
    status, details = check_determinism(episode, row, tolerance_m=0.1)
    assert status == "not_evaluable"
    assert details["reason"] == "no evaluable endpoints in episode row"
    assert details["checks_performed"] == []
    assert details["checks_informational"] == ["final_progress recorded"]
    assert details["original_final_progress"] == 0.85


def test_check_determinism_no_endpoints_is_not_evaluable() -> None:
    """No position and no progress -> ``not_evaluable`` with the pinned reason."""
    episode = _replay_episode([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)])
    status, details = check_determinism(episode, _row(), tolerance_m=0.1)
    assert status == "not_evaluable"
    assert details["reason"] == "no evaluable endpoints in episode row"


# ---------------------------------------------------------------------------
# SHA-256 helpers: exact digests
# ---------------------------------------------------------------------------


def test_compute_bytes_sha256_known_digests() -> None:
    """Pin the exact SHA-256 digests for known byte inputs."""
    assert compute_bytes_sha256(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert compute_bytes_sha256(b"test") == (
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )


def test_compute_file_sha256_matches_bytes_helper(tmp_path: Path) -> None:
    """File hashing over the same bytes matches the bytes helper exactly."""
    payload = b"deterministic-replay-bytes"
    path = tmp_path / "episode.jsonl"
    path.write_bytes(payload)
    assert compute_file_sha256(path) == compute_bytes_sha256(payload)
    assert len(compute_file_sha256(path)) == 64  # hex digest length
