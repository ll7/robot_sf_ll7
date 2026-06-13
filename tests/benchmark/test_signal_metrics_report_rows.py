"""Tests for signal_metrics_report_rows and compliance eligibility gate.

Covers all four row types:
- red_required_stop: observable row where robot crosses stop line under red
- green_proceed: observable row where robot crosses during green
- unavailable_no_claim: no signal metadata
- proxy_only_denominator_excluded: proxy diagnostic planner, excluded from denominator
"""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.signal_metrics import (
    SignalEpisode,
    render_report_rows_markdown,
    signal_metrics_report_rows,
)


class _MockEpisode(SignalEpisode):
    """Minimal episode implementation for testing."""

    def __init__(
        self,
        robot_pos: np.ndarray,
        peds_pos: np.ndarray,
        dt: float,
        episode_metadata: dict | None,
    ):
        self.robot_pos = robot_pos
        self.peds_pos = peds_pos
        self.dt = dt
        self.episode_metadata = episode_metadata


def _observable_red_metadata() -> dict:
    """Metadata for a planner_observable red-phase crossing."""
    return {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 5.0},
                {"state": "green", "duration": 5.0},
            ],
            "stop_line": [[10.0, 10.0], [10.0, -10.0]],
            "crosswalk_polygon": [
                [11.0, 10.0],
                [15.0, 10.0],
                [15.0, -10.0],
                [11.0, -10.0],
            ],
        }
    }


def _observable_green_metadata() -> dict:
    """Metadata for a planner_observable green-phase crossing."""
    return {
        "signal_state": {
            "contract_state": "planner_observable",
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 0.05},
                {"state": "green", "duration": 5.0},
            ],
            "stop_line": [[0.0, 1.0], [0.0, -1.0]],
            "crosswalk_polygon": [
                [1.0, 1.0],
                [5.0, 1.0],
                [5.0, -1.0],
                [1.0, -1.0],
            ],
        }
    }


def _red_robot_trajectory() -> np.ndarray:
    """Robot trajectory that crosses stop line during red phase."""
    return np.array(
        [
            [0.0, 0.0],
            [11.0, 0.0],
            [12.0, 0.0],
            [13.0, 0.0],
            [14.0, 0.0],
            [15.0, 0.0],
        ]
    )


def _green_robot_trajectory() -> np.ndarray:
    """Robot trajectory that crosses stop line only after green onset."""
    return np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.5, 0.0],
            [1.5, 0.0],
            [2.5, 0.0],
        ]
    )


# ---------------------------------------------------------------------------
# Row type tests
# ---------------------------------------------------------------------------


def test_report_rows_red_required_stop():
    """Observable red crossing is classified as red_required_stop."""
    episode = _MockEpisode(
        _red_robot_trajectory(),
        np.zeros((6, 0, 2)),
        1.0,
        _observable_red_metadata(),
    )
    rows = signal_metrics_report_rows([("ep_red_001", episode)])

    assert len(rows) == 1
    r = rows[0]
    assert r["episode_id"] == "ep_red_001"
    assert r["row_type"] == "red_required_stop"
    assert r["planner_observable"] is True
    assert r["benchmark_evidence"] is True
    assert r["signal_compliance_eligible"] is True
    assert r["signal_metrics_denominator"] == 1
    assert r["stop_line_behaviour"]["crossed_under_red"] is True
    assert r["stop_line_behaviour"]["red_violation_count"] == 1
    assert r["pedestrian_conflict"]["eligible"] is True
    assert r["exclusion_reason"] == ""


def test_report_rows_green_proceed():
    """Observable green crossing is classified as green_proceed."""
    episode = _MockEpisode(
        _green_robot_trajectory(),
        np.zeros((5, 0, 2)),
        0.1,
        _observable_green_metadata(),
    )
    rows = signal_metrics_report_rows([("ep_green_001", episode)])

    assert len(rows) == 1
    r = rows[0]
    assert r["episode_id"] == "ep_green_001"
    assert r["row_type"] == "green_proceed"
    assert r["planner_observable"] is True
    assert r["benchmark_evidence"] is True
    assert r["signal_compliance_eligible"] is True
    assert r["signal_metrics_denominator"] == 1
    assert r["stop_line_behaviour"]["crossed_under_red"] is False
    assert r["stop_line_behaviour"]["red_violation_count"] == 0
    assert r["pedestrian_conflict"]["eligible"] is True
    assert r["exclusion_reason"] == ""


def test_report_rows_unavailable_no_claim():
    """Missing signal metadata is classified as unavailable_no_claim."""
    episode = _MockEpisode(
        np.zeros((10, 2)),
        np.zeros((10, 0, 2)),
        0.1,
        None,
    )
    rows = signal_metrics_report_rows([("ep_unavail_001", episode)])

    assert len(rows) == 1
    r = rows[0]
    assert r["episode_id"] == "ep_unavail_001"
    assert r["row_type"] == "unavailable_no_claim"
    assert r["planner_observable"] is False
    assert r["benchmark_evidence"] is False
    assert r["signal_compliance_eligible"] is False
    assert r["signal_metrics_denominator"] == 0
    assert r["exclusion_reason"] == "signal_state_metadata_absent"
    assert r["stop_line_behaviour"]["crossed_under_red"] is False
    assert r["pedestrian_conflict"]["eligible"] is False


def test_report_rows_proxy_only_denominator_excluded():
    """Proxy diagnostic planner is classified as proxy_only_denominator_excluded."""
    episode = _MockEpisode(
        np.zeros((10, 2)),
        np.zeros((10, 0, 2)),
        0.1,
        {
            "signal_state": {
                "contract_state": "proxy_diagnostic",
            }
        },
    )
    rows = signal_metrics_report_rows([("ep_proxy_001", episode)])

    assert len(rows) == 1
    r = rows[0]
    assert r["episode_id"] == "ep_proxy_001"
    assert r["row_type"] == "proxy_only_denominator_excluded"
    assert r["planner_observable"] is False
    assert r["benchmark_evidence"] is False
    assert r["signal_compliance_eligible"] is False
    assert r["signal_metrics_denominator"] == 0
    assert r["exclusion_reason"] == "signal_state_not_benchmark_evidence"
    assert r["stop_line_behaviour"]["crossed_under_red"] is False
    assert r["pedestrian_conflict"]["eligible"] is False


# ---------------------------------------------------------------------------
# Compliance eligibility gate
# ---------------------------------------------------------------------------


def test_compliance_eligible_requires_planner_observable_and_benchmark_evidence():
    """Only rows with planner_observable=true AND benchmark_evidence=true are eligible."""
    # Observable + benchmark evidence -> eligible
    obs_episode = _MockEpisode(
        _green_robot_trajectory(),
        np.zeros((5, 0, 2)),
        0.1,
        _observable_green_metadata(),
    )
    # Observable but no benchmark evidence -> not eligible
    no_evidence_episode = _MockEpisode(
        np.zeros((2, 2)),
        np.zeros((2, 0, 2)),
        1.0,
        {
            "signal_state": {
                "contract_state": "planner_observable",
                "benchmark_evidence": False,
                "timeline": [{"state": "green", "duration": 1.0}],
                "stop_line": [[0.0, 1.0], [0.0, -1.0]],
            }
        },
    )

    rows = signal_metrics_report_rows(
        [
            ("eligible", obs_episode),
            ("not_eligible", no_evidence_episode),
        ]
    )

    eligible_rows = [r for r in rows if r["signal_compliance_eligible"]]
    ineligible_rows = [r for r in rows if not r["signal_compliance_eligible"]]

    assert len(eligible_rows) == 1
    assert eligible_rows[0]["episode_id"] == "eligible"
    assert len(ineligible_rows) == 1
    assert ineligible_rows[0]["episode_id"] == "not_eligible"
    assert ineligible_rows[0]["row_type"] == "unavailable_no_claim"
    assert ineligible_rows[0]["exclusion_reason"] == "signal_state_not_benchmark_evidence"


def test_proxy_row_never_claims_compliance():
    """Proxy diagnostic rows must never claim traffic-light compliance."""
    episodes = [
        (
            "proxy_a",
            _MockEpisode(
                np.zeros((5, 2)),
                np.zeros((5, 0, 2)),
                0.1,
                {"signal_state": {"contract_state": "proxy_diagnostic"}},
            ),
        ),
        (
            "proxy_b",
            _MockEpisode(
                np.zeros((5, 2)),
                np.zeros((5, 0, 2)),
                0.1,
                {
                    "signal_state": {
                        "contract_state": "proxy_diagnostic",
                        "planner_observable": True,
                        "benchmark_evidence": True,
                    }
                },
            ),
        ),
    ]
    rows = signal_metrics_report_rows(episodes)

    for r in rows:
        assert r["signal_compliance_eligible"] is False
        assert r["row_type"] == "proxy_only_denominator_excluded"
        assert r["signal_metrics_denominator"] == 0


def test_unavailable_row_never_claims_compliance():
    """Unavailable rows must never claim traffic-light compliance."""
    rows = signal_metrics_report_rows(
        [
            (
                "no_meta",
                _MockEpisode(
                    np.zeros((3, 2)),
                    np.zeros((3, 0, 2)),
                    0.5,
                    None,
                ),
            ),
            (
                "empty_meta",
                _MockEpisode(
                    np.zeros((3, 2)),
                    np.zeros((3, 0, 2)),
                    0.5,
                    {},
                ),
            ),
        ]
    )

    for r in rows:
        assert r["signal_compliance_eligible"] is False
        assert r["row_type"] == "unavailable_no_claim"
        assert r["signal_metrics_denominator"] == 0


# ---------------------------------------------------------------------------
# Report structure tests
# ---------------------------------------------------------------------------


def test_report_rows_contain_all_required_fields():
    """Every row must contain all required report fields."""
    episodes = [
        (
            "ep1",
            _MockEpisode(
                _red_robot_trajectory(),
                np.zeros((6, 0, 2)),
                1.0,
                _observable_red_metadata(),
            ),
        ),
        (
            "ep2",
            _MockEpisode(
                np.zeros((3, 2)),
                np.zeros((3, 0, 2)),
                0.1,
                None,
            ),
        ),
        (
            "ep3",
            _MockEpisode(
                np.zeros((5, 2)),
                np.zeros((5, 0, 2)),
                0.1,
                {"signal_state": {"contract_state": "proxy_diagnostic"}},
            ),
        ),
    ]
    rows = signal_metrics_report_rows(episodes)

    required_keys = {
        "episode_id",
        "row_type",
        "planner_observable",
        "benchmark_evidence",
        "signal_compliance_eligible",
        "signal_unavailable_exclusion_count",
        "signal_metrics_denominator",
        "exclusion_reason",
        "stop_line_behaviour",
        "pedestrian_conflict",
        "delay_after_green_onset_s",
        "signal_metrics_evidence",
    }

    for r in rows:
        assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - r.keys()}"

        stop = r["stop_line_behaviour"]
        assert "crossed_under_red" in stop
        assert "min_distance_m" in stop
        assert "red_violation_count" in stop

        ped = r["pedestrian_conflict"]
        assert "count" in ped
        assert "label" in ped
        assert "eligible" in ped


def test_report_rows_exclude_reason_for_non_observable():
    """Non-observable rows must carry a non-empty exclusion reason."""
    episodes = [
        (
            "unavail",
            _MockEpisode(
                np.zeros((3, 2)),
                np.zeros((3, 0, 2)),
                0.5,
                None,
            ),
        ),
        (
            "proxy",
            _MockEpisode(
                np.zeros((3, 2)),
                np.zeros((3, 0, 2)),
                0.5,
                {"signal_state": {"contract_state": "proxy_diagnostic"}},
            ),
        ),
    ]
    rows = signal_metrics_report_rows(episodes)

    for r in rows:
        if not r["signal_compliance_eligible"]:
            assert r["exclusion_reason"], (
                f"Non-compliance-eligible row {r['episode_id']} must have exclusion_reason"
            )


def test_markdown_rendering_produces_table():
    """render_report_rows_markdown should produce a Markdown table."""
    episodes = [
        (
            "ep_red",
            _MockEpisode(
                _red_robot_trajectory(),
                np.zeros((6, 0, 2)),
                1.0,
                _observable_red_metadata(),
            ),
        ),
        (
            "ep_none",
            _MockEpisode(
                np.zeros((3, 2)),
                np.zeros((3, 0, 2)),
                0.5,
                None,
            ),
        ),
    ]
    rows = signal_metrics_report_rows(episodes)
    md = render_report_rows_markdown(rows)

    assert md.startswith("| episode_id")
    assert "|---|" in md
    assert "ep_red" in md
    assert "ep_none" in md
    assert "red_required_stop" in md
    assert "unavailable_no_claim" in md
