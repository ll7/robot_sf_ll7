"""Tests for issue #4803: benchmark plot helpers guard against empty / non-finite inputs.

Acceptance criteria from the issue:
- Both helpers filter None/non-finite values and handle empty input without raising.
- Each helper called with empty and NaN-laden inputs renders without error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers: _filter_finite, _ax_no_data
# ---------------------------------------------------------------------------


def test_filter_finite_removes_none():
    """None values are stripped from the output list."""
    from robot_sf.benchmark.visualization import _filter_finite

    assert _filter_finite([None, 1.0, None]) == [1.0]


def test_filter_finite_removes_nan():
    """NaN values are stripped from the output list."""
    from robot_sf.benchmark.visualization import _filter_finite

    assert _filter_finite([float("nan"), 0.5]) == [0.5]


def test_filter_finite_removes_inf():
    """±Inf values are stripped from the output list."""
    from robot_sf.benchmark.visualization import _filter_finite

    assert _filter_finite([float("inf"), float("-inf"), 0.1]) == pytest.approx([0.1])


def test_filter_finite_all_non_finite():
    """All-invalid inputs produce an empty list."""
    from robot_sf.benchmark.visualization import _filter_finite

    assert _filter_finite([None, float("nan"), float("inf")]) == []


def test_filter_finite_empty_input():
    """Empty input returns empty list without error."""
    from robot_sf.benchmark.visualization import _filter_finite

    assert _filter_finite([]) == []


def test_pair_finite_keeps_episode_alignment():
    """_pair_finite drops a pair when either side is non-finite, keeping x/y aligned."""
    from robot_sf.benchmark.visualization import _pair_finite

    # episode 0: success finite, snqi NaN -> drop; episode 1: both finite -> keep;
    # episode 2: success None -> drop.
    xs, ys = _pair_finite([0.2, 0.5, None], [float("nan"), 0.9, 0.3])
    assert xs == pytest.approx([0.5])
    assert ys == pytest.approx([0.9])


def test_pair_finite_empty_when_no_overlap():
    """No pair survives when finite values never coincide on the same index."""
    from robot_sf.benchmark.visualization import _pair_finite

    xs, ys = _pair_finite([0.1, float("nan")], [float("nan"), 0.2])
    assert xs == []
    assert ys == []


def test_ax_no_data_does_not_raise():
    """_ax_no_data renders placeholder text and sets axis labels."""
    from robot_sf.benchmark.visualization import _ax_no_data

    ax = MagicMock()
    _ax_no_data(ax, "My Title", xlabel="X", ylabel="Y")
    ax.set_title.assert_called_once_with("My Title")
    ax.set_xlabel.assert_called_once_with("X")
    ax.set_ylabel.assert_called_once_with("Y")
    ax.text.assert_called_once()


# ---------------------------------------------------------------------------
# _populate_metrics_axes: normal and degenerate inputs
# ---------------------------------------------------------------------------


def _make_axes():
    """Return four MagicMock axes."""
    return MagicMock(), MagicMock(), MagicMock(), MagicMock()


@pytest.mark.parametrize(
    "collisions, success_rates, snqi_scores",
    [
        # All empty
        ([], [], []),
        # All None
        ([None, None], [None], [None]),
        # Mixed NaN / Inf / None
        ([float("nan"), None, float("inf")], [None], [float("nan")]),
        # Some valid, some invalid
        ([None, 0.1, float("nan")], [0.5, None], [float("inf"), 0.7]),
    ],
)
def test_populate_metrics_axes_degenerate_no_raise(collisions, success_rates, snqi_scores):
    """_populate_metrics_axes must not raise for degenerate inputs."""
    from robot_sf.benchmark.visualization import _populate_metrics_axes

    ax1, ax2, ax3, ax4 = _make_axes()
    # Should complete without raising
    _populate_metrics_axes(ax1, ax2, ax3, ax4, collisions, success_rates, snqi_scores)


def test_populate_metrics_axes_empty_uses_no_data_placeholder():
    """When all series are empty, _ax_no_data is called for each subplot."""
    from robot_sf.benchmark.visualization import _populate_metrics_axes

    ax1, ax2, ax3, ax4 = _make_axes()
    _populate_metrics_axes(ax1, ax2, ax3, ax4, [], [], [])

    # No hist/scatter calls when data is absent
    ax1.hist.assert_not_called()
    ax2.hist.assert_not_called()
    ax3.hist.assert_not_called()
    ax4.scatter.assert_not_called()

    # Placeholder text rendered on each axis
    for ax in (ax1, ax2, ax3, ax4):
        ax.text.assert_called_once()


def test_populate_metrics_axes_valid_data_uses_plots():
    """With valid data, hist and scatter are called (not placeholders)."""
    from robot_sf.benchmark.visualization import _populate_metrics_axes

    ax1, ax2, ax3, ax4 = _make_axes()
    _populate_metrics_axes(
        ax1,
        ax2,
        ax3,
        ax4,
        [0.1, 0.2],
        [0.8, 0.9],
        [0.6, 0.7],
    )

    ax1.hist.assert_called_once()
    ax2.hist.assert_called_once()
    ax3.hist.assert_called_once()
    ax4.scatter.assert_called_once()


def test_populate_metrics_axes_nan_filtered_valid_data_uses_plots():
    """Valid values survive NaN filtering and still reach hist/scatter."""
    from robot_sf.benchmark.visualization import _populate_metrics_axes

    ax1, ax2, ax3, ax4 = _make_axes()
    _populate_metrics_axes(
        ax1,
        ax2,
        ax3,
        ax4,
        [0.1, float("nan"), None],  # one finite value
        [float("inf"), 0.5],  # one finite value
        [None, 0.9],  # one finite value
    )

    ax1.hist.assert_called_once()
    ax2.hist.assert_called_once()
    ax3.hist.assert_called_once()
    ax4.scatter.assert_called_once()


# ---------------------------------------------------------------------------
# _populate_scenario_comparison_axes: normal and degenerate inputs
# ---------------------------------------------------------------------------


def test_populate_scenario_comparison_axes_empty_no_raise():
    """_populate_scenario_comparison_axes with empty scenarios must not raise."""
    from robot_sf.benchmark.visualization import _populate_scenario_comparison_axes

    ax1, ax2, ax3 = MagicMock(), MagicMock(), MagicMock()
    _populate_scenario_comparison_axes(ax1, ax2, ax3, [], [], [], [])

    # bar must not be called with empty sequences
    ax1.bar.assert_not_called()
    ax2.bar.assert_not_called()
    ax3.bar.assert_not_called()

    # Placeholder text rendered on each axis
    for ax in (ax1, ax2, ax3):
        ax.text.assert_called_once()


def test_populate_scenario_comparison_axes_valid_uses_bar():
    """With valid scenarios, bar is called for each subplot."""
    from robot_sf.benchmark.visualization import _populate_scenario_comparison_axes

    ax1, ax2, ax3 = MagicMock(), MagicMock(), MagicMock()
    _populate_scenario_comparison_axes(
        ax1,
        ax2,
        ax3,
        ["scen_a", "scen_b"],
        [0.8, 0.7],
        [0.1, 0.2],
        [0.9, 0.85],
    )

    ax1.bar.assert_called_once()
    ax2.bar.assert_called_once()
    ax3.bar.assert_called_once()
