"""Unit tests for shared Matplotlib plotting style helpers."""

import matplotlib as mpl

from robot_sf.benchmark.plotting_style import apply_latex_style

EXPECTED_RCPARAMS = {
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def test_apply_latex_style_sets_default_rcparams():
    """Apply every shared baseline value inside an isolated rcParams context."""
    with mpl.rc_context():
        apply_latex_style()

        for name, expected in EXPECTED_RCPARAMS.items():
            assert mpl.rcParams[name] == expected


def test_apply_latex_style_overrides_take_precedence():
    """Let per-figure overrides win when they overlap shared defaults."""
    overrides = {"font.size": 12, "legend.fontsize": 11}

    with mpl.rc_context():
        apply_latex_style(overrides)

        for name, expected in overrides.items():
            assert mpl.rcParams[name] == expected
        assert mpl.rcParams["axes.labelsize"] == EXPECTED_RCPARAMS["axes.labelsize"]


def test_apply_latex_style_rcparams_do_not_leak_from_test_context():
    """Restore global rcParams after applying the mutating style helper."""
    original_rcparams = mpl.rcParams.copy()

    with mpl.rc_context():
        apply_latex_style({"font.size": 13})
        assert mpl.rcParams["font.size"] == 13

    assert mpl.rcParams == original_rcparams
