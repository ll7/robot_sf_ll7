"""Centralized Matplotlib plotting style helpers.

Provides a single function to apply LaTeX-friendly rcParams consistently
across all figures in the project (see docs/dev_guide.md).

Usage
-----
from robot_sf.benchmark.plotting_style import apply_latex_style

apply_latex_style()

Optionally pass overrides to tweak specific rcParams per figure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_RCPARAMS: dict[str, object] = {
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # Editable text in vector PDFs
    # Typography (pt)
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


def apply_latex_style(overrides: Mapping[str, object] | None = None) -> None:
    """Apply LaTeX-friendly rcParams for consistent figure exports.

    Args:
        overrides: Optional mapping of rcParams to override defaults.
    """
    params = DEFAULT_RCPARAMS.copy()
    if overrides:
        params.update(dict(overrides))
    mpl.rcParams.update(params)


__all__ = ["DEFAULT_RCPARAMS", "apply_latex_style"]
