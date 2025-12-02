"""Scenario thumbnail rendering utilities.

Render per-scenario static thumbnails (PNG and optional PDF) and an optional
montage grid. Designed to be lightweight: relies only on scenario parameters
and the deterministic generator to get initial state and obstacles; does not
step the simulator. Headless-safe via MPL Agg backend (see seed_utils).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # pillow

from robot_sf.benchmark.plotting_style import apply_latex_style
from robot_sf.benchmark.scenario_generator import AREA_HEIGHT, AREA_WIDTH, generate_scenario
from robot_sf.common.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass
class ThumbMeta:
    """ThumbMeta class."""

    scenario_id: str
    png: str
    pdf: str | None


def _latex_rcparams():
    """Latex rcparams.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Maintain backward compatibility for existing imports; delegate to shared helper
    apply_latex_style(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )


def _scenario_seed(base_seed: int, scenario_id: str) -> int:
    """Scenario seed.

    Args:
        base_seed: Auto-generated placeholder description.
        scenario_id: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    # Small stable hash to offset base seed per scenario
    import hashlib

    h = hashlib.sha256(scenario_id.encode()).hexdigest()[:8]
    return (base_seed + int(h, 16)) % (2**31 - 1)


def _draw_obstacles(ax, obstacles: Sequence[tuple[float, float, float, float]]):
    """Draw obstacles.

    Args:
        ax: Auto-generated placeholder description.
        obstacles: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    for x1, y1, x2, y2 in obstacles:
        ax.plot([x1, x2], [y1, y2], color="#444", lw=1.2, alpha=0.9)


def _draw_agents(ax, pos: np.ndarray, goals: np.ndarray | None = None):
    """Draw agents.

    Args:
        ax: Auto-generated placeholder description.
        pos: Auto-generated placeholder description.
        goals: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    if pos.size == 0:
        return
    ax.scatter(pos[:, 0], pos[:, 1], s=10, c="#1f77b4", alpha=0.7, edgecolors="none")
    if goals is not None and goals.shape == pos.shape:
        # Thin lines toward goals to hint flow
        for i in range(pos.shape[0]):
            ax.plot(
                [pos[i, 0], goals[i, 0]],
                [pos[i, 1], goals[i, 1]],
                color="#1f77b4",
                lw=0.4,
                alpha=0.25,
            )


def _extract_goals_from_state(state: np.ndarray) -> np.ndarray:
    """Extract goals from state.

    Args:
        state: Auto-generated placeholder description.

    Returns:
        np.ndarray: Auto-generated placeholder description.
    """
    # state cols: [x,y,vx,vy,goalx,goaly,tau]
    if state.shape[1] >= 6:
        return state[:, 4:6]
    return np.zeros_like(state[:, 0:2])


def render_scenario_thumbnail(
    params: dict[str, object],
    seed: int,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    figsize: tuple[float, float] = (3.2, 2.0),
) -> ThumbMeta:
    """Render a single scenario thumbnail to disk.

    Returns ThumbMeta with written paths.
    """
    _latex_rcparams()
    set_global_seed(seed, deterministic=True)

    gen = generate_scenario(dict(params), seed=seed)
    state = gen.state
    pos = state[:, 0:2]
    goals = _extract_goals_from_state(state)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf_path: Path | None = Path(out_pdf) if out_pdf else None

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(0, AREA_WIDTH)
    ax.set_ylim(0, AREA_HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    # Background
    ax.set_facecolor("#fafafa")
    _draw_obstacles(ax, gen.obstacles)
    _draw_agents(ax, pos, goals)
    # Subtle border
    for spine in ax.spines.values():
        spine.set_color("#999")
        spine.set_linewidth(0.5)
    # Minimal ticks
    ax.set_xticks([])
    ax.set_yticks([])
    sid = str(params.get("id", "scenario"))
    title_bits = [sid]
    if "flow" in params:
        title_bits.append(str(params["flow"]))
    if "density" in params:
        title_bits.append(str(params["density"]))
    ax.set_title(" · ".join(title_bits))

    fig.savefig(out_png, dpi=300)
    pdf_path_str: str | None = None
    if out_pdf_path is not None:
        fig.savefig(out_pdf_path)
        pdf_path_str = str(out_pdf_path)
    plt.close(fig)

    return ThumbMeta(scenario_id=sid, png=str(out_png), pdf=pdf_path_str)


def save_scenario_thumbnails(
    scenarios: Iterable[dict[str, object]],
    out_dir: str | Path,
    base_seed: int = 0,
    out_pdf: bool = False,
    figsize: tuple[float, float] = (3.2, 2.0),
) -> list[ThumbMeta]:
    """Save scenario thumbnails.

    Args:
        scenarios: Auto-generated placeholder description.
        out_dir: Auto-generated placeholder description.
        base_seed: Auto-generated placeholder description.
        out_pdf: Auto-generated placeholder description.
        figsize: Auto-generated placeholder description.

    Returns:
        list[ThumbMeta]: Auto-generated placeholder description.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: list[ThumbMeta] = []
    for sc in scenarios:
        sid = str(sc.get("id", "scenario"))
        seed = _scenario_seed(base_seed, sid)
        png = out_dir / f"{sid}.png"
        pdf = (out_dir / f"{sid}.pdf") if out_pdf else None
        meta = render_scenario_thumbnail(sc, seed=seed, out_png=png, out_pdf=pdf, figsize=figsize)
        metas.append(meta)
    return metas


def save_montage(
    metas: Sequence[ThumbMeta],
    out_png: str | Path,
    cols: int = 3,
    out_pdf: str | Path | None = None,
    pad: float = 0.2,
) -> dict[str, str]:
    """Compose a simple montage grid from already-rendered thumbnails.

    Loads PNGs from metas to avoid re-rendering. Returns dict of written paths.
    """
    if len(metas) == 0:
        return {"png": str(out_png)}
    cols = max(1, int(cols))
    rows = int(math.ceil(len(metas) / cols))
    # Load images
    imgs = [Image.open(m.png) for m in metas]
    w, h = imgs[0].size
    # Padding in pixels
    pad_px = int(pad * 100)  # heuristic
    grid_w = cols * w + (cols - 1) * pad_px
    grid_h = rows * h + (rows - 1) * pad_px
    montage = Image.new("RGB", (grid_w, grid_h), color=(250, 250, 250))
    for idx, im in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        x = c * (w + pad_px)
        y = r * (h + pad_px)
        montage.paste(im, (x, y))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_png)

    out_pdf_path: Path | None = Path(out_pdf) if out_pdf else None
    pdf_path_str: str | None = None
    if out_pdf_path is not None:
        # Save via matplotlib to get vectorized PDF wrapper
        _latex_rcparams()
        dpi = 300
        fig_w = grid_w / dpi
        fig_h = grid_h / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.axis("off")
        ax.imshow(montage)
        fig.savefig(out_pdf_path)
        plt.close(fig)
        pdf_path_str = str(out_pdf_path)
    # Close images
    for im in imgs:
        im.close()
    return {"png": str(out_png), **({"pdf": pdf_path_str} if pdf_path_str else {})}


__all__ = [
    "ThumbMeta",
    "render_scenario_thumbnail",
    "save_montage",
    "save_scenario_thumbnails",
]
