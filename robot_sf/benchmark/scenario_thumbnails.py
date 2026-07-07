"""Scenario thumbnail rendering utilities.

Render per-scenario static thumbnails (PNG and optional PDF) and an optional
montage grid. Designed to be lightweight: relies only on scenario parameters
and the deterministic generator to get initial state and obstacles; does not
step the simulator. Headless-safe via MPL Agg backend (see seed_utils).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from PIL import Image  # pillow

from robot_sf.benchmark.plotting_style import apply_latex_style
from robot_sf.benchmark.scenario_generator import (
    AREA_HEIGHT,
    AREA_WIDTH,
    generate_scenario,
)
from robot_sf.common.seed import set_global_seed

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass
class ThumbMeta:
    """Metadata for a rendered scenario thumbnail."""

    scenario_id: str
    png: str
    pdf: str | None


_SAFE_SCENARIO_ID_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def _latex_rcparams():
    # Maintain backward compatibility for existing imports; delegate to shared helper
    """Apply LaTeX-style rcparams for thumbnail plots."""
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
    # Small stable hash to offset base seed per scenario
    """Derive a deterministic per-scenario seed from a base seed.

    Returns:
        Deterministic seed value.
    """
    h = hashlib.sha256(scenario_id.encode()).hexdigest()[:8]
    return (base_seed + int(h, 16)) % (2**31 - 1)


def resolve_scenario_label(params: dict[str, object]) -> str:
    """Resolve scenario label with explicit fallback priority.

    Priority: ``id`` -> ``name`` -> ``scenario_id`` -> stable hash fallback.

    Returns:
        Human-readable scenario label before filename sanitization.
    """

    for key in ("id", "name", "scenario_id"):
        raw = params.get(key)
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped:
                return stripped
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    fallback = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"scenario_{fallback}"


def sanitize_scenario_label(label: str) -> str:
    """Sanitize scenario labels for filesystem-safe output filenames.

    Returns:
        Lowercase, sanitized basename suitable for PNG/PDF output.
    """

    cleaned = _SAFE_SCENARIO_ID_PATTERN.sub("_", label.strip())
    cleaned = cleaned.strip("._-").lower()
    return cleaned or "scenario"


def _resolve_unique_scenario_ids(
    scenarios: list[dict[str, object]],
) -> list[str]:
    """Resolve deterministic unique scenario identifiers for output files.

    Returns:
        Sanitized, unique per-scenario identifiers aligned with input order.
    """

    counts: dict[str, int] = {}
    emitted_ids: set[str] = set()
    ids: list[str] = []
    for scenario in scenarios:
        raw = resolve_scenario_label(scenario)
        base = sanitize_scenario_label(raw)
        count = counts.get(base, 0) + 1
        unique = base if count == 1 else f"{base}__{count}"
        while unique in emitted_ids:
            count += 1
            unique = f"{base}__{count}"
        counts[base] = count
        if unique != base:
            logger.warning(
                "Scenario thumbnail id collision after sanitization: '{}' -> '{}' (using '{}')",
                raw,
                base,
                unique,
            )
        emitted_ids.add(unique)
        ids.append(unique)
    return ids


def _draw_obstacles(ax, obstacles: Sequence[tuple[float, float, float, float]]):
    """Draw obstacle segments on the axes."""
    for x1, y1, x2, y2 in obstacles:
        ax.plot([x1, x2], [y1, y2], color="#444", lw=1.2, alpha=0.9)


def _draw_agents(ax, pos: np.ndarray, goals: np.ndarray | None = None):
    """Draw agent positions and optional goal vectors."""
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
    # state cols: [x,y,vx,vy,goalx,goaly,tau]
    """Extract goal positions from a scenario state array.

    Returns:
        Goal position array with shape (n, 2).
    """
    if state.shape[1] >= 6:
        return state[:, 4:6]
    return np.zeros_like(state[:, 0:2])


def _draw_thumbnail_axes(ax, gen, pos, goals, title: str) -> None:
    """Draw the shared thumbnail content (limits, obstacles, agents, title).

    Factored so the legacy raster path and the publication path render identical
    data; only styling/format/provenance differ.
    """
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
    ax.set_title(title)


def render_scenario_thumbnail(  # noqa: PLR0913
    params: dict[str, object],
    seed: int,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    figsize: tuple[float, float] = (3.2, 2.0),
    *,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption_fragment: str | None = None,
    size: str = "single",
    source_artifacts: list[dict[str, object]] | None = None,
    generator_command: str | None = None,
) -> ThumbMeta:
    """Render a single scenario thumbnail to disk.

    When ``publication`` is False (default) the legacy raster path is used
    (PNG and optional PDF). When True, the thumbnail is rendered inside
    :func:`publication_style` and saved via :func:`save_publication_figure`
    with a provenance sidecar and optional caption fragment. A PNG is always
    included in publication output so montages can composite it.

    Returns
    -------
    ThumbMeta
        Metadata object containing paths to the rendered PNG and optional PDF files.
    """
    set_global_seed(seed, deterministic=True)

    gen = generate_scenario(dict(params), seed=seed)
    state = gen.state
    pos = state[:, 0:2]
    goals = _extract_goals_from_state(state)

    out_png_path = Path(out_png)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    raw_id = params.get("id")
    sid = str(raw_id) if raw_id else resolve_scenario_label(params)
    title_bits = [sid]
    if "flow" in params:
        title_bits.append(str(params["flow"]))
    if "density" in params:
        title_bits.append(str(params["density"]))
    title = " · ".join(title_bits)

    if publication:
        return _render_thumbnail_publication(
            out_png_path,
            gen=gen,
            pos=pos,
            goals=goals,
            sid=sid,
            title=title,
            figsize=figsize,
            formats=formats,
            caption_fragment=caption_fragment,
            size=size,
            source_artifacts=source_artifacts,
            generator_command=generator_command,
        )

    _latex_rcparams()
    out_pdf_path: Path | None = Path(out_pdf) if out_pdf else None

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    _draw_thumbnail_axes(ax, gen, pos, goals, title)

    fig.savefig(out_png_path, dpi=300)
    pdf_path_str: str | None = None
    if out_pdf_path is not None:
        fig.savefig(out_pdf_path)
        pdf_path_str = str(out_pdf_path)
    plt.close(fig)

    return ThumbMeta(scenario_id=sid, png=str(out_png_path), pdf=pdf_path_str)


def _render_thumbnail_publication(  # noqa: PLR0913
    out_png_path: Path,
    *,
    gen,
    pos: np.ndarray,
    goals: np.ndarray,
    sid: str,
    title: str,
    figsize: tuple[float, float],
    formats: Sequence[str],
    caption_fragment: str | None,
    size: str,
    source_artifacts: list[dict[str, object]] | None,
    generator_command: str | None,
) -> ThumbMeta:
    """Render a thumbnail in publication style with provenance sidecar.

    Lazy-imports the figures pack to avoid a circular import with the
    ``robot_sf.benchmark.figures`` package (this module is imported by it).

    Returns:
        Metadata object for the rendered thumbnail.
    """
    from robot_sf.benchmark.figures.export import save_publication_figure  # noqa: PLC0415
    from robot_sf.benchmark.figures.provenance import build_provenance  # noqa: PLC0415
    from robot_sf.benchmark.figures.style import publication_style  # noqa: PLC0415

    # Thumbnails are raster-friendly and montages composite PNGs, so always
    # include png alongside any requested vector formats.
    pub_formats: tuple[str, ...] = tuple(dict.fromkeys((*formats, "png")))

    with publication_style(size=size):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        _draw_thumbnail_axes(ax, gen, pos, goals, title)

        output_base = out_png_path.with_suffix("")
        provenance = build_provenance(
            generator_command=generator_command or "render_scenario_thumbnail",
            figure_formats=list(pub_formats),
            source_artifacts=source_artifacts or [],
            claim_boundary="Scenario thumbnail visualization; not benchmark evidence.",
        )
        save_publication_figure(
            fig,
            output_base,
            formats=pub_formats,
            provenance=provenance,
            caption_fragment=caption_fragment,
        )
    plt.close(fig)

    png_path = str(output_base.with_suffix(".png"))
    pdf_path: str | None = str(output_base.with_suffix(".pdf")) if "pdf" in pub_formats else None
    return ThumbMeta(scenario_id=sid, png=png_path, pdf=pdf_path)


def save_scenario_thumbnails(  # noqa: PLR0913
    scenarios: Iterable[dict[str, object]],
    out_dir: str | Path,
    base_seed: int = 0,
    out_pdf: bool = False,
    figsize: tuple[float, float] = (3.2, 2.0),
    *,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption: bool = False,
    campaign: str | None = None,
    size: str = "single",
    generator_command: str | None = None,
) -> list[ThumbMeta]:
    """Render thumbnails for multiple scenarios.

    When ``publication`` is True, each thumbnail is rendered in publication
    style with a provenance sidecar; when ``caption`` is also True, a
    per-scenario ``.caption.tex`` is written.

    Returns:
        List of thumbnail metadata entries.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_list = [dict(sc) for sc in scenarios]
    scenario_ids = _resolve_unique_scenario_ids(scenarios_list)

    caption_builder = None
    if caption:
        from robot_sf.benchmark.figures.provenance import build_caption_fragment  # noqa: PLC0415

        caption_builder = build_caption_fragment

    metas: list[ThumbMeta] = []
    for sc, sid in zip(scenarios_list, scenario_ids, strict=False):
        seed = _scenario_seed(base_seed, sid)
        png = out_dir / f"{sid}.png"
        pdf = (out_dir / f"{sid}.pdf") if out_pdf else None
        render_payload = dict(sc)
        render_payload["id"] = sid
        caption_fragment = (
            caption_builder(scenario_id=sid, campaign_name=campaign)
            if caption_builder is not None
            else None
        )
        meta = render_scenario_thumbnail(
            render_payload,
            seed=seed,
            out_png=png,
            out_pdf=pdf,
            figsize=figsize,
            publication=publication,
            formats=formats,
            caption_fragment=caption_fragment,
            size=size,
            generator_command=generator_command,
        )
        metas.append(meta)
    return metas


def save_montage(  # noqa: PLR0913
    metas: Sequence[ThumbMeta],
    out_png: str | Path,
    cols: int = 3,
    out_pdf: str | Path | None = None,
    pad: float = 0.2,
    *,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption_fragment: str | None = None,
    size: str = "single",
    source_artifacts: list[dict[str, object]] | None = None,
    generator_command: str | None = None,
) -> dict[str, str]:
    """Compose a simple montage grid from already-rendered thumbnails.

    Loads PNGs from metas to avoid re-rendering. Returns dict of written paths.

    When ``publication`` is True, the PIL composite is wrapped in a
    publication-styled matplotlib figure and saved via
    :func:`save_publication_figure` with a provenance sidecar and optional
    caption fragment; the composite image itself is unchanged.

    Returns:
        Dictionary mapping written format keys (e.g., 'png') to file paths.
    """
    if len(metas) == 0:
        return {"png": str(out_png)}
    cols = max(1, int(cols))
    rows = math.ceil(len(metas) / cols)
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

    if publication:
        result = _save_montage_publication(
            out_png,
            montage=montage,
            formats=formats,
            caption_fragment=caption_fragment,
            size=size,
            source_artifacts=source_artifacts,
            generator_command=generator_command,
        )
        for im in imgs:
            im.close()
        return result

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


def _save_montage_publication(
    out_png: Path,
    *,
    montage: Image.Image,
    formats: Sequence[str],
    caption_fragment: str | None,
    size: str,
    source_artifacts: list[dict[str, object]] | None,
    generator_command: str | None,
) -> dict[str, str]:
    """Save the montage composite as a publication figure with provenance.

    Lazy-imports the figures pack to avoid a circular import with the
    ``robot_sf.benchmark.figures`` package.

    Returns:
        Dictionary mapping written format keys (e.g., 'png') to file paths.
    """
    from robot_sf.benchmark.figures.export import save_publication_figure  # noqa: PLC0415
    from robot_sf.benchmark.figures.provenance import build_provenance  # noqa: PLC0415
    from robot_sf.benchmark.figures.style import publication_style  # noqa: PLC0415

    output_base = out_png.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with publication_style(size=size):
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(montage)
        provenance = build_provenance(
            generator_command=generator_command or "save_montage",
            figure_formats=list(formats),
            source_artifacts=source_artifacts or [],
            claim_boundary="Scenario montage visualization; not benchmark evidence.",
        )
        saved = save_publication_figure(
            fig,
            output_base,
            formats=tuple(formats),
            provenance=provenance,
            caption_fragment=caption_fragment,
        )
    plt.close(fig)

    result: dict[str, str] = {}
    for path in saved:
        if path.suffix == ".png":
            result["png"] = str(path)
        elif path.suffix == ".pdf":
            result["pdf"] = str(path)
        elif path.suffix == ".svg":
            result["svg"] = str(path)
    # Fall back to the requested PNG path if png was not among the formats.
    result.setdefault("png", str(out_png))
    return result


__all__ = [
    "ThumbMeta",
    "render_scenario_thumbnail",
    "resolve_scenario_label",
    "sanitize_scenario_label",
    "save_montage",
    "save_scenario_thumbnails",
]
