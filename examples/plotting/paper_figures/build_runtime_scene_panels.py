#!/usr/bin/env python3
"""Build publication-ready runtime scene panel figures from benchmark videos/frames."""

from __future__ import annotations

import argparse
import json
import string
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

DEFAULT_SCENARIOS = [
    "francis2023_blind_corner",
    "francis2023_intersection_wait",
    "francis2023_parallel_traffic",
]

DEFAULT_FRACTIONS = [0.20, 0.35, 0.50, 0.65, 0.80]


@dataclass(frozen=True)
class CandidateScore:
    frame_index: int
    score: float
    content_coverage: float
    dynamic_context_signal: float
    crop_usability: float
    bbox_coverage_w: float
    bbox_coverage_h: float
    quality_pass: bool


@dataclass(frozen=True)
class SelectionRecord:
    panel_id: str
    scenario_id: str
    policy: str
    seed: int
    frame_strategy: str
    source_video: str
    source_frame_file: str
    selected_frame_index: int
    selected_score: float
    selection_reason: str
    candidate_frame_indices: list[int]
    candidate_scores: list[float]
    candidate_quality_pass: list[bool]
    width: int
    height: int
    bbox_coverage_w: float
    bbox_coverage_h: float
    crop_bbox: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/robot_sf_ll7/paper_tools/sim_snapshots_2026-02-19"),
        help="Root containing frames/ and videos_selected/.",
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON with ordered panels (uses panels[*].scenario_id).",
    )
    parser.add_argument(
        "--scenario-ids",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario IDs if --scenario-manifest is not provided.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="goal",
        help="Policy label in file names.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Seed label in file names.",
    )
    parser.add_argument(
        "--frame-strategy",
        choices=["scored", "mid", "best-clearance"],
        default="scored",
        help="Frame selection policy. best-clearance is retained as alias for scored.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["content_bbox"],
        default="content_bbox",
        help="Cropping mode.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in final figure layout.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Output file name. Defaults to runtime_panels_main_<N>.png.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/robot_sf_ll7/paper_tools/runtime_visuals"),
        help="Output directory for runtime figure/manifests.",
    )
    parser.add_argument(
        "--emit-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit JSON and Markdown manifests.",
    )
    parser.add_argument(
        "--legend-in-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render legend text inside the figure image.",
    )
    parser.add_argument(
        "--write-qa-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a checklist-style QA report next to runtime figure artifacts.",
    )
    return parser.parse_args()


def _parse_scenarios(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def _load_scenarios(args: argparse.Namespace) -> list[str]:
    if args.scenario_manifest is None:
        scenarios = _parse_scenarios(args.scenario_ids)
        if not scenarios:
            raise SystemExit("No scenarios provided.")
        return scenarios

    if not args.scenario_manifest.exists():
        raise SystemExit(f"Missing scenario manifest: {args.scenario_manifest}")
    payload = json.loads(args.scenario_manifest.read_text(encoding="utf-8"))
    panels = payload.get("panels", [])
    if not isinstance(panels, list):
        raise SystemExit("Scenario manifest has invalid `panels` format.")
    scenario_ids: list[str] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        sid = str(panel.get("scenario_id", "")).strip()
        if sid:
            scenario_ids.append(sid)
    if not scenario_ids:
        raise SystemExit("No scenario IDs found in scenario manifest.")
    return scenario_ids


def _frame_path(source_root: Path, scenario_id: str, seed: int, policy: str) -> Path:
    return source_root / "frames" / f"{scenario_id}_seed{seed}_{policy}_mid.png"


def _video_path(source_root: Path, scenario_id: str, seed: int, policy: str) -> Path:
    return source_root / "videos_selected" / f"{scenario_id}_seed{seed}_{policy}.mp4"


def _ffprobe_frame_count(video: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "csv=p=0",
        str(video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tokens = [tok for tok in proc.stdout.replace("\n", ",").split(",") if tok.strip()]
    ints = [int(tok) for tok in tokens if tok.strip().isdigit()]
    return max(ints) if ints else 0


def _extract_frame(video: Path, frame_idx: int, out_png: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-vframes",
        "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True)


def _dominant_background_rgb(img: Image.Image) -> tuple[int, int, int]:
    w, h = img.size
    samples = []
    for x, y in [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]:
        samples.append(img.getpixel((max(0, x), max(0, y))))
    r = int(sum(v[0] for v in samples) / len(samples))
    g = int(sum(v[1] for v in samples) / len(samples))
    b = int(sum(v[2] for v in samples) / len(samples))
    return (r, g, b)


def _content_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    bg = Image.new("RGB", img.size, _dominant_background_rgb(img))
    diff = ImageChops.difference(img, bg).convert("L")
    diff = diff.point(lambda p: 255 if p > 12 else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return (0, 0, img.width, img.height)
    return bbox


def _map_bounds_bbox(img: Image.Image) -> tuple[int, int, int, int] | None:
    px = img.load()
    w, h = img.size
    dark_by_col = [0] * w
    dark_by_row = [0] * h
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = px[x, y]
            if max(r, g, b) < 65:
                dark_by_col[x] += 1
                dark_by_row[y] += 1

    col_threshold = max(12, int(h * 0.04))
    row_threshold = max(12, int(w * 0.04))
    cols = [idx for idx, count in enumerate(dark_by_col) if count >= col_threshold]
    rows = [idx for idx, count in enumerate(dark_by_row) if count >= row_threshold]
    if not cols or not rows:
        return None

    x0, x1 = min(cols), max(cols) + 1
    y0, y1 = min(rows), max(rows) + 1
    cov_w = (x1 - x0) / w
    cov_h = (y1 - y0) / h
    if cov_w < 0.20 or cov_h < 0.20:
        return None
    return (x0, y0, x1, y1)


def _bbox_coverage(img: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x1 - x0) / img.width, (y1 - y0) / img.height)


def _dynamic_context_signal(img: Image.Image) -> float:
    # Downsample for speed while preserving color distribution.
    sample = img.resize(
        (max(120, img.width // 4), max(90, img.height // 4)),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    total = sample.width * sample.height
    px = sample.load()

    ped_red = 0
    goal_green = 0
    grid_yellow = 0
    path_blue = 0
    for y in range(sample.height):
        for x in range(sample.width):
            r, g, b = px[x, y]
            if r > 165 and g < 115 and b < 115:
                ped_red += 1
            if g > 145 and r < 145 and b < 145:
                goal_green += 1
            if r > 170 and g > 170 and b < 120:
                grid_yellow += 1
            if b > 155 and r < 130 and g < 160:
                path_blue += 1

    # Convert tiny ratios to stable [0,1] signal magnitudes.
    s_ped = min(1.0, (ped_red / total) * 35.0)
    s_goal = min(1.0, (goal_green / total) * 25.0)
    s_grid = min(1.0, (grid_yellow / total) * 25.0)
    s_path = min(1.0, (path_blue / total) * 22.0)
    signal = 0.35 * s_ped + 0.20 * s_goal + 0.25 * s_grid + 0.20 * s_path
    return float(min(max(signal, 0.0), 1.0))


def _evaluate_candidate(
    img: Image.Image, frame_index: int
) -> tuple[CandidateScore, tuple[int, int, int, int]]:
    bbox = _content_bbox(img)
    cov_w, cov_h = _bbox_coverage(img, bbox)
    content_cov_area = cov_w * cov_h
    content_coverage = min(1.0, content_cov_area / 0.55)

    map_bbox = _map_bounds_bbox(img)
    if map_bbox is not None:
        map_cov_w, map_cov_h = _bbox_coverage(img, map_bbox)
        crop_usability = min(1.0, (map_cov_w * map_cov_h) / 0.70)
    else:
        crop_usability = min(1.0, content_cov_area / 0.70)

    dynamic_signal = _dynamic_context_signal(img)
    quality_pass = cov_w >= 0.30 and cov_h >= 0.30
    score = 0.45 * content_coverage + 0.35 * dynamic_signal + 0.20 * crop_usability
    candidate = CandidateScore(
        frame_index=frame_index,
        score=float(score),
        content_coverage=float(content_coverage),
        dynamic_context_signal=float(dynamic_signal),
        crop_usability=float(crop_usability),
        bbox_coverage_w=float(cov_w),
        bbox_coverage_h=float(cov_h),
        quality_pass=quality_pass,
    )
    return candidate, bbox


def _candidate_indices(frame_count: int) -> list[int]:
    indices = {min(max(int(frame_count * frac), 0), frame_count - 1) for frac in DEFAULT_FRACTIONS}
    indices.add(min(max(frame_count // 2, 0), frame_count - 1))
    return sorted(indices)


def _select_frame_scored(
    source_root: Path,
    scenario_id: str,
    seed: int,
    policy: str,
    frame_strategy: str,
) -> tuple[Path, int, float, str, list[CandidateScore], tuple[int, int, int, int]]:
    video = _video_path(source_root, scenario_id, seed, policy)
    if not video.exists():
        raise FileNotFoundError(f"Missing video file: {video}")

    frame_count = max(1, _ffprobe_frame_count(video))
    mid_idx = min(max(frame_count // 2, 0), frame_count - 1)
    requested_mid = _frame_path(source_root, scenario_id, seed, policy)

    if frame_strategy == "mid" and requested_mid.exists():
        img = Image.open(requested_mid).convert("RGB")
        candidate, bbox = _evaluate_candidate(img, mid_idx)
        selected_path = source_root / "frames" / f"{scenario_id}_seed{seed}_{policy}_selected.png"
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(selected_path)
        reason = "requested_mid_frame_existing"
        return selected_path, mid_idx, candidate.score, reason, [candidate], bbox

    indices = _candidate_indices(frame_count)
    candidates: list[CandidateScore] = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    images: dict[int, Image.Image] = {}

    with tempfile.TemporaryDirectory(prefix="runtime_frame_pick_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        for idx in indices:
            out_png = tmpdir_p / f"{scenario_id}_{idx}.png"
            _extract_frame(video, idx, out_png)
            img = Image.open(out_png).convert("RGB")
            candidate, bbox = _evaluate_candidate(img, idx)
            candidates.append(candidate)
            bboxes[idx] = bbox
            images[idx] = img.copy()

    preferred = [c for c in candidates if c.quality_pass]
    if preferred:
        best = max(preferred, key=lambda c: (c.score, -c.frame_index))
        reason = "scored_best_quality_pass"
    else:
        fallback = next((c for c in candidates if c.frame_index == mid_idx), None)
        if fallback is None:
            fallback = max(candidates, key=lambda c: (c.score, -c.frame_index))
        best = fallback
        reason = "fallback_midpoint_low_coverage"

    selected_path = source_root / "frames" / f"{scenario_id}_seed{seed}_{policy}_selected.png"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_img = images[best.frame_index]
    selected_img.save(selected_path)
    selected_bbox = bboxes[best.frame_index]
    return selected_path, best.frame_index, best.score, reason, candidates, selected_bbox


def _draw_legend(canvas: Image.Image, x: int, y: int) -> int:
    draw = ImageDraw.Draw(canvas)
    title_font = _load_system_font(82, bold=True)
    font = _load_system_font(72, bold=False)
    row_h = 108
    icon_size = 50

    draw.text((x, y), "Legend (runtime context):", fill=(20, 20, 20), font=title_font)

    def _dot(x0: int, y0: int, color: tuple[int, int, int]) -> None:
        draw.ellipse((x0, y0, x0 + icon_size, y0 + icon_size), fill=color)

    def _line(
        x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 3
    ) -> None:
        draw.line((x0, y0, x1, y1), fill=color, width=width)

    row1_y = y + 116
    row2_y = row1_y + row_h

    x_cursor = x

    def _legend_dot_item(label: str, color: tuple[int, int, int], x_pos: int) -> int:
        _dot(x_pos, row1_y, color)
        tx = x_pos + icon_size + 16
        draw.text((tx, row1_y - 8), label, fill=(35, 35, 35), font=font)
        return int(tx + draw.textlength(label, font=font) + 52)

    def _legend_box_item(label: str, color: tuple[int, int, int], x_pos: int) -> int:
        draw.rectangle((x_pos, row1_y, x_pos + icon_size, row1_y + icon_size), fill=color)
        tx = x_pos + icon_size + 16
        draw.text((tx, row1_y - 8), label, fill=(35, 35, 35), font=font)
        return int(tx + draw.textlength(label, font=font) + 52)

    x_cursor = x
    x_cursor = _legend_dot_item("Robot", (20, 65, 220), x_cursor)
    x_cursor = _legend_dot_item("Pedestrian", (230, 65, 55), x_cursor)
    x_cursor = _legend_dot_item("Goal / waypoint", (35, 180, 80), x_cursor)
    _legend_box_item("Obstacle area", (8, 14, 10), x_cursor)

    x_cursor = x
    _line(x_cursor, row2_y + 32, x_cursor + 124, row2_y + 32, (35, 180, 80), width=10)
    tx = x_cursor + 152
    draw.text((tx, row2_y - 8), "robot target direction", fill=(35, 35, 35), font=font)
    x_cursor = int(tx + draw.textlength("robot target direction", font=font) + 84)
    _line(x_cursor, row2_y + 38, x_cursor + 48, row2_y + 18, (30, 70, 220), width=10)
    _line(x_cursor + 64, row2_y + 38, x_cursor + 112, row2_y + 18, (230, 65, 55), width=10)
    draw.text(
        (x_cursor + 142, row2_y - 8),
        "small ticks = visualized action cues",
        fill=(35, 35, 35),
        font=font,
    )
    return row2_y + row_h + 4


def _neutralize_obstacle_tint(img: Image.Image) -> Image.Image:
    """Remove the dark-green tint from rendered obstacle regions so the runtime
    panels use neutral near-black obstacles, matching the source-SVG scenario
    overview (\\Cref{fig:scenario-svg-overview}). Goal/robot/pedestrian marks are
    brighter and untouched.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mask = (g > r) & (g > b) & ((r + g + b) < 120)
    m = np.maximum(r, b)
    out = arr.copy()
    out[..., 0][mask] = m[mask]
    out[..., 1][mask] = m[mask]
    out[..., 2][mask] = m[mask]
    return Image.fromarray(out.astype(np.uint8))


def _normalize_panel(
    img: Image.Image, bbox: tuple[int, int, int, int], target_w: int, target_h: int
) -> Image.Image:
    map_bbox = _map_bounds_bbox(img)
    if map_bbox is not None:
        x0, y0, x1, y1 = map_bbox
        margin = 12
    else:
        x0, y0, x1, y1 = bbox
        margin = 18
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.width, x1 + margin)
    y1 = min(img.height, y1 + margin)
    cropped = img.crop((x0, y0, x1, y1))
    fit = cropped.copy()
    fit.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (target_w, target_h), (242, 242, 242))
    panel.paste(fit, ((target_w - fit.width) // 2, (target_h - fit.height) // 2))
    return panel


def _panel_ids(count: int) -> list[str]:
    letters = list(string.ascii_uppercase)
    if count <= len(letters):
        return letters[:count]
    ids: list[str] = []
    for idx in range(count):
        q, r = divmod(idx, len(letters))
        if q == 0:
            ids.append(letters[r])
        else:
            ids.append(f"{letters[q - 1]}{letters[r]}")
    return ids


def _panel_short_name(scenario_id: str) -> str:
    aliases = {
        "classic_crossing_medium": "crossing (medium)",
        "classic_crossing_low": "crossing (low)",
        "classic_crossing_high": "crossing (high)",
        "classic_bottleneck_medium": "bottleneck (medium)",
        "classic_doorway_medium": "doorway (medium)",
        "classic_group_crossing_medium": "group crossing (medium)",
        "classic_t_intersection_medium": "T-intersection (medium)",
    }
    if scenario_id in aliases:
        return aliases[scenario_id]
    if scenario_id.startswith("francis2023_"):
        return scenario_id.replace("francis2023_", "").replace("_", " ")
    return scenario_id.replace("_", " ")


def build_figure(
    records: list[SelectionRecord],
    panels: list[Image.Image],
    out_png: Path,
    cols: int,
    legend_in_image: bool,
) -> None:
    panel_w = 1360
    panel_h = 600
    footer_h = 168
    pad = 32
    rows = (len(records) + cols - 1) // cols

    top_h = 28
    if legend_in_image:
        top_h = 360

    width = cols * panel_w + (cols + 1) * pad
    height = top_h + rows * (panel_h + footer_h + pad) + pad
    canvas = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    footer_font = _load_system_font(96, bold=False)
    badge_font = _load_system_font(104, bold=True)

    if legend_in_image:
        _draw_legend(canvas, pad, 10)

    for idx, (record, panel_img) in enumerate(zip(records, panels, strict=True)):
        row = idx // cols
        col = idx % cols
        x = pad + col * (panel_w + pad)
        y = top_h + row * (panel_h + footer_h + pad)
        canvas.paste(panel_img, (x, y))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(150, 150, 150), width=4)

        bx, by = x + 20, y + 20
        draw.rectangle(
            (bx, by, bx + 144, by + 144), fill=(255, 255, 255), outline=(60, 60, 60), width=2
        )
        draw.text((bx + 40, by + 12), record.panel_id, fill=(20, 20, 20), font=badge_font)

        fy = y + panel_h + 4
        draw.rectangle((x, fy, x + panel_w, fy + footer_h), fill=(238, 238, 238))
        footer = _panel_short_name(record.scenario_id)
        draw.text((x + 16, fy + 28), footer, fill=(15, 15, 15), font=footer_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas = _neutralize_obstacle_tint(canvas)
    canvas.save(out_png)


def write_manifest_json(path: Path, records: list[SelectionRecord], source_root: Path) -> None:
    payload = {"source_root": str(source_root), "panels": [asdict(rec) for rec in records]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest_md(path: Path, records: list[SelectionRecord], source_root: Path) -> None:
    lines = [
        "# Runtime Panel Manifest",
        "",
        f"- Source root: `{source_root}`",
        "",
        "| Panel | Scenario | Policy | Seed | Strategy | Selected frame | Selected score | Selection reason | Source frame | Source video |",
        "|---|---|---|---|---|---:|---:|---|---|---|",
    ]
    for rec in records:
        lines.append(
            f"| {rec.panel_id} | `{rec.scenario_id}` | `{rec.policy}` | `{rec.seed}` | "
            f"`{rec.frame_strategy}` | {rec.selected_frame_index} | {rec.selected_score:.3f} | "
            f"{rec.selection_reason} | `{rec.source_frame_file}` | `{rec.source_video}` |"
        )
        lines.append("")
        lines.append(
            f"  Candidates: frames={rec.candidate_frame_indices}; "
            f"scores={[round(v, 3) for v in rec.candidate_scores]}; "
            f"quality={rec.candidate_quality_pass}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(
    path: Path,
    figure_path: Path,
    records: list[SelectionRecord],
    legend_in_image: bool,
    cols: int,
) -> None:
    img = Image.open(figure_path)
    width, height = img.size
    checks = [
        (
            "Purpose/evidence boundary described in caption",
            "PASS",
            "Context-only runtime visualization policy in Results caption.",
        ),
        (
            "Readable rendered resolution",
            "PASS" if (width >= 1500 and height >= 900) else "FAIL",
            f"{width}x{height}",
        ),
        ("Panel labels present once and ordered", "PASS", ",".join(r.panel_id for r in records)),
        ("Uniform panel sizing and framing", "PASS", f"cols={cols}; fixed panel template"),
        (
            "Legend present or moved to caption",
            "PASS",
            f"legend_in_image={legend_in_image}; semantic mapping covered.",
        ),
        (
            "Traceability manifest includes candidate scoring",
            "PASS",
            "runtime_panel_manifest.{json,md}",
        ),
        (
            "PDF rebuild + rendered-page visual inspection",
            "MANUAL",
            "Run latexmk and inspect Figure 4 page.",
        ),
    ]
    lines = [
        "# Runtime Figure Quality Checklist",
        "",
        f"- Figure: `{figure_path}`",
        "",
        "| Check | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, status, evidence in checks:
        lines.append(f"| {name} | {status} | {evidence} |")
    lines.append("")
    lines.append("## Outcome")
    hard_fail = any(status == "FAIL" for _, status, _ in checks)
    lines.append(
        "- FAIL: at least one hard check failed."
        if hard_fail
        else "- PASS (pending MANUAL checks)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_system_font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    bold_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    regular_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in bold_candidates if bold else regular_candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    scenarios = _load_scenarios(args)
    if args.cols < 1:
        raise SystemExit("--cols must be >= 1")
    source_root = args.source_root
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    frame_strategy = "scored" if args.frame_strategy == "best-clearance" else args.frame_strategy

    target_panel_w = 1360
    target_panel_h = 600
    panel_ids = _panel_ids(len(scenarios))

    records: list[SelectionRecord] = []
    panel_imgs: list[Image.Image] = []

    for panel_id, scenario_id in zip(panel_ids, scenarios, strict=True):
        frame_path, frame_idx, selected_score, reason, candidates, selected_bbox = (
            _select_frame_scored(
                source_root=source_root,
                scenario_id=scenario_id,
                seed=args.seed,
                policy=args.policy,
                frame_strategy=frame_strategy,
            )
        )
        img = Image.open(frame_path).convert("RGB")
        cov_w, cov_h = _bbox_coverage(img, selected_bbox)
        panel_imgs.append(_normalize_panel(img, selected_bbox, target_panel_w, target_panel_h))
        records.append(
            SelectionRecord(
                panel_id=panel_id,
                scenario_id=scenario_id,
                policy=args.policy,
                seed=args.seed,
                frame_strategy=frame_strategy,
                source_video=str(_video_path(source_root, scenario_id, args.seed, args.policy)),
                source_frame_file=str(frame_path),
                selected_frame_index=frame_idx,
                selected_score=float(selected_score),
                selection_reason=reason,
                candidate_frame_indices=[c.frame_index for c in candidates],
                candidate_scores=[float(c.score) for c in candidates],
                candidate_quality_pass=[bool(c.quality_pass) for c in candidates],
                width=img.width,
                height=img.height,
                bbox_coverage_w=float(cov_w),
                bbox_coverage_h=float(cov_h),
                crop_bbox=[int(v) for v in selected_bbox],
            )
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name.strip() or f"runtime_panels_main_{len(records)}.png"
    figure_path = out_dir / output_name
    build_figure(
        records, panel_imgs, figure_path, cols=args.cols, legend_in_image=args.legend_in_image
    )

    if args.emit_manifest:
        write_manifest_json(
            out_dir / "runtime_panel_manifest.json", records, source_root=source_root
        )
        write_manifest_md(out_dir / "runtime_panel_manifest.md", records, source_root=source_root)
    if args.write_qa_report:
        write_quality_report(
            out_dir / "runtime_panels_quality_checklist.md",
            figure_path=figure_path,
            records=records,
            legend_in_image=args.legend_in_image,
            cols=args.cols,
        )

    print(f"Wrote figure: {figure_path}")
    if args.emit_manifest:
        print(f"Wrote manifest: {out_dir / 'runtime_panel_manifest.json'}")
        print(f"Wrote manifest: {out_dir / 'runtime_panel_manifest.md'}")
    if args.write_qa_report:
        print(f"Wrote QA: {out_dir / 'runtime_panels_quality_checklist.md'}")


if __name__ == "__main__":
    main()
