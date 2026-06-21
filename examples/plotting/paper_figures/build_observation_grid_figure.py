#!/usr/bin/env python3
"""Build a PPO occupancy-grid observation figure from rendered scenario videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

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
    obstacle_grid_ratio: float
    pedestrian_grid_ratio: float


@dataclass(frozen=True)
class ObservationSelectionRecord:
    scenario_id: str
    policy: str
    seed: int
    source_video: str
    source_frame_file: str
    selected_frame_index: int
    selected_score: float
    selection_reason: str
    candidate_frame_indices: list[int]
    candidate_scores: list[float]
    candidate_quality_pass: list[bool]
    obstacle_grid_ratio: float
    pedestrian_grid_ratio: float
    width: int
    height: int
    crop_bbox: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root containing frames/ and videos_selected/.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="classic_t_intersection_medium",
        help="Scenario ID to visualize.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="ppo",
        help="Policy label in video file names.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Seed label in video file names.",
    )
    parser.add_argument(
        "--frame-strategy",
        choices=["scored", "mid"],
        default="scored",
        help="Frame selection strategy.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/robot_sf_ll7/paper_tools/runtime_visuals"),
        help="Output directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="observation_grid_ppo_classic_t_intersection_medium.png",
        help="Output image file name.",
    )
    return parser.parse_args()


def _video_path(source_root: Path, scenario_id: str, seed: int, policy: str) -> Path:
    return source_root / "videos_selected" / f"{scenario_id}_seed{seed}_{policy}.mp4"


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


def _bbox_coverage(img: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x1 - x0) / img.width, (y1 - y0) / img.height)


def _occupancy_channel_ratios(img: Image.Image) -> tuple[float, float]:
    sample = img.resize(
        (max(120, img.width // 4), max(90, img.height // 4)),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    total = sample.width * sample.height
    obs_count = 0
    ped_count = 0
    px = sample.load()
    for y in range(sample.height):
        for x in range(sample.width):
            r, g, b = px[x, y]
            if r > 175 and g > 165 and b < 120:
                obs_count += 1
            if r > 170 and g < 110 and b < 110:
                ped_count += 1
    return obs_count / total, ped_count / total


def _dynamic_context_signal(img: Image.Image) -> float:
    obs_ratio, ped_ratio = _occupancy_channel_ratios(img)
    sample = img.resize(
        (max(120, img.width // 4), max(90, img.height // 4)),
        Image.Resampling.BILINEAR,
    ).convert("RGB")
    total = sample.width * sample.height
    route_blue_count = 0
    goal_green_count = 0
    px = sample.load()
    for y in range(sample.height):
        for x in range(sample.width):
            r, g, b = px[x, y]
            if b > 155 and r < 130 and g < 160:
                route_blue_count += 1
            if g > 145 and r < 145 and b < 145:
                goal_green_count += 1
    route_blue = route_blue_count / total
    goal_green = goal_green_count / total
    s_obs = min(1.0, obs_ratio * 30.0)
    s_ped = min(1.0, ped_ratio * 40.0)
    s_route = min(1.0, route_blue * 24.0)
    s_goal = min(1.0, goal_green * 20.0)
    return 0.35 * s_obs + 0.30 * s_ped + 0.20 * s_route + 0.15 * s_goal


def _evaluate_candidate(
    img: Image.Image, frame_index: int
) -> tuple[CandidateScore, tuple[int, int, int, int]]:
    bbox = _content_bbox(img)
    cov_w, cov_h = _bbox_coverage(img, bbox)
    content_cov_area = cov_w * cov_h
    content_coverage = min(1.0, content_cov_area / 0.55)
    crop_usability = min(1.0, content_cov_area / 0.70)
    dynamic_signal = _dynamic_context_signal(img)
    obs_ratio, ped_ratio = _occupancy_channel_ratios(img)
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
        obstacle_grid_ratio=float(obs_ratio),
        pedestrian_grid_ratio=float(ped_ratio),
    )
    return candidate, bbox


def _candidate_indices(frame_count: int) -> list[int]:
    indices = {min(max(int(frame_count * frac), 0), frame_count - 1) for frac in DEFAULT_FRACTIONS}
    indices.add(min(max(frame_count // 2, 0), frame_count - 1))
    return sorted(indices)


def _select_frame(
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
    indices = [mid_idx] if frame_strategy == "mid" else _candidate_indices(frame_count)

    candidates: list[CandidateScore] = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    images: dict[int, Image.Image] = {}
    with tempfile.TemporaryDirectory(prefix="obs_grid_pick_") as tmpdir:
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

    selected_path = (
        source_root / "frames" / f"{scenario_id}_seed{seed}_{policy}_observation_selected.png"
    )
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    images[best.frame_index].save(selected_path)
    return selected_path, best.frame_index, best.score, reason, candidates, bboxes[best.frame_index]


def _normalize_panel(
    img: Image.Image, bbox: tuple[int, int, int, int], width: int, height: int
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    margin = 16
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.width, x1 + margin)
    y1 = min(img.height, y1 + margin)
    cropped = img.crop((x0, y0, x1, y1))
    fit = cropped.copy()
    fit.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), (242, 242, 242))
    panel.paste(fit, ((width - fit.width) // 2, (height - fit.height) // 2))
    return panel


def _resize_crop_to_panel(
    img: Image.Image, bbox: tuple[int, int, int, int], width: int, height: int
) -> Image.Image:
    x0, y0, x1, y1 = bbox
    cropped = img.crop((x0, y0, x1, y1))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


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


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> int:
    title_font = _load_system_font(46, bold=True)
    subtitle_font = _load_system_font(32, bold=False)
    font = _load_system_font(34, bold=False)
    draw.text((x, y), "PPO planner-facing observation", fill=(20, 20, 20), font=title_font)
    draw.text(
        (x, y + 58),
        "Occupancy channels, map geometry, robot state, and goal context shown at one selected frame.",
        fill=(55, 55, 55),
        font=subtitle_font,
    )
    y0 = y + 118
    x_cursor = x
    marker_size = 34

    def _box(label: str, color: tuple[int, int, int]) -> None:
        nonlocal x_cursor
        draw.rectangle(
            (x_cursor, y0 + 4, x_cursor + marker_size, y0 + 4 + marker_size),
            fill=color,
            outline=(20, 20, 20),
            width=2,
        )
        x_cursor += marker_size + 14
        draw.text((x_cursor, y0 - 2), label, fill=(30, 30, 30), font=font)
        x_cursor += int(draw.textlength(label, font=font)) + 34

    def _dot(label: str, color: tuple[int, int, int]) -> None:
        nonlocal x_cursor
        draw.ellipse(
            (x_cursor, y0 + 4, x_cursor + marker_size, y0 + 4 + marker_size),
            fill=color,
        )
        x_cursor += marker_size + 14
        draw.text((x_cursor, y0 - 2), label, fill=(30, 30, 30), font=font)
        x_cursor += int(draw.textlength(label, font=font)) + 34

    _box("Obstacle occupancy", (255, 255, 0))
    _box("Pedestrian occupancy", (255, 0, 0))
    _box("Map geometry", (8, 14, 10))
    _dot("Robot", (20, 65, 220))
    _dot("Goal / waypoint", (35, 180, 80))
    return y0 + 48


def _draw_callout(
    draw: ImageDraw.ImageDraw,
    label: str,
    text_xy: tuple[int, int],
    target_xy: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = text_xy
    tx, ty = target_xy
    padding_x = 16
    padding_y = 10
    lines = label.split("\n")
    widths = [draw.textlength(line, font=font) for line in lines]
    line_h = 40
    box_w = int(max(widths) + 2 * padding_x)
    box_h = int(len(lines) * line_h + 2 * padding_y)
    draw.rectangle(
        (x, y, x + box_w, y + box_h), fill=(255, 255, 255), outline=(90, 90, 90), width=2
    )
    for idx, line in enumerate(lines):
        draw.text((x + padding_x, y + padding_y + idx * line_h), line, fill=(30, 30, 30), font=font)
    anchor = (
        min(max(tx, x), x + box_w),
        min(max(ty, y), y + box_h),
    )
    if anchor == (tx, ty):
        anchor = (x + box_w // 2, y + box_h)
    draw.line((anchor[0], anchor[1], tx, ty), fill=(90, 90, 90), width=4)


def build_figure(
    frame: Image.Image,
    record: ObservationSelectionRecord,
    out_path: Path,
) -> None:
    panel_w = 1800
    panel_h = 1464
    pad = 24
    top_h = 218
    width = panel_w + 2 * pad
    height = top_h + panel_h + 2 * pad
    canvas = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    _draw_legend(draw, pad, 10)
    canvas.paste(frame, (pad, top_h))
    draw.rectangle((pad, top_h, pad + panel_w, top_h + panel_h), outline=(150, 150, 150), width=3)
    callout_font = _load_system_font(34, bold=False)
    _draw_callout(
        draw,
        "local observation\nwindow boundary",
        (pad + 1160, top_h + 160),
        (pad + 1510, top_h + 105),
        callout_font,
    )
    _draw_callout(
        draw,
        "robot state +\ngoal vector",
        (pad + 1080, top_h + 670),
        (pad + 930, top_h + 760),
        callout_font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_manifest_json(path: Path, record: ObservationSelectionRecord, source_root: Path) -> None:
    payload = {"source_root": str(source_root), "selection": asdict(record)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_manifest_md(path: Path, record: ObservationSelectionRecord, source_root: Path) -> None:
    lines = [
        "# Observation Grid Figure Manifest",
        "",
        f"- Source root: `{source_root}`",
        "",
        f"- Scenario: `{record.scenario_id}`",
        f"- Policy: `{record.policy}`",
        f"- Seed: `{record.seed}`",
        f"- Selected frame: `{record.selected_frame_index}`",
        f"- Selection score: `{record.selected_score:.3f}` ({record.selection_reason})",
        f"- Obstacle occupancy ratio: `{record.obstacle_grid_ratio:.5f}`",
        f"- Pedestrian occupancy ratio: `{record.pedestrian_grid_ratio:.5f}`",
        "",
        f"- Source frame: `{record.source_frame_file}`",
        f"- Source video: `{record.source_video}`",
        "",
        f"- Candidate frames: `{record.candidate_frame_indices}`",
        f"- Candidate scores: `{[round(v, 3) for v in record.candidate_scores]}`",
        f"- Candidate quality pass: `{record.candidate_quality_pass}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(path: Path, figure_path: Path, record: ObservationSelectionRecord) -> None:
    img = Image.open(figure_path)
    width, height = img.size
    has_obs = record.obstacle_grid_ratio > 0.0005
    has_ped = record.pedestrian_grid_ratio > 0.0005
    checks = [
        (
            "Figure resolution readable",
            "PASS" if width >= 1200 and height >= 700 else "FAIL",
            f"{width}x{height}",
        ),
        (
            "Obstacle occupancy channel visible (yellow)",
            "PASS" if has_obs else "MANUAL_REVIEW",
            f"ratio={record.obstacle_grid_ratio:.5f}",
        ),
        (
            "Pedestrian occupancy channel visible (red)",
            "PASS" if has_ped else "MANUAL_REVIEW",
            f"ratio={record.pedestrian_grid_ratio:.5f}",
        ),
        ("Legend semantics included", "PASS", "in-image legend drawn"),
        ("PDF rebuild + page inspection", "MANUAL", "Run latexmk and inspect Figure 3 page."),
    ]
    lines = [
        "# Observation Grid Figure Quality Checklist",
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


def main() -> None:
    args = parse_args()
    source_root = args.source_root
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    frame_path, frame_idx, selected_score, reason, candidates, bbox = _select_frame(
        source_root=source_root,
        scenario_id=args.scenario_id,
        seed=args.seed,
        policy=args.policy,
        frame_strategy=args.frame_strategy,
    )
    img = Image.open(frame_path).convert("RGB")
    if args.scenario_id == "classic_t_intersection_medium" and args.policy == "ppo":
        # The manuscript observation frame includes simulator UI margins around
        # the planner-facing map. Tightening this known crop keeps the evidence
        # content unchanged while reducing empty space in the paper figure.
        bbox = (180, 0, 1065, 720)
        panel = _resize_crop_to_panel(img, bbox, width=1800, height=1464)
    else:
        panel = _normalize_panel(img, bbox, width=1800, height=1464)
    obs_ratio, ped_ratio = _occupancy_channel_ratios(img)

    record = ObservationSelectionRecord(
        scenario_id=args.scenario_id,
        policy=args.policy,
        seed=args.seed,
        source_video=str(_video_path(source_root, args.scenario_id, args.seed, args.policy)),
        source_frame_file=str(frame_path),
        selected_frame_index=frame_idx,
        selected_score=float(selected_score),
        selection_reason=reason,
        candidate_frame_indices=[c.frame_index for c in candidates],
        candidate_scores=[float(c.score) for c in candidates],
        candidate_quality_pass=[bool(c.quality_pass) for c in candidates],
        obstacle_grid_ratio=float(obs_ratio),
        pedestrian_grid_ratio=float(ped_ratio),
        width=img.width,
        height=img.height,
        crop_bbox=[int(v) for v in bbox],
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / args.output_name
    build_figure(panel, record, figure_path)
    write_manifest_json(out_dir / "observation_grid_manifest.json", record, source_root=source_root)
    write_manifest_md(out_dir / "observation_grid_manifest.md", record, source_root=source_root)
    write_quality_report(out_dir / "observation_grid_quality_checklist.md", figure_path, record)

    print(f"Wrote figure: {figure_path}")
    print(f"Wrote manifest: {out_dir / 'observation_grid_manifest.json'}")
    print(f"Wrote manifest: {out_dir / 'observation_grid_manifest.md'}")
    print(f"Wrote QA: {out_dir / 'observation_grid_quality_checklist.md'}")


if __name__ == "__main__":
    main()
