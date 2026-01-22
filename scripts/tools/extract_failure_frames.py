#!/usr/bin/env python3
"""Extract key frames from failed scenario videos.

This helper reads a summary/manifest produced by render_scenario_videos.py and
extracts the first frame plus a set of frame offsets from the end of each
failed video. It is intended for quick visual triage of failures without
watching full videos.

Example:
  uv run python scripts/tools/extract_failure_frames.py \
    --summary output/recordings/scenario_videos_francis2023_socnav_social_force_20260122_153356/summary.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from PIL import Image


@dataclass
class SummaryRow:
    """Row parsed from the video summary table."""

    scenario: str
    seed: int
    policy: str
    steps: int
    max_steps: int
    stop_reason: str
    success: bool
    collision: bool
    status: str
    video: str


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.md produced by render_scenario_videos.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory for extracted frames (defaults next to summary).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        action="append",
        default=[-50, -30, -20, -10, -5],
        help="Frame offsets from the end to capture (repeatable).",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        default=True,
        help="Only extract frames for entries where success=false (default: true).",
    )
    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        default=True,
        help="Generate a contact sheet per video (default: true).",
    )
    parser.add_argument(
        "--sheet-cols",
        type=int,
        default=4,
        help="Number of columns in contact sheets (default: 4).",
    )
    parser.add_argument(
        "--sheet-width",
        type=int,
        default=320,
        help="Resize width for each tile in the contact sheet (default: 320).",
    )
    return parser


def _parse_summary(summary_path: Path) -> list[SummaryRow]:
    """Parse summary.md into structured rows."""
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        if line.strip().startswith("| ---"):
            continue
        if "scenario" in line and "seed" in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 10:
            continue
        rows.append(
            SummaryRow(
                scenario=parts[0],
                seed=int(parts[1]),
                policy=parts[2],
                steps=int(parts[3]),
                max_steps=int(parts[4]),
                stop_reason=parts[5],
                success=parts[6].lower() == "true",
                collision=parts[7].lower() == "true",
                status=parts[8],
                video=parts[9],
            )
        )
    return rows


def _get_frame_count(video_path: Path) -> int:
    """Return the total frame count using ffprobe."""
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
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"ffprobe failed for {video_path}: {msg}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if count is None:
        raise RuntimeError(f"Frame count unavailable for {video_path}")
    return int(count)


def _extract_frame(video_path: Path, frame_index: int, out_path: Path) -> None:
    """Extract a single frame using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vframes",
        "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        msg = result.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg failed for {video_path}: {msg}")


def _frame_label(frame_index: int, total_frames: int) -> str:
    """Label a frame based on its relation to the end of the video."""
    if frame_index <= 0:
        return "start"
    if frame_index >= total_frames - 1:
        return "end"
    offset = frame_index - total_frames
    return f"end{offset:+d}"


def _make_contact_sheet(
    frame_paths: list[Path],
    *,
    output_path: Path,
    cols: int,
    tile_width: int,
) -> None:
    """Create a simple contact sheet image from extracted frames."""
    if not frame_paths:
        return
    images = []
    for path in frame_paths:
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            continue
        if tile_width > 0:
            ratio = tile_width / img.width
            height = max(1, int(img.height * ratio))
            img = img.resize((tile_width, height), Image.BILINEAR)
        images.append(img)
    if not images:
        return

    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    tile_h = max(img.height for img in images)
    tile_w = max(img.width for img in images)
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * tile_h
        sheet.paste(img, (x, y))

    sheet.save(output_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    summary_path = args.summary.resolve()
    rows = _parse_summary(summary_path)

    if not rows:
        logger.warning("No rows found in {}", summary_path)
        return 0

    output_root = args.output or (summary_path.parent / "failure_frames")
    output_root.mkdir(parents=True, exist_ok=True)

    offsets = sorted(set(int(val) for val in args.offset))

    extracted = 0
    for row in rows:
        if args.only_failures and row.success:
            continue
        video_path = summary_path.parent / row.video
        if not video_path.exists():
            logger.warning("Missing video: {}", video_path)
            continue
        try:
            total_frames = _get_frame_count(video_path)
        except RuntimeError as exc:
            logger.warning("Skipping {}: {}", video_path.name, exc)
            continue

        frame_indices = {0, max(0, total_frames - 1)}
        for offset in offsets:
            idx = max(0, min(total_frames - 1, total_frames + offset))
            frame_indices.add(idx)
        frame_indices = sorted(frame_indices)

        out_dir = output_root / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_outputs: list[Path] = []
        for frame_index in frame_indices:
            label = _frame_label(frame_index, total_frames)
            out_path = out_dir / f"{video_path.stem}_{label}_frame{frame_index:06d}.png"
            try:
                _extract_frame(video_path, frame_index, out_path)
                frame_outputs.append(out_path)
                extracted += 1
            except RuntimeError as exc:
                logger.warning("Failed to extract frame {}: {}", frame_index, exc)
        if args.contact_sheets:
            sheet_path = out_dir / "contact_sheet.png"
            _make_contact_sheet(
                frame_outputs,
                output_path=sheet_path,
                cols=int(args.sheet_cols),
                tile_width=int(args.sheet_width),
            )

    logger.info("Extracted {} frames into {}", extracted, output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
