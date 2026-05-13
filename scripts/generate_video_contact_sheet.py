"""Generate thumbnail contact sheets from episode frame artifacts.

The first supported input contract is intentionally lightweight: each episode
JSONL row may provide image frame paths in either ``frame_paths`` or
``video.frame_paths``. Relative paths resolve next to the JSONL file. Rows that
only contain MP4 paths should be pre-extracted to frame images before calling
this helper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def _iter_jsonl_rows(path: Path) -> list[dict]:
    """Load dictionary rows from an episode JSONL file."""
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Episode row {line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _resolve_frame_paths(row: dict, base_dir: Path) -> list[Path]:
    """Resolve frame image paths from a JSONL episode row."""
    video_meta = row.get("video") if isinstance(row.get("video"), dict) else {}
    raw_paths = row.get("frame_paths") or video_meta.get("frame_paths") or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list):
        return []

    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = base_dir / path
        paths.append(path)
    return paths


def _load_frame_images(frame_paths: list[Path]) -> list[Image.Image]:
    """Open frame image paths as RGB Pillow images."""
    images: list[Image.Image] = []
    for path in frame_paths:
        if not path.exists():
            raise FileNotFoundError(f"Frame image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def _make_sheet(images: list[Image.Image], *, columns: int) -> Image.Image:
    """Assemble RGB images into a fixed-size grid."""
    if not images:
        raise ValueError("No frame image paths found in episode JSONL")
    columns = max(1, columns)
    rows = (len(images) + columns - 1) // columns
    tile_width = max(image.width for image in images)
    tile_height = max(image.height for image in images)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(255, 255, 255))
    for index, image in enumerate(images):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y))
    return sheet


def generate_contact_sheet(
    episodes_jsonl: Path,
    output_path: Path,
    *,
    columns: int = 3,
) -> Path:
    """Generate a contact-sheet PNG from episode frame image metadata.

    Returns:
        Path to the written contact-sheet image.
    """
    base_dir = episodes_jsonl.parent
    rows = _iter_jsonl_rows(episodes_jsonl)
    frame_paths: list[Path] = []
    for row in rows:
        frame_paths.extend(_resolve_frame_paths(row, base_dir))
    if not frame_paths:
        raise ValueError("No frame image paths found in episode JSONL")

    images = _load_frame_images(frame_paths)
    sheet = _make_sheet(images, columns=columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for contact-sheet generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_jsonl", type=Path, help="Episode JSONL with frame_paths metadata")
    parser.add_argument("output_path", type=Path, help="PNG path for the contact sheet")
    parser.add_argument("--columns", type=int, default=3, help="Number of columns in the grid")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the contact-sheet generator CLI."""
    args = _build_parser().parse_args(argv)
    generate_contact_sheet(args.episodes_jsonl, args.output_path, columns=args.columns)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
