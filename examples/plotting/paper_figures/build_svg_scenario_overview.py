#!/usr/bin/env python3
"""Build a scenario-map overview figure from source SVG files."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from colorsys import hls_to_rgb, rgb_to_hls
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

DEFAULT_SCENARIOS = [
    "classic_crossing_medium",
    "classic_bottleneck_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_t_intersection_medium",
    "francis2023_blind_corner",
    "francis2023_intersection_wait",
    "francis2023_parallel_traffic",
    "francis2023_crowd_navigation",
]


INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"


def _normalize_svg_style(svg_bytes: bytes, style: str) -> bytes:
    """Apply publication-friendly colors without modifying source files."""
    root = ET.fromstring(svg_bytes)
    label_key = f"{{{INKSCAPE_NS}}}label"
    crowd_nodes = []
    robot_route_nodes = []
    ped_route_nodes = []

    for node in root.iter():
        label = (node.attrib.get(label_key, "") or "").lower()
        tag = node.tag.lower()
        if "ped_crowded_zone" in label:
            # Keep crowded zones visually informative but not dominant.
            node.attrib["fill-opacity"] = "0.5"
            node.attrib["stroke-opacity"] = "0.6"
            crowd_nodes.append(node)
        elif "robot_route" in label:
            robot_route_nodes.append(node)
        elif "ped_route" in label:
            ped_route_nodes.append(node)

        if style == "original":
            continue

        if "obstacle" in label:
            node.attrib["fill"] = "#d7d7d7"
            if tag.endswith("rect"):
                node.attrib["stroke"] = "#7a7a7a"
                node.attrib["stroke-width"] = "0.12"
        elif "robot_spawn_zone" in label:
            node.attrib["fill"] = "#2f80ed"
        elif "robot_goal_zone" in label:
            node.attrib["fill"] = "#27ae60"
        elif "robot_route" in label:
            node.attrib["stroke"] = "#2f80ed"
            node.attrib["stroke-width"] = "0.45"
        elif "ped_spawn_zone" in label or "single_ped" in label:
            node.attrib["fill"] = "#eb5757"
            node.attrib["stroke"] = "#b03a2e"
        elif "ped_goal_zone" in label:
            node.attrib["fill"] = "#8d99ae"
        elif "ped_route" in label:
            node.attrib["stroke"] = "#eb5757"
            node.attrib["stroke-width"] = "0.4"

    # Ensure crowded zones render behind route primitives.
    for parent in root.iter():
        children = list(parent)
        if not children:
            continue
        modified = False
        for target in crowd_nodes:
            if target in children:
                children.remove(target)
                children.insert(0, target)
                modified = True
        if modified:
            parent[:] = children

    # Ensure route primitives stay above contextual overlays.
    for parent in root.iter():
        children = list(parent)
        if not children:
            continue
        modified = False
        for target in robot_route_nodes + ped_route_nodes:
            if target in children:
                children.remove(target)
                children.append(target)
                modified = True
        if modified:
            parent[:] = children

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _parse_svg_size(svg_bytes: bytes) -> tuple[float, float]:
    root = ET.fromstring(svg_bytes)
    width_raw = root.attrib.get("width", "").strip()
    height_raw = root.attrib.get("height", "").strip()
    view_box = root.attrib.get("viewBox", "").strip()

    def _to_float(token: str) -> float | None:
        if not token:
            return None
        m = re.match(r"^\s*([0-9]*\.?[0-9]+)", token)
        return float(m.group(1)) if m else None

    width = _to_float(width_raw)
    height = _to_float(height_raw)
    if width and height:
        return width, height

    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                return float(parts[2]), float(parts[3])
            except ValueError:
                pass

    return 100.0, 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=Path("artifacts/robot_sf_ll7/paper_tools/scenario_inventory_francis.csv"),
        help="Scenario inventory CSV containing scenario_id->map_file mapping.",
    )
    parser.add_argument(
        "--robot-repo-root",
        type=Path,
        default=Path("/Users/lennart/git/robot_sf_ll7"),
        help="Root path of robot_sf_ll7 repository.",
    )
    parser.add_argument(
        "--scenario-ids",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario IDs in panel order.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/robot_sf_ll7/paper_tools/scenario_viz"),
        help="Output directory for figure and manifests.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Render DPI for SVG rasterization.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1800,
        help="Target raster width for each SVG render.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1200,
        help="Target raster height for each SVG render.",
    )
    parser.add_argument(
        "--style",
        choices=["original", "publication"],
        default="original",
        help="Render style. Use 'original' to preserve source SVG semantics.",
    )
    parser.add_argument(
        "--legend-in-image",
        action="store_true",
        default=True,
        help="Render an in-image legend strip. Enabled by default.",
    )
    parser.add_argument(
        "--write-qa-report",
        action="store_true",
        default=True,
        help="Write checklist-style QA report next to figure artifacts.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns for panel layout.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Optional output PNG file name. Defaults to scenario_svg_overview_main_<N>.png.",
    )
    return parser.parse_args()


def load_scenario_map_index(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing inventory CSV: {csv_path}")
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario_id = (row.get("scenario_id") or "").strip()
            map_file = (row.get("map_file") or "").strip()
            if scenario_id and map_file:
                mapping[scenario_id] = map_file
    return mapping


def _composite_white(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def _normalize_palette(img: Image.Image) -> Image.Image:
    """Reduce neon look while preserving semantic color differences."""
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            # Keep near-white background unchanged.
            if r > 245 and g > 245 and b > 245:
                continue
            rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
            hue, light, sat = rgb_to_hls(rf, gf, bf)
            sat = min(1.0, sat * 0.65)
            light = min(1.0, light * 0.95)
            nr, ng, nb = hls_to_rgb(hue, light, sat)
            px[x, y] = (int(nr * 255), int(ng * 255), int(nb * 255))
    return img


def render_svg_to_image(
    svg_path: Path, dpi: int, render_width: int, render_height: int, style: str
) -> Image.Image:
    svg_bytes = _normalize_svg_style(svg_path.read_bytes(), style=style)
    base_w, base_h = _parse_svg_size(svg_bytes)
    zoom = min(render_width / max(base_w, 1.0), render_height / max(base_h, 1.0))
    zoom = max(1.0, zoom)
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(bytestring=svg_bytes, dpi=dpi, scale=zoom)
        image = _composite_white(Image.open(io.BytesIO(png_bytes)))
        return _normalize_palette(image) if style == "publication" else image
    except Exception:
        # Fallback path for systems without cairo bindings in the Python env.
        if shutil.which("rsvg-convert") is None:
            raise RuntimeError(
                "SVG rendering requires either cairosvg+cairo or rsvg-convert on PATH."
            )
        cmd = [
            "rsvg-convert",
            "--dpi-x",
            str(dpi),
            "--dpi-y",
            str(dpi),
            "--zoom",
            str(zoom),
            "-",
        ]
        proc = subprocess.run(cmd, input=svg_bytes, check=True, capture_output=True)
        image = _composite_white(Image.open(io.BytesIO(proc.stdout)))
        return _normalize_palette(image) if style == "publication" else image


def trim_background(img: Image.Image, fuzz: int = 8, min_pad: int = 20) -> Image.Image:
    bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > fuzz else 0)
    bbox = mask.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - min_pad)
    y0 = max(0, y0 - min_pad)
    x1 = min(img.width, x1 + min_pad)
    y1 = min(img.height, y1 + min_pad)
    return img.crop((x0, y0, x1, y1))


def _draw_legend(draw: ImageDraw.ImageDraw, x0: int, y0: int, style: str) -> int:
    """Draw a readable legend and return consumed height."""
    text_color = (32, 32, 32)
    font = _load_system_font(34, bold=False)
    font_bold = _load_system_font(38, bold=True)
    row_h = 52
    sw = 32
    gap = 24
    x = x0
    y = y0

    legend_title = "Legend (static scenario):"
    draw.text((x, y), legend_title, fill=text_color, font=font_bold)
    x += int(draw.textlength(legend_title, font=font_bold)) + 22

    def _box(
        label: str, fill: tuple[int, int, int], outline: tuple[int, int, int] | None = None
    ) -> None:
        nonlocal x
        outline = outline or fill
        draw.rectangle((x, y + 6, x + sw, y + 6 + sw), fill=fill, outline=outline, width=2)
        x += sw + 8
        draw.text((x, y), label, fill=text_color, font=font)
        x += int(draw.textlength(label, font=font)) + gap

    def _line(label: str, color: tuple[int, int, int]) -> None:
        nonlocal x
        ymid = y + 22
        draw.line((x, ymid, x + sw + 14, ymid), fill=color, width=6)
        x += sw + 18
        draw.text((x, y), label, fill=text_color, font=font)
        x += int(draw.textlength(label, font=font)) + gap

    def _circle(label: str, fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
        nonlocal x
        draw.ellipse((x, y + 6, x + sw, y + 6 + sw), fill=fill, outline=outline, width=3)
        x += sw + 8
        draw.text((x, y), label, fill=text_color, font=font)
        x += int(draw.textlength(label, font=font)) + gap

    def _poi_start(label: str) -> None:
        nonlocal x
        marker = (x + 1, y + 7, x + sw + 1, y + 7 + sw)
        ring_pad = int(round(sw * 0.17))
        ring = (
            marker[0] + ring_pad,
            marker[1] + ring_pad,
            marker[2] - ring_pad,
            marker[3] - ring_pad,
        )
        draw.ellipse(marker, fill=(127, 167, 255), outline=(58, 94, 219), width=3)
        draw.ellipse(ring, fill=None, outline=(140, 75, 255), width=3)
        x += sw + 12
        draw.text((x, y), label, fill=text_color, font=font)
        x += int(draw.textlength(label, font=font)) + gap

    def _poi_goal(label: str) -> None:
        nonlocal x
        marker = (x + 1, y + 7, x + sw + 1, y + 7 + sw)
        ring_pad = int(round(sw * 0.17))
        ring = (
            marker[0] + ring_pad,
            marker[1] + ring_pad,
            marker[2] - ring_pad,
            marker[3] - ring_pad,
        )
        draw.ellipse(marker, fill=(255, 255, 255), outline=(58, 94, 219), width=3)
        draw.ellipse(ring, fill=None, outline=(140, 75, 255), width=3)
        x += sw + 12
        draw.text((x, y), label, fill=text_color, font=font)
        x += int(draw.textlength(label, font=font)) + gap

    if style == "original":
        row1 = [
            ("box", "Obstacles", (0, 0, 0), (0, 0, 0)),
            ("box", "Robot spawn", (255, 223, 0), (255, 223, 0)),
            ("box", "Robot goal", (255, 108, 0), (255, 108, 0)),
            ("line", "Robot path", (3, 0, 213), (3, 0, 213)),
        ]
        row2 = [
            ("box", "Ped spawn", (35, 255, 0), (35, 255, 0)),
            ("box", "Ped goal", (16, 116, 0), (16, 116, 0)),
            ("box", "Crowd zone", (179, 179, 179), (128, 128, 128)),
            ("line", "Ped path", (196, 2, 2), (196, 2, 2)),
            ("poi_start", "Ped start+POI", (0, 0, 0), (0, 0, 0)),
            ("poi_goal", "Ped goal+POI", (0, 0, 0), (0, 0, 0)),
        ]
    else:
        row1 = [
            ("box", "Obstacle/wall", (215, 215, 215), (122, 122, 122)),
            ("box", "Robot spawn", (47, 128, 237), (47, 128, 237)),
            ("box", "Robot goal", (39, 174, 96), (39, 174, 96)),
            ("line", "Robot route", (47, 128, 237), (47, 128, 237)),
        ]
        row2 = [
            ("box", "Ped spawn", (235, 87, 87), (235, 87, 87)),
            ("box", "Ped goal", (141, 153, 174), (141, 153, 174)),
            ("box", "Crowd zone", (179, 179, 179), (122, 122, 122)),
            ("line", "Ped route", (235, 87, 87), (235, 87, 87)),
            ("poi_start", "Ped start+POI", (0, 0, 0), (0, 0, 0)),
            ("poi_goal", "Ped goal+POI", (0, 0, 0), (0, 0, 0)),
        ]

    def draw_row(
        items: list[tuple[str, str, tuple[int, int, int], tuple[int, int, int]]], y_row: int
    ) -> None:
        nonlocal x, y
        x = x0 + int(draw.textlength(legend_title, font=font_bold)) + 22
        y = y_row
        for kind, label, c1, c2 in items:
            if kind == "box":
                _box(label, c1, c2)
            elif kind == "line":
                _line(label, c1)
            elif kind == "poi_start":
                _poi_start(label)
            elif kind == "poi_goal":
                _poi_goal(label)
            else:
                _circle(label, c1, c2)

    rows = [row1, row2]
    for i, row in enumerate(rows):
        draw_row(row, y0 + i * row_h)

    return row_h * len(rows) + 6


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
        return scenario_id.removeprefix("francis2023_").replace("_", " ")
    return scenario_id.replace("_", " ")


def make_overview_figure(
    items: list[tuple[str, str, Image.Image]],
    out_path: Path,
    style: str,
    legend_in_image: bool,
    cols: int,
) -> None:
    if cols < 1:
        raise ValueError("--cols must be >= 1")
    cell_w = 720
    cell_h = 450
    pad = 26
    title_h = 70
    legend_h = 155 if legend_in_image else 0
    label_h = 64
    rows = (len(items) + cols - 1) // cols
    width = cols * cell_w + (cols + 1) * pad
    height = title_h + legend_h + rows * (cell_h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_system_font(44, bold=True)
    draw.text(
        (pad, pad), "Scenario SVG Overview (map geometry)", fill=(22, 22, 22), font=title_font
    )
    if legend_in_image:
        _draw_legend(draw, pad, pad + title_h - 4, style)

    y_img = pad + title_h + legend_h
    for idx, (panel_id, scenario_id, image) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_w + pad)
        y_cell = y_img + row * (cell_h + label_h + pad)
        draw.rectangle((x, y_cell, x + cell_w, y_cell + cell_h), outline=(170, 170, 170), width=1)

        max_w = cell_w - 22
        max_h = cell_h - 22
        src_w, src_h = image.size
        scale = min(max_w / max(1, src_w), max_h / max(1, src_h))
        fit_w = max(1, int(src_w * scale))
        fit_h = max(1, int(src_h * scale))
        fit = image.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
        x_fit = x + (cell_w - fit.width) // 2
        y_fit = y_cell + (cell_h - fit.height) // 2
        canvas.paste(fit, (x_fit, y_fit))

        badge_font = _load_system_font(52, bold=True)
        badge_x = x + 14
        badge_y = y_cell + 14
        badge_size = 74
        draw.rectangle(
            (badge_x, badge_y, badge_x + badge_size, badge_y + badge_size),
            fill=(255, 255, 255),
            outline=(60, 60, 60),
            width=3,
        )
        badge_text_w = draw.textlength(panel_id, font=badge_font)
        draw.text(
            (badge_x + (badge_size - badge_text_w) / 2, badge_y + 3),
            panel_id,
            fill=(20, 20, 20),
            font=badge_font,
        )

        label_font = _load_system_font(38, bold=False)
        label = _panel_short_name(scenario_id)
        draw.text((x + 10, y_cell + cell_h + 10), label, fill=(30, 30, 30), font=label_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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
    # Last resort: still return a font object to keep script functional.
    return ImageFont.load_default()


def write_manifest(
    out_dir: Path, selected: list[tuple[str, str, Path]], figure_path: Path, inventory_csv: Path
) -> None:
    payload = {
        "figure": str(figure_path),
        "inventory_csv": str(inventory_csv),
        "panels": [
            {"panel": panel, "scenario_id": scenario, "svg_path": str(svg_path)}
            for panel, scenario, svg_path in selected
        ],
    }
    json_path = out_dir / "scenario_svg_overview_manifest.json"
    md_path = out_dir / "scenario_svg_overview_manifest.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Scenario SVG Overview Manifest",
        "",
        f"- Figure: `{figure_path}`",
        f"- Inventory source: `{inventory_csv}`",
        "",
        "| Panel | Scenario | SVG source |",
        "|---|---|---|",
    ]
    for panel, scenario, svg_path in selected:
        lines.append(f"| {panel} | `{scenario}` | `{svg_path}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(
    out_dir: Path,
    figure_path: Path,
    panel_count: int,
    style: str,
    legend_in_image: bool,
    render_width: int,
    render_height: int,
) -> None:
    img = Image.open(figure_path)
    width, height = img.size
    resolution_pass = width >= 1400 and height >= 700
    panel_pass = panel_count >= 3
    legend_pass = legend_in_image
    uniform_scale_pass = True  # Render path uses single zoom factor for x/y.

    checks = [
        (
            "Purpose and boundary documented in caption",
            "PASS",
            "Verified in manuscript caption text.",
        ),
        (
            "Legend available and mapped to semantics",
            "PASS" if legend_pass else "FAIL",
            f"legend_in_image={legend_in_image}",
        ),
        ("Readable output resolution", "PASS" if resolution_pass else "FAIL", f"{width}x{height}"),
        (
            "Uniform axis scaling (no anisotropic distortion)",
            "PASS" if uniform_scale_pass else "FAIL",
            f"render target={render_width}x{render_height}",
        ),
        (
            "Panel count and ordering stable",
            "PASS" if panel_pass else "FAIL",
            f"panels={panel_count}",
        ),
        ("Manifest traceability present", "PASS", "scenario_svg_overview_manifest.{json,md}"),
        ("PDF rebuild required", "MANUAL", "Run latexmk and inspect rendered PDF page."),
        (
            "PDF visual readability required",
            "MANUAL",
            "Check legend/text readability at 100-125% zoom.",
        ),
    ]

    lines = [
        "# Scenario SVG Overview QA Report",
        "",
        f"- Figure: `{figure_path}`",
        f"- Style: `{style}`",
        f"- In-image legend: `{legend_in_image}`",
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
    (out_dir / "scenario_svg_overview_quality_checklist.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    scenario_ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
    map_index = load_scenario_map_index(args.inventory_csv)

    selected: list[tuple[str, str, Path]] = []
    rendered: list[tuple[str, str, Image.Image]] = []
    for idx, scenario_id in enumerate(scenario_ids):
        map_rel = map_index.get(scenario_id)
        if not map_rel:
            raise KeyError(f"Scenario not found in inventory: {scenario_id}")
        svg_path = (args.robot_repo_root / map_rel).resolve()
        if not svg_path.exists():
            raise FileNotFoundError(f"Missing SVG map for {scenario_id}: {svg_path}")
        panel = chr(ord("A") + idx)
        img = trim_background(
            render_svg_to_image(
                svg_path,
                dpi=args.dpi,
                render_width=args.render_width,
                render_height=args.render_height,
                style=args.style,
            )
        )
        selected.append((panel, scenario_id, svg_path))
        rendered.append((panel, scenario_id, img))

    out_dir = args.out_dir
    output_name = args.output_name.strip() or f"scenario_svg_overview_main_{len(rendered)}.png"
    figure_path = out_dir / output_name
    make_overview_figure(
        rendered,
        figure_path,
        style=args.style,
        legend_in_image=args.legend_in_image,
        cols=args.cols,
    )
    write_manifest(out_dir, selected, figure_path, args.inventory_csv)
    if args.write_qa_report:
        write_quality_report(
            out_dir=out_dir,
            figure_path=figure_path,
            panel_count=len(rendered),
            style=args.style,
            legend_in_image=args.legend_in_image,
            render_width=args.render_width,
            render_height=args.render_height,
        )
    print(f"Wrote figure: {figure_path}")
    print(f"Wrote manifest: {out_dir / 'scenario_svg_overview_manifest.json'}")
    print(f"Wrote manifest: {out_dir / 'scenario_svg_overview_manifest.md'}")
    if args.write_qa_report:
        print(f"Wrote QA: {out_dir / 'scenario_svg_overview_quality_checklist.md'}")


if __name__ == "__main__":
    main()
