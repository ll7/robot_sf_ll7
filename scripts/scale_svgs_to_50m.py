"""Scale SVG scenario files to have dimensions < 50m.

Converts a 400x400 viewBox to 40x40 and scales all coordinates accordingly.
Every input path is explicit: a bare invocation fails instead of rewriting the
tracked maps that originally motivated this one-off utility.
"""

import argparse
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path


def scale_coordinate(value_str: str, scale_factor: float) -> str:
    """Scale a coordinate value string."""
    try:
        value = float(value_str)
        scaled = value * scale_factor
        # Keep precision but remove unnecessary decimals
        if scaled == int(scaled):
            return str(int(scaled))
        return f"{scaled:.1f}"
    except ValueError:
        return value_str


def scale_path_data(d: str, scale_factor: float) -> str:
    """Scale all coordinates in SVG path data."""

    # Match numbers (including negative and decimals)
    def replace_number(match):
        """Scale one numeric regex match by the enclosing scale factor.

        Args:
            match: Regex match whose full text is a single coordinate token.

        Returns:
            The scaled coordinate string; non-numeric tokens are returned
            unchanged by :func:`scale_coordinate`.
        """
        return scale_coordinate(match.group(0), scale_factor)

    return re.sub(r"-?\d+\.?\d*", replace_number, d)


def scale_svg_file(input_path: str, output_path: str, scale_factor: float = 0.1):
    """Scale an entire SVG file by the given factor."""
    input_path_obj = Path(input_path)
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Update viewBox (this is the actual coordinate space in meters)
    viewbox = root.get("viewBox", "")
    if viewbox:
        parts = viewbox.split()
        if len(parts) == 4:
            scaled_parts = [scale_coordinate(p, scale_factor) for p in parts]
            root.set("viewBox", " ".join(scaled_parts))

    # Keep width and height attributes for display (don't scale these)
    # These are just for rendering, not for physical dimensions

    # Scale all elements
    for elem in root.iter():
        # Scale geometric attributes
        for attr in ["x", "y", "cx", "cy", "r", "width", "height", "stroke-width"]:
            attr_value = elem.get(attr)
            if attr_value is not None:
                elem.set(attr, scale_coordinate(attr_value, scale_factor))

        # Scale path data
        path_data = elem.get("d")
        if path_data is not None:
            elem.set("d", scale_path_data(path_data, scale_factor))

    # Write to output with proper formatting
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    # Fix formatting (ET doesn't preserve it well)
    with open(output_path, encoding="utf-8") as f:
        content = f.read()

    # Add the comment back at the top
    scenario_name = input_path_obj.stem.replace("_", " ").title()
    comment = (
        f"<!-- {scenario_name} | Dimensions < 50m | Scale: 1 SVG unit = 1 meter (SI units) -->\n"
    )

    # Remove the old comment if present and add new one
    content = re.sub(r"<!--[\s\S]*?-->\s*", "", content, count=1)
    content = content.replace(
        "<?xml version='1.0' encoding='UTF-8'?>\n",
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n{comment}',
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Inputs are always explicit; a bare run never rewrites tracked maps."""
    parser = argparse.ArgumentParser(
        prog="scale_svgs_to_50m",
        description=(
            "Scale SVG scenario files below 50 m (400x400 viewBox to 40x40). "
            "Every input path must be named explicitly."
        ),
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="SVG files to scale.")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.1,
        help="Multiplier applied to coordinates and the viewBox (default: 0.1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write scaled copies here instead of overwriting each input in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned writes without touching any file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Scale the explicitly named SVGs and return the process exit code."""
    args = _build_parser().parse_args(argv)
    if not args.scale_factor:
        raise SystemExit("--scale-factor must be non-zero")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for input_path in args.inputs:
        if not input_path.is_file():
            raise SystemExit(f"input is not a file: {input_path}")
        target = input_path if args.output_dir is None else args.output_dir / input_path.name
        if args.dry_run:
            print(f"would scale {input_path} -> {target}")
            continue
        scale_svg_file(str(input_path), str(target), scale_factor=args.scale_factor)
        written.append(target)
        print(f"scaled {input_path} -> {target}")
    if args.dry_run:
        print(f"dry run: {len(args.inputs)} input(s) unchanged")
    else:
        print(f"scaled {len(written)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
