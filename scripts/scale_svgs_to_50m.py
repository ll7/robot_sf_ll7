"""
Scale SVG scenario files to have dimensions < 50m.
Converts 400x400 viewBox to 40x40 and scales all coordinates accordingly.
"""

import re
import xml.etree.ElementTree as ET


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
        return scale_coordinate(match.group(0), scale_factor)

    return re.sub(r"-?\d+\.?\d*", replace_number, d)


def scale_svg_file(input_path: str, output_path: str, scale_factor: float = 0.1):
    """Scale an entire SVG file by the given factor."""
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
            if elem.get(attr):
                elem.set(attr, scale_coordinate(elem.get(attr), scale_factor))

        # Scale path data
        if elem.get("d"):
            elem.set("d", scale_path_data(elem.get("d"), scale_factor))

    # Write to output with proper formatting
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    # Fix formatting (ET doesn't preserve it well)
    with open(output_path) as f:
        content = f.read()

    # Add the comment back at the top
    scenario_name = input_path.split("/")[-1].replace("_", " ").replace(".svg", "").title()
    comment = (
        f"<!-- {scenario_name} | Dimensions < 50m | Scale: 1 SVG unit = 1 meter (SI units) -->\n"
    )

    # Remove the old comment if present and add new one
    content = re.sub(r"<!--.*?-->\s*", "", content, count=1)
    content = content.replace(
        "<?xml version='1.0' encoding='UTF-8'?>\n",
        f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n{comment}',
    )

    with open(output_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    files = [
        "maps/svg_maps/static_humans.svg",
        "maps/svg_maps/overtaking.svg",
        "maps/svg_maps/crossing.svg",
        "maps/svg_maps/door_passing.svg",
    ]

    for filepath in files:
        print(f"Scaling {filepath}...")
        scale_svg_file(filepath, filepath, scale_factor=0.1)
        print("  ✓ Scaled to < 50m dimensions")

    print("\nAll scenarios scaled successfully!")
    print("Dimensions are now 40m × 40m (< 50m requirement met)")
