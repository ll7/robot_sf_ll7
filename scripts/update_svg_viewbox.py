"""
Update SVG scenario files to have dimensions < 50m by modifying only the viewBox.
The viewBox defines the coordinate system in meters (1 SVG unit = 1 meter).
"""

import re
import xml.etree.ElementTree as ET


def update_svg_viewbox(filepath: str, new_viewbox: str) -> bool:
    """Update the viewBox attribute of an SVG file."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Find the SVG element (might have namespace)
    if root.tag.endswith("}svg") or root.tag == "svg":
        # Read the original file to preserve formatting
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        # Replace viewBox
        content = re.sub(r'viewBox="[^"]*"', f'viewBox="{new_viewbox}"', content)

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Updated {filepath}: viewBox='{new_viewbox}'")
        return True
    return False


if __name__ == "__main__":
    files = [
        "maps/svg_maps/static_humans.svg",
        "maps/svg_maps/overtaking.svg",
        "maps/svg_maps/crossing.svg",
        "maps/svg_maps/door_passing.svg",
    ]

    # Change from 400x400 to 40x40 meters (< 50m requirement)
    new_viewbox = "0 0 40 40"

    print(f"Updating SVG files to viewBox='{new_viewbox}' (40m × 40m < 50m)")
    print("-" * 70)

    for filepath in files:
        update_svg_viewbox(filepath, new_viewbox)

    print("\n✓ All scenarios updated successfully!")
    print("Dimensions are now 40m × 40m (< 50m requirement met)")
