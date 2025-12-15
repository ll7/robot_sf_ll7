"""Plot a MapDefinition as a quick visual sanity check."""

from pathlib import Path

from robot_sf.common.logging import configure_logging
from robot_sf.maps import visualize_map_definition
from robot_sf.nav.svg_map_parser import convert_map


def main() -> None:
    """Render a default SVG map to `output/map.png`."""
    configure_logging()

    map_def = convert_map("maps/templates/map_template.svg")
    if map_def is None:
        raise RuntimeError("Failed to load SVG map")

    output_path = Path("output/map.png")
    visualize_map_definition(map_def, output_path=output_path, title="Map Overview")


if __name__ == "__main__":
    main()
