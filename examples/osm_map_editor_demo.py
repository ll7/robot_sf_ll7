#!/usr/bin/env python3
"""OSM Zones Editor interactive demonstration (T034).

End-to-end workflow demonstrating the visual zone editor:
1. Render OSM background map from PBF
2. Launch interactive editor
3. Draw zones and routes interactively
4. Save to YAML format
5. Reload and verify

This script demonstrates the complete map editing pipeline introduced in Issue #392.

Usage:
    # Interactive mode (requires display)
    uv run python examples/osm_map_editor_demo.py

    # Headless mode (CI/automated testing)
    DISPLAY= MPLBACKEND=Agg uv run python examples/osm_map_editor_demo.py --headless

Keyboard Shortcuts (when editor is open):
    H           Show help menu with all shortcuts
    P           Switch to zone drawing mode
    R           Switch to route drawing mode
    Click       Add vertex to polygon/route
    Drag        Move existing vertex
    Right-Click Delete nearest vertex
    Enter       Finish current polygon/route
    Escape      Cancel current drawing
    Ctrl+Z      Undo last action
    Ctrl+Y      Redo last undone action
    Shift       Toggle vertex snapping to boundaries
    Ctrl+S      Save zones/routes to YAML

Expected Workflow:
    1. Script renders background map (output/maps/editor_demo/background.png)
    2. Interactive editor window opens with rendered map
    3. User draws zones and routes interactively (or auto-creates in headless mode)
    4. Press Ctrl+S to save to YAML (output/maps/editor_demo/zones.yaml)
    5. Script reloads YAML to verify round-trip works
    6. Close editor window to exit

Dependencies:
    - OSM PBF file at test_scenarios/osm_fixtures/sample_block.pbf
    - Interactive display (unless --headless)
"""

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.common.matplotlib_utils import ensure_interactive_backend
from robot_sf.maps.osm_background_renderer import render_osm_background
from robot_sf.maps.osm_zones_editor import OSMZonesEditor
from robot_sf.maps.osm_zones_yaml import Zone, load_zones_yaml, save_zones_yaml
from robot_sf.nav.osm_map_builder import osm_to_map_definition

configure_logging()


def setup_output_directory() -> Path:
    """Create output directory for demo artifacts.

    Returns:
        Path to output directory
    """
    output_dir = Path("output/maps/editor_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_background(
    pbf_file: str,
    output_dir: Path,
    *,
    pixels_per_meter: float | None,
    max_pixels: int,
    min_pixels_per_meter: float,
    max_pixels_per_meter: float,
) -> tuple[Path, dict]:
    """Render OSM background map from PBF file.

    Args:
        pbf_file: Path to OSM PBF file
        output_dir: Directory for output files

    Returns:
        Tuple of (png_path, affine_data)
    """
    print("=" * 70)
    print("STEP 1: Rendering OSM Background Map")
    print("=" * 70)

    try:
        render_result = render_osm_background(
            pbf_file=pbf_file,
            output_dir=str(output_dir),
            pixels_per_meter=pixels_per_meter,
            dpi=100,
            max_pixels=max_pixels,
            min_pixels_per_meter=min_pixels_per_meter,
            max_pixels_per_meter=max_pixels_per_meter,
        )
        affine_data = render_result["affine_transform"]

        png_path = Path(render_result["png_path"])
        logger.info(f"✅ Background rendered: {png_path}")
        logger.info(f"   Pixel dimensions: {affine_data['pixel_dimensions']}")
        logger.info(f"   Bounds (meters): {affine_data['bounds_meters']}")

        return png_path, affine_data
    except (SystemError, RuntimeError) as e:
        logger.warning(f"Background rendering failed (headless mode): {e}")
        logger.info("   Skipping background render in headless mode")
        # Return dummy values for headless mode
        png_path = output_dir / "background.png"
        affine_data = {
            "pixel_dimensions": (800, 600),
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "pixel_per_meter": 1.0,
            "origin": "upper",
        }
        return png_path, affine_data


def launch_interactive_editor(
    png_path: Path, output_yaml: Path, map_definition=None
) -> OSMZonesEditor:
    """Launch the interactive zone editor.

    Args:
        png_path: Path to background PNG
        output_yaml: Path where zones YAML will be saved
        map_definition: Optional MapDefinition for validation

    Returns:
        Editor instance
    """
    print("")
    print("=" * 70)
    print("STEP 2: Launching Interactive Editor")
    print("=" * 70)
    print("The editor window will open shortly.")
    print("Press 'H' in the editor window to see all keyboard shortcuts.")
    print("")
    print("Quick start:")
    print("  1. Press 'P' to enter zone drawing mode")
    print("  2. Click to add vertices to your zone")
    print("  3. Press 'Enter' to finish the zone")
    print("  4. Press 'Ctrl+S' to save to YAML")
    print("  5. Close the window when done")
    print("")

    editor = OSMZonesEditor(
        png_file=str(png_path),
        output_yaml=str(output_yaml),
        map_definition=map_definition,
    )

    return editor


def create_demo_zones_headless(output_yaml: Path) -> None:
    """Create sample zones programmatically for headless mode.

    Args:
        output_yaml: Path where zones YAML will be saved
    """
    print("")
    print("=" * 70)
    print("HEADLESS MODE: Creating Demo Zones Programmatically")
    print("=" * 70)

    from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, Route

    # Create sample zones
    config = OSMZonesConfig(
        zones={
            "spawn_zone": Zone(
                name="spawn_zone",
                type="spawn",
                polygon=[(10, 10), (30, 10), (30, 30), (10, 30)],
                priority=1,
            ),
            "goal_zone": Zone(
                name="goal_zone",
                type="goal",
                polygon=[(70, 70), (90, 70), (90, 90), (70, 90)],
                priority=1,
            ),
        },
        routes={
            "route_1": Route(
                name="route_1",
                waypoints=[(20, 20), (50, 50), (80, 80)],
            )
        },
    )

    save_zones_yaml(config, str(output_yaml))
    logger.info(f"✅ Demo zones saved: {output_yaml}")
    logger.info(f"   Zones: {len(config.zones)}")
    logger.info(f"   Routes: {len(config.routes)}")


def verify_yaml_roundtrip(output_yaml: Path) -> None:
    """Load saved YAML and verify contents.

    Args:
        output_yaml: Path to zones YAML file
    """
    print("")
    print("=" * 70)
    print("STEP 3: Verifying YAML Round-Trip")
    print("=" * 70)

    if not output_yaml.exists():
        logger.warning(f"❌ YAML file not found: {output_yaml}")
        logger.info("   (This is expected if you didn't save in the editor)")
        return

    config = load_zones_yaml(str(output_yaml))
    logger.info(f"✅ YAML loaded successfully: {output_yaml}")
    logger.info(f"   Zones: {len(config.zones)}")
    logger.info(f"   Routes: {len(config.routes)}")

    if config.zones:
        logger.info("")
        logger.info("   Zone details:")
        for zone_name, zone in config.zones.items():
            logger.info(f"     - {zone_name}: {zone.type}, {len(zone.polygon)} vertices")

    if config.routes:
        logger.info("")
        logger.info("   Route details:")
        for route_name, route in config.routes.items():
            logger.info(f"     - {route_name}: {len(route.waypoints)} waypoints")


def main() -> None:
    """Run the complete OSM map editor demonstration."""
    # Ensure interactive matplotlib backend for editor GUI
    interactive_available = ensure_interactive_backend(verbose=True)
    if not interactive_available:
        logger.warning(
            "No interactive matplotlib backend available, will fall back to headless mode if needed"
        )

    parser = argparse.ArgumentParser(description="OSM Zones Editor Demo")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (create demo zones without interactive editor)",
    )
    parser.add_argument(
        "--pbf-file",
        type=str,
        default="test_scenarios/osm_fixtures/sample_block.pbf",
        help="Path to OSM PBF file (default: test_scenarios/osm_fixtures/sample_block.pbf)",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=None,
        help="Pixels per meter for background rendering (auto-fit when omitted)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=4000,
        help="Max pixel dimension before downscaling (raise to reduce clamping)",
    )
    parser.add_argument(
        "--min-pixels-per-meter",
        type=float,
        default=0.5,
        help="Lower bound when auto-selecting pixels-per-meter",
    )
    parser.add_argument(
        "--max-pixels-per-meter",
        type=float,
        default=10.0,
        help="Upper bound when auto-selecting pixels-per-meter",
    )
    args = parser.parse_args()

    # Verify PBF file exists
    pbf_path = Path(args.pbf_file)
    if not pbf_path.exists():
        logger.error(f"❌ PBF file not found: {args.pbf_file}")
        logger.info("Download a sample from: https://extract.bbbike.org/")
        return

    print("")
    print(f"╔{'=' * 68}╗")
    print(f"║{' ' * 15}OSM ZONES EDITOR DEMONSTRATION{' ' * 23}║")
    print(f"╚{'=' * 68}╝")
    logger.info("")

    # Setup output directory
    output_dir = setup_output_directory()
    output_yaml = output_dir / "zones.yaml"

    # Render background map
    png_path, _ = render_background(
        args.pbf_file,
        output_dir,
        pixels_per_meter=args.pixels_per_meter,
        max_pixels=args.max_pixels,
        min_pixels_per_meter=args.min_pixels_per_meter,
        max_pixels_per_meter=args.max_pixels_per_meter,
    )

    # Check if PNG was actually created (rendering might fail in headless environment)
    if not png_path.exists():
        logger.warning("Background PNG not created - likely running in headless environment")
        logger.info("Switching to headless mode automatically")
        args.headless = True

    # Load map definition for validation
    try:
        map_def = osm_to_map_definition(pbf_file=args.pbf_file, line_buffer_m=1.5)
        logger.info(f"✅ MapDefinition loaded for validation ({len(map_def.obstacles)} obstacles)")
    except Exception as e:
        logger.warning(f"Could not load MapDefinition: {e}")
        map_def = None

    if args.headless:
        # Headless mode: create demo zones programmatically
        create_demo_zones_headless(output_yaml)
    else:
        # Interactive mode: launch editor
        editor = launch_interactive_editor(png_path, output_yaml, map_def)
        editor.run(blocking=True)
        logger.info("")
        logger.info("Editor closed.")

    # Verify YAML was saved and can be loaded
    verify_yaml_roundtrip(output_yaml)

    # Summary
    print("")
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("  - background.png: Rendered OSM map")
    print("  - affine_transform.json: Coordinate transform data")
    if output_yaml.exists():
        print("  - zones.yaml: Saved zones/routes configuration")
    print("")
    print("Next steps:")
    print("  1. Inspect the generated files in the output directory")
    print("  2. Edit zones.yaml manually or re-run the editor")
    print("  3. Use the zones configuration in your robot navigation environment")
    print("")


if __name__ == "__main__":
    main()
