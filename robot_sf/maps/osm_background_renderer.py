"""OSM PBF background renderer for map visualization.

Renders OSM PBF data to PNG format with affine transform metadata for
pixel-to-world coordinate mapping. Supports multi-layer PBF files.

Key features:
- Load multi-layer PBF (lines, multipolygons, multilinestrings)
- Render buildings (gray), water (blue), streets (yellow)
- Generate PNG output with affine transform metadata (JSON)
- Validate round-trip pixel↔world coordinate transformations
- Support custom DPI and pixel/meter scale

Output format:
- PNG: Rasterized map background
- JSON: Affine transform with pixel_per_meter, bounds_meters, pixel_dimensions
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def render_osm_background(
    pbf_file: str,
    output_dir: str = "output/maps/",
    pixels_per_meter: float = 2.0,
    dpi: int = 100,
    figure_size: tuple[float, float] = (10, 10),
) -> dict:
    """Render OSM PBF to PNG background with affine transform metadata.

    Loads OSM PBF file and renders buildings, water, and streets to PNG format.
    Generates JSON metadata with affine transform for coordinate mapping.

    Args:
        pbf_file: Path to OSM PBF file
        output_dir: Output directory for PNG and JSON files
        pixels_per_meter: Scale factor (default 2.0 pixels/meter)
        dpi: DPI for figure rendering (default 100)
        figure_size: Base figure size in inches (default 10x10)

    Returns:
        Dictionary with affine transform metadata:
        {
            "pixel_per_meter": float,
            "bounds_meters": [minx, miny, maxx, maxy],
            "pixel_dimensions": [width_px, height_px],
            "dpi": int
        }

    Raises:
        FileNotFoundError: If pbf_file does not exist
        ValueError: If PBF is empty or invalid
    """
    pbf_path = Path(pbf_file)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_file}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load OSM data from multiple layers
    try:
        layers_to_load = ["lines", "multipolygons", "multilinestrings"]
        gdfs = []
        for layer in layers_to_load:
            try:
                gdf_layer = gpd.read_file(pbf_file, layer=layer)
                if not gdf_layer.empty:
                    gdfs.append(gdf_layer)
                    logger.info(f"Loaded {len(gdf_layer)} features from layer '{layer}'")
            except Exception:
                continue

        if not gdfs:
            gdf = gpd.read_file(pbf_file)
        else:
            gdf = pd.concat(gdfs, ignore_index=False).reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Failed to load PBF: {e}") from e

    if gdf.empty:
        raise ValueError(f"PBF file is empty: {pbf_file}")

    # Get bounds
    bounds = gdf.total_bounds
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]

    if not (width_m > 0 and height_m > 0):
        raise ValueError(f"Invalid bounds: width={width_m}, height={height_m}")

    # Compute pixel dimensions
    pixel_width = int(width_m * pixels_per_meter)
    pixel_height = int(height_m * pixels_per_meter)

    # Clamp to prevent memory issues
    max_pixels = 4000
    if pixel_width > max_pixels or pixel_height > max_pixels:
        scale_factor = min(max_pixels / pixel_width, max_pixels / pixel_height)
        pixel_width = int(pixel_width * scale_factor)
        pixel_height = int(pixel_height * scale_factor)
        logger.warning(f"Clamped pixel dimensions to {pixel_width}x{pixel_height}")

    figure_inches_x = pixel_width / dpi
    figure_inches_y = pixel_height / dpi

    # Create figure
    fig, ax = plt.subplots(figsize=(figure_inches_x, figure_inches_y), dpi=dpi)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()

    # Plot buildings (gray)
    if "building" in gdf.columns:
        buildings = gdf[gdf["building"].notna()]
        if not buildings.empty:
            try:
                buildings.plot(ax=ax, color="lightgray", edgecolor="gray", linewidth=0.5, zorder=1)
            except Exception:
                pass

    # Plot water (blue)
    water_mask = False
    if "waterway" in gdf.columns:
        water_mask = gdf["waterway"].notna()
    if "natural" in gdf.columns:
        water_mask = water_mask | (gdf["natural"] == "water")
    if isinstance(water_mask, pd.Series) and water_mask.any():
        try:
            gdf[water_mask].plot(
                ax=ax, color="lightblue", edgecolor="blue", linewidth=0.5, zorder=2
            )
        except Exception:
            pass

    # Plot streets (yellow)
    if "highway" in gdf.columns:
        streets = gdf[gdf["highway"].notna()]
        if not streets.empty:
            try:
                streets.plot(
                    ax=ax, color="lightyellow", edgecolor="orange", linewidth=0.5, zorder=3
                )
            except Exception:
                pass

    # Save PNG
    png_path = output_path / "background.png"
    fig.savefig(str(png_path), bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved PNG: {png_path}")

    # Save affine transform
    affine_data = {
        "pixel_per_meter": pixels_per_meter,
        "bounds_meters": list(bounds),
        "pixel_dimensions": [pixel_width, pixel_height],
        "dpi": dpi,
    }

    json_path = output_path / "affine_transform.json"
    with open(json_path, "w") as f:
        json.dump(affine_data, f, indent=2)
    logger.info(f"Saved affine JSON: {json_path}")

    return affine_data


def validate_affine_transform(
    point_pixel: tuple[float, float],
    affine_data: dict,
    tolerance_pixels: float = 1.0,
) -> bool:
    """Validate round-trip pixel↔world transformation."""
    world = pixel_to_world(point_pixel, affine_data)
    point_pixel_recovered = world_to_pixel(world, affine_data)
    error = (
        (point_pixel[0] - point_pixel_recovered[0]) ** 2
        + (point_pixel[1] - point_pixel_recovered[1]) ** 2
    ) ** 0.5
    return error <= tolerance_pixels


def pixel_to_world(
    point_pixel: tuple[float, float],
    affine_data: dict,
) -> tuple[float, float]:
    """Transform pixel coordinates to world coordinates."""
    px, py = point_pixel
    pixel_per_meter = affine_data["pixel_per_meter"]
    minx, miny, maxx, maxy = affine_data["bounds_meters"]
    world_x = minx + (px / pixel_per_meter)
    world_y = miny + (py / pixel_per_meter)
    return (world_x, world_y)


def world_to_pixel(
    point_world: tuple[float, float],
    affine_data: dict,
) -> tuple[float, float]:
    """Transform world coordinates to pixel coordinates."""
    world_x, world_y = point_world
    pixel_per_meter = affine_data["pixel_per_meter"]
    minx, miny, maxx, maxy = affine_data["bounds_meters"]
    pixel_x = (world_x - minx) * pixel_per_meter
    pixel_y = (world_y - miny) * pixel_per_meter
    return (pixel_x, pixel_y)


def save_affine_transform(affine_data: dict, output_path: str) -> None:
    """Save affine transform to JSON file."""
    with open(output_path, "w") as f:
        json.dump(affine_data, f, indent=2)
    logger.info(f"Saved: {output_path}")


def load_affine_transform(input_path: str) -> dict:
    """Load affine transform from JSON file."""
    with open(input_path) as f:
        affine_data = json.load(f)
    logger.info(f"Loaded: {input_path}")
    return affine_data
