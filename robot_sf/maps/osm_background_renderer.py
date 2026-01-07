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
- JSON: Metadata with nested affine_transform (pixel_per_meter, bounds_meters, pixel_dimensions)
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from shapely.affinity import translate

from robot_sf.nav.osm_map_builder import project_to_utm


def _load_osm_data(pbf_file: str) -> gpd.GeoDataFrame:
    """Load OSM data from the available PBF layers.

    Returns:
        GeoDataFrame containing merged OSM features.
    """
    layers_to_load = ["lines", "multipolygons", "multilinestrings"]
    gdfs: list[gpd.GeoDataFrame] = []

    for layer in layers_to_load:
        try:
            gdf_layer = gpd.read_file(pbf_file, layer=layer)
        except (OSError, ValueError) as exc:
            logger.debug(f"Skipping layer '{layer}': {exc}")
            continue
        if not gdf_layer.empty:
            gdfs.append(gdf_layer)
            logger.info(f"Loaded {len(gdf_layer)} features from layer '{layer}'")

    if not gdfs:
        return gpd.read_file(pbf_file)

    return pd.concat(gdfs, ignore_index=False).reset_index(drop=True)


def _compute_pixel_dimensions(
    bounds: list[float] | tuple[float, float, float, float], pixels_per_meter: float, dpi: int
) -> tuple[int, int, float, float]:
    """Derive pixel dimensions and figure size from bounds and scale.

    Returns:
        Tuple of (pixel_width, pixel_height, figure_inches_x, figure_inches_y).
    """
    minx, miny, maxx, maxy = bounds
    width_m = maxx - minx
    height_m = maxy - miny

    if not (width_m > 0 and height_m > 0):
        raise ValueError(f"Invalid bounds: width={width_m}, height={height_m}")

    pixel_width = max(1, int(round(width_m * pixels_per_meter)))
    pixel_height = max(1, int(round(height_m * pixels_per_meter)))

    max_pixels = 4000
    if pixel_width > max_pixels or pixel_height > max_pixels:
        scale_factor = min(max_pixels / pixel_width, max_pixels / pixel_height)
        pixel_width = int(pixel_width * scale_factor)
        pixel_height = int(pixel_height * scale_factor)
        logger.warning(f"Clamped pixel dimensions to {pixel_width}x{pixel_height}")

    figure_inches_x = pixel_width / dpi
    figure_inches_y = pixel_height / dpi
    return pixel_width, pixel_height, figure_inches_x, figure_inches_y


def _plot_buildings(gdf, ax) -> None:
    """Plot building geometries if present."""
    if "building" not in gdf.columns:
        return
    buildings = gdf[gdf["building"].notna()]
    if buildings.empty:
        return
    try:
        buildings.plot(ax=ax, color="lightgray", edgecolor="gray", linewidth=0.5, zorder=1)
    except (ValueError, TypeError) as exc:
        logger.debug(f"Skipping building plotting: {exc}")


def _plot_water(gdf, ax) -> None:
    """Plot water geometries if present."""
    water_mask = False
    if "waterway" in gdf.columns:
        water_mask = gdf["waterway"].notna()
    if "natural" in gdf.columns:
        water_mask = water_mask | (gdf["natural"] == "water")
    if not (isinstance(water_mask, pd.Series) and water_mask.any()):
        return
    try:
        gdf[water_mask].plot(ax=ax, color="lightblue", edgecolor="blue", linewidth=0.5, zorder=2)
    except (ValueError, TypeError) as exc:
        logger.debug(f"Skipping water plotting: {exc}")


def _plot_streets(gdf, ax) -> None:
    """Plot street geometries if present."""
    if "highway" not in gdf.columns:
        return
    streets = gdf[gdf["highway"].notna()]
    if streets.empty:
        return
    try:
        streets.plot(ax=ax, color="lightyellow", edgecolor="orange", linewidth=0.5, zorder=3)
    except (ValueError, TypeError) as exc:
        logger.debug(f"Skipping street plotting: {exc}")


def _plot_layers(gdf, ax) -> None:
    """Plot buildings, water, and streets if present."""
    _plot_buildings(gdf, ax)
    _plot_water(gdf, ax)
    _plot_streets(gdf, ax)


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
        Dictionary with png_path and affine_transform metadata.

    Raises:
        FileNotFoundError: If pbf_file does not exist
        ValueError: If PBF is empty or invalid
    """
    pbf_path = Path(pbf_file)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF file not found: {pbf_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gdf = _load_osm_data(pbf_file)
    if gdf.empty:
        raise ValueError(f"PBF file is empty: {pbf_file}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)

    gdf_utm, utm_zone = project_to_utm(gdf)
    bounds = gdf_utm.total_bounds
    minx, miny, maxx, maxy = bounds
    width_m = maxx - minx
    height_m = maxy - miny

    # Translate to a local frame so pixel→world transforms align with MapDefinition.
    def _shift_geometry(geom):
        if geom is None or geom.is_empty:
            return geom
        return translate(geom, xoff=-minx, yoff=-miny)

    gdf_local = gdf_utm.copy()
    gdf_local["geometry"] = gdf_local["geometry"].apply(_shift_geometry)
    bounds_local = (0.0, 0.0, width_m, height_m)
    pixel_width, pixel_height, figure_inches_x, figure_inches_y = _compute_pixel_dimensions(
        bounds_local, pixels_per_meter, dpi
    )

    fig, ax = plt.subplots(figsize=(figure_inches_x, figure_inches_y), dpi=dpi)
    ax.set_xlim(0.0, width_m)
    ax.set_ylim(0.0, height_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    _plot_layers(gdf_local, ax)

    png_path = output_path / "background.png"
    fig.savefig(str(png_path), dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved PNG: {png_path}")

    affine_transform = {
        "pixel_origin": [0.0, 0.0],
        "pixel_per_meter": pixels_per_meter,
        "bounds_meters": [0.0, 0.0, width_m, height_m],
        "pixel_dimensions": [pixel_width, pixel_height],
        "dpi": dpi,
        "origin": "upper",
    }

    metadata = {
        "pbf_file": str(pbf_file),
        "utm_zone": utm_zone,
        "affine_transform": affine_transform,
    }

    json_path = output_path / "affine_transform.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info(f"Saved affine JSON: {json_path}")

    return {
        "png_path": str(png_path),
        "affine_transform": affine_transform,
    }


def validate_affine_transform(
    affine_data: dict,
    point_pixel: tuple[float, float] | None = None,
    tolerance_pixels: float = 1.0,
    tolerance_meters: float = 0.1,
) -> bool:
    """Validate round-trip pixel↔world transformation.

    Returns:
        True when round-trip pixel→world→pixel and world→pixel→world stay within tolerance.
    """
    if point_pixel is None:
        point_pixel = (0.0, 0.0)

    world = pixel_to_world(point_pixel, affine_data)
    point_pixel_recovered = world_to_pixel(world, affine_data)
    pixel_error = (
        (point_pixel[0] - point_pixel_recovered[0]) ** 2
        + (point_pixel[1] - point_pixel_recovered[1]) ** 2
    ) ** 0.5

    world_recovered = pixel_to_world(point_pixel_recovered, affine_data)
    world_error = (
        (world[0] - world_recovered[0]) ** 2 + (world[1] - world_recovered[1]) ** 2
    ) ** 0.5

    return pixel_error <= tolerance_pixels and world_error <= tolerance_meters


def pixel_to_world(
    point_pixel: tuple[float, float],
    affine_data: dict,
) -> tuple[float, float]:
    """Transform pixel coordinates to world coordinates.

    Returns:
        Tuple of (x, y) in world meters.
    """
    px, py = point_pixel
    pixel_per_meter = affine_data["pixel_per_meter"]
    minx, miny, maxx, maxy = affine_data["bounds_meters"]
    origin_x, origin_y = affine_data.get("pixel_origin", (0.0, 0.0))
    origin = affine_data.get("origin", "lower")

    world_x = minx + ((px - origin_x) / pixel_per_meter)
    if origin == "upper":
        world_y = maxy - ((py - origin_y) / pixel_per_meter)
    else:
        world_y = miny + ((py - origin_y) / pixel_per_meter)
    return (world_x, world_y)


def world_to_pixel(
    point_world: tuple[float, float],
    affine_data: dict,
) -> tuple[float, float]:
    """Transform world coordinates to pixel coordinates.

    Returns:
        Tuple of (px, py) in pixel space.
    """
    world_x, world_y = point_world
    pixel_per_meter = affine_data["pixel_per_meter"]
    minx, miny, maxx, maxy = affine_data["bounds_meters"]
    origin_x, origin_y = affine_data.get("pixel_origin", (0.0, 0.0))
    origin = affine_data.get("origin", "lower")

    pixel_x = origin_x + ((world_x - minx) * pixel_per_meter)
    if origin == "upper":
        pixel_y = origin_y + ((maxy - world_y) * pixel_per_meter)
    else:
        pixel_y = origin_y + ((world_y - miny) * pixel_per_meter)
    return (pixel_x, pixel_y)


def save_affine_transform(affine_data: dict, output_path: str) -> None:
    """Save affine transform to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(affine_data, f, indent=2, sort_keys=True)
    logger.info(f"Saved: {output_path}")


def load_affine_transform(input_path: str) -> dict:
    """Load affine transform from JSON file.

    Returns:
        Dictionary with affine metadata loaded from disk.
    """
    with open(input_path, encoding="utf-8") as f:
        affine_data = json.load(f)
    if isinstance(affine_data, dict) and "affine_transform" in affine_data:
        logger.info(f"Loaded: {input_path}")
        return affine_data["affine_transform"]
    logger.info(f"Loaded: {input_path}")
    return affine_data
