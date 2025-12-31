"""Matplotlib-based interactive visual editor for OSM zones and routes.

This module provides a GUI for editing zones (spawn, goal, crowded areas) and routes
(navigation corridors) overlaid on OSM-derived background PNG maps.

Key Features:
- Interactive click-and-drag zone/route creation
- Vertex editing with drag and delete (right-click)
- Undo/redo history (Ctrl+Z, Ctrl+Y)
- Snapping to map boundaries (Shift toggle)
- Real-time validation warnings (out-of-bounds, obstacle crossing)
- Save/load YAML integration (Ctrl+S)
- Keyboard shortcuts for mode switching (P=polygon, R=route)

Interactive Editor Flow:
1. Load OSM-derived background PNG and MapDefinition
2. Click mode (P for polygon/zone, R for route)
3. Click to add vertices; right-click to delete
4. Shift to toggle snapping
5. Ctrl+Z/Ctrl+Y for undo/redo
6. Ctrl+S to save zones/routes to YAML

Architecture:
- State machine: IDLE, DRAW, EDIT modes
- Event handlers: on_click, on_motion, on_key_press, on_scroll
- Undo/redo: Stack-based history
- Validation: Real-time checks with matplotlib text annotations
- Snapping: Shapely-based nearest-boundary detection

Usage Example:
    from robot_sf.maps.osm_zones_editor import OSMZonesEditor
    from robot_sf.nav.osm_map_builder import load_pbf, osm_to_map_definition
    from robot_sf.maps.osm_background_renderer import render_osm_background

    # Load map
    gdf = load_pbf("map.pbf", osm_tags_to_ways={...})
    map_def = osm_to_map_definition(gdf, ...)
    png_file = render_osm_background(gdf, "background.png")

    # Launch editor
    editor = OSMZonesEditor(
        png_file=png_file,
        map_definition=map_def,
        output_yaml="zones.yaml"
    )
    editor.run()

Test Coverage:
- T026: Class instantiation, state machine basics
- T027: Click handlers for vertex placement
- T028: Vertex dragging and deletion (manual/interactive)
- T029: Undo/redo stack operations
- T030: Snapping logic with boundary detection
- T031: Validation warnings and highlighting
- T032: Save to YAML integration
- T033: Keyboard shortcuts and mode switching (manual/interactive)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.image import imread
from shapely.geometry import Point, Polygon

try:
    from shapely.ops import nearest_points
except ImportError:
    nearest_points = None  # Fallback for older Shapely versions

from robot_sf.common.types import Vec2D
from robot_sf.maps.osm_background_renderer import (
    load_affine_transform,
    pixel_to_world,
    world_to_pixel,
)
from robot_sf.maps.osm_zones_yaml import (
    OSMZonesConfig,
    Route,
    Zone,
    load_zones_yaml,
    save_zones_yaml,
)

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

ShapelyPolygon = Polygon

# ============================================================================
# Enums & Type Definitions
# ============================================================================


class EditorMode(Enum):
    """Editor operational modes."""

    IDLE = "idle"
    """No operation in progress."""

    DRAW = "draw"
    """Drawing a zone or route."""

    EDIT = "edit"
    """Editing existing zone or route."""


class DrawMode(Enum):
    """What type of object to draw."""

    ZONE = "zone"
    """Drawing a zone polygon."""

    ROUTE = "route"
    """Drawing a route path."""


# ============================================================================
# Action History for Undo/Redo (T029)
# ============================================================================


@dataclass
class EditorAction:
    """Single undo/redo action."""

    action_type: str
    """Type: 'add_vertex', 'delete_vertex', 'move_vertex', 'finish_polygon', etc."""

    target: str
    """'zone' or 'route'."""

    target_name: str | None = None
    """Name of zone/route being edited."""

    data: dict[str, Any] = field(default_factory=dict)
    """Action-specific data (coordinates, indices, etc.)."""

    def execute(self, editor: "OSMZonesEditor") -> None:
        """Execute action (redo)."""
        raise NotImplementedError(f"Execute not implemented for {self.action_type}")

    def undo(self, editor: "OSMZonesEditor") -> None:
        """Undo action."""
        raise NotImplementedError(f"Undo not implemented for {self.action_type}")


@dataclass
class UndoRedoStack:
    """Stack-based history with max size (T029 foundation)."""

    undo_stack: list[EditorAction] = field(default_factory=list)
    redo_stack: list[EditorAction] = field(default_factory=list)
    max_size: int = 100
    """Maximum undo history depth."""

    def push_action(self, action: EditorAction) -> None:
        """Add action to undo stack and clear redo stack."""
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self, editor: "OSMZonesEditor") -> bool:
        """Execute undo.

        Returns:
            True if an action was undone, False when stack is empty.
        """
        if not self.undo_stack:
            logger.warning("Undo stack is empty")
            return False
        action = self.undo_stack.pop()
        action.undo(editor)
        self.redo_stack.append(action)
        return True

    def redo(self, editor: "OSMZonesEditor") -> bool:
        """Execute redo.

        Returns:
            True if an action was redone, False when stack is empty.
        """
        if not self.redo_stack:
            logger.warning("Redo stack is empty")
            return False
        action = self.redo_stack.pop()
        action.execute(editor)
        self.undo_stack.append(action)
        return True


# ============================================================================
# EditorAction Subclasses (T029 Implementations)
# ============================================================================


class AddVertexAction(EditorAction):
    """Undo/redo for adding a vertex to current polygon."""

    def __init__(self, vertex: Vec2D, index: int):
        """Initialize AddVertexAction.

        Args:
            vertex: (x, y) coordinates in world space
            index: Position in polygon (usually len(current_polygon) before add)
        """
        super().__init__(
            action_type="add_vertex",
            target="current_polygon",
            data={"vertex": vertex, "index": index},
        )

    def execute(self, editor: "OSMZonesEditor") -> None:
        """Re-add vertex to current polygon (redo)."""
        vertex = self.data["vertex"]
        editor.current_polygon.append(vertex)
        logger.debug(f"Redo: Added vertex {vertex}")

    def undo(self, editor: "OSMZonesEditor") -> None:
        """Remove vertex from current polygon."""
        if editor.current_polygon:
            removed = editor.current_polygon.pop()
            logger.debug(f"Undo: Removed vertex {removed}")


class DeleteVertexAction(EditorAction):
    """Undo/redo for deleting a vertex from current polygon."""

    def __init__(self, vertex: Vec2D, index: int):
        """Initialize DeleteVertexAction.

        Args:
            vertex: (x, y) coordinates in world space
            index: Position in polygon
        """
        super().__init__(
            action_type="delete_vertex",
            target="current_polygon",
            data={"vertex": vertex, "index": index},
        )

    def execute(self, editor: "OSMZonesEditor") -> None:
        """Remove vertex from current polygon (redo)."""
        index = self.data["index"]
        if 0 <= index < len(editor.current_polygon):
            removed = editor.current_polygon.pop(index)
            logger.debug(f"Redo: Deleted vertex at index {index}: {removed}")

    def undo(self, editor: "OSMZonesEditor") -> None:
        """Re-add deleted vertex to current polygon."""
        vertex = self.data["vertex"]
        index = self.data["index"]
        editor.current_polygon.insert(index, vertex)
        logger.debug(f"Undo: Re-added vertex {vertex} at index {index}")


class MoveVertexAction(EditorAction):
    """Undo/redo for moving a vertex within current polygon."""

    def __init__(self, index: int, old_pos: Vec2D, new_pos: Vec2D):
        """Initialize MoveVertexAction.

        Args:
            index: Vertex index in current_polygon
            old_pos: Original (x, y) in world space
            new_pos: New (x, y) in world space
        """
        super().__init__(
            action_type="move_vertex",
            target="current_polygon",
            data={"index": index, "old_pos": old_pos, "new_pos": new_pos},
        )

    def execute(self, editor: "OSMZonesEditor") -> None:
        """Move vertex to new position (redo)."""
        index = self.data["index"]
        new_pos = self.data["new_pos"]
        if 0 <= index < len(editor.current_polygon):
            editor.current_polygon[index] = new_pos
            logger.debug(f"Redo: Moved vertex {index} to {new_pos}")

    def undo(self, editor: "OSMZonesEditor") -> None:
        """Move vertex back to original position."""
        index = self.data["index"]
        old_pos = self.data["old_pos"]
        if 0 <= index < len(editor.current_polygon):
            editor.current_polygon[index] = old_pos
            logger.debug(f"Undo: Moved vertex {index} back to {old_pos}")


class FinishPolygonAction(EditorAction):
    """Undo/redo for finishing (committing) a polygon as a zone or route."""

    def __init__(
        self, polygon: list[Vec2D], name: str, draw_type: DrawMode, zone_type: str = "spawn"
    ):
        """Initialize FinishPolygonAction.

        Args:
            polygon: List of vertices defining the zone/route
            name: Name to assign to zone or route
            draw_type: DrawMode.ZONE or DrawMode.ROUTE
            zone_type: Type of zone ('spawn', 'goal', etc.) - only used if draw_type is ZONE
        """
        super().__init__(
            action_type="finish_polygon",
            target=draw_type.value,
            target_name=name,
            data={"polygon": list(polygon), "draw_type": draw_type.value, "zone_type": zone_type},
        )

    def execute(self, editor: "OSMZonesEditor") -> None:
        """Commit polygon as zone or route (redo)."""
        name = self.target_name
        polygon = self.data["polygon"]
        draw_type = self.data["draw_type"]
        zone_type = self.data.get("zone_type", "spawn")

        if draw_type == "zone" and name:
            zone = Zone(name=name, type=zone_type, polygon=polygon)
            editor.config.zones[name] = zone
            logger.debug(f"Redo: Created zone '{name}' with {len(polygon)} vertices")
        elif draw_type == "route" and name:
            route = Route(name=name, waypoints=polygon)
            editor.config.routes[name] = route
            logger.debug(f"Redo: Created route '{name}' with {len(polygon)} waypoints")

    def undo(self, editor: "OSMZonesEditor") -> None:
        """Remove committed zone or route."""
        name = self.target_name
        draw_type = self.data["draw_type"]

        if draw_type == "zone" and name and name in editor.config.zones:
            del editor.config.zones[name]
            logger.debug(f"Undo: Deleted zone '{name}'")
        elif draw_type == "route" and name and name in editor.config.routes:
            del editor.config.routes[name]
            logger.debug(f"Undo: Deleted route '{name}'")


# ============================================================================
# OSMZonesEditor Main Class (T026)
# ============================================================================


class OSMZonesEditor:
    """Interactive Matplotlib-based visual editor for OSM zones and routes (T026-T033).

    Core Responsibilities:
    - Display OSM background PNG with coordinate transforms
    - Handle mouse/keyboard events for zone/route creation
    - Manage editor state (IDLE, DRAW, EDIT)
    - Maintain undo/redo history
    - Validate geometry in real-time
    - Save/load configurations to YAML

    Attributes:
        png_file: Path to background PNG (from osm_background_renderer)
        map_definition: Optional MapDefinition for validation/snapping
        config: Current zones/routes configuration
        history: Undo/redo stack
        current_polygon: Vertices being drawn (in DRAW mode)
        current_draw_type: Whether drawing zone or route
        snap_enabled: Snapping to boundary enabled (Shift toggle)
        output_yaml: Path to save zones/routes YAML
    """

    def __init__(
        self,
        png_file: str,
        map_definition: Any | None = None,
        output_yaml: str | None = None,
        initial_config: OSMZonesConfig | None = None,
        affine_json: str | None = None,
    ):
        """Initialize editor.

        Args:
            png_file: Path to background PNG from osm_background_renderer
            map_definition: Optional MapDefinition for validation
            output_yaml: Path to save zones YAML
            initial_config: Optional existing config to load (T026 basic version just stores it)
            affine_json: Optional path to affine_transform.json from osm_background_renderer (T027)
        """
        self.png_file = Path(png_file)
        self.map_definition = map_definition
        self.output_yaml = output_yaml
        self.config = initial_config or OSMZonesConfig()

        # Editor state
        self.mode = EditorMode.IDLE
        self.draw_mode = DrawMode.ZONE
        self.current_polygon: list[Vec2D] = []
        self.snap_enabled = False
        self.history = UndoRedoStack()

        # Matplotlib setup (T026 skeleton - no display yet)
        self.fig: Figure | None = None
        self.ax: Axes | None = None
        self.background_image: np.ndarray | None = None

        # Coordinate transform (T027 click handlers)
        self.affine_data: dict | None = None
        if affine_json:
            try:
                self.affine_data = load_affine_transform(affine_json)
                logger.info(f"Loaded affine transform from {affine_json}")
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"Failed to load affine transform: {e}")
        else:
            # Try to auto-detect affine.json next to PNG
            affine_candidate = self.png_file.parent / "affine_transform.json"
            if affine_candidate.exists():
                try:
                    self.affine_data = load_affine_transform(str(affine_candidate))
                    logger.info(f"Auto-loaded affine transform from {affine_candidate}")
                except (OSError, ValueError, KeyError) as e:
                    logger.warning(f"Failed to auto-load affine: {e}")

        # Vertex markers cache (T027)
        self._vertex_markers: list[mpatches.Circle] = []

        # Vertex editing state (T028)
        self._dragging_vertex_idx: int | None = None  # Index of vertex being dragged
        self._drag_start_pos: Vec2D | None = None  # Original position when drag began
        self._hovered_vertex_idx: int | None = None  # Index of hovered vertex (for visual feedback)
        self._vertex_drag_threshold = 15  # pixels - distance to detect vertex click

        # Validation state (T031)
        self._last_validation: dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }  # Cache last validation result

        logger.info(
            f"OSMZonesEditor initialized: png={self.png_file}, output_yaml={self.output_yaml}, "
            f"affine={'loaded' if self.affine_data else 'not loaded'}"
        )

    # ========================================================================
    # Initialization & Display (T026)
    # ========================================================================

    def setup_display(self) -> None:
        """Set up Matplotlib figure and axes (T026)."""
        logger.info("Setting up matplotlib display")

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.suptitle("OSM Zones & Routes Editor")

        # Load and display background PNG
        try:
            self.background_image = imread(str(self.png_file))
            self.ax.imshow(self.background_image, origin="upper")
            self.ax.set_title(f"Background: {self.png_file.name}")
            logger.info(f"Loaded background image: {self.background_image.shape}")
        except FileNotFoundError:
            logger.error(f"Background PNG not found: {self.png_file}")
            # Continue with empty axes
            self.ax.set_xlim(0, 1000)
            self.ax.set_ylim(0, 1000)

        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        # Initial UI state
        self._update_title()
        self._redraw()

    def _update_title(self) -> None:
        """Update figure title with current mode and state."""
        if self.fig:
            mode_str = (
                f"Mode: {self.draw_mode.value} | Snap: {'ON' if self.snap_enabled else 'OFF'}"
            )
            status_str = f"Zones: {len(self.config.zones)}, Routes: {len(self.config.routes)}"
            self.fig.suptitle(f"OSM Editor | {mode_str} | {status_str}", fontsize=12)

    def _show_help(self) -> None:
        """Display keyboard shortcuts help menu (T033)."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║              OSM ZONES EDITOR - KEYBOARD SHORTCUTS          ║
╠══════════════════════════════════════════════════════════════╣
║ MODE SWITCHING                                               ║
║   P          Switch to ZONE drawing mode                     ║
║   R          Switch to ROUTE drawing mode                    ║
║                                                              ║
║ DRAWING CONTROLS                                             ║
║   Click      Add vertex to current polygon/route            ║
║   Right-Click Delete nearest vertex                          ║
║   Drag       Move existing vertex (ZONE mode)                ║
║   Enter      Finish current polygon/route                    ║
║   Escape     Cancel current drawing                          ║
║                                                              ║
║ EDITING                                                      ║
║   Ctrl+Z     Undo last action                                ║
║   Ctrl+Y     Redo last undone action                         ║
║   Shift      Toggle vertex snapping to boundaries            ║
║                                                              ║
║ FILE OPERATIONS                                              ║
║   Ctrl+S     Save zones/routes to YAML                       ║
║                                                              ║
║ HELP                                                         ║
║   H          Show this help menu                             ║
╚══════════════════════════════════════════════════════════════╝
        """
        logger.info(help_text)

    def _redraw(self) -> None:
        """Redraw axes with current state."""
        if not self.ax or not self.fig:
            return

        # Clear overlays (keep background)
        artists_to_remove = [
            a for a in self.ax.get_children() if isinstance(a, (mpatches.Polygon, mpatches.Circle))
        ]
        for artist in artists_to_remove:
            artist.remove()

        # Redraw zones
        for zone_name, zone in self.config.zones.items():
            self._draw_zone(zone, label=zone_name)

        # Redraw routes
        for route_name, route in self.config.routes.items():
            self._draw_route(route, label=route_name)

        # Redraw current polygon if drawing
        if self.current_polygon:
            self._draw_current_polygon()

        self._update_title()
        self.fig.canvas.draw_idle()

    def _draw_zone(self, zone: Zone, label: str = "") -> None:
        """Draw a zone polygon on axes (T027 with world→pixel conversion)."""
        if not self.ax or len(zone.polygon) < 3:
            return

        # Convert world → pixel for display
        polygon_pixels = zone.polygon
        if self.affine_data:
            try:
                polygon_pixels = [world_to_pixel(pt, self.affine_data) for pt in zone.polygon]
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to convert zone polygon: {e}")

        polygon_patch = mpatches.Polygon(
            polygon_pixels,
            alpha=0.3,
            color="blue" if zone.type == "spawn" else "green",
            edgecolor="black",
            linewidth=1.5,
            label=label or zone.name,
        )
        self.ax.add_patch(polygon_patch)

    def _draw_route(self, route: Route, label: str = "") -> None:
        """Draw a route as a polyline on axes (T027 with world→pixel conversion)."""
        if not self.ax or len(route.waypoints) < 2:
            return

        # Convert world → pixel for display
        waypoints_pixels = route.waypoints
        if self.affine_data:
            try:
                waypoints_pixels = [world_to_pixel(pt, self.affine_data) for pt in route.waypoints]
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to convert route waypoints: {e}")

        xs = [p[0] for p in waypoints_pixels]
        ys = [p[1] for p in waypoints_pixels]

        self.ax.plot(
            xs,
            ys,
            "o-",
            color="red",
            linewidth=2,
            markersize=5,
            label=label or route.name,
        )

    def _draw_current_polygon(self) -> None:
        """Draw vertices being added in DRAW mode with markers (T027-T028).

        Converts world coordinates back to pixel for display on PNG.
        Shows vertex indices and world coordinates.
        Highlights hovered/dragged vertices for visual feedback (T028).
        """
        if not self.ax or not self.current_polygon:
            return

        pixel_points = self._current_polygon_pixels()
        if not pixel_points:
            return

        if len(pixel_points) > 1:
            self._draw_polygon_lines(pixel_points)
            self._draw_vertex_markers(pixel_points)
        else:
            self._draw_single_vertex(pixel_points[0])

    def _current_polygon_pixels(self) -> list[Vec2D]:
        """Convert current polygon to pixel coordinates with fallback.

        Returns:
            Pixel-space vertices; falls back to original coordinates on failure.
        """
        if not self.current_polygon:
            return []
        if not self.affine_data:
            return self.current_polygon
        try:
            return [world_to_pixel(world_pt, self.affine_data) for world_pt in self.current_polygon]
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Failed to convert world to pixel: {e}")
            return self.current_polygon

    def _draw_polygon_lines(self, pixel_points: list[Vec2D]) -> None:
        """Draw connecting lines for the in-progress polygon."""
        xs = [p[0] for p in pixel_points]
        ys = [p[1] for p in pixel_points]
        line_color = "red" if not self._last_validation.get("valid", True) else "yellow"
        self.ax.plot(xs, ys, "o-", color=line_color, linewidth=2, markersize=8, label="Current")

    def _draw_vertex_markers(self, pixel_points: list[Vec2D]) -> None:
        """Draw vertex markers and labels for multi-point polygons."""
        for i, (px, py) in enumerate(pixel_points):
            color, radius, linewidth = self._vertex_style(i)
            circle = mpatches.Circle(
                (px, py), radius=radius, color=color, alpha=0.7, ec="black", linewidth=linewidth
            )
            self.ax.add_patch(circle)

            world_x, world_y = self.current_polygon[i]
            label_text = f"V{i}\n({world_x:.1f},{world_y:.1f})"
            self.ax.text(px + 15, py, label_text, fontsize=8, color="white", weight="bold")

    def _draw_single_vertex(self, pixel_point: Vec2D) -> None:
        """Draw a single vertex marker with label."""
        px, py = pixel_point
        color, radius, linewidth = self._vertex_style(0)
        circle = mpatches.Circle(
            (px, py), radius=radius, color=color, alpha=0.7, ec="black", linewidth=linewidth
        )
        self.ax.add_patch(circle)

        world_x, world_y = self.current_polygon[0]
        label_text = f"V0\n({world_x:.1f},{world_y:.1f})"
        self.ax.text(px + 15, py, label_text, fontsize=8, color="white", weight="bold")

    def _vertex_style(self, idx: int) -> tuple[str, int, float]:
        """Style tuple for a vertex index.

        Returns:
            Color, radius, and line width to render the vertex marker.
        """
        if idx == self._dragging_vertex_idx:
            return "cyan", 14, 2.5
        if idx == self._hovered_vertex_idx:
            return "lime", 12, 2.0
        return "yellow", 10, 1.5

    # ========================================================================
    # Helper Methods for Vertex Editing (T028)
    # ========================================================================

    def _find_vertex_at_pixel(self, pixel_x: float, pixel_y: float) -> int | None:
        """Find which vertex (if any) is near the given pixel coordinates (T028).

        Uses drag threshold distance to determine if click is close enough to a vertex.

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels

        Returns:
            Index of nearby vertex, or None if no vertex within threshold
        """
        if not self.current_polygon or not self.affine_data:
            return None

        for idx, world_pt in enumerate(self.current_polygon):
            try:
                pixel_pt = world_to_pixel(world_pt, self.affine_data)
                dx = pixel_x - pixel_pt[0]
                dy = pixel_y - pixel_pt[1]
                distance = (dx**2 + dy**2) ** 0.5

                if distance <= self._vertex_drag_threshold:
                    return idx
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Failed to convert vertex {idx} to pixel: {e}")
                continue

        return None

    def _move_vertex(
        self, vertex_idx: int, world_x: float, world_y: float, record_action: bool = True
    ) -> Vec2D | None:
        """Move a vertex to new world coordinates (T028 & T029).

        Args:
            vertex_idx: Index of vertex to move
            world_x: New X coordinate in world meters
            world_y: New Y coordinate in world meters
            record_action: If True, record MoveVertexAction in history (T029)

        Returns:
            Old position if move was successful, None otherwise
        """
        if 0 <= vertex_idx < len(self.current_polygon):
            old_pos = self.current_polygon[vertex_idx]
            self.current_polygon[vertex_idx] = (world_x, world_y)
            if record_action and old_pos != (world_x, world_y):
                # Only record action if vertex actually moved (T029)
                action = MoveVertexAction(vertex_idx, old_pos, (world_x, world_y))
                self.history.push_action(action)
            logger.info(
                f"Moved vertex {vertex_idx} from ({old_pos[0]:.2f}, {old_pos[1]:.2f}) "
                f"to ({world_x:.2f}, {world_y:.2f})"
            )
            return old_pos
        return None

    def _delete_vertex_at_index(self, vertex_idx: int) -> None:
        """Delete a vertex by index (T028 & T029: with history tracking).

        Args:
            vertex_idx: Index of vertex to delete
        """
        if 0 <= vertex_idx < len(self.current_polygon):
            deleted = self.current_polygon.pop(vertex_idx)
            # Push DeleteVertexAction to history (T029)
            action = DeleteVertexAction(deleted, vertex_idx)
            self.history.push_action(action)
            logger.info(
                f"Deleted vertex {vertex_idx} at ({deleted[0]:.2f}, {deleted[1]:.2f}), "
                f"remaining: {len(self.current_polygon)}"
            )
            if not self.current_polygon:
                self.mode = EditorMode.IDLE
                logger.info("Drawing cancelled (no vertices left)")

    def _snap_to_boundary(
        self, world_x: float, world_y: float, tolerance_m: float = 0.5
    ) -> tuple[float, float]:
        """Snap vertex to nearest allowed_areas boundary if enabled (T030).

        Uses Shapely nearest-point detection to find closest point on boundary polygon
        within specified tolerance distance.

        Args:
            world_x: Original X coordinate in world meters
            world_y: Original Y coordinate in world meters
            tolerance_m: Maximum snap distance in meters (default 0.5m)

        Returns:
            Snapped coordinates (world_x, world_y) if snapping enabled and boundary found,
            otherwise original coordinates.
        """
        if not self.snap_enabled or not self.map_definition:
            return world_x, world_y

        # Get allowed_areas from map_definition (if available)
        allowed_areas = getattr(self.map_definition, "allowed_areas", None)
        if not allowed_areas or len(allowed_areas) == 0:
            logger.debug("No allowed_areas available for snapping")
            return world_x, world_y

        try:
            # Create point for current vertex position
            current_point = Point(world_x, world_y)

            snap_candidate = self._nearest_boundary_point(current_point, allowed_areas)
            if snap_candidate:
                nearest_pt, snap_dist = snap_candidate
                if snap_dist < tolerance_m:
                    logger.info(
                        f"Snapped vertex from ({world_x:.2f}, {world_y:.2f}) to "
                        f"({nearest_pt.x:.2f}, {nearest_pt.y:.2f}) [distance: {snap_dist:.3f}m]"
                    )
                    return (nearest_pt.x, nearest_pt.y)

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Snapping failed: {e}")

        return world_x, world_y

    def _nearest_boundary_point(
        self, current_point: Point, allowed_areas: list
    ) -> tuple[Point, float] | None:
        """Find the nearest boundary point among allowed areas.

        Returns:
            Tuple of nearest point and distance, or None when no boundary found.
        """
        best_snap_dist = float("inf")
        best_point: Point | None = None

        for area in allowed_areas:
            try:
                boundary_poly = Polygon(area) if isinstance(area, (list, tuple)) else area
                nearest_pt = (
                    nearest_points(current_point, boundary_poly.boundary)[1]
                    if nearest_points is not None
                    else self._nearest_manual_point(current_point, boundary_poly)
                )
                snap_dist = current_point.distance(nearest_pt)
                if snap_dist < best_snap_dist:
                    best_snap_dist = snap_dist
                    best_point = nearest_pt
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error snapping to boundary {area}: {e}")
                continue

        if best_point is None:
            return None
        return best_point, best_snap_dist

    def _nearest_manual_point(self, current_point: Point, boundary_poly: Polygon) -> Point:
        """Fallback nearest-point search using exterior coordinates.

        Returns:
            Nearest point on the polygon boundary to the provided point.
        """
        min_dist = float("inf")
        nearest_pt = current_point
        for coord in boundary_poly.exterior.coords:
            dist = current_point.distance(Point(coord))
            if dist < min_dist:
                min_dist = dist
                nearest_pt = Point(coord)
        return nearest_pt

    def _validate_polygon(self, polygon: list[Vec2D] | None = None) -> dict[str, list[str]]:
        """Validate current or provided polygon for out-of-bounds and obstacle crossing (T031).

        Checks:
        1. Out-of-bounds: All vertices and polygon edges must be within allowed_areas
        2. Obstacle crossing: Polygon must not overlap with obstacles

        Args:
            polygon: Polygon to validate. If None, uses current_polygon.

        Returns:
            Dict with validation status:
                {
                    "valid": bool,
                    "errors": list of error messages (out-of-bounds, obstacle crossing),
                    "warnings": list of warning messages
                }
        """
        result = {"valid": True, "errors": [], "warnings": []}

        if not self.map_definition:
            logger.debug("No map_definition available for validation")
            return result

        poly = polygon or self.current_polygon
        if len(poly) < 3:
            logger.debug("Polygon has < 3 vertices, skipping validation")
            return result

        poly_shape = self._build_polygon_shape(poly, result)
        if poly_shape is None:
            return result

        self._check_out_of_bounds(poly_shape, result)
        self._check_obstacle_crossings(poly_shape, result)

        if not result["valid"]:
            logger.info(f"Validation failed: {'; '.join(result['errors'])}")

        return result

    def _build_polygon_shape(
        self, vertices: list[Vec2D], result: dict[str, list[str]]
    ) -> Polygon | None:
        """Create a valid Shapely polygon or record the error.

        Returns:
            Valid polygon geometry or None when creation/repair fails.
        """
        try:
            poly_shape = ShapelyPolygon(vertices)
            if poly_shape.is_valid:
                return poly_shape
            poly_shape = poly_shape.buffer(0)
            if poly_shape.is_valid:
                return poly_shape
            result["errors"].append("Polygon is self-intersecting (invalid geometry)")
        except (ValueError, TypeError) as e:
            result["errors"].append(f"Failed to create polygon geometry: {e}")

        result["valid"] = False
        return None

    def _check_out_of_bounds(self, poly_shape: Polygon, result: dict[str, list[str]]) -> None:
        """Check that polygon stays within allowed areas, if provided."""
        allowed_areas = getattr(self.map_definition, "allowed_areas", None)
        if not allowed_areas:
            return

        try:
            allowed_union = self._union_allowed_areas(allowed_areas)
            if allowed_union and not poly_shape.within(allowed_union):
                difference = poly_shape.difference(allowed_union)
                if difference.area > 0:
                    result["errors"].append(
                        "Zone extends outside allowed areas "
                        f"(out-of-bounds area: {difference.area:.3f} m²)"
                    )
                    result["valid"] = False
                    logger.warning("Out-of-bounds zone detected")
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error checking bounds: {e}")

    def _union_allowed_areas(self, allowed_areas: list[Any]) -> Polygon | None:
        """Union allowed areas into a single geometry.

        Returns:
            Combined polygon of all allowed areas, or None if none are usable.
        """
        allowed_union = None
        for area in allowed_areas:
            try:
                area_poly = self._to_polygon(area)
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error processing allowed area: {e}")
                continue

            allowed_union = area_poly if allowed_union is None else allowed_union.union(area_poly)
        return allowed_union

    def _to_polygon(self, area: Any) -> Polygon:
        """Convert list/tuple areas to shapely polygons.

        Returns:
            Shapely polygon for the provided area definition.
        """
        if isinstance(area, (list, tuple)):
            return ShapelyPolygon(area)
        return area

    def _check_obstacle_crossings(self, poly_shape: Polygon, result: dict[str, list[str]]) -> None:
        """Check for overlaps between polygon and known obstacles."""
        obstacles = getattr(self.map_definition, "obstacles", None)
        if not obstacles:
            return

        try:
            for obstacle_idx, obstacle in enumerate(obstacles):
                try:
                    obs_poly = self._obstacle_polygon(obstacle)
                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Error processing obstacle {obstacle_idx}: {e}")
                    continue

                if not poly_shape.intersects(obs_poly):
                    continue

                intersection = poly_shape.intersection(obs_poly)
                if intersection.area > 0:
                    result["errors"].append(
                        "Zone crosses obstacle "
                        f"#{obstacle_idx} (overlap area: {intersection.area:.3f} m²)"
                    )
                    result["valid"] = False
                    logger.warning(f"Obstacle crossing detected at obstacle {obstacle_idx}")
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error checking obstacles: {e}")

    def _obstacle_polygon(self, obstacle: Any) -> Polygon:
        """Return a shapely polygon for an obstacle definition.

        Returns:
            Shapely polygon describing the obstacle.
        """
        if hasattr(obstacle, "vertices"):
            return ShapelyPolygon(obstacle.vertices)
        return self._to_polygon(obstacle)

    # ========================================================================
    # Event Handlers (T027-T033 foundation)
    # ========================================================================

    def _on_click(self, event: Any) -> None:
        """Click handler for adding vertices and mode switching (T027).

        Converts pixel coordinates to world coordinates using affine transform.
        Left click: add vertex. Right click: delete vertex.
        """
        if not event.inaxes or event.button is None:
            return

        pixel_x, pixel_y = event.xdata, event.ydata
        if pixel_x is None or pixel_y is None:
            return

        logger.debug(f"Click at pixel ({pixel_x:.1f}, {pixel_y:.1f}), mode={self.mode.value}")

        world_x, world_y = self._pixel_to_world_safe(pixel_x, pixel_y)

        if event.button == 1:
            self._handle_left_click(world_x, world_y, pixel_x, pixel_y)
        elif event.button == 3:
            self._handle_right_click(pixel_x, pixel_y)

        self._redraw()

    def _pixel_to_world_safe(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """Convert pixel coordinates to world space with logging and fallback.

        Returns:
            Converted world coordinates, or the original pixel values if conversion fails.
        """
        if self.affine_data:
            try:
                world_x, world_y = pixel_to_world((pixel_x, pixel_y), self.affine_data)
                logger.debug(f"  → World coordinates: ({world_x:.2f}, {world_y:.2f})")
                return world_x, world_y
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Failed to convert pixel to world: {e}")
        else:
            logger.debug("  (using pixel coordinates, no affine transform loaded)")
        return pixel_x, pixel_y

    def _handle_left_click(
        self, world_x: float, world_y: float, pixel_x: float, pixel_y: float
    ) -> None:
        """Handle left-click interactions in draw/idle modes."""
        if self.mode == EditorMode.DRAW:
            vertex_idx = self._find_vertex_at_pixel(pixel_x, pixel_y)
            if vertex_idx is not None:
                self._dragging_vertex_idx = vertex_idx
                self._drag_start_pos = self.current_polygon[vertex_idx]
                logger.info(f"Started dragging vertex {vertex_idx}")
                return

            index = len(self.current_polygon)
            self.current_polygon.append((world_x, world_y))
            action = AddVertexAction((world_x, world_y), index)
            self.history.push_action(action)
            logger.info(
                f"Added vertex #{len(self.current_polygon)}: ({world_x:.2f}, {world_y:.2f})"
            )
            return

        if self.mode == EditorMode.IDLE:
            self.current_polygon = [(world_x, world_y)]
            self.mode = EditorMode.DRAW
            logger.info(f"Started drawing {self.draw_mode.value} at ({world_x:.2f}, {world_y:.2f})")

    def _handle_right_click(self, pixel_x: float, pixel_y: float) -> None:
        """Handle right-click deletion logic."""
        if self.mode != EditorMode.DRAW or not self.current_polygon:
            return

        vertex_idx = self._find_vertex_at_pixel(pixel_x, pixel_y)
        if vertex_idx is not None:
            self._delete_vertex_at_index(vertex_idx)
            return

        deleted = self.current_polygon.pop()
        logger.info(
            f"Deleted last vertex ({deleted[0]:.2f}, {deleted[1]:.2f}), "
            f"remaining: {len(self.current_polygon)}"
        )
        if not self.current_polygon:
            self.mode = EditorMode.IDLE
            logger.info("Drawing cancelled (no vertices left)")

    def _on_motion(self, event: Any) -> None:
        """Motion handler for vertex dragging and hover feedback (T028, T030).

        Handles:
        - Dragging vertices to new positions (if _dragging_vertex_idx is set)
        - Snapping vertices to allowed areas boundaries if enabled (T030)
        - Visual hover feedback (highlight vertex under cursor)
        """
        if not event.inaxes or self.mode != EditorMode.DRAW:
            self._dragging_vertex_idx = None
            self._drag_start_pos = None
            self._hovered_vertex_idx = None
            return

        pixel_x = event.xdata
        pixel_y = event.ydata
        if pixel_x is None or pixel_y is None:
            return

        # If dragging a vertex, move it to new position (T028 drag completion)
        if self._dragging_vertex_idx is not None:
            try:
                if self.affine_data:
                    world_x, world_y = pixel_to_world((pixel_x, pixel_y), self.affine_data)
                else:
                    world_x, world_y = pixel_x, pixel_y

                # Apply snapping if enabled (T030)
                world_x, world_y = self._snap_to_boundary(world_x, world_y)

                self._move_vertex(self._dragging_vertex_idx, world_x, world_y, record_action=False)
                logger.debug(
                    f"Dragging vertex {self._dragging_vertex_idx} to ({world_x:.2f}, {world_y:.2f})"
                )

                # Validate polygon after movement (T031)
                self._last_validation = self._validate_polygon()
                if not self._last_validation["valid"]:
                    logger.warning(f"Polygon validation failed: {self._last_validation['errors']}")
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.warning(f"Failed during drag: {e}")
                self._dragging_vertex_idx = None
                self._drag_start_pos = None

        # Update hover feedback (T028 visual feedback)
        hovered_idx = self._find_vertex_at_pixel(pixel_x, pixel_y)
        if hovered_idx != self._hovered_vertex_idx:
            self._hovered_vertex_idx = hovered_idx
            if hovered_idx is not None:
                logger.debug(f"Hovering over vertex {hovered_idx}")
            self._redraw()

    def _on_key_press(self, event: Any) -> None:
        """Keyboard handler for mode switching, undo/redo, save (T029, T032, T033, T028).

        Also handles stopping drag when releasing mouse buttons or other events.
        """
        if event.key is None:
            return

        logger.debug(f"Key pressed: {event.key}")

        if self._stop_drag_on_key_press():
            return

        key_actions = {
            "p": self._activate_zone_mode,
            "r": self._activate_route_mode,
            "shift": self._toggle_snapping,
            "h": self._show_help,
            "ctrl+z": self._undo_action,
            "ctrl+y": self._redo_action,
            "ctrl+s": self._save_if_configured,
            "escape": self._cancel_drawing,
        }

        action = key_actions.get(event.key)
        if action:
            action()
            return

        if event.key == "enter" and self.mode == EditorMode.DRAW:
            self._finish_polygon_if_ready()

    def _stop_drag_on_key_press(self) -> bool:
        """Stop vertex dragging on key press.

        Returns:
            True if a drag was cancelled, False otherwise.
        """
        if self._dragging_vertex_idx is None:
            return False
        self._finish_drag()
        logger.debug("Drag cancelled due to key press")
        self._redraw()
        return True

    def _finish_drag(self) -> None:
        """Record a single move action for the completed drag."""
        if self._dragging_vertex_idx is not None and self._drag_start_pos is not None:
            try:
                new_pos = self.current_polygon[self._dragging_vertex_idx]
            except IndexError:
                new_pos = None
            else:
                if self._drag_start_pos != new_pos:
                    action = MoveVertexAction(
                        self._dragging_vertex_idx, self._drag_start_pos, new_pos
                    )
                    self.history.push_action(action)

        self._dragging_vertex_idx = None
        self._drag_start_pos = None

    def _on_button_release(self, event: Any) -> None:
        """Finalize drag and record history on mouse button release."""
        if event.button != 1:
            return
        if self._dragging_vertex_idx is None:
            return
        self._finish_drag()
        self._redraw()

    def _activate_zone_mode(self) -> None:
        """Switch editor to zone drawing mode."""
        self.draw_mode = DrawMode.ZONE
        logger.info("Switched to ZONE mode (P)")
        self._update_title()
        self._redraw()

    def _activate_route_mode(self) -> None:
        """Switch editor to route drawing mode."""
        self.draw_mode = DrawMode.ROUTE
        logger.info("Switched to ROUTE mode (R)")
        self._update_title()
        self._redraw()

    def _toggle_snapping(self) -> None:
        """Toggle snapping to allowed areas."""
        self.snap_enabled = not self.snap_enabled
        logger.info(f"Snapping: {self.snap_enabled}")
        self._update_title()
        self._redraw()

    def _undo_action(self) -> None:
        """Undo last change and refresh display."""
        if self.history.undo(self):
            logger.info("Undo executed")
        self._redraw()

    def _redo_action(self) -> None:
        """Redo last undone change and refresh display."""
        if self.history.redo(self):
            logger.info("Redo executed")
        self._redraw()

    def _save_if_configured(self) -> None:
        """Save YAML if output path is available."""
        if self.output_yaml:
            self._save_yaml()
        else:
            logger.warning("No output YAML path specified")

    def _finish_polygon_if_ready(self) -> None:
        """Finish polygon if enough vertices exist."""
        if len(self.current_polygon) >= 3:
            self._finish_current_polygon()
        else:
            logger.warning("Polygon needs at least 3 vertices")

    def _cancel_drawing(self) -> None:
        """Cancel current drawing session."""
        self.current_polygon = []
        self.mode = EditorMode.IDLE
        logger.info("Drawing cancelled")
        self._redraw()

    # ========================================================================
    # Polygon Management
    # ========================================================================

    def _finish_current_polygon(self) -> None:
        """Finish drawing current polygon and add to config."""
        # Zones need 3+ points, routes need 2+
        min_points = 3 if self.draw_mode == DrawMode.ZONE else 2
        if len(self.current_polygon) < min_points:
            logger.warning(f"{self.draw_mode.value} needs at least {min_points} points")
            return

        # Preserve polygon for history actions
        polygon_copy = list(self.current_polygon)

        if self.draw_mode == DrawMode.ZONE:
            zone_name = f"zone_{len(self.config.zones) + 1}"
            zone = Zone(
                name=zone_name,
                type="spawn",  # Default; could be made configurable
                polygon=polygon_copy,
            )
            self.config.zones[zone_name] = zone
            action = FinishPolygonAction(polygon_copy, zone_name, DrawMode.ZONE, "spawn")
            self.history.push_action(action)
            logger.info(f"Added zone: {zone_name} with {len(self.current_polygon)} vertices")
        else:
            route_name = f"route_{len(self.config.routes) + 1}"
            route = Route(
                name=route_name,
                waypoints=polygon_copy,
                route_type="pedestrian",
            )
            self.config.routes[route_name] = route
            action = FinishPolygonAction(polygon_copy, route_name, DrawMode.ROUTE)
            self.history.push_action(action)
            logger.info(f"Added route: {route_name} with {len(self.current_polygon)} waypoints")

        self.current_polygon = []
        self.mode = EditorMode.IDLE
        self._redraw()

    # ========================================================================
    # YAML I/O (T032)
    # ========================================================================

    def _save_yaml(self) -> None:
        """Save current config to YAML file."""
        if not self.output_yaml:
            logger.error("No output YAML path specified")
            return

        try:
            save_zones_yaml(self.config, self.output_yaml)
            logger.info(f"Saved zones to {self.output_yaml}")
        except (OSError, ValueError) as e:
            logger.error(f"Failed to save YAML: {e}")

    def load_yaml(self, yaml_file: str) -> None:
        """Load zones from YAML file."""
        try:
            self.config = load_zones_yaml(yaml_file)
            logger.info(f"Loaded zones from {yaml_file}")
            self._redraw()
        except (OSError, ValueError) as e:
            logger.error(f"Failed to load YAML: {e}")

    # ========================================================================
    # Run & Cleanup
    # ========================================================================

    def run(self, blocking: bool = True) -> None:
        """Launch the interactive editor.

        Args:
            blocking: If True, blocks until window is closed
        """
        self.setup_display()
        logger.info("Editor window opened. Press 'H' for help with keyboard shortcuts.")
        logger.info("Quick start: 'P' for zone mode, 'R' for route mode, click to draw.")
        logger.info("Ctrl+S to save, Ctrl+Z/Ctrl+Y for undo/redo.")

        if blocking:
            plt.show()

    def close(self) -> None:
        """Close the editor window."""
        if self.fig:
            plt.close(self.fig)
            logger.info("Editor window closed")
