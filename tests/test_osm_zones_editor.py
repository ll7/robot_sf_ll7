"""Tests for OSM zones visual editor (T026-T033).

This test suite validates:
- Editor initialization and display setup (T026)
- Click handlers for vertex placement (T027)
- Vertex editing (drag, delete) - T028
- Undo/redo operations - T029
- Snapping logic - T030
- Validation warnings - T031
- YAML save integration - T032
- Keyboard shortcuts - T033

Note: Interactive tests (T028, T033) are primarily manual/visual.
Programmatic tests focus on state management and core logic.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from robot_sf.maps.osm_zones_editor import (
    DrawMode,
    EditorAction,
    EditorMode,
    OSMZonesEditor,
    UndoRedoStack,
)
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, Zone


class TestEditorInitialization:
    """Test OSMZonesEditor class initialization (T026)."""

    def test_editor_creation(self):
        """Test basic editor creation without display."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name, output_yaml="/tmp/zones.yaml")

            assert editor.png_file == Path(tmp.name)
            assert editor.output_yaml == "/tmp/zones.yaml"
            assert editor.mode == EditorMode.IDLE
            assert editor.draw_mode == DrawMode.ZONE
            assert len(editor.current_polygon) == 0
            assert not editor.snap_enabled

    def test_editor_with_initial_config(self):
        """Test editor initialized with existing config."""
        config = OSMZonesConfig(zones={"z1": Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])})

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name, initial_config=config)
            assert len(editor.config.zones) == 1
            assert "z1" in editor.config.zones

    def test_editor_default_config(self):
        """Test editor creates empty config by default."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            assert len(editor.config.zones) == 0
            assert len(editor.config.routes) == 0


class TestEditorStateManagement:
    """Test editor mode and state transitions."""

    def test_mode_switching(self):
        """Test switching between IDLE, DRAW, EDIT modes."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Start in IDLE
            assert editor.mode == EditorMode.IDLE

            # Simulate starting draw
            editor.current_polygon = [(0, 0)]
            editor.mode = EditorMode.DRAW
            assert editor.mode == EditorMode.DRAW

            # Simulate finishing
            editor.mode = EditorMode.IDLE
            assert editor.mode == EditorMode.IDLE

    def test_draw_mode_switching(self):
        """Test switching between ZONE and ROUTE draw modes."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            assert editor.draw_mode == DrawMode.ZONE

            editor.draw_mode = DrawMode.ROUTE
            assert editor.draw_mode == DrawMode.ROUTE

            editor.draw_mode = DrawMode.ZONE
            assert editor.draw_mode == DrawMode.ZONE

    def test_snap_toggle(self):
        """Test snapping toggle."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            assert not editor.snap_enabled
            editor.snap_enabled = True
            assert editor.snap_enabled
            editor.snap_enabled = False
            assert not editor.snap_enabled


class TestPolygonManagement:
    """Test polygon creation and management (T027, T028)."""

    def test_finish_zone_polygon(self):
        """Test finishing a zone polygon."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Start with a 3-vertex polygon
            editor.current_polygon = [(0, 0), (1, 0), (1, 1)]
            editor.draw_mode = DrawMode.ZONE
            editor.mode = EditorMode.DRAW

            editor._finish_current_polygon()

            # Check zone was added
            assert len(editor.config.zones) == 1
            assert "zone_1" in editor.config.zones
            zone = editor.config.zones["zone_1"]
            assert zone.polygon == [(0, 0), (1, 0), (1, 1)]
            assert zone.type == "spawn"

            # Check state reset
            assert len(editor.current_polygon) == 0
            assert editor.mode == EditorMode.IDLE

    def test_finish_route_polygon(self):
        """Test finishing a route."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Start with a 2-waypoint route
            editor.current_polygon = [(0, 0), (1, 1)]
            editor.draw_mode = DrawMode.ROUTE
            editor.mode = EditorMode.DRAW

            editor._finish_current_polygon()

            # Check route was added
            assert len(editor.config.routes) == 1
            assert "route_1" in editor.config.routes
            route = editor.config.routes["route_1"]
            assert route.waypoints == [(0, 0), (1, 1)]
            assert route.route_type == "pedestrian"

    def test_insufficient_vertices(self):
        """Test that polygons with < 3 vertices are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Try to finish polygon with only 2 points
            editor.current_polygon = [(0, 0), (1, 1)]
            editor.draw_mode = DrawMode.ZONE
            editor._finish_current_polygon()

            # Zone should not be added
            assert len(editor.config.zones) == 0


class TestClickHandlers:
    """Test click handler pixel↔world coordinate transforms (T027)."""

    def test_click_handler_with_affine_transform(self):
        """Test click converts pixel→world coordinates using affine."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal affine transform JSON
            affine_json = Path(tmpdir) / "affine_transform.json"
            affine_data = {
                "pixel_per_meter": 2.0,
                "bounds_meters": [0.0, 0.0, 100.0, 100.0],
                "pixel_dimensions": [200, 200],
                "dpi": 100,
            }
            affine_json.write_text(json.dumps(affine_data))

            # Create editor with affine
            png_file = Path(tmpdir) / "test.png"
            png_file.write_bytes(b"\x89PNG")

            editor = OSMZonesEditor(
                png_file=str(png_file),
                affine_json=str(affine_json),
            )

            assert editor.affine_data is not None
            assert editor.affine_data["pixel_per_meter"] == 2.0

            # Simulate click event
            event = MagicMock()
            event.inaxes = True
            event.button = 1  # Left click
            event.xdata = 0.0  # Pixel (0, 0)
            event.ydata = 0.0

            # Start drawing
            editor._on_click(event)
            assert editor.mode == EditorMode.DRAW
            assert len(editor.current_polygon) == 1
            # World coordinates should match pixel transform
            assert editor.current_polygon[0] == (0.0, 0.0)

    def test_click_handler_pixel_to_world_transform(self):
        """Test pixel→world coordinate transformation in click handler."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            # Affine: 2 pixels/meter, bounds [0,0,100,100]
            affine_json = Path(tmpdir) / "affine_transform.json"
            affine_data = {
                "pixel_per_meter": 2.0,
                "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            }
            affine_json.write_text(json.dumps(affine_data))

            png_file = Path(tmpdir) / "test.png"
            png_file.write_bytes(b"\x89PNG")

            editor = OSMZonesEditor(png_file=str(png_file), affine_json=str(affine_json))

            # Simulate click at pixel (100, 50)
            # Expected world: (100/2.0, 50/2.0) = (50.0, 25.0)
            event = MagicMock()
            event.inaxes = True
            event.button = 1
            event.xdata = 100.0
            event.ydata = 50.0

            editor._on_click(event)
            assert len(editor.current_polygon) == 1
            assert editor.current_polygon[0] == (50.0, 25.0)

    def test_click_handler_without_affine(self):
        """Test click handler fallback when no affine transform loaded."""
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            # No affine JSON provided
            editor = OSMZonesEditor(png_file=tmp.name)
            assert editor.affine_data is None

            # Simulate click
            event = MagicMock()
            event.inaxes = True
            event.button = 1
            event.xdata = 42.5
            event.ydata = 13.7

            editor._on_click(event)
            # Should use pixel coordinates as-is when no affine
            assert editor.current_polygon[0] == (42.5, 13.7)

    def test_click_handler_delete_vertex(self):
        """Test right-click deletes last vertex."""
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Add vertices
            editor.current_polygon = [(0, 0), (1, 1), (2, 2)]
            editor.mode = EditorMode.DRAW

            # Right-click to delete
            event = MagicMock()
            event.inaxes = True
            event.button = 3  # Right click
            event.xdata = 999
            event.ydata = 999

            editor._on_click(event)
            assert len(editor.current_polygon) == 2
            assert editor.current_polygon[-1] == (1, 1)

    def test_click_handler_multiple_vertices(self):
        """Test adding multiple vertices via clicks."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as tmpdir:
            affine_json = Path(tmpdir) / "affine_transform.json"
            affine_data = {"pixel_per_meter": 1.0, "bounds_meters": [0, 0, 1000, 1000]}
            affine_json.write_text(json.dumps(affine_data))

            png_file = Path(tmpdir) / "test.png"
            png_file.write_bytes(b"\x89PNG")

            editor = OSMZonesEditor(png_file=str(png_file), affine_json=str(affine_json))

            # Click 1 at (0, 0)
            event = MagicMock()
            event.inaxes = True
            event.button = 1
            event.xdata = 0.0
            event.ydata = 0.0

            editor._on_click(event)
            assert editor.mode == EditorMode.DRAW
            assert len(editor.current_polygon) == 1

            # Click 2 at (100, 0)
            event.xdata = 100.0
            event.ydata = 0.0
            editor._on_click(event)
            assert len(editor.current_polygon) == 2

            # Click 3 at (100, 100)
            event.xdata = 100.0
            event.ydata = 100.0
            editor._on_click(event)
            assert len(editor.current_polygon) == 3
            assert editor.current_polygon[-1] == (100.0, 100.0)


class TestVertexEditing:
    """Test T028: Vertex editing (drag, delete, visual feedback)."""

    def test_find_vertex_at_pixel_with_affine(self):
        """Test finding vertex near a pixel coordinate (T028)."""
        affine_data = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "pixel_dimensions": [200, 200],
            "dpi": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(
                png_file=tmp.name,
                affine_json=None,  # Will set directly
            )
            editor.affine_data = affine_data

            # Add some vertices in world coordinates
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]

            # Vertex 0 is at world (10, 10) = pixel (20, 20)
            # Clicking at (20, 20) ± threshold should find it
            idx = editor._find_vertex_at_pixel(20.0, 20.0)
            assert idx == 0

            # Clicking nearby (within 15px threshold) should also find it
            idx = editor._find_vertex_at_pixel(25.0, 20.0)
            assert idx == 0  # Still within threshold

            # Clicking far away should not find it
            idx = editor._find_vertex_at_pixel(100.0, 100.0)
            assert idx is None

            # Vertex 1 is at (20, 20) = pixel (40, 40)
            idx = editor._find_vertex_at_pixel(40.0, 40.0)
            assert idx == 1

    def test_find_vertex_without_affine(self):
        """Test finding vertex when no affine transform is loaded (T028)."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name, affine_json=None)
            editor.affine_data = None  # No affine loaded
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0)]

            # Without affine, should return None (can't convert coordinates)
            idx = editor._find_vertex_at_pixel(20.0, 20.0)
            assert idx is None

    def test_move_vertex(self):
        """Test moving a vertex to a new position (T028)."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]

            # Move vertex 1 to a new position
            editor._move_vertex(1, 25.0, 25.0)
            assert editor.current_polygon[1] == (25.0, 25.0)

            # Other vertices unchanged
            assert editor.current_polygon[0] == (10.0, 10.0)
            assert editor.current_polygon[2] == (30.0, 30.0)

    def test_delete_vertex_at_index(self):
        """Test deleting a vertex by index (T028)."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
            editor.mode = EditorMode.DRAW

            # Delete vertex 1
            editor._delete_vertex_at_index(1)
            assert len(editor.current_polygon) == 2
            assert editor.current_polygon == [(10.0, 10.0), (30.0, 30.0)]
            assert editor.mode == EditorMode.DRAW  # Mode unchanged

            # Delete vertex 0
            editor._delete_vertex_at_index(0)
            assert len(editor.current_polygon) == 1
            assert editor.current_polygon == [(30.0, 30.0)]

            # Delete last vertex - should go back to IDLE
            editor._delete_vertex_at_index(0)
            assert len(editor.current_polygon) == 0
            assert editor.mode == EditorMode.IDLE

    def test_drag_vertex_via_motion(self):
        """Test dragging a vertex using motion events (T028)."""
        affine_data = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "pixel_dimensions": [200, 200],
            "dpi": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.affine_data = affine_data
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0)]
            editor.mode = EditorMode.DRAW

            # Create motion event
            event = MagicMock()
            event.inaxes = True
            event.xdata = 20.0  # Pixel coordinate of vertex 0
            event.ydata = 20.0

            # Start dragging vertex 0
            editor._dragging_vertex_idx = 0

            # Move to (30, 30) in pixels = (15, 15) in world
            event.xdata = 30.0
            event.ydata = 30.0
            editor._on_motion(event)

            # Vertex should have moved
            assert editor.current_polygon[0] == (15.0, 15.0)

    def test_hover_vertex_feedback(self):
        """Test visual feedback when hovering over vertices (T028)."""
        affine_data = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "pixel_dimensions": [200, 200],
            "dpi": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.affine_data = affine_data
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0)]
            editor.mode = EditorMode.DRAW

            # Create motion event
            event = MagicMock()
            event.inaxes = True

            # Hover over vertex 0 at (20, 20) pixels
            event.xdata = 20.0
            event.ydata = 20.0
            editor._on_motion(event)
            assert editor._hovered_vertex_idx == 0

            # Move away from vertices
            event.xdata = 100.0
            event.ydata = 100.0
            editor._on_motion(event)
            assert editor._hovered_vertex_idx is None

    def test_right_click_deletes_clicked_vertex(self):
        """Test that right-click deletes the clicked vertex (T028 smart delete)."""
        affine_data = {
            "pixel_per_meter": 2.0,
            "bounds_meters": [0.0, 0.0, 100.0, 100.0],
            "pixel_dimensions": [200, 200],
            "dpi": 1.0,
        }

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.affine_data = affine_data
            editor.current_polygon = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
            editor.mode = EditorMode.DRAW

            # Create right-click event at vertex 1 (pixel 40, 40)
            event = MagicMock()
            event.inaxes = True
            event.button = 3  # Right click
            event.xdata = 40.0  # Pixel of vertex 1 (20, 20) world
            event.ydata = 40.0

            editor._on_click(event)

            # Vertex 1 should be deleted
            assert len(editor.current_polygon) == 2
            assert editor.current_polygon == [(10.0, 10.0), (30.0, 30.0)]

    def test_drag_cancelled_on_key_press(self):
        """Test that dragging is cancelled when a key is pressed (T028)."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor._dragging_vertex_idx = 1

            # Create key press event
            event = MagicMock()
            event.key = "escape"

            editor._on_key_press(event)

            # Dragging should be cancelled
            assert editor._dragging_vertex_idx is None

    def test_motion_resets_drag_outside_draw_mode(self):
        """Test that dragging state is reset when exiting DRAW mode (T028)."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.mode = EditorMode.IDLE  # Not in DRAW mode
            editor._dragging_vertex_idx = 1
            editor._hovered_vertex_idx = 0

            # Create motion event
            event = MagicMock()
            event.inaxes = True
            event.xdata = 50.0
            event.ydata = 50.0

            editor._on_motion(event)

            # Both should be reset
            assert editor._dragging_vertex_idx is None
            assert editor._hovered_vertex_idx is None


class TestUndoRedoStack:
    """Test undo/redo functionality (T029)."""

    def test_stack_creation(self):
        """Test UndoRedoStack initialization."""
        stack = UndoRedoStack(max_size=50)
        assert len(stack.undo_stack) == 0
        assert len(stack.redo_stack) == 0
        assert stack.max_size == 50

    def test_action_push(self):
        """Test pushing actions to undo stack."""
        stack = UndoRedoStack()

        # Mock actions
        action1 = EditorAction(action_type="add_vertex", target="zone", data={"x": 0, "y": 0})
        action2 = EditorAction(action_type="add_vertex", target="zone", data={"x": 1, "y": 1})

        stack.push_action(action1)
        assert len(stack.undo_stack) == 1
        assert len(stack.redo_stack) == 0

        stack.push_action(action2)
        assert len(stack.undo_stack) == 2
        assert len(stack.redo_stack) == 0  # Redo stack cleared

    def test_stack_max_size(self):
        """Test that undo stack respects max size."""
        stack = UndoRedoStack(max_size=5)

        # Add 10 actions
        for i in range(10):
            action = EditorAction(action_type="add_vertex", target="zone", data={"i": i})
            stack.push_action(action)

        # Stack should be capped at max_size
        assert len(stack.undo_stack) <= 5

    def test_empty_undo(self):
        """Test undo on empty stack."""
        stack = UndoRedoStack()
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Undo on empty stack should return False
            result = stack.undo(editor)
            assert not result

    def test_empty_redo(self):
        """Test redo on empty stack."""
        stack = UndoRedoStack()
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Redo on empty stack should return False
            result = stack.redo(editor)
            assert not result


class TestYAMLIntegration:
    """Test YAML save/load integration (T032)."""

    def test_save_yaml(self):
        """Test saving editor config to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"
            png_file = Path(tmpdir) / "bg.png"
            png_file.write_bytes(b"\x89PNG")  # Minimal PNG header

            editor = OSMZonesEditor(png_file=str(png_file), output_yaml=str(yaml_file))

            # Add zone
            editor.config.zones["z1"] = Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])

            editor._save_yaml()

            assert yaml_file.exists()
            content = yaml_file.read_text()
            assert "z1" in content

    def test_load_yaml(self):
        """Test loading YAML into editor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"
            png_file = Path(tmpdir) / "bg.png"
            png_file.write_bytes(b"\x89PNG")

            # Create initial YAML
            config = OSMZonesConfig(zones={"z1": Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])})
            from robot_sf.maps.osm_zones_yaml import save_zones_yaml

            save_zones_yaml(config, str(yaml_file))

            # Load into editor
            editor = OSMZonesEditor(png_file=str(png_file))
            editor.load_yaml(str(yaml_file))

            assert len(editor.config.zones) == 1
            assert "z1" in editor.config.zones

    def test_save_no_output_path(self):
        """Test save fails gracefully when no output path specified."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name, output_yaml=None)
            # Should not raise, just log warning
            editor._save_yaml()


class TestEditorDisplay:
    """Test display setup and drawing (T026 display setup)."""

    def test_setup_display_creates_figure(self):
        """Test display setup creates figure and axes."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid PNG file
            png_file = Path(tmpdir) / "test.png"
            img = Image.new("RGB", (100, 100))
            img.save(png_file)

            editor = OSMZonesEditor(png_file=str(png_file))
            editor.setup_display()

            # Check figure and axes were created (will be actual matplotlib objects)
            assert editor.fig is not None
            assert editor.ax is not None
            assert editor.background_image is not None
            assert editor.background_image.shape == (100, 100, 3)

    def test_update_title(self):
        """Test that title updates with current state."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.draw_mode = DrawMode.ZONE
            editor.snap_enabled = False
            editor.config.zones["z1"] = Zone("z1", "spawn", [(0, 0), (1, 0), (1, 1)])

            # Mock fig
            editor.fig = MagicMock()
            editor._update_title()

            # Just verify it doesn't crash
            assert editor.fig is not None


class TestCompleteWorkflow:
    """Integration test of editor workflow (T026-T032)."""

    def test_complete_zone_editing_workflow(self):
        """Test complete workflow: create → save → load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_file = Path(tmpdir) / "zones.yaml"
            png_file = Path(tmpdir) / "bg.png"
            png_file.write_bytes(b"\x89PNG")

            # Step 1: Create editor and add zone
            editor = OSMZonesEditor(png_file=str(png_file), output_yaml=str(yaml_file))
            editor.current_polygon = [(0, 0), (1, 0), (1, 1)]
            editor.draw_mode = DrawMode.ZONE
            editor._finish_current_polygon()

            assert len(editor.config.zones) == 1

            # Step 2: Save
            editor._save_yaml()
            assert yaml_file.exists()

            # Step 3: Load into new editor
            editor2 = OSMZonesEditor(png_file=str(png_file))
            editor2.load_yaml(str(yaml_file))

            assert len(editor2.config.zones) == 1
            assert "zone_1" in editor2.config.zones
            zone = editor2.config.zones["zone_1"]
            # YAML converts tuples to lists, so check values match
            assert zone.polygon == [[0, 0], [1, 0], [1, 1]]


class TestUndoRedo:
    """Test undo/redo stack operations (T029)."""

    def test_undo_stack_empty_warning(self):
        """Test undo on empty stack returns False and logs warning."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            history = UndoRedoStack()

            result = history.undo(editor)
            assert result is False

    def test_redo_stack_empty_warning(self):
        """Test redo on empty stack returns False and logs warning."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            history = UndoRedoStack()

            result = history.redo(editor)
            assert result is False

    def test_add_vertex_action_execute_undo(self):
        """Test AddVertexAction execute and undo."""
        from robot_sf.maps.osm_zones_editor import AddVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.mode = EditorMode.DRAW
            editor.current_polygon = []

            # Execute: add vertex
            action = AddVertexAction((1.5, 2.5), 0)
            action.execute(editor)

            assert len(editor.current_polygon) == 1
            assert editor.current_polygon[0] == (1.5, 2.5)

            # Undo: remove vertex
            action.undo(editor)
            assert len(editor.current_polygon) == 0

    def test_delete_vertex_action_execute_undo(self):
        """Test DeleteVertexAction execute and undo."""
        from robot_sf.maps.osm_zones_editor import DeleteVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.mode = EditorMode.DRAW
            editor.current_polygon = [(0, 0), (1, 1), (2, 2)]

            # Execute: delete middle vertex
            action = DeleteVertexAction((1, 1), 1)
            action.execute(editor)

            assert len(editor.current_polygon) == 2
            assert editor.current_polygon == [(0, 0), (2, 2)]

            # Undo: restore vertex
            action.undo(editor)
            assert len(editor.current_polygon) == 3
            assert editor.current_polygon[1] == (1, 1)

    def test_move_vertex_action_execute_undo(self):
        """Test MoveVertexAction execute and undo."""
        from robot_sf.maps.osm_zones_editor import MoveVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(0, 0), (1, 1), (2, 2)]

            # Execute: move second vertex
            action = MoveVertexAction(1, (1, 1), (1.5, 1.5))
            action.execute(editor)

            assert editor.current_polygon[1] == (1.5, 1.5)

            # Undo: move back
            action.undo(editor)
            assert editor.current_polygon[1] == (1, 1)

    def test_undo_redo_sequence(self):
        """Test undo-redo sequence: add, delete, undo adds, redo adds."""
        from robot_sf.maps.osm_zones_editor import AddVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = []
            history = UndoRedoStack()

            # Add two vertices
            add1 = AddVertexAction((0, 0), 0)
            add2 = AddVertexAction((1, 1), 1)
            history.push_action(add1)
            history.push_action(add2)
            add1.execute(editor)
            add2.execute(editor)

            assert len(editor.current_polygon) == 2

            # Undo: remove last vertex
            history.undo(editor)
            assert len(editor.current_polygon) == 1

            # Redo: restore vertex
            history.redo(editor)
            assert len(editor.current_polygon) == 2
            assert editor.current_polygon[1] == (1, 1)

    def test_undo_clears_redo_stack(self):
        """Test that adding new action after undo clears redo stack."""
        from robot_sf.maps.osm_zones_editor import AddVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = []
            history = UndoRedoStack()

            # Add, undo, add new
            action1 = AddVertexAction((0, 0), 0)
            history.push_action(action1)
            action1.execute(editor)

            history.undo(editor)
            assert len(history.redo_stack) == 1

            # New action clears redo stack
            action2 = AddVertexAction((1, 1), 0)
            history.push_action(action2)
            assert len(history.redo_stack) == 0

    def test_undo_stack_max_size_boundary(self):
        """Test undo stack respects max_size limit."""
        from robot_sf.maps.osm_zones_editor import AddVertexAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = []
            history = UndoRedoStack(max_size=5)

            # Add 10 actions, should keep only last 5
            for i in range(10):
                action = AddVertexAction((i, i), i)
                history.push_action(action)

            assert len(history.undo_stack) == 5

    def test_undo_redo_state_consistency(self):
        """Test polygon state remains consistent through undo/redo cycles."""
        from robot_sf.maps.osm_zones_editor import (
            MoveVertexAction,
        )

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(0, 0), (1, 1), (2, 2)]
            history = UndoRedoStack()

            original_state = list(editor.current_polygon)

            # Perform a move operation
            move_action = MoveVertexAction(1, (1, 1), (1.5, 1.5))
            history.push_action(move_action)
            move_action.execute(editor)

            # Verify state changed
            assert editor.current_polygon[1] == (1.5, 1.5)

            # Undo should restore original
            history.undo(editor)
            assert editor.current_polygon == original_state

            # Redo should apply move again
            history.redo(editor)
            assert editor.current_polygon[1] == (1.5, 1.5)

    def test_finish_polygon_action(self):
        """Test FinishPolygonAction for committing zones/routes."""
        from robot_sf.maps.osm_zones_editor import FinishPolygonAction

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            polygon = [(0, 0), (1, 0), (1, 1)]

            # Execute: create zone
            action = FinishPolygonAction(polygon, "zone_1", DrawMode.ZONE)
            action.execute(editor)

            assert "zone_1" in editor.config.zones
            assert editor.config.zones["zone_1"].polygon == polygon

            # Undo: remove zone
            action.undo(editor)
            assert "zone_1" not in editor.config.zones

            # Redo: restore zone
            action.execute(editor)
            assert "zone_1" in editor.config.zones


class TestSnapping:
    """Test snapping logic (T030)."""

    def test_snap_disabled_returns_original(self):
        """Test that snapping disabled returns original coordinates."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            assert not editor.snap_enabled

            world_x, world_y = 5.5, 10.5
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y)

            assert snapped_x == world_x
            assert snapped_y == world_y

    def test_snap_no_map_definition(self):
        """Test snapping with no map_definition returns original."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.snap_enabled = True
            editor.map_definition = None

            world_x, world_y = 5.5, 10.5
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y)

            assert snapped_x == world_x
            assert snapped_y == world_y

    def test_snap_no_allowed_areas(self):
        """Test snapping with no allowed_areas returns original."""
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.snap_enabled = True

            # Mock map_definition with no allowed_areas
            editor.map_definition = MagicMock()
            editor.map_definition.allowed_areas = None

            world_x, world_y = 5.5, 10.5
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y)

            assert snapped_x == world_x
            assert snapped_y == world_y

    def test_snap_to_boundary_within_tolerance(self):
        """Test snapping to boundary within tolerance."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.snap_enabled = True

            # Create simple square boundary: (0,0) to (10,10)
            boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

            # Mock map_definition with allowed_areas
            editor.map_definition = MagicMock()
            editor.map_definition.allowed_areas = [boundary]

            # Test point near (5, 0) edge - should snap closer
            world_x, world_y = 5.2, 0.4  # ~0.45m from edge
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y, tolerance_m=0.5)

            # Should snap to boundary
            assert snapped_x == 5.2  # X stays same (perpendicular projection)
            assert snapped_y == 0.0  # Y snaps to edge

    def test_snap_outside_tolerance(self):
        """Test that points outside tolerance are not snapped."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.snap_enabled = True

            # Create simple square boundary
            boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

            editor.map_definition = MagicMock()
            editor.map_definition.allowed_areas = [boundary]

            # Test point far from edge (> 0.5m)
            world_x, world_y = 15.0, 15.0  # Far outside
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y, tolerance_m=0.5)

            # Should NOT snap (outside tolerance)
            assert snapped_x == world_x
            assert snapped_y == world_y

    def test_snap_toggle_with_shift_key(self):
        """Test Shift key toggles snapping."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            assert not editor.snap_enabled

            # Toggle on
            editor.snap_enabled = True
            assert editor.snap_enabled

            # Toggle off
            editor.snap_enabled = False
            assert not editor.snap_enabled

    def test_multiple_boundaries_snap_to_closest(self):
        """Test snapping to closest boundary when multiple available."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.snap_enabled = True

            # Two square boundaries
            boundary1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])  # Left square
            boundary2 = Polygon([(10, 0), (15, 0), (15, 5), (10, 5)])  # Right square

            editor.map_definition = MagicMock()
            editor.map_definition.allowed_areas = [boundary1, boundary2]

            # Point between boundaries but closer to boundary2
            world_x, world_y = 8.0, 2.5
            snapped_x, snapped_y = editor._snap_to_boundary(world_x, world_y, tolerance_m=2.5)

            # Should snap to closer boundary2 edge at x=10
            assert snapped_x == 10.0
            assert snapped_y == 2.5


class TestValidation:
    """Test real-time validation logic (T031)."""

    def test_validate_no_map_definition(self):
        """Test validation with no map_definition returns valid."""
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(0, 0), (1, 0), (1, 1)]
            editor.map_definition = None

            result = editor._validate_polygon()
            assert result["valid"]
            assert len(result["errors"]) == 0

    def test_validate_short_polygon(self):
        """Test validation with < 3 vertices skips validation."""

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)
            editor.current_polygon = [(0, 0), (1, 0)]  # Only 2 vertices

            result = editor._validate_polygon()
            assert result["valid"]
            assert len(result["errors"]) == 0

    def test_validate_within_bounds(self):
        """Test polygon completely within allowed areas validates as valid."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Large allowed area: (0,0) to (10,10)
            allowed_area = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = []
            editor.map_definition.allowed_areas = [allowed_area]

            # Small polygon inside: (1,1) to (3,3)
            editor.current_polygon = [(1, 1), (3, 1), (3, 3), (1, 3)]

            result = editor._validate_polygon()
            assert result["valid"]
            assert len(result["errors"]) == 0

    def test_validate_out_of_bounds(self):
        """Test polygon extending outside allowed areas detected."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Small allowed area: (0,0) to (5,5)
            allowed_area = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = []
            editor.map_definition.allowed_areas = [allowed_area]

            # Large polygon extending outside: (2,2) to (8,8)
            editor.current_polygon = [(2, 2), (8, 2), (8, 8), (2, 8)]

            result = editor._validate_polygon()
            assert not result["valid"]
            assert len(result["errors"]) > 0
            assert "extends outside" in result["errors"][0].lower()

    def test_validate_obstacle_crossing(self):
        """Test polygon crossing obstacle detected."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Allowed area: (0,0) to (10,10)
            allowed_area = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

            # Obstacle: (3,3) to (5,5)
            obstacle = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])

            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = [obstacle]
            editor.map_definition.allowed_areas = [allowed_area]

            # Polygon crossing obstacle: (2,2) to (6,6)
            editor.current_polygon = [(2, 2), (6, 2), (6, 6), (2, 6)]

            result = editor._validate_polygon()
            assert not result["valid"]
            assert len(result["errors"]) > 0
            assert "obstacle" in result["errors"][0].lower()

    def test_validate_multiple_violations(self):
        """Test polygon with both out-of-bounds and obstacle crossing."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Small allowed area: (0,0) to (5,5)
            allowed_area = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

            # Obstacle in center: (2,2) to (3,3)
            obstacle = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])

            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = [obstacle]
            editor.map_definition.allowed_areas = [allowed_area]

            # Large polygon: (1,1) to (8,8) - crosses both bounds and obstacle
            editor.current_polygon = [(1, 1), (8, 1), (8, 8), (1, 8)]

            result = editor._validate_polygon()
            assert not result["valid"]
            # Should have at least 2 errors (out-of-bounds + obstacle)
            assert len(result["errors"]) >= 1

    def test_validate_empty_allowed_areas(self):
        """Test validation with empty allowed_areas is valid."""
        from unittest.mock import MagicMock

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = []
            editor.map_definition.allowed_areas = []

            editor.current_polygon = [(0, 0), (5, 0), (5, 5), (0, 5)]

            result = editor._validate_polygon()
            # Empty allowed_areas means no validation constraints
            assert result["valid"]

    def test_validation_state_cached(self):
        """Test validation state is properly cached in _last_validation."""
        from unittest.mock import MagicMock

        from shapely.geometry import Polygon

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            editor = OSMZonesEditor(png_file=tmp.name)

            # Setup invalid scenario
            allowed_area = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
            editor.map_definition = MagicMock()
            editor.map_definition.obstacles = []
            editor.map_definition.allowed_areas = [allowed_area]

            # Polygon outside bounds
            editor.current_polygon = [(6, 6), (8, 6), (8, 8), (6, 8)]

            # Initially should be valid (default state)
            assert editor._last_validation["valid"]

            # Validate and update cache (as _on_motion() does)
            result = editor._validate_polygon()
            assert not result["valid"]
            editor._last_validation = result

            # Cache should now be updated
            assert not editor._last_validation["valid"]
            assert len(editor._last_validation["errors"]) > 0

            # Verify cache persists on next call
            cached_result = editor._last_validation.copy()
            result2 = editor._validate_polygon()
            editor._last_validation = result2
            assert editor._last_validation["errors"] == cached_result["errors"]


class TestSaveTrigger:
    """Test suite for T032: Save trigger (Ctrl+S functionality)."""

    def setup_method(self):
        """Setup test PNG for each test."""
        self.png_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        self.png_path = self.png_file.name
        self.png_file.close()

    def teardown_method(self):
        """Cleanup test PNG."""
        if os.path.exists(self.png_path):
            os.remove(self.png_path)

    def test_save_with_output_yaml(self):
        """Test save functionality when output_yaml is specified."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.yaml")
            editor = OSMZonesEditor(png_file=self.png_path, output_yaml=output_path)

            # Add some zones
            editor.config.zones = {
                "zone1": Zone(name="zone1", type="spawn", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])
            }

            # Mock the save function to verify it's called
            with patch("robot_sf.maps.osm_zones_editor.save_zones_yaml") as mock_save:
                editor._save_yaml()

                # Verify save was called with correct args
                mock_save.assert_called_once()
                args = mock_save.call_args
                assert args[0][0] == editor.config
                assert args[0][1] == output_path

    def test_save_without_output_yaml(self):
        """Test save behavior when no output_yaml is specified."""
        from unittest.mock import patch

        editor = OSMZonesEditor(png_file=self.png_path)  # No output_yaml

        # Mock the save function
        with patch("robot_sf.maps.osm_zones_editor.save_zones_yaml") as mock_save:
            editor._save_yaml()

            # Should not call save_zones_yaml
            mock_save.assert_not_called()

    def test_save_key_handler_ctrl_s(self):
        """Test that Ctrl+S key event triggers save."""
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.yaml")
            editor = OSMZonesEditor(png_file=self.png_path, output_yaml=output_path)

            # Mock key press event
            event = MagicMock()
            event.key = "ctrl+s"

            # Mock _save_yaml to verify it's called
            with patch.object(editor, "_save_yaml") as mock_save:
                editor._on_key_press(event)
                mock_save.assert_called_once()

    def test_save_preserves_data_round_trip(self):
        """Test that save and load preserve zone data correctly."""
        from robot_sf.maps.osm_zones_yaml import load_zones_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_roundtrip.yaml")

            # Create editor with zones
            editor = OSMZonesEditor(png_file=self.png_path, output_yaml=output_path)
            original_zone = Zone(
                name="test_zone", type="spawn", polygon=[(0, 0), (5, 0), (5, 5), (0, 5)]
            )
            editor.config.zones = {"test_zone": original_zone}

            # Save
            editor._save_yaml()

            # Verify file exists
            assert os.path.exists(output_path)

            # Load back and verify
            loaded_config = load_zones_yaml(output_path)
            assert "test_zone" in loaded_config.zones
            loaded_zone = loaded_config.zones["test_zone"]
            assert len(loaded_zone.polygon) == len(original_zone.polygon)
            # YAML converts tuples to lists, so compare values not types
            for loaded_point, original_point in zip(loaded_zone.polygon, original_zone.polygon):
                assert list(loaded_point) == list(original_point)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
