# Telemetry Pane Display Fix

## Problem
The matplotlib graph in the telemetry pane was only appearing for very short periods of time, making it difficult to observe metrics during simulation.

## Root Cause Analysis
The issue was caused by **display persistence loss** in the Pygame rendering pipeline:

1. **Throttled Rendering**: The telemetry pane was configured with `refresh_hz=2.0`, meaning it only re-rendered every 500ms
2. **No Caching**: When rendering was throttled and returned `None`, nothing was blitted to the screen for that frame
3. **Frame Buffer Clearing**: Since Pygame clears the frame buffer each frame, the lack of blitting meant the telemetry pane disappeared from view until the next refresh
4. **Result**: The graph was visible for a few frames when rendered, then disappeared for the next 500ms until the next render cycle

## Solution Implemented

### 1. Surface Caching (Primary Fix)
Modified `TelemetryPane.render_surface()` to cache and reuse the last rendered surface:

```python
def render_surface(self):
    """Render the pane as a pygame.Surface, or return cached surface if throttled."""
    now = _timestamp_ms()
    min_interval_ms = 1000.0 / max(self.refresh_hz, 0.1)
    
    if self._last_render_ms and now - self._last_render_ms < min_interval_ms:
        # Return cached surface if available for display persistence
        return self._last_surface
    
    # ... render new surface and cache it ...
    self._last_surface = surface
    return surface
```

**Benefit**: The telemetry pane now displays continuously between render intervals, providing visual persistence while reducing computational overhead.

### 2. Buffer Management Fixes
Enhanced buffer handling in `visualization.py`:

- **Immediate Buffer Copy**: Copy matplotlib canvas buffer immediately to prevent garbage collection issues
- **Proper RGBA Handling**: Ensure RGBA array is copied after color channel reordering
- **Surface Standalone Copy**: Convert pygame surface references to standalone copies to prevent buffer reference issues
- **Contiguous Memory Layout**: Enforce C-contiguous memory layout for pygame compatibility

```python
# Copy buffer immediately to preserve it
buffer = buffer.copy()

# Ensure standalone surface (not a reference)
surface = pygame.image.frombuffer(rgba, ...)
return surface.copy()  # Standalone copy
```

### 3. Layout Fixes
Replaced `fig.tight_layout()` with `fig.subplots_adjust()` for more reliable small-figure rendering:

```python
# More reliable for small figures
fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1)
```

## Files Modified
- `robot_sf/telemetry/pane.py`: Added surface caching mechanism
- `robot_sf/telemetry/visualization.py`: Improved buffer management and layout handling

## User Experience Improvement
- ✅ Telemetry graph now displays continuously without flickering
- ✅ Metrics remain visible between render intervals
- ✅ Smoother visual experience with persistent display
- ✅ Computational efficiency maintained through refresh throttling

## Testing
All telemetry tests pass:
- `test_render_metric_panel_shape`: PASSED
- `test_telemetry_session_appends_and_writes`: PASSED
- `test_live_pane_refresh_and_fps_budget`: PASSED
- `test_replay_alignment_within_tolerance`: PASSED
- `test_export_combined_image`: PASSED

## Technical Notes
The fix leverages Pygame's capability to blit the same surface multiple times per frame update. By caching the last rendered surface, we achieve:
1. **Continuous Display**: Graph visible on every frame, not just during render intervals
2. **Lower Latency Perception**: Smoother visual experience despite throttled rendering
3. **Energy Efficiency**: Computational work (matplotlib rendering) only performed at configured frequency
4. **Memory Safety**: Proper buffer management prevents garbage collection issues
