# Test Map Fixtures

**Purpose**: Provide test data for map loading and validation tests in the fast-pysf test suite.

**Location**: `fast-pysf/tests/test_maps/`

## Available Fixtures

### Valid Map Fixtures

#### `map_regular.json`
Complete map with all components (obstacles, routes, crowded zones).

**Structure**:
```json
{
    "version": "1.0",
    "name": "test_map_regular",
    "bounds": {
        "x_margin": [min_x, max_x],
        "y_margin": [min_y, max_y]
    },
    "ped_routes": [
        {
            "name": "route_name",
            "waypoints": [[x1, y1], [x2, y2], ...],
            "reversible": "true" | "false"
        }
    ],
    "crowded_zones": [
        {
            "name": "zone_name",
            "zone_rect": [[x1, y1], [x2, y2], [x3, y3]]
        }
    ],
    "obstacles": [
        {
            "name": "obstacle_name",
            "vertices": [[x1, y1], [x2, y2], ...]
        }
    ]
}
```

**Usage**: Tests complete map loading functionality

---

#### `map_no_obstacles.json`
Map with routes and crowded zones but no obstacles.

**Usage**: Tests optional obstacles handling

---

#### `map_no_routes.json`
Map with obstacles and crowded zones but no pedestrian routes.

**Usage**: Tests optional routes handling

---

#### `map_no_crowded_zone.json`
Map with obstacles and routes but no crowded zones.

**Usage**: Tests optional crowded zones handling

---

### Invalid Map Fixtures

#### `invalid_json_file.json`
Malformed JSON file (not valid JSON syntax).

**Content**: Plain text that cannot be parsed as JSON

**Usage**: Tests error handling for invalid JSON files

---

## Adding New Test Maps

To add a new test map fixture:

1. **Create the JSON file** in this directory following the schema above
2. **Use descriptive naming**: `map_<feature_description>.json`
3. **Add test case** in `fast-pysf/tests/test_map_loader.py`:
   ```python
   def test_my_new_map():
       map_file = str(MAPS_DIR / 'map_my_feature.json')
       map_definition = load_map(map_file)
       # Add assertions
   ```
4. **Document the fixture** in this README

## JSON Schema

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Map format version (currently "1.0") |
| `name` | string | Yes | Human-readable map identifier |
| `bounds` | object | Yes | Map boundary constraints |
| `ped_routes` | array | No | Pedestrian route definitions |
| `crowded_zones` | array | No | High-density area definitions |
| `obstacles` | array | No | Static obstacle polygons |

### Bounds Object

```json
{
    "x_margin": [min_x, max_x],
    "y_margin": [min_y, max_y]
}
```

- Both `x_margin` and `y_margin` are required
- Values must be numeric (float or int)
- `max` must be greater than `min`

### Route Object

```json
{
    "name": "route_1",
    "waypoints": [[x1, y1], [x2, y2], ...],
    "reversible": "true"
}
```

- `name`: Unique identifier for the route
- `waypoints`: Array of [x, y] coordinate pairs (minimum 2 points)
- `reversible`: String "true" or "false" (pedestrians can walk both directions)

### Crowded Zone Object

```json
{
    "name": "zone_1",
    "zone_rect": [[x1, y1], [x2, y2], [x3, y3]]
}
```

- `name`: Unique identifier for the zone
- `zone_rect`: Triangle vertices (exactly 3 coordinate pairs)

### Obstacle Object

```json
{
    "name": "obstacle_1",
    "vertices": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
}
```

- `name`: Unique identifier for the obstacle
- `vertices`: Polygon vertices (minimum 3 points for a closed shape)

## Test Integration

These fixtures are used by `test_map_loader.py` which uses dynamic path resolution:

```python
from pathlib import Path

TEST_DIR = Path(__file__).parent
MAPS_DIR = TEST_DIR / 'test_maps'

# Load fixture
map_file = str(MAPS_DIR / 'map_regular.json')
map_definition = load_map(map_file)
```

This pattern ensures tests work regardless of where pytest is invoked from (repository root or subdirectory).

## Related Files

- **Test file**: `fast-pysf/tests/test_map_loader.py`
- **Map loader**: `fast-pysf/pysocialforce/map_loader.py`
- **Map config**: `fast-pysf/pysocialforce/map_config.py`
- **Production maps**: `fast-pysf/maps/` (not used in tests)

## Notes

- All fixture files are tracked in version control
- Fixtures should be minimal but representative (small coordinates, simple shapes)
- Invalid fixtures should clearly demonstrate specific error cases
- Path resolution uses `Path(__file__).parent` to work from any directory
