
# OpenStreetMap SVG Map Import

## Usage

### Download Maps From OSM
1. Go to https://www.openstreetmap.org/?mlat=48.383&mlon=10.883&zoom=11#map=18/48.33420/10.89496
2. Click Share (right side panel) -> Image -> choose SVG format -> click Download

Info: This location shows the campus of Uni Augsburg.

### Convert Maps To JSON Format

```sh
python3 svg_conv.py osm_input_file.svg output_file.json
```

Now, the file can be further edited with the MapEditor.

```sh
cd ..
python3 -m map_editor
```
