# Documentation Tooling

This subtree holds documentation-adjacent tooling and legacy assets that do not belong in the
repository root.

The relocation decision comes from the root-layout inventory in
[docs/context/issue_1573_root_layout_inventory.md](../context/issue_1573_root_layout_inventory.md)
and the low-risk tooling move tracked by Issue #1579.

## Contents

| Path | Status | Use |
| --- | --- | --- |
| [class_diagram/](class_diagram/) | generated docs tooling | Stores generated Pyreverse SVG class/package diagrams and the helper script that refreshes them from the repository root. |
| [svg_conv/](svg_conv/) | legacy tooling | Preserves an older OpenStreetMap SVG-to-JSON converter plus sample files for historical reference. Prefer the maintained parser in [robot_sf/nav/svg_map_parser.py](../../robot_sf/nav/svg_map_parser.py) for current map ingestion work. |

## Notes

Run the UML helper from the repository root:

```bash
./docs/tooling/class_diagram/generate_uml.sh
```

Treat `svg_conv/` as compatibility/reference material, not the preferred implementation path for
new SVG map parser features.
