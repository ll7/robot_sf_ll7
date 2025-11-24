#!/usr/bin/env python3
"""Snapshot map IDs for validation and regression testing.

This script captures the current state of the map registry and saves it
to a snapshot file for validation purposes.

Usage:
    python scripts/validation/snapshot_map_ids.py [--output OUTPUT_FILE]
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def main():
    """Snapshot current map IDs."""
    parser = argparse.ArgumentParser(description="Snapshot map IDs from registry")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path",
        default="output/tmp/maps_snapshot.json",
    )
    args = parser.parse_args()
    
    try:
        # Import registry after path setup
        from robot_sf.maps import registry
        
        # Build registry and get IDs
        ids = registry.list_ids()
        reg = registry.get_registry()
        
        # Create snapshot
        snapshot = {
            "total_maps": len(ids),
            "map_ids": ids,
            "maps_with_metadata": [
                map_id for map_id, (svg, json_path) in reg.items() if json_path is not None
            ],
            "maps_without_metadata": [
                map_id for map_id, (svg, json_path) in reg.items() if json_path is None
            ],
        }
        
        # Save snapshot
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        
        logger.info(f"Snapshot saved to: {output_path}")
        logger.info(f"Total maps: {snapshot['total_maps']}")
        logger.info(f"Maps with metadata: {len(snapshot['maps_with_metadata'])}")
        logger.info(f"Maps without metadata: {len(snapshot['maps_without_metadata'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
