"""Tests for the capability-aware map catalog verifier."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from scripts.tools.generate_map_registry import generate_registry
from scripts.validation.verify_map_catalog import validate_registry

if TYPE_CHECKING:
    from pathlib import Path


def _write_svg(path: Path, body: str) -> Path:
    """Write a minimal SVG wrapper with provided body content.

    Args:
        path: Destination SVG path.
        body: Inner SVG XML content.

    Returns:
        Path: The written SVG path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg"\n'
            '     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n'
            '     width="12" height="12">\n'
            f"{body}\n"
            "</svg>\n"
        ),
        encoding="utf-8",
    )
    return path


def _write_registry(path: Path, payload: dict[str, Any]) -> Path:
    """Write a YAML registry payload.

    Args:
        path: Destination registry path.
        payload: YAML-serializable registry payload.

    Returns:
        Path: The written registry path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _runtime_svg_body() -> str:
    """Return an SVG body with robot/pedestrian routes, zones, and obstacles."""
    return """
  <rect id="robot_spawn_zone_0" inkscape:label="robot_spawn_zone_0" x="1" y="1" width="1" height="1" />
  <rect id="robot_goal_zone_0" inkscape:label="robot_goal_zone_0" x="10" y="10" width="1" height="1" />
  <rect id="ped_spawn_zone_0" inkscape:label="ped_spawn_zone_0" x="2" y="2" width="1" height="1" />
  <rect id="ped_goal_zone_0" inkscape:label="ped_goal_zone_0" x="9" y="9" width="1" height="1" />
  <rect id="wall" inkscape:label="obstacle" x="1" y="10" width="1" height="1" />
  <path id="robot_path" inkscape:label="robot_route_0_0" d="M 1 1 L 10 10" />
  <path id="ped_path" inkscape:label="ped_route_0_0" d="M 2 2 L 9 9" />
""".strip()


def test_generator_preserves_review_fields_and_validator_accepts(tmp_path: Path) -> None:
    """Generation should preserve reviewed fields while refreshing computed fields."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(map_root / "demo.svg", _runtime_svg_body())
    _write_registry(
        registry_path,
        {
            "maps": [
                {
                    "map_id": "demo",
                    "path": "svg_maps/demo.svg",
                    "role": "reviewed-role",
                    "profile": "reviewed-profile",
                    "capabilities": {
                        "robot_runtime": True,
                        "pedestrian_runtime": True,
                        "route_only": False,
                        "obstacle_source": True,
                        "benchmark_candidate": True,
                    },
                    "limitations": ["reviewed-limitation"],
                    "validation": {"status": "reviewed", "rule_ids": ["manual"]},
                },
            ],
        },
    )

    registry = generate_registry(map_root, registry_path)
    _write_registry(registry_path, registry)

    row = registry["maps"][0]
    assert row["role"] == "reviewed-role"
    assert row["profile"] == "reviewed-profile"
    assert row["limitations"] == ["reviewed-limitation"]
    assert row["validation"] == {"status": "reviewed", "rule_ids": ["manual"]}
    assert validate_registry(registry_path, map_root) == []


def test_validator_reports_duplicate_map_ids(tmp_path: Path) -> None:
    """Duplicate map IDs should fail synchronization validation."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(map_root / "demo.svg", _runtime_svg_body())
    registry = generate_registry(map_root, registry_path)
    registry["maps"].append(dict(registry["maps"][0]))
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any("duplicate map_id" in error for error in errors)


def test_validator_reports_stale_hash_and_unregistered_svg(tmp_path: Path) -> None:
    """Changed source files and missing rows should fail validation."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(map_root / "demo.svg", _runtime_svg_body())
    registry = generate_registry(map_root, registry_path)
    registry["maps"][0]["source_sha256"] = "not-current"
    _write_svg(map_root / "extra.svg", _runtime_svg_body())
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any("stale source_sha256" in error for error in errors)
    assert any("unregistered SVG map: svg_maps/extra.svg" in error for error in errors)


def test_validator_reports_capability_parser_contradiction(tmp_path: Path) -> None:
    """Reviewed capability claims cannot exceed parser-derived facts."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(
        map_root / "obstacle_only.svg",
        '<rect id="wall" inkscape:label="obstacle" x="4" y="4" width="1" height="1" />',
    )
    registry = generate_registry(map_root, registry_path)
    registry["maps"][0]["capabilities"]["robot_runtime"] = True
    registry["maps"][0]["capabilities"]["benchmark_candidate"] = True
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any(
        "capabilities.robot_runtime contradicts parser metadata" in error for error in errors
    )


def test_validator_rejects_benchmark_profile_without_capability(tmp_path: Path) -> None:
    """Benchmark role/profile labels should not exceed reviewed capabilities."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(map_root / "demo.svg", _runtime_svg_body())
    registry = generate_registry(map_root, registry_path)
    registry["maps"][0]["role"] = "benchmark_candidate"
    registry["maps"][0]["profile"] = "benchmark_candidate"
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any(
        "role benchmark_candidate requires benchmark capability" in error for error in errors
    )
    assert any(
        "profile benchmark_candidate requires benchmark capability" in error for error in errors
    )


def test_validator_reports_non_mapping_capabilities_for_benchmark_profile(
    tmp_path: Path,
) -> None:
    """Malformed capabilities should be reported instead of crashing validation."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    _write_svg(map_root / "demo.svg", _runtime_svg_body())
    registry = generate_registry(map_root, registry_path)
    registry["maps"][0]["role"] = "benchmark_candidate"
    registry["maps"][0]["profile"] = "benchmark_candidate"
    registry["maps"][0]["capabilities"] = []
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any("capabilities must be a mapping" in error for error in errors)
    assert any(
        "role benchmark_candidate requires benchmark capability" in error for error in errors
    )
    assert any(
        "profile benchmark_candidate requires benchmark capability" in error for error in errors
    )


def test_validator_rejects_absolute_catalog_paths(tmp_path: Path) -> None:
    """Catalog paths should remain portable relative paths."""
    map_root = tmp_path / "maps" / "svg_maps"
    registry_path = tmp_path / "maps" / "registry.yaml"
    svg_path = _write_svg(map_root / "demo.svg", _runtime_svg_body())
    registry = generate_registry(map_root, registry_path)
    registry["maps"][0]["path"] = str(svg_path)
    _write_registry(registry_path, registry)

    errors = validate_registry(registry_path, map_root)

    assert any("path must be relative to the registry directory" in error for error in errors)
