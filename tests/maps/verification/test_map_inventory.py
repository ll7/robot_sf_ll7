"""Unit tests locking the public map-inventory contracts.

These tests cover :class:`MapRecord` and :class:`MapInventory` from
:mod:`robot_sf.maps.verification.map_inventory` using only pytest temporary
directories and synthetic minimal SVG files. No repository map assets are read
or modified; the inventory is exercised purely through its public surface.

See Also
--------
- robot_sf.maps.verification.map_inventory : Module under test
- specs/001-map-verification : Feature specification
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from robot_sf.maps.verification.map_inventory import MapInventory, MapRecord

if TYPE_CHECKING:
    from pathlib import Path


def _write_svg(
    path: Path,
    content: str = '<svg xmlns="http://www.w3.org/2000/svg"/>',
) -> Path:
    """Create a minimal SVG file (and parent directories) and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def populated_maps_root(tmp_path: Path) -> Path:
    """Maps root with one classic, pedestrian-only, simple, and untagged map."""
    _write_svg(tmp_path / "classic_arena.svg")
    _write_svg(tmp_path / "ped_only_corridor.svg")
    _write_svg(tmp_path / "simple_room.svg")
    _write_svg(tmp_path / "big_plaza.svg")
    return tmp_path


class TestMapRecord:
    """Lock MapRecord validation and metadata-preservation contracts."""

    def test_rejects_missing_file(self, tmp_path: Path):
        """A MapRecord whose file does not exist must raise FileNotFoundError."""
        missing = tmp_path / "ghost.svg"
        with pytest.raises(FileNotFoundError):
            MapRecord(map_id="ghost", file_path=missing)

    def test_preserves_supplied_metadata(self, tmp_path: Path):
        """All constructor-supplied metadata must round-trip onto the record."""
        svg = _write_svg(tmp_path / "alpha.svg")
        tags = {"benchmark", "classic"}
        metadata = {"goals": 3, "spawn_zones": 2}
        stamp = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)

        record = MapRecord(
            map_id="alpha",
            file_path=svg,
            tags=tags,
            ci_enabled=False,
            metadata=metadata,
            last_modified=stamp,
        )

        assert record.map_id == "alpha"
        assert record.file_path == svg
        assert record.tags == tags
        assert record.ci_enabled is False
        assert record.metadata == metadata
        assert record.last_modified == stamp

    def test_defaults(self, tmp_path: Path):
        """Omitted optional fields fall back to empty tags/metadata and ci_enabled."""
        svg = _write_svg(tmp_path / "beta.svg")
        record = MapRecord(map_id="beta", file_path=svg)

        assert record.tags == set()
        assert record.ci_enabled is True
        assert record.metadata == {}
        assert isinstance(record.last_modified, datetime)


class TestMapInventory:
    """Lock discovery, tag inference, lookup, filtering, len, and iteration."""

    def test_recursive_svg_discovery(self, tmp_path: Path):
        """Nested SVG files are discovered recursively; non-SVG files are ignored."""
        _write_svg(tmp_path / "top.svg")
        _write_svg(tmp_path / "sub" / "nested.svg")
        _write_svg(tmp_path / "sub" / "deep" / "deeper.svg")
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

        inventory = MapInventory(maps_root=tmp_path)

        assert sorted(r.map_id for r in inventory) == ["deeper", "nested", "top"]

    def test_empty_root_yields_empty_inventory(self, tmp_path: Path):
        """An existing but empty maps root produces an empty inventory."""
        inventory = MapInventory(maps_root=tmp_path)

        assert len(inventory) == 0
        assert inventory.get_all_maps() == []

    def test_missing_root_yields_empty_inventory(self, tmp_path: Path):
        """A maps root that does not exist produces an empty inventory.

        The test points the inventory at a genuinely absent path (it does not
        create the directory itself). Only the empty-outcome contract is locked
        here; the implementation's directory-creation side effect is intentionally
        not asserted to avoid changing current public behavior (see issue #6339
        stop-condition #5).
        """
        missing_root = tmp_path / "absent"
        assert not missing_root.exists()

        inventory = MapInventory(maps_root=missing_root)

        assert len(inventory) == 0
        assert inventory.get_all_maps() == []
        assert inventory.get_ci_enabled_maps() == []

    @pytest.mark.parametrize(
        ("map_id", "expected_tags"),
        [
            ("classic_arena", {"classic"}),
            ("ped_only_corridor", {"pedestrian_only"}),
            ("simple_room", {"simple"}),
            ("big_plaza", set()),
            ("classic_simple_ped_only", {"classic", "pedestrian_only", "simple"}),
            ("SIMPLE_room", {"simple"}),
            ("PED_ONLY", {"pedestrian_only"}),
        ],
    )
    def test_tag_inference(self, map_id: str, expected_tags: set[str], tmp_path: Path):
        """Tags are inferred from filename patterns (case-insensitive for ped/simple)."""
        _write_svg(tmp_path / f"{map_id}.svg")

        inventory = MapInventory(maps_root=tmp_path)
        record = inventory.get_map_by_id(map_id)

        assert record is not None
        assert record.tags == expected_tags

    def test_get_map_by_id(self, populated_maps_root: Path):
        """get_map_by_id returns the matching record and None for unknown ids."""
        inventory = MapInventory(maps_root=populated_maps_root)

        found = inventory.get_map_by_id("classic_arena")
        missing = inventory.get_map_by_id("does_not_exist")

        assert found is not None
        assert found.map_id == "classic_arena"
        assert found.file_path.name == "classic_arena.svg"
        assert found.file_path.exists()
        # Lookup is stable: repeated calls return the same record object.
        assert inventory.get_map_by_id("classic_arena") is found
        assert missing is None

    def test_get_maps_by_tag(self, populated_maps_root: Path):
        """get_maps_by_tag returns maps carrying the tag and [] for unknown tags."""
        inventory = MapInventory(maps_root=populated_maps_root)

        classic = sorted(m.map_id for m in inventory.get_maps_by_tag("classic"))
        ped = sorted(m.map_id for m in inventory.get_maps_by_tag("pedestrian_only"))
        simple = sorted(m.map_id for m in inventory.get_maps_by_tag("simple"))
        unknown = inventory.get_maps_by_tag("nonexistent")

        assert classic == ["classic_arena"]
        assert ped == ["ped_only_corridor"]
        assert simple == ["simple_room"]
        assert unknown == []

    def test_all_maps_are_ci_enabled_by_default(self, populated_maps_root: Path):
        """The inventory marks every discovered map as CI-enabled by default."""
        inventory = MapInventory(maps_root=populated_maps_root)

        assert all(m.ci_enabled for m in inventory)
        assert sorted(m.map_id for m in inventory.get_ci_enabled_maps()) == sorted(
            m.map_id for m in inventory
        )

    def test_get_ci_enabled_maps_filters_by_flag(self, populated_maps_root: Path):
        """get_ci_enabled_maps honors the ci_enabled flag rather than returning all."""
        inventory = MapInventory(maps_root=populated_maps_root)
        target = inventory.get_map_by_id("simple_room")
        assert target is not None
        target.ci_enabled = False

        enabled_ids = sorted(m.map_id for m in inventory.get_ci_enabled_maps())

        assert "simple_room" not in enabled_ids
        assert enabled_ids == ["big_plaza", "classic_arena", "ped_only_corridor"]

    def test_len_matches_record_count(self, populated_maps_root: Path):
        """len(inventory) equals the number of discovered maps."""
        inventory = MapInventory(maps_root=populated_maps_root)

        assert len(inventory) == 4
        assert len(inventory) == len(inventory.get_all_maps())

    def test_iteration_yields_each_record_once(self, populated_maps_root: Path):
        """Iterating the inventory yields each MapRecord exactly once."""
        inventory = MapInventory(maps_root=populated_maps_root)

        records = list(inventory)

        assert len(records) == len(inventory)
        assert all(isinstance(r, MapRecord) for r in records)
        assert sorted(r.map_id for r in records) == [
            "big_plaza",
            "classic_arena",
            "ped_only_corridor",
            "simple_room",
        ]
