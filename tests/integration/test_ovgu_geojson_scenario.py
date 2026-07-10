"""End-to-end proof for the tracked real OpenStreetMap GeoJSON import."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.nav.geojson_map_provenance import validate_import_provenance
from robot_sf.training.scenario_loader import load_scenarios, resolve_map_definition
from scripts.validation.check_geojson_import import main as check_geojson_import

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "maps/imported/ovgu_campus_walk.geojson"
PROVENANCE = REPO_ROOT / "maps/imported/ovgu_campus_walk.provenance.yaml"
SEGMENT_MAP = REPO_ROOT / "maps/imported/ovgu_campus_walk.yaml"
SCENARIO = REPO_ROOT / "configs/scenarios/single/ovgu_campus_walk.yaml"
EPISODE_SCHEMA = REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"


def test_real_extract_round_trips_to_the_tracked_scenario_map(tmp_path: Path) -> None:
    """The reviewed public input reproduces the committed loadable segment map."""
    manifest = validate_import_provenance(PROVENANCE, SOURCE)
    generated_map = tmp_path / "ovgu_campus_walk.yaml"

    assert manifest["classification"] == "exploratory_only"
    assert check_geojson_import([str(SOURCE), str(PROVENANCE), str(generated_map)]) == 0
    assert generated_map.read_bytes() == SEGMENT_MAP.read_bytes()

    scenarios = load_scenarios(SCENARIO)
    assert [scenario["name"] for scenario in scenarios] == ["ovgu_campus_walk_exploratory"]
    map_def = resolve_map_definition(scenarios[0]["map_file"], scenario_path=SCENARIO)
    assert map_def is not None
    assert len(map_def.robot_routes) == 2
    assert map_def.obstacles


def test_real_extract_scenario_executes_one_cpu_smoke(tmp_path: Path) -> None:
    """The actual map-based runner executes the imported scenario without an exception."""
    episodes = tmp_path / "episodes.jsonl"
    summary = run_map_batch(
        SCENARIO,
        episodes,
        EPISODE_SCHEMA,
        horizon=5,
        record_forces=False,
        algo="goal",
        workers=1,
        resume=False,
    )

    assert summary["total_jobs"] == 1
    assert summary["written"] == 1
    assert summary["failed_jobs"] == 0
    assert summary["failures"] == []
    record = json.loads(episodes.read_text(encoding="utf-8").splitlines()[0])
    assert record["scenario_id"] == "ovgu_campus_walk_exploratory"
    assert record["seed"] == 4980
