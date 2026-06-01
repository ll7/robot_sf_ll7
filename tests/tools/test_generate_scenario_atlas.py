"""Tests for the scenario atlas generator."""

from __future__ import annotations

import csv
import json
import shlex
from typing import TYPE_CHECKING

import yaml

from scripts.tools import generate_scenario_atlas as atlas

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_scenario_atlas_writes_rows_thumbnails_cards_manifest_and_gaps(
    tmp_path: Path,
) -> None:
    """A tiny scenario matrix should produce a provenance-rich atlas pack."""
    matrix = _write_fixture_matrix(tmp_path)
    out_dir = tmp_path / "atlas"

    exit_code = atlas.main(
        [
            "--matrix",
            str(matrix),
            "--output",
            str(out_dir),
            "--run-id",
            "fixture_atlas",
            "--base-seed",
            "7",
        ],
    )

    assert exit_code == 0
    atlas_md = out_dir / "scenario_atlas.md"
    atlas_csv = out_dir / "scenario_atlas.csv"
    manifest_path = out_dir / "atlas_manifest.json"
    assert atlas_md.exists()
    assert atlas_csv.exists()
    assert manifest_path.exists()

    rows = list(csv.DictReader(atlas_csv.open(newline="", encoding="utf-8")))
    assert [row["scenario_id"] for row in rows] == ["fixture_crossing", "fixture_gap"]
    assert rows[0]["authored_intent"] == "orthogonal crossing fixture"
    assert rows[0]["certification_status"] == "not_certified"
    assert rows[0]["executed_evidence"] == "not_executed"
    assert rows[0]["map_ref"].endswith("fixture_map.svg")
    assert rows[0]["benchmark_ref"] == "configs/benchmarks/fixture_benchmark.yaml"
    assert rows[1]["coverage_gaps"] == "missing_contract;missing_certificate;missing_hazard_mapping"

    markdown = atlas_md.read_text(encoding="utf-8")
    assert "fixture_crossing" in markdown
    assert "not benchmark evidence" in markdown
    assert "configs/scenarios/fixture_atlas.yaml#fixture_crossing" in markdown

    thumb = out_dir / "thumbnails" / "fixture_crossing.png"
    card = out_dir / "mechanism_cards" / "fixture_crossing.md"
    assert thumb.exists()
    assert card.exists()
    assert "orthogonal crossing fixture" in card.read_text(encoding="utf-8")

    gaps = json.loads((out_dir / "coverage_gaps.json").read_text(encoding="utf-8"))
    assert {
        "scenario_id": "fixture_gap",
        "gap": "missing_contract",
        "status": "not_available",
    } in gaps["records"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "scenario_atlas_manifest.v1"
    assert manifest["run_id"] == "fixture_atlas"
    assert manifest["claim_boundary"].startswith("Scenario atlas is a discoverability")
    manifest_paths = {entry["path"] for entry in manifest["outputs"]}
    assert "scenario_atlas.md" in manifest_paths
    assert "thumbnails/fixture_crossing.png" in manifest_paths
    assert all(len(entry["sha256"]) == 64 for entry in manifest["outputs"])


def test_generate_scenario_atlas_records_map_id_file_and_stable_repeated_outputs(
    tmp_path: Path,
) -> None:
    """Repeated runs with the same inputs should produce stable thumbnails."""
    matrix = _write_fixture_matrix(tmp_path)
    out_a = tmp_path / "atlas-a"
    out_b = tmp_path / "atlas-b"

    for out_dir in (out_a, out_b):
        assert (
            atlas.main(
                [
                    "--matrix",
                    str(matrix),
                    "--output",
                    str(out_dir),
                    "--run-id",
                    "fixture_atlas",
                    "--base-seed",
                    "7",
                ],
            )
            == 0
        )

    rows = list(csv.DictReader((out_a / "scenario_atlas.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["map_ref"].endswith("fixture_map.svg")
    assert rows[1]["benchmark_ref"] == "configs/benchmarks/fixture_gap_benchmark.yaml"

    thumb_a = (out_a / "thumbnails" / "fixture_crossing.png").read_bytes()
    thumb_b = (out_b / "thumbnails" / "fixture_crossing.png").read_bytes()
    assert thumb_a == thumb_b

    manifest_a = json.loads((out_a / "atlas_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_b / "atlas_manifest.json").read_text(encoding="utf-8"))
    checksums_a = {entry["path"]: entry["sha256"] for entry in manifest_a["outputs"]}
    checksums_b = {entry["path"]: entry["sha256"] for entry in manifest_b["outputs"]}
    assert (
        checksums_a["thumbnails/fixture_crossing.png"]
        == checksums_b["thumbnails/fixture_crossing.png"]
    )


def test_command_quotes_shell_sensitive_paths_and_run_id(tmp_path: Path) -> None:
    """Recorded commands should be replayable when arguments need quoting."""
    matrix = tmp_path / "matrix with spaces.yaml"
    output = tmp_path / "out dir"
    args = atlas._build_parser().parse_args(
        [
            "--matrix",
            str(matrix),
            "--output",
            str(output),
            "--run-id",
            "fixture run; rm -rf nope",
            "--base-seed",
            "11",
        ]
    )

    command = atlas._command(args)

    assert shlex.split(command) == [
        "uv",
        "run",
        "python",
        "scripts/tools/generate_scenario_atlas.py",
        "--matrix",
        str(matrix),
        "--output",
        str(output),
        "--run-id",
        "fixture run; rm -rf nope",
        "--base-seed",
        "11",
    ]
    assert shlex.quote("fixture run; rm -rf nope") in command


def test_map_ref_preserves_map_id_and_map_file(tmp_path: Path) -> None:
    """Map-id rows should still backlink the resolved map file when available."""
    matrix = tmp_path / "fixture_atlas.yaml"
    map_file = tmp_path / "fixture_map.svg"
    matrix.write_text("scenarios: []\n", encoding="utf-8")
    map_file.write_text("<svg></svg>\n", encoding="utf-8")

    ref = atlas._map_ref({"map_id": "fixture_map", "map_file": "fixture_map.svg"}, matrix)

    assert ref.startswith("map_id:fixture_map;")
    assert ref.endswith("fixture_map.svg")


def test_path_ref_normalizes_paths_outside_repo(tmp_path: Path) -> None:
    """Non-repo path references should not depend on caller lexical spelling."""
    outside = tmp_path / "outside.yaml"
    outside.write_text("scenarios: []\n", encoding="utf-8")

    assert atlas._path_ref(outside) == outside.resolve().as_posix()


def _write_fixture_matrix(tmp_path: Path) -> Path:
    """Create a tiny scenario matrix with one complete and one gap-heavy row."""
    (tmp_path / "fixture_map.svg").write_text("<svg></svg>\n", encoding="utf-8")
    matrix = tmp_path / "fixture_atlas.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "fixture_crossing",
                        "map_file": "fixture_map.svg",
                        "density": "low",
                        "flow": "crossing",
                        "metadata": {
                            "authored_intent": "orthogonal crossing fixture",
                            "hazards": ["crossing_conflict"],
                            "known_failure_modes": ["stall"],
                            "runnable_command": "uv run robot_sf_bench run --matrix configs/scenarios/fixture_atlas.yaml",
                        },
                        "atlas": {
                            "source_path": "configs/scenarios/fixture_atlas.yaml",
                            "benchmark_config": "configs/benchmarks/fixture_benchmark.yaml",
                            "contract": "docs/scenario_contracts.md#fixture_crossing",
                            "certificate": "docs/scenario_certification.md#fixture_crossing",
                            "hazard_mapping": "docs/hazard_traceability.md#fixture_crossing",
                        },
                    },
                    {
                        "name": "fixture_gap",
                        "map_file": "fixture_map.svg",
                        "density": "medium",
                        "flow": "head_on",
                        "metadata": {
                            "intended_mechanism": "head-on gap fixture",
                        },
                        "atlas": {
                            "source_path": "configs/scenarios/fixture_atlas.yaml",
                            "benchmark_configs": ["configs/benchmarks/fixture_gap_benchmark.yaml"],
                        },
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return matrix
