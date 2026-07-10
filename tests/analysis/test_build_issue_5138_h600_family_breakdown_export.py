"""Tests for the issue #5138 h600 family-breakdown export builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analysis.build_issue_5138_h600_family_breakdown_export import (
    CELL_COLUMNS,
    FAMILY_COLUMNS,
    HARD_SUMMARY_COLUMNS,
    SCHEMA_VERSION,
    UNIVERSALLY_HARD_FAMILIES,
    RunSource,
    build_export,
)


def _write_family_breakdown(reports_dir: Path, *, planners: tuple[str, ...]) -> None:
    """Write a minimal canonical scenario_family_breakdown.csv for fixtures.

    The cell granularity is encoded by giving different ``episodes`` per family
    so the per-cell ``n`` assertion is meaningful; success is chosen so that
    ``bottleneck`` is the only universally-zero family across the two planners.
    """

    reports_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "planner_key",
        "algo",
        "scenario_family",
        "use_case",
        "context",
        "speed_regime",
        "maneuver_type",
        "episodes",
        "success_mean",
        "collisions_mean",
        "ped_collision_count_mean",
        "obstacle_collision_count_mean",
        "total_collision_count_mean",
        "near_misses_mean",
        "time_to_goal_norm_mean",
        "path_efficiency_mean",
        "comfort_exposure_mean",
        "jerk_mean",
        "snqi_mean",
    ]
    # success by family/planner: bottleneck is universally zero; others vary.
    success_by_family = {
        "bottleneck": "0.0000",
        "cross_trap": "0.3333",
        "head_on_corridor": "0.6667",
        "accompanying_peer": "1.0000",
    }
    near_by_family = {
        "bottleneck": "12.0000",
        "cross_trap": "8.0000",
        "head_on_corridor": "5.0000",
        "accompanying_peer": "0.0000",
    }
    rows = []
    for planner in planners:
        for family in ("bottleneck", "cross_trap", "head_on_corridor", "accompanying_peer"):
            rows.append(
                {
                    "planner_key": planner,
                    "algo": planner,
                    "scenario_family": family,
                    "use_case": "",
                    "context": "",
                    "speed_regime": "",
                    "maneuver_type": "",
                    "episodes": "6",
                    "success_mean": success_by_family[family],
                    "collisions_mean": "1.0000",
                    "ped_collision_count_mean": "1.0000",
                    "obstacle_collision_count_mean": "0.0000",
                    "total_collision_count_mean": "1.0000",
                    "near_misses_mean": near_by_family[family],
                    "time_to_goal_norm_mean": "1.0000",
                    "path_efficiency_mean": "1.0000",
                    "comfort_exposure_mean": "0.0100",
                    "jerk_mean": "",
                    "snqi_mean": "-0.3000",
                }
            )
    with (reports_dir / "scenario_family_breakdown.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_cell_breakdown(reports_dir: Path, *, planners: tuple[str, ...]) -> None:
    """Write a minimal canonical scenario_breakdown.csv (per-cell) for fixtures."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "planner_key",
        "algo",
        "scenario_family",
        "scenario_id",
        "use_case",
        "context",
        "speed_regime",
        "maneuver_type",
        "episodes",
        "success_mean",
        "collisions_mean",
        "ped_collision_count_mean",
        "obstacle_collision_count_mean",
        "total_collision_count_mean",
        "near_misses_mean",
        "time_to_goal_norm_mean",
        "path_efficiency_mean",
        "comfort_exposure_mean",
        "jerk_mean",
        "snqi_mean",
    ]
    cell_spec = (
        ("bottleneck", "classic_bottleneck_low", "0.0000"),
        ("bottleneck", "classic_bottleneck_high", "0.0000"),
        ("cross_trap", "classic_cross_trap_low", "0.0000"),
        ("cross_trap", "classic_cross_trap_high", "0.6667"),
        ("head_on_corridor", "classic_head_on_corridor_low", "0.6667"),
        ("accompanying_peer", "francis2023_accompanying_peer", "1.0000"),
    )
    rows = []
    for planner in planners:
        for family, scenario_id, success in cell_spec:
            rows.append(
                {
                    "planner_key": planner,
                    "algo": planner,
                    "scenario_family": family,
                    "scenario_id": scenario_id,
                    "use_case": "",
                    "context": "",
                    "speed_regime": "",
                    "maneuver_type": "",
                    "episodes": "3",
                    "success_mean": success,
                    "collisions_mean": "1.0000",
                    "ped_collision_count_mean": "1.0000",
                    "obstacle_collision_count_mean": "0.0000",
                    "total_collision_count_mean": "1.0000",
                    "near_misses_mean": "4.0000",
                    "time_to_goal_norm_mean": "1.0000",
                    "path_efficiency_mean": "1.0000",
                    "comfort_exposure_mean": "0.0100",
                    "jerk_mean": "",
                    "snqi_mean": "-0.3000",
                }
            )
    with (reports_dir / "scenario_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_seed_episode_rows(reports_dir: Path) -> None:
    """Write a minimal seed_episode_rows.csv (provenance checksum source)."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "scenario_id",
        "planner_key",
        "seed",
        "success",
        "collision",
        "near_miss",
        "snqi",
    ]
    with (reports_dir / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "x-1-0",
                "scenario_id": "classic_bottleneck_low",
                "planner_key": "goal",
                "seed": "1",
                "success": "0.0",
                "collision": "1.0",
                "near_miss": "4.0",
                "snqi": "-0.3",
            }
        )


def _write_campaign_summary(reports_dir: Path, *, job_id: str) -> None:
    """Write a minimal campaign_summary.json carrying provenance fields."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "campaign": {
            "campaign_id": f"issue3810_h600_{job_id}",
            "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
            "scenario_matrix_hash": "c10df617a87c",
            "comparability_mapping_hash": "6f349046993d",
            "created_at_utc": "2026-07-02T11:15:01.146142Z",
            "git_hash": "abc123",
            "evidence_status": "valid",
            "benchmark_success": True,
        },
        "planner_rows": [{"planner_key": "goal"}, {"planner_key": "orca"}],
    }
    (reports_dir / "campaign_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _make_bundle(tmp_path: Path) -> tuple[Path, list[RunSource]]:
    """Create a fixture evidence bundle + two h600 legs with canonical breakdowns."""

    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    # Seed an existing source_manifest + README so the builder extends rather
    # than recreates them.
    (output_dir / "source_manifest.json").write_text(
        json.dumps({"runs": [], "generated_outputs": ["planner_metric_summary.csv"]}),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text("# Bundle\n\nExisting content.\n", encoding="utf-8")
    (output_dir / "planner_metric_summary.csv").write_text(
        "job_id,metric\n13268,success\n", encoding="utf-8"
    )

    runs = []
    for job_id, run_label, planners in (
        ("13268", "confirm", ("goal", "orca")),
        ("13273", "extended_roster", ("goal", "orca")),
    ):
        reports_dir = tmp_path / "runs" / job_id / "reports"
        _write_family_breakdown(reports_dir, planners=planners)
        _write_cell_breakdown(reports_dir, planners=planners)
        _write_seed_episode_rows(reports_dir)
        _write_campaign_summary(reports_dir, job_id=job_id)
        runs.append(RunSource(job_id=job_id, run_label=run_label, reports_dir=reports_dir))
    return output_dir, runs


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_export_produces_all_artifacts_with_release_schema(tmp_path: Path) -> None:
    """All six artifacts are written and the family table uses the release schema."""
    output_dir, runs = _make_bundle(tmp_path)

    report = build_export(output_dir, runs)

    family_csv = output_dir / "h600_scenario_family_breakdown.csv"
    cell_csv = output_dir / "h600_scenario_cell_breakdown.csv"
    hard_csv = output_dir / "h600_universally_hard_families_summary.csv"
    for artifact in (
        family_csv,
        output_dir / "h600_scenario_family_breakdown.md",
        cell_csv,
        output_dir / "h600_scenario_cell_breakdown.md",
        hard_csv,
        output_dir / "h600_universally_hard_families_summary.md",
    ):
        assert artifact.exists(), f"missing artifact {artifact.name}"

    # Per-family table uses the release schema (run provenance columns prepended).
    family_rows = _read_csv(family_csv)
    assert tuple(family_rows[0].keys()) == FAMILY_COLUMNS
    # Two legs x two planners x four families = 16 rows.
    assert len(family_rows) == 16
    # Both legs are represented and distinguishable.
    run_labels = {row["run_label"] for row in family_rows}
    assert run_labels == {"confirm", "extended_roster"}
    # The shared matrix hash is carried on every row.
    assert {row["scenario_matrix_hash"] for row in family_rows} == {"c10df617a87c"}

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["family_row_count"] == 16


def test_per_cell_episode_count_present_on_every_row(tmp_path: Path) -> None:
    """Every family-level AND cell-level row must carry a non-empty ``episodes``."""

    output_dir, runs = _make_bundle(tmp_path)
    build_export(output_dir, runs)

    family_rows = _read_csv(output_dir / "h600_scenario_family_breakdown.csv")
    cell_rows = _read_csv(output_dir / "h600_scenario_cell_breakdown.csv")
    assert family_rows, "family table is empty"
    assert cell_rows, "cell table is empty"
    for row in family_rows:
        assert row["episodes"], f"family row missing episodes: {row}"
        assert int(row["episodes"]) > 0
    for row in cell_rows:
        assert row["episodes"], f"cell row missing episodes: {row}"
        assert int(row["episodes"]) > 0
        # Cell table carries the scenario_id granularity.
        assert row["scenario_id"]
    # Cell columns include scenario_id after the family.
    assert "scenario_id" in CELL_COLUMNS


def test_universally_hard_families_surfaced_with_zero_completion_flag(tmp_path: Path) -> None:
    """The crux families are filtered and flagged so the question is answerable."""

    output_dir, runs = _make_bundle(tmp_path)
    build_export(output_dir, runs)

    hard_rows = _read_csv(output_dir / "h600_universally_hard_families_summary.csv")
    families = {row["scenario_family"] for row in hard_rows}
    assert families == set(UNIVERSALLY_HARD_FAMILIES)
    assert tuple(hard_rows[0].keys()) == HARD_SUMMARY_COLUMNS
    # bottleneck is universally zero across both planners in the fixture.
    bottleneck = [r for r in hard_rows if r["scenario_family"] == "bottleneck"]
    assert bottleneck, "bottleneck missing from hard-families summary"
    assert all(r["zero_completion"] == "yes" for r in bottleneck)
    # cross_trap and head_on_corridor are NOT universally zero in the fixture.
    non_zero = [r for r in hard_rows if r["scenario_family"] in ("cross_trap", "head_on_corridor")]
    assert all(r["zero_completion"] == "no" for r in non_zero)


def test_source_manifest_and_sha256sums_updated_and_gate_coverage(tmp_path: Path) -> None:
    """New artifacts must be registered + checksummed so the F-C4(ii) gate holds."""

    from robot_sf.benchmark.identity.hash_utils import sha256_file

    output_dir, runs = _make_bundle(tmp_path)
    build_export(output_dir, runs)

    manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
    block = manifest["issue_5138_family_breakdown_export"]
    assert block["schema_version"] == SCHEMA_VERSION
    assert block["claim_boundary"].startswith("diagnostic-only")
    assert len(block["runs"]) == 2
    # Provenance carries source SHA256 for both breakdowns + seed rows.
    for run in block["runs"]:
        sources = run["source_files"]
        assert "scenario_family_breakdown.csv" in sources
        assert "scenario_breakdown.csv" in sources
        assert "seed_episode_rows.csv" in sources
        for entry in sources.values():
            assert len(entry["sha256"]) == 64

    # SHA256SUMS covers every .md/.json/.csv file in the bundle with matching digests.
    sums = {}
    for line in (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        digest, rel = line.split()
        sums[Path(rel).name] = digest
    checksummed = [
        p for p in output_dir.iterdir() if p.is_file() and p.suffix in {".md", ".json", ".csv"}
    ]
    assert checksummed, "fixture bundle has no checksummable files"
    for path in checksummed:
        assert path.name in sums, f"{path.name} not in SHA256SUMS"
        assert sums[path.name] == sha256_file(path), f"{path.name} digest mismatch"


def test_readme_section_appended_and_is_idempotent(tmp_path: Path) -> None:
    """The #5138 README section is appended once and not duplicated on re-run."""
    output_dir, runs = _make_bundle(tmp_path)
    build_export(output_dir, runs)
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "Issue #5138 per-family + per-cell h600 breakdown export" in readme
    assert "diagnostic-only" in readme
    # Re-running does not duplicate the section.
    build_export(output_dir, runs)
    assert (output_dir / "README.md").read_text(encoding="utf-8").count(
        "## Issue #5138 per-family + per-cell h600 breakdown export"
    ) == 1


def test_generated_evidence_preserves_csv_schema_and_records_review_provenance(
    tmp_path: Path,
) -> None:
    """CSV exports retain their schema while the manifest records their review marker."""
    output_dir, runs = _make_bundle(tmp_path)
    build_export(output_dir, runs)
    manifest = json.loads((output_dir / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert set(manifest["issue_5138_family_breakdown_export"]["generated_outputs"]) == {
        "h600_scenario_cell_breakdown.csv",
        "h600_scenario_cell_breakdown.md",
        "h600_scenario_family_breakdown.csv",
        "h600_scenario_family_breakdown.md",
        "h600_universally_hard_families_summary.csv",
        "h600_universally_hard_families_summary.md",
    }
    for name, columns in (
        ("h600_scenario_cell_breakdown.csv", CELL_COLUMNS),
        ("h600_scenario_family_breakdown.csv", FAMILY_COLUMNS),
        ("h600_universally_hard_families_summary.csv", HARD_SUMMARY_COLUMNS),
    ):
        with (output_dir / name).open(newline="", encoding="utf-8") as handle:
            assert tuple(csv.DictReader(handle).fieldnames or ()) == columns


def test_build_export_fails_closed_when_canonical_breakdown_missing(tmp_path: Path) -> None:
    """A missing canonical breakdown makes the builder fail closed, not emit a partial export."""
    output_dir, runs = _make_bundle(tmp_path)
    # Delete one canonical family breakdown -> builder must fail closed.
    (runs[0].reports_dir / "scenario_family_breakdown.csv").unlink()
    with pytest.raises(FileNotFoundError):
        build_export(output_dir, runs)
