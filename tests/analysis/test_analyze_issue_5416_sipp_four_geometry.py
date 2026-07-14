"""Tests for the issue #5416 paired-outcome analyzer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.analysis import analyze_issue_5416_sipp_four_geometry as analyzer

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKET = REPO_ROOT / "configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml"
SCENARIOS = (
    "classic_head_on_corridor_low",
    "classic_doorway_low",
    "classic_station_platform_medium",
    "classic_merging_low",
)
SEEDS = (111, 112, 113, 114, 115)
PLANNERS = (
    "sipp_lattice",
    "hybrid_rule_v0_minimal",
    "teb",
    "nmpc_social",
    "dwa",
)


def _row(planner: str, scenario: str, seed: int, *, success: bool = True) -> dict:
    algorithm = "hybrid_rule_local_planner" if planner == "hybrid_rule_v0_minimal" else planner
    return {
        "version": "v1",
        "scenario_id": scenario,
        "seed": seed,
        "horizon": 500,
        "metrics": {
            "ped_collision_count": 0.0,
            "obstacle_collision_count": 0.0,
            "time_to_goal_norm": 0.4 if success else 1.0,
            "path_efficiency": 0.9 if success else 0.2,
            "deadlock": not success,
        },
        "outcome": {
            "route_complete": success,
            "collision_event": False,
            "timeout_event": not success,
        },
        "integrity": {"contradictions": []},
        "algorithm_metadata": {
            "algorithm": algorithm,
            "status": "ok",
            "config": {"planner_variant": planner} if planner == "hybrid_rule_v0_minimal" else {},
            "planner_kinematics": {"execution_mode": "native"},
            "planner_diagnostics": {
                "expansion_limit_hits": 1 if planner == "sipp_lattice" and not success else 0,
                "runtime_bound_exits": 0,
                "fallback_count": 0,
                "commitment_invalidations": 1 if planner == "sipp_lattice" else 0,
                "planner_step_runtime_seconds": [0.01, 0.02, 0.03],
            },
        },
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": scenario,
            "seed": seed,
            "config_hash": "b" * 40,
            "repo_commit": "a" * 40,
            "simulator_settings": {"horizon": 500, "dt": 0.1},
        },
    }


def _write_bundle(
    tmp_path: Path, *, fallback_key: tuple[str, str, int] | None = None
) -> tuple[Path, list[Path]]:
    episodes = tmp_path / "episodes.jsonl"
    rows = []
    for planner in PLANNERS:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                row = _row(planner, scenario, seed, success=planner == "sipp_lattice")
                if fallback_key == (planner, scenario, seed):
                    row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = "adapter"
                rows.append(row)
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifests = []
    for planner in PLANNERS:
        manifest = tmp_path / f"{planner}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": analyzer.EXECUTION_PROVENANCE_SCHEMA,
                    "planner_id": planner,
                    "episodes_path": str(episodes),
                    "episodes_sha256": hashlib.sha256(episodes.read_bytes()).hexdigest(),
                    "execution_provenance": {
                        "exact_command": "uv run python scripts/validation/run_issue_5416_campaign.py",
                        "environment_manifest": "environment.json",
                        "cpu_route": "local-cpu",
                        "job_id": f"local-{planner}",
                        "resource_request": {"cpus": 1, "memory_gb": 4},
                        "wall_time_seconds": 12.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifests.append(manifest)
    return episodes, manifests


def _refresh_manifest_digests(episodes: Path, manifests: list[Path]) -> None:
    """Keep synthetic manifest hashes aligned after mutating the episode fixture."""
    digest = hashlib.sha256(episodes.read_bytes()).hexdigest()
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["episodes_sha256"] = digest
        manifest.write_text(json.dumps(payload), encoding="utf-8")


def _analyze(tmp_path: Path, episodes: Path, manifests: tuple[Path, ...] = ()) -> dict:
    return analyzer.build_analysis(
        episode_paths=[episodes],
        provenance_paths=manifests,
        output_dir=tmp_path / "report",
        packet_path=PACKET,
    )


def test_complete_native_bundle_emits_paired_summary_and_rule_inputs(tmp_path: Path) -> None:
    """A complete native fixture produces paired summary artifacts and a pass decision."""
    episodes, manifests = _write_bundle(tmp_path)

    report = _analyze(tmp_path, episodes, tuple(manifests))

    assert (
        report["status"],
        report["matrix"]["eligible_rows"],
        len(report["paired_comparisons"]),
    ) == ("complete", 100, 80)
    assert report["advancement_rule"]["status"] == "pass"
    assert (tmp_path / "report" / "summary.json").is_file()


def test_missing_diagnostics_and_provenance_do_not_use_episode_wall_time(tmp_path: Path) -> None:
    """Missing diagnostics stay blocked instead of using episode wall time."""
    episodes, _ = _write_bundle(tmp_path, fallback_key=("sipp_lattice", SCENARIOS[0], SEEDS[0]))
    rows = [json.loads(line) for line in episodes.read_text().splitlines()]
    rows[1]["algorithm_metadata"].pop("planner_diagnostics")
    rows[1]["wall_time_sec"] = 99.0
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = _analyze(tmp_path, episodes)

    assert (report["status"], report["matrix"]["eligible_rows"]) == ("partial", 99)
    assert any(
        "execution_mode is not native" in reason
        for row in report["denominator_exclusions"]
        for reason in row["reasons"]
    )
    assert [report["criteria"][index]["status"] for index in (2, 3)] == ["blocked"] * 2
    assert all(
        row.get("planner_step_runtime_median_seconds") is None
        for row in report["planner_summaries"]
        if row.get("planner_id") == "sipp_lattice"
    )


def test_unkeyed_extra_row_blocks_complete_report(tmp_path: Path) -> None:
    """An invalid extra input must block interpretation even when the matrix is full."""
    episodes, manifests = _write_bundle(tmp_path)
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    extra = _row("dwa", SCENARIOS[0], SEEDS[0])
    extra["planner_id"] = "unexpected"
    extra["algorithm_metadata"]["algorithm"] = "unregistered"
    rows.append(extra)
    episodes.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    _refresh_manifest_digests(episodes, manifests)

    report = _analyze(tmp_path, episodes, tuple(manifests))

    assert report["status"] == "partial"
    assert report["advancement_rule"]["status"] == "blocked"
    assert report["matrix"]["excluded_rows"] == 1
    assert report["denominator_exclusions"][-1]["planner_id"] is None


def test_conflicting_planner_identity_is_excluded(tmp_path: Path) -> None:
    """Contradictory planner fields must not be silently attributed to one roster key."""
    episodes, manifests = _write_bundle(tmp_path)
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    row["planner_id"] = "sipp_lattice"
    row["algorithm_metadata"]["config"] = {"planner_variant": "sipp_lattice"}
    row["algorithm_metadata"]["algorithm"] = "dwa"
    episodes.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    _refresh_manifest_digests(episodes, manifests)

    report = _analyze(tmp_path, episodes, tuple(manifests))

    assert report["status"] == "partial"
    assert any(
        "planner identity conflict" in reason
        for row in report["denominator_exclusions"]
        for reason in row["reasons"]
    )


def test_invalid_manifest_field_type_blocks_provenance(tmp_path: Path) -> None:
    """Malformed execution metadata must not satisfy the provenance gate."""
    episodes, manifests = _write_bundle(tmp_path)
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    payload["execution_provenance"]["cpu_route"] = []
    manifests[0].write_text(json.dumps(payload), encoding="utf-8")

    report = _analyze(tmp_path, episodes, tuple(manifests))

    assert report["status"] == "partial"
    assert any(
        "cpu_route" in error and "invalid" in error for error in report["provenance"]["errors"]
    )
