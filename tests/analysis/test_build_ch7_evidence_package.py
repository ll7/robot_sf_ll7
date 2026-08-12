"""Contract tests for the release-level Chapter 7 evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from typing import TYPE_CHECKING

import pytest

from scripts.analysis import build_ch7_evidence_package as builder

if TYPE_CHECKING:
    from pathlib import Path


ARMS = (
    "goal",
    "guarded_ppo",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "orca",
    "ppo",
    "prediction_planner",
    "predictive_mppi",
    "risk_dwa",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "social_force",
    "socnav_sampling",
)
SCENARIOS = (
    "classic_realworld_double_bottleneck_high",
    "francis2023_blind_corner",
    "francis2023_narrow_doorway",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_sums(root: Path) -> str:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{_sha256(path)}  {path.relative_to(root).as_posix()}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(rows) + "\n", encoding="ascii")
    return _sha256(sums)


def _make_source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    _write_json(
        root / "package_manifest.json",
        {"n_requested": 90, "n_admitted": 88, "n_excluded": 2, "visualization_only": True},
    )
    _write_json(
        root / "package_complete.json",
        {"n_requested": 90, "n_admitted": 88, "n_excluded": 2, "visualization_only": True},
    )
    rows = [{"admission_status": "admitted", "episode_id": f"episode-{i}"} for i in range(88)]
    rows.extend({"admission_status": "excluded", "episode_id": f"excluded-{i}"} for i in range(2))
    _write_json(
        root / "mapping_receipt.json",
        {
            "schema_version": "mapping.v1",
            "n_rows": 90,
            "rows": rows,
            "provenance": {"release_bundle_sha256": "a" * 64},
        },
    )
    return root, _write_sums(root)


def _scenario_values(scenario: str, planner: str) -> tuple[float, float, float, float]:
    if scenario == "classic_realworld_double_bottleneck_high":
        if planner in builder.HYBRID_ARMS:
            return 1.0, 0.0, 0.0, 0.0
        return 0.0, 1.0, 1.0, 0.0
    if scenario == "francis2023_blind_corner":
        if planner == "ppo":
            return 22 / 30, 8 / 30, 0.0, 8 / 30
        return 0.0, 1.0, 1.0, 0.0
    if planner in {"orca", "social_force"}:
        return 0.0, 0.0, 0.0, 0.0
    if planner == "hybrid_rule_v3_fast_progress_static_escape_continuous":
        return 0.0, 23 / 30, 0.0, 23 / 30
    return 0.0, 1.0, 0.0, 1.0


def _make_release(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "release-root"
    payload = root / "release" / "payload"
    _write_json(payload / "campaign_manifest.json", {"campaign_id": "fixture-release"})
    _write_json(
        payload / "release/release_manifest.resolved.json",
        {"release_tag": "0.0.3", "release_id": "fixture"},
    )
    _write_json(
        payload / "reports/matrix_summary.json",
        {
            "rows": [
                {
                    "planner_key": arm,
                    "algo": arm,
                    "planner_group": "fixture",
                    "kinematics": "differential_drive",
                    "config_hash": f"config-{i}",
                    "campaign_id": "fixture",
                    "git_commit": "commit",
                    "horizon": 600,
                    "resolved_seeds": list(range(111, 141)),
                }
                for i, arm in enumerate(ARMS)
            ]
        },
    )
    rows = []
    for scenario in SCENARIOS:
        for planner in ARMS:
            success, collision, ped_collision, obstacle_collision = _scenario_values(
                scenario, planner
            )
            rows.append(
                {
                    "planner_key": planner,
                    "scenario_family": "fixture",
                    "scenario_id": scenario,
                    "episodes": "30",
                    "success_mean": f"{success:.4f}",
                    "collisions_mean": f"{collision:.4f}",
                    "ped_collision_count_mean": f"{ped_collision:.4f}",
                    "obstacle_collision_count_mean": f"{obstacle_collision:.4f}",
                    "total_collision_count_mean": f"{collision:.4f}",
                    "near_misses_mean": "0.0",
                    "time_to_goal_norm_mean": "1.0",
                    "path_efficiency_mean": "1.0",
                    "snqi_mean": "'0.0",
                }
            )
    scenario_path = payload / "reports/scenario_breakdown.csv"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    with scenario_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for planner in ARMS:
        episodes = payload / "runs" / f"{planner}__differential_drive" / "episodes.jsonl"
        episodes.parent.mkdir(parents=True, exist_ok=True)
        with episodes.open("w", encoding="utf-8") as stream:
            for scenario in SCENARIOS:
                for index in range(30):
                    route = False
                    timeout = False
                    if scenario == SCENARIOS[0]:
                        route = planner in builder.HYBRID_ARMS
                    elif scenario == SCENARIOS[1]:
                        route = planner == "ppo" and index < 22
                    elif planner in {"orca", "social_force"}:
                        timeout = True
                    elif planner == "hybrid_rule_v3_fast_progress_static_escape_continuous":
                        timeout = index >= 23
                    record = {
                        "scenario_id": scenario,
                        "algo": planner,
                        "outcome": {
                            "route_complete": route,
                            "collision_event": not route and not timeout,
                        },
                        "termination_reason": "route_complete"
                        if route
                        else ("terminated" if timeout else "collision"),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root / "release", arcname="fixture-release")
    return archive, _sha256(archive)


def _make_compact(tmp_path: Path) -> Path:
    root = tmp_path / "compact"
    _write_json(
        root / "compact_packet.json",
        {"schema_version": "issue_6814_compact_packet.v1", "disposition": "unsupported"},
    )
    _write_sums(root)
    return root


@pytest.fixture
def fixture_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    source, source_digest = _make_source(tmp_path)
    archive, archive_digest = _make_release(tmp_path)
    compact = _make_compact(tmp_path)
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text("schema_version: ch7_case_portfolio.v2\n", encoding="utf-8")
    monkeypatch.setattr(builder, "EXPECTED_SOURCE_SHA256SUMS", source_digest)
    monkeypatch.setattr(builder, "EXPECTED_RELEASE_ARCHIVE_SHA256", archive_digest)
    return {"source": source, "archive": archive, "compact": compact, "portfolio": portfolio}


def test_build_is_deterministic_and_retains_unavailable_trace_boundaries(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "package"
    manifest = builder.build_ch7_evidence_package(
        source_package=fixture_inputs["source"],
        release_archive=fixture_inputs["archive"],
        issue6814_compact=fixture_inputs["compact"],
        output=output,
        portfolio_config=fixture_inputs["portfolio"],
        check_determinism=True,
    )
    assert manifest["status"] == "blocked_pending_domain_approval"
    assert manifest["counts"] == {"requested": 90, "admitted": 88, "excluded": 2}
    assert manifest["atlas"]["audit_cells"] == 42
    assert manifest["atlas"]["publication_cells"] == 20
    assert (output / "publication/chapter7_release_cells.pdf").stat().st_size > 0
    assert (output / "publication/chapter7_release_cells.svg").stat().st_size > 0
    assert (
        json.loads((output / "unavailable/doorway_ppo_seed113_114.json").read_text())["status"]
        == "unavailable"
    )
    assert (
        json.loads((output / "unavailable/double_bottleneck_goal_ppo_seed118.json").read_text())[
            "status"
        ]
        == "unavailable"
    )
    assert json.loads((output / "mapping_ledger.json").read_text())["counts"] == {
        "requested": 90,
        "admitted": 88,
        "excluded": 2,
    }
    assert not list(output.rglob("*.jsonl"))
    assert not list(output.rglob("*.tar.gz"))
    assert len((output / "SHA256SUMS").read_text().splitlines()) >= 10


def test_terminal_signature_fixture_preserves_timeout_and_collision_counts(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    output = tmp_path / "package"
    builder.build_ch7_evidence_package(
        source_package=fixture_inputs["source"],
        release_archive=fixture_inputs["archive"],
        issue6814_compact=fixture_inputs["compact"],
        output=output,
        portfolio_config=fixture_inputs["portfolio"],
    )
    payload = json.loads((output / "publication/reduced_atlas.json").read_text())
    rows = {
        row["planner_key"]: row
        for row in payload["cells"]
        if row["scenario_id"] == "francis2023_narrow_doorway"
    }
    assert rows["orca"]["terminal_counts"] == {"timeout": 30}
    assert rows["social_force"]["terminal_counts"] == {"timeout": 30}
    assert rows["hybrid_rule_v3_fast_progress_static_escape_continuous"]["terminal_counts"] == {
        "collision_event": 23,
        "timeout": 7,
    }


def test_source_digest_mismatch_stops_before_package_creation(
    fixture_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "EXPECTED_SOURCE_SHA256SUMS", "0" * 64)
    with pytest.raises(builder.Ch7EvidencePackageError, match="SHA256SUMS digest mismatch"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=fixture_inputs["compact"],
            output=tmp_path / "package",
            portfolio_config=fixture_inputs["portfolio"],
        )
    assert not (tmp_path / "package").exists()
