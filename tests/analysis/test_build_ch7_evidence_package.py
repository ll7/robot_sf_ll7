"""Contract tests for the release-level Chapter 7 evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from scripts.analysis import build_ch7_evidence_package as builder

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


def _write_sums(root: Path, *, exclude: set[str] | None = None) -> str:
    exclude = exclude or set()
    rows = []
    for path in sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and p.relative_to(root).as_posix() not in exclude
    ):
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
        {
            "n_requested": 90,
            "n_admitted": 88,
            "n_excluded": 2,
            "visualization_only": True,
            "sha256sums_sha256": "pending",
        },
    )
    rows = [{"admission_status": "admitted", "episode_id": f"episode-{i}"} for i in range(88)]
    rows.extend({"admission_status": "excluded", "episode_id": f"excluded-{i}"} for i in range(2))
    _write_json(
        root / "mapping_receipt.json",
        {
            "schema_version": "mapping.v1",
            "n_rows": 90,
            "rows": rows,
            "provenance": {},
        },
    )
    source_digest = _write_sums(root, exclude={"package_complete.json"})
    complete = json.loads((root / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = source_digest
    _write_json(root / "package_complete.json", complete)
    return root, source_digest


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


def _make_compact(tmp_path: Path, source_digest: str) -> Path:
    root = tmp_path / "compact"
    _write_json(
        root / "compact_packet.json",
        {
            "schema_version": "issue_6814_compact_packet.v1",
            "issue": 6814,
            "source_package": {
                "source_issue": 6412,
                "source_package_sha256sums_sha256": source_digest,
            },
            "full_packet": {
                "manifest_retrieval_key": "issue6814/full/manifest",
                "manifest_sha256": "a" * 64,
            },
            "disposition": "unsupported",
            "check_results": {
                "artifact_integrity_ok": True,
                "deterministic_rebuild_ok": True,
                "package_digest_ok": True,
                "row_contract_digest_ok": True,
            },
            "source_contracts": [
                {
                    "path": path,
                    "sha256": f"{i + 1:064x}",
                    "status": "unsupported",
                    "trace_identity": {},
                }
                for i, path in enumerate(
                    (
                        "source_contracts/doorway_ppo_113.json",
                        "source_contracts/doorway_ppo_114.json",
                        "source_contracts/double_bottleneck_goal_118.json",
                        "source_contracts/double_bottleneck_ppo_118.json",
                    )
                )
            ],
            "pairs": [
                {
                    "pair_id": pair_id,
                    "comparison_grammar": "matched_start",
                    "comparison_grain": "matched_planner_pair",
                    "full_receipt": {
                        "retrieval_key": f"issue6814/full/pair-{i}",
                        "sha256": f"{i + 11:064x}",
                    },
                    "pair_compatibility": {
                        "status": "incompatible",
                        "shared_prefix": {"shared_prefix": False},
                    },
                    "semantic_inputs": {},
                    "process_validation": {},
                    "renderer_admission": {"disposition": "unsupported"},
                }
                for i, pair_id in enumerate(
                    (
                        "classic_doorway_medium--ppo--113-114",
                        "classic_realworld_double_bottleneck_high--goal--118-118",
                    )
                )
            ],
            "evidence_boundary": {
                "visualization_only": True,
                "new_simulation_performed": False,
                "episode_substitution_performed": False,
                "tolerance_profile_modified": False,
            },
        },
    )
    _write_sums(root)
    return root


@pytest.fixture
def fixture_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    source, source_digest = _make_source(tmp_path)
    archive, archive_digest = _make_release(tmp_path)
    mapping_path = source / "mapping_receipt.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["provenance"] = {
        "release_bundle_sha256": archive_digest,
        "release_tag": "0.0.3",
    }
    _write_json(mapping_path, mapping)
    source_digest = _write_sums(source, exclude={"package_complete.json"})
    complete = json.loads((source / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = source_digest
    _write_json(source / "package_complete.json", complete)
    compact = _make_compact(tmp_path, source_digest)
    monkeypatch.setattr(
        builder, "EXPECTED_COMPACT_PACKET_SHA256", _sha256(compact / "compact_packet.json")
    )
    monkeypatch.setattr(
        builder, "EXPECTED_COMPACT_SHA256SUMS_SHA256", _sha256(compact / "SHA256SUMS")
    )
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(
        """schema_version: ch7_case_portfolio.v2
selection:
  required_roles: [planner_upset, seed_sensitivity, feasibility_criticism, metric_disagreement]
  frozen_role_targets:
    planner_upset: ch7-role-planner-upset--classic-realworld-double-bottleneck-high--goal-vs-ppo--seed-118
    seed_sensitivity: ch7-role-seed-sensitivity--classic-doorway-medium--ppo--seeds-113-114
    feasibility_criticism: ch7-role-feasibility-criticism--francis2023-narrow-doorway
    metric_disagreement: ch7-role-cross-cell-inversion--hybrid-vs-ppo--double-bottleneck-vs-blind-corner
release_cell_selection:
  scenarios:
    - classic_realworld_double_bottleneck_high
    - francis2023_blind_corner
    - francis2023_narrow_doorway
  non_doorway_planners:
    - ppo
    - hybrid_rule_v3_fast_progress_static_escape
    - hybrid_rule_v3_fast_progress_static_escape_continuous
  doorway_planners:
    - goal
    - guarded_ppo
    - hybrid_rule_v3_fast_progress_static_escape
    - hybrid_rule_v3_fast_progress_static_escape_continuous
    - orca
    - ppo
    - prediction_planner
    - predictive_mppi
    - risk_dwa
    - sacadrl
    - scenario_adaptive_hybrid_orca_v1
    - scenario_adaptive_hybrid_orca_v2_collision_guard
    - social_force
    - socnav_sampling
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_SOURCE_SHA256SUMS", source_digest)
    monkeypatch.setattr(builder, "EXPECTED_RELEASE_ARCHIVE_SHA256", archive_digest)
    monkeypatch.setattr(
        builder,
        "EXPECTED_APPROVED_PACKAGE_COMPLETE_SHA256",
        _sha256(source / "package_complete.json"),
    )
    return {
        "source": source,
        "source_digest": source_digest,
        "archive": archive,
        "compact": compact,
        "portfolio": portfolio,
    }


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
    assert (
        json.loads((output / "review/source_verification.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "verified_but_domain_approval_pending"
    )
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
    assert {path.name for path in (output / "sketches").glob("*.md")} == set(
        builder.WIREFRAME_FILES
    )


def test_package_schema_rejects_contradictory_admission_states(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Schema consumers must not accept a blocked/admitted state contradiction."""

    output = tmp_path / "package"
    manifest = builder.build_ch7_evidence_package(
        source_package=fixture_inputs["source"],
        release_archive=fixture_inputs["archive"],
        issue6814_compact=fixture_inputs["compact"],
        output=output,
        portfolio_config=fixture_inputs["portfolio"],
    )
    schema = json.loads(
        (
            Path(__file__).parents[2] / "robot_sf/benchmark/schemas/ch7-evidence-package.v1.json"
        ).read_text(encoding="utf-8")
    )
    for field, value in (
        ("status", "admitted"),
        ("admission_status", "admitted"),
        ("source_integrity_gate", "passed"),
    ):
        mutated = dict(manifest)
        mutated[field] = value
        errors = list(builder.Draft202012Validator(schema).iter_errors(mutated))
        assert errors, field


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


def test_compact_schema_and_checksum_path_are_fail_closed(
    fixture_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compact = fixture_inputs["compact"]
    payload_path = compact / "compact_packet.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("pairs")
    _write_json(payload_path, payload)
    monkeypatch.setattr(builder, "EXPECTED_COMPACT_PACKET_SHA256", _sha256(payload_path))
    monkeypatch.setattr(builder, "EXPECTED_COMPACT_SHA256SUMS_SHA256", _write_sums(compact))
    with pytest.raises(builder.Ch7EvidencePackageError, match="compact input schema error"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=compact,
            output=tmp_path / "package",
            portfolio_config=fixture_inputs["portfolio"],
        )


def test_compact_directory_rejects_unlisted_artifact(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    """Reject compact input siblings outside its two-file digest boundary."""

    (fixture_inputs["compact"] / "UNLISTED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.Ch7EvidencePackageError, match="unlisted or missing"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=fixture_inputs["compact"],
            output=tmp_path / "package",
            portfolio_config=fixture_inputs["portfolio"],
        )


def test_compact_digest_pin_rejects_self_consistent_forgery(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    compact = fixture_inputs["compact"]
    payload_path = compact / "compact_packet.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["full_packet"]["manifest_sha256"] = "b" * 64
    _write_json(payload_path, payload)
    _write_sums(compact)
    with pytest.raises(builder.Ch7EvidencePackageError, match="approved digest"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=compact,
            output=tmp_path / "package",
            portfolio_config=fixture_inputs["portfolio"],
        )


def test_compact_unsupported_semantics_are_fail_closed(
    fixture_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compact = fixture_inputs["compact"]
    payload_path = compact / "compact_packet.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["renderer_admission"]["disposition"] = "supported"
    _write_json(payload_path, payload)
    monkeypatch.setattr(builder, "EXPECTED_COMPACT_PACKET_SHA256", _sha256(payload_path))
    monkeypatch.setattr(builder, "EXPECTED_COMPACT_SHA256SUMS_SHA256", _write_sums(compact))
    with pytest.raises(builder.Ch7EvidencePackageError, match="renderer admission"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=compact,
            output=tmp_path / "package",
            portfolio_config=fixture_inputs["portfolio"],
        )


def test_unlisted_package_complete_requires_binding(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    source = fixture_inputs["source"]
    complete = json.loads((source / "package_complete.json").read_text(encoding="utf-8"))
    complete.pop("sha256sums_sha256")
    _write_json(source / "package_complete.json", complete)
    with pytest.raises(builder.Ch7EvidencePackageError, match="package_complete"):
        builder.verify_source_package(source, fixture_inputs["source_digest"])


def test_cli_converts_manifest_schema_error_to_typed_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise_schema_error(**_: object) -> dict[str, object]:
        raise builder.ValidationError("invalid generated manifest", instance={})

    monkeypatch.setattr(builder, "build_ch7_evidence_package", _raise_schema_error)
    status = builder.main(
        [
            "--source-package",
            str(tmp_path / "source"),
            "--release-archive",
            str(tmp_path / "release.tar.gz"),
            "--issue6814-compact",
            str(tmp_path / "compact"),
            "--output",
            str(tmp_path / "package"),
        ]
    )
    assert status == 2
    assert "ch7 evidence package unavailable" in capsys.readouterr().out


def test_unlisted_source_file_is_rejected(fixture_inputs: dict[str, Path], tmp_path: Path) -> None:
    (fixture_inputs["source"] / "unlisted-trace.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(builder.Ch7EvidencePackageError, match="unlisted files"):
        builder.verify_source_package(fixture_inputs["source"], fixture_inputs["source_digest"])


def test_frozen_portfolio_mutation_is_rejected(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    portfolio = fixture_inputs["portfolio"]
    portfolio.write_text(
        portfolio.read_text(encoding="utf-8").replace(
            "required_roles: [planner_upset, seed_sensitivity, feasibility_criticism, metric_disagreement]",
            "required_roles: [seed_sensitivity]",
        ),
        encoding="utf-8",
    )
    with pytest.raises(builder.Ch7EvidencePackageError, match="required roles"):
        builder.build_ch7_evidence_package(
            source_package=fixture_inputs["source"],
            release_archive=fixture_inputs["archive"],
            issue6814_compact=fixture_inputs["compact"],
            output=tmp_path / "package",
            portfolio_config=portfolio,
        )


def test_source_release_provenance_is_required(
    fixture_inputs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping_path = fixture_inputs["source"] / "mapping_receipt.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping.pop("provenance")
    _write_json(mapping_path, mapping)
    digest = _write_sums(fixture_inputs["source"], exclude={"package_complete.json"})
    complete = json.loads(
        (fixture_inputs["source"] / "package_complete.json").read_text(encoding="utf-8")
    )
    complete["sha256sums_sha256"] = digest
    _write_json(fixture_inputs["source"] / "package_complete.json", complete)
    monkeypatch.setattr(builder, "EXPECTED_SOURCE_SHA256SUMS", digest)
    monkeypatch.setattr(
        builder,
        "EXPECTED_APPROVED_PACKAGE_COMPLETE_SHA256",
        _sha256(fixture_inputs["source"] / "package_complete.json"),
    )
    with pytest.raises(builder.Ch7EvidencePackageError, match="provenance"):
        builder.verify_source_package(fixture_inputs["source"], digest)


def test_duplicate_selected_cell_is_rejected(
    fixture_inputs: dict[str, Path], tmp_path: Path
) -> None:
    # Exercise the selection gate directly so the fixture remains small and deterministic.
    cells = [
        {"scenario_id": "classic_realworld_double_bottleneck_high", "planner_key": "ppo"},
        {"scenario_id": "classic_realworld_double_bottleneck_high", "planner_key": "ppo"},
    ]
    with pytest.raises(builder.Ch7EvidencePackageError, match="duplicate"):
        builder._validate_selected_cells(cells)


def test_selected_cell_matrix_and_denominator_are_fail_closed() -> None:
    cells = [
        {"scenario_id": scenario, "planner_key": planner, "episodes": 30}
        for scenario in builder.REQUIRED_SCENARIOS[:2]
        for planner in ("ppo", *builder.HYBRID_ARMS)
    ]
    cells.extend(
        {
            "scenario_id": builder.REQUIRED_SCENARIOS[2],
            "planner_key": planner,
            "episodes": 30,
        }
        for planner in builder.DOORWAY_ARMS
    )
    with pytest.raises(builder.Ch7EvidencePackageError, match="does not match"):
        builder._validate_selected_cells(cells[:-1])
    cells[-1]["episodes"] = 29
    with pytest.raises(builder.Ch7EvidencePackageError, match="exactly 30"):
        builder._validate_selected_cells(cells)
