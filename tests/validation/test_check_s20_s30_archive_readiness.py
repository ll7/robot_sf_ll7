"""Tests issue #1554 S20/S30 archive-readiness checker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.tools.campaign_result_store import write_result_store
from scripts.validation import check_s20_s30_archive_readiness as readiness

if TYPE_CHECKING:
    from pathlib import Path


def _write_seed_sets(path: Path) -> Path:
    payload = {
        "paper_eval_s20": list(range(111, 131)),
        "paper_eval_s30": list(range(111, 141)),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_packet(
    path: Path,
    repo_root: Path,
    *,
    result_store: str = "store",
    seed_sets_path: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
    full_campaign_in_this_issue: object = False,
    submit_slurm_from_this_issue: object = False,
) -> Path:
    seed_sets = _write_seed_sets(repo_root / "seed_sets.yaml")
    payload = {
        "schema_version": "s20-s30-seed-budget-launch-packet.v1",
        "campaign_id": "issue_1554_test",
        "evidence_status": "proposal",
        "no_benchmark_result_claim": True,
        "no_claim_statement": "No S20/S30 paper-facing claim exists.",
        "claim_map_gate": {
            "target_claim": "S20 ordering remains stable; S30 only if rank flips.",
            "status": "to_be_confirmed_by_maintainer",
            "required_metric_surface": [
                "success",
                "collisions",
                "near_misses",
                "time_to_goal_norm",
            ],
            "planner_rows_to_confirm": ["goal", "orca"],
            "seed_tier": "s20_then_s30_if_rankflip",
            "why_s10_insufficient": "S10 seed instability remains too large for paper claims.",
        },
        "methodology_reference": "docs/context/issue_1545_power_aware_seed_budget_planning.md",
        "seed_policy": {
            "mode": "seed-set",
            "seed_sets_path": seed_sets_path or seed_sets.relative_to(repo_root).as_posix(),
            "primary_seed_set": "paper_eval_s20",
            "escalation_seed_set": "paper_eval_s30",
            "escalate_to_s30_when": "rank flip observed",
        },
        "fail_closed_policy": {
            "valid_row_statuses": ["native", "adapter"],
            "fail_closed_statuses": ["fallback", "degraded", "unavailable", "failed"],
        },
        "expected_artifacts": {
            "result_store": result_store,
            "result_store_required_files": [
                "episodes.parquet",
                "summary.json",
                "analysis.json",
                "claim_card.yaml",
                "reproduction.md",
                "tables/manifest.json",
                "figures/manifest.json",
            ],
        "bundle": [
            "output/issue_1554_s20_s30/bundle.json",
            "output/issue_1554_s20_s30/BUNDLE.md",
        ],
        },
        "execution_boundary": {
            "full_campaign_in_this_issue": full_campaign_in_this_issue,
            "submit_slurm_from_this_issue": submit_slurm_from_this_issue,
            "bundle_status_until_run": "blocked_until_run",
        },
    }
    if metadata_overrides:
        for dotted_key, value in metadata_overrides.items():
            section, key = dotted_key.split(".", maxsplit=1)
            payload[section][key] = value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _complete_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for planner in ("goal", "orca"):
        for seed in range(111, 131):
            rows.append(
                {
                    "run_id": f"{planner}-{seed}",
                    "episode_id": f"{planner}-{seed}-crossing",
                    "planner": planner,
                    "scenario_id": "crossing",
                    "scenario_family": "classic",
                    "seed": seed,
                    "row_status": "native",
                    "artifact_uri": f"wandb://robot-sf/{planner}-{seed}",
                    "artifact_sha256": "a" * 64,
                    "success": 1.0,
                    "collision": 0.0,
                    "near_misses": 0.0,
                    "time_to_goal_norm": 0.5,
                }
            )
    return rows


def test_complete_archive_metadata_and_store_pass(tmp_path: Path) -> None:
    """Complete S20 result-store metadata is archive-ready without claiming paper evidence."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
        analysis={"seed_resampling_rank_flip": {"rank_flip_observed": False}},
    )
    (repo / "output/issue_1554_s20_s30/bundle.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (repo / "output/issue_1554_s20_s30/bundle.json").write_text(
        "{}", encoding="utf-8"
    )
    (repo / "output/issue_1554_s20_s30/BUNDLE.md").write_text(
        "# S20/S30 Bundle", encoding="utf-8"
    )

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.READY
    assert "no full benchmark campaign run" in report["claim_boundary"]
    assert report["target_claim_metadata"]["status"] == "to_be_confirmed_by_maintainer"
    assert report["target_claim_metadata"]["expected_status"] == "to_be_confirmed_by_maintainer"
    assert (
        "No S20/S30 paper-facing claim exists."
        in report["target_claim_metadata"]["no_claim_statement"]
    )
    assert report["seed_tier"]["primary_seed_count"] == 20
    assert report["seed_tier"]["escalation_seed_count"] == 30
    assert report["seed_tier"]["mode"] == "seed-set"
    assert report["seed_tier"]["claim_gate_seed_tier"] == "s20_then_s30_if_rankflip"
    assert report["seed_tier"]["escalate_to_s30_when"] == "rank flip observed"
    assert report["metric_coverage"] == {
        "success": True,
        "collisions": True,
        "near_misses": True,
        "time_to_goal_norm": True,
    }
    assert report["missing_artifact_diagnostics"] == []
    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 0


def test_missing_result_store_fails_closed_with_artifact_diagnostics(tmp_path: Path) -> None:
    """Absent result-store artifacts are blocked, not silently paper-ready."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    assert any(
        "missing result-store files" in item for item in report["missing_artifact_diagnostics"]
    )
    assert report["expected_result_store_files"]["episodes.parquet"] is False
    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 1


def test_malformed_archive_packet_returns_malformed_exit(tmp_path: Path) -> None:
    """Malformed launch-packet metadata has a distinct fail-closed exit code."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = repo / "packet.yaml"
    packet.write_text(
        yaml.safe_dump(
            {
                "schema_version": "s20-s30-seed-budget-launch-packet.v1",
                "campaign_id": "issue_1554_test",
                "no_benchmark_result_claim": False,
            }
        ),
        encoding="utf-8",
    )

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_missing_no_claim_statement_is_malformed(tmp_path: Path) -> None:
    """Paper-facing archive packets must explicitly preserve no-claim wording."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    payload = yaml.safe_load(packet.read_text(encoding="utf-8"))
    payload.pop("no_claim_statement")
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_claim_and_seed_tier_metadata_drift_blocks_readiness(tmp_path: Path) -> None:
    """Claim-gate and seed-tier drift is a blocked archive state, not paper-ready."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(
        repo / "packet.yaml",
        repo,
        metadata_overrides={
            "claim_map_gate.status": "established",
            "seed_policy.mode": "inline",
            "claim_map_gate.seed_tier": "s30_required",
            "seed_policy.escalate_to_s30_when": "maintainer asks",
        },
    )
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
        analysis={"seed_resampling_rank_flip": {"rank_flip_observed": False}},
    )

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    assert any(
        "claim_map_gate.status must remain" in item
        for item in report["missing_artifact_diagnostics"]
    )
    assert any(
        "seed_policy.mode must remain" in item for item in report["missing_artifact_diagnostics"]
    )
    assert any(
        "claim_map_gate.seed_tier must remain" in item
        for item in report["missing_artifact_diagnostics"]
    )
    assert any(
        "escalate_to_s30_when must describe rank-flip" in item
        for item in report["missing_artifact_diagnostics"]
    )


def test_fail_closed_rows_and_missing_metrics_block_readiness(tmp_path: Path) -> None:
    """Fallback rows or missing primary metrics keep the bundle blocked."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    rows = _complete_rows()
    rows[0]["row_status"] = "fallback"
    for row in rows:
        row.pop("time_to_goal_norm", None)
    write_result_store(
        repo / "store",
        rows,
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
    )

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    assert any(
        "fail-closed/non-promotable" in item for item in report["missing_artifact_diagnostics"]
    )
    assert (
        "missing required metric columns: ['time_to_goal_norm']"
        in report["missing_artifact_diagnostics"]
    )


def test_archive_reader_uses_campaign_result_store_parquet_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness uses shared result-store parquet reader instead of pandas directly."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
    )
    (repo / "output/issue_1554_s20_s30/bundle.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (repo / "output/issue_1554_s20_s30/bundle.json").write_text(
        "{}", encoding="utf-8"
    )
    (repo / "output/issue_1554_s20_s30/BUNDLE.md").write_text(
        "# S20/S30 Bundle", encoding="utf-8"
    )
    calls: list[Path] = []
    original = readiness.read_parquet_frame

    def _recording_reader(path: Path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(readiness, "read_parquet_frame", _recording_reader)

    assert readiness.build_report(packet, repo)["status"] == readiness.READY
    assert calls == [repo / "store" / "episodes.parquet"]


def test_non_finite_seed_cells_are_reported_and_block_readiness(tmp_path: Path) -> None:
    """Invalid seed cells are counted explicitly instead of silently disappearing."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
    )
    episodes_path = repo / "store" / "episodes.parquet"
    episodes = readiness.read_parquet_frame(episodes_path)
    episodes["seed"] = episodes["seed"].astype(float)
    episodes.loc[0, "seed"] = float("inf")
    episodes.to_parquet(episodes_path, index=False)

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    goal_coverage = report["planner_seed_coverage"]["goal"]
    assert goal_coverage["skipped_non_finite_seed_count"] == 1
    assert 111 in goal_coverage["missing_primary_seeds"]
    assert any(
        "skipped non-finite/unparseable seed values: 1" in item
        for item in report["missing_artifact_diagnostics"]
    )


def test_json_cli_output_is_parseable(tmp_path: Path, capsys) -> None:
    """The JSON CLI surface emits machine-readable readiness reports."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == readiness.SCHEMA_VERSION
    assert payload["status"] == readiness.BLOCKED


def test_packet_paths_must_stay_inside_repo_root(tmp_path: Path) -> None:
    """Launch-packet paths cannot traverse outside the repository root."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo, seed_sets_path="../seed_sets.yaml")

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_result_store_member_paths_must_be_relative(tmp_path: Path) -> None:
    """Expected result-store members cannot escape the result-store directory."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    payload = yaml.safe_load(packet.read_text(encoding="utf-8"))
    payload["expected_artifacts"]["result_store_required_files"][0] = "../episodes.parquet"
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_execution_boundary_requires_actual_booleans(tmp_path: Path) -> None:
    """String booleans in launch-packet execution boundaries are malformed."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo, full_campaign_in_this_issue="false")

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_full_campaign_boundary_blocks_complete_archive_readiness(tmp_path: Path) -> None:
    """Archive readiness cannot pass when packet moves campaign execution into this issue."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo, full_campaign_in_this_issue=True)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
    )

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    assert (
        "execution_boundary.full_campaign_in_this_issue must be false for archive-readiness"
        in report["missing_artifact_diagnostics"]
    )


def test_slurm_submission_boundary_blocks_complete_archive_readiness(tmp_path: Path) -> None:
    """Archive readiness cannot pass when packet authorizes Slurm submission in this issue."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo, submit_slurm_from_this_issue=True)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
    )

    report = readiness.build_report(packet, repo)

    assert report["status"] == readiness.BLOCKED
    assert (
        "execution_boundary.submit_slurm_from_this_issue must be false for archive-readiness"
        in report["missing_artifact_diagnostics"]
    )


def test_seed_tiers_must_be_unique_integers(tmp_path: Path) -> None:
    """Duplicate or non-integer seed tiers are malformed instead of coerced."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    seeds = yaml.safe_load((repo / "seed_sets.yaml").read_text(encoding="utf-8"))
    seeds["paper_eval_s20"][0] = seeds["paper_eval_s20"][1]
    (repo / "seed_sets.yaml").write_text(yaml.safe_dump(seeds, sort_keys=False), encoding="utf-8")

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2


def test_missing_bundle_outputs_are_reported_as_blockers(tmp_path: Path) -> None:
    """Missing declared bundle files blocks readiness as missing-artifact diagnostics."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(repo / "packet.yaml", repo)
    write_result_store(
        repo / "store",
        _complete_rows(),
        study_id="issue_1554_test",
        command="uv run python scripts/tools/run_camera_ready_benchmark.py ...",
        analysis={"seed_resampling_rank_flip": {"rank_flip_observed": False}},
    )

    report = readiness.build_report(packet, repo)
    assert report["status"] == readiness.BLOCKED
    assert report["missing_bundle_outputs"] == [
        str((repo / "output/issue_1554_s20_s30/bundle.json").resolve()),
        str((repo / "output/issue_1554_s20_s30/BUNDLE.md").resolve()),
    ]
    assert any("missing bundle outputs" in item for item in report["missing_artifact_diagnostics"])


def test_duplicate_planner_rows_in_contract_is_malformed(tmp_path: Path) -> None:
    """Duplicate planner rows in claim metadata is malformed and fail-closed."""

    repo = tmp_path / "repo"
    repo.mkdir()
    packet = _write_packet(
        repo / "packet.yaml",
        repo,
        metadata_overrides={
            "claim_map_gate.planner_rows_to_confirm": ["goal", "goal"],
        },
    )

    assert readiness.main(["--packet", str(packet), "--repo-root", str(repo)]) == 2
