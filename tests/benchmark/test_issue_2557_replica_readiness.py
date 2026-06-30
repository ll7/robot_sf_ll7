"""Tests for issue #2557 fixed-seed replica artifact-readiness preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from robot_sf.benchmark.issue_2557_replica_readiness import (
    assess_issue_2557_replica_readiness,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "benchmark" / "check_issue_2557_replica_readiness.py"
TRACKED_SEED_SUMMARY = (
    REPO_ROOT
    / "docs/context/evidence/issue_2557_reward_curriculum_partial_2026-06-08/seed_summary.json"
)


def _row(seed: int) -> dict:
    return {
        "seed": seed,
        "job_id": 12000 + seed,
        "partition": "a30" if seed % 2 else "l40s",
        "run_id": f"issue2557_seed{seed}",
        "eval_step": 10_000_000,
        "success_rate": 0.9,
        "collision_rate": 0.1,
        "snqi": 0.2,
        "wandb_url": f"https://wandb.ai/ll7/robot_sf/runs/seed{seed}",
        "run_summary_sha256": "a" * 64,
        "eval_timeline_sha256": "b" * 64,
        "source_config_sha256": "c" * 64,
    }


def _write_manifest(
    path: Path, *, seeds: tuple[int, ...] = (501, 502), partial: bool = False
) -> Path:
    payload = {
        "schema_version": "issue_2557_reward_curriculum_partial.v2",
        "generated_utc": "2026-06-10T13:17:36Z",
        "source_commit": "fixture",
        "branch": "fixture",
        "source_worktrees": [
            {
                "branch": "issue-2557-fixture",
                "commit": "1" * 40,
                "seeds": list(seeds),
            }
        ],
        "claim_boundary": "diagnostic_training_evidence",
        "incomplete_or_pending_seeds": [999] if partial else [],
        "aggregate": {"count": len(seeds)},
        "rows": [_row(seed) for seed in seeds],
    }
    if partial:
        payload["claim_boundary"] = "expanded_partial_training_evidence_not_full_seed_batch"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_configs(root: Path, seeds: tuple[int, ...]) -> str:
    template = "configs/seed{seed}.yaml"
    for seed in seeds:
        config = root / template.format(seed=seed)
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"seed: {seed}\n", encoding="utf-8")
    return template


def test_ready_when_required_seed_rows_and_provenance_present(tmp_path: Path) -> None:
    """A complete compact manifest with expected configs is ready."""

    manifest = _write_manifest(tmp_path / "seed_summary.json")
    template = _write_configs(tmp_path, (501, 502))

    result = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
    )

    assert result.ready is True
    assert result.blockers == ()
    assert result.present_seeds == (501, 502)


def test_missing_replica_row_blocks_readiness(tmp_path: Path) -> None:
    """Expected seed coverage fails closed when a replica row is absent."""

    manifest = _write_manifest(tmp_path / "seed_summary.json", seeds=(501,))
    template = _write_configs(tmp_path, (501, 502))

    result = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
    )

    assert result.ready is False
    assert "seed 502: missing replica row" in result.blockers


def test_missing_required_provenance_blocks_readiness(tmp_path: Path) -> None:
    """Each replica row must carry downstream provenance pointers."""

    manifest = _write_manifest(tmp_path / "seed_summary.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["rows"][0]["wandb_url"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    template = _write_configs(tmp_path, (501, 502))

    result = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
    )

    assert result.ready is False
    assert "rows[0] missing wandb_url" in result.blockers


def test_malformed_replica_manifest_reports_field_blockers(tmp_path: Path) -> None:
    """Malformed rows and source-worktree metadata produce explicit blockers."""

    manifest = _write_manifest(tmp_path / "seed_summary.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "wrong.v1"
    payload["claim_boundary"] = ""
    payload["source_worktrees"] = [{"branch": "fixture", "commit": "short", "seeds": []}]
    payload["aggregate"] = {"count": 99}
    payload["rows"] = [
        "not-a-row",
        {
            **_row(501),
            "seed": "501",
            "job_id": True,
            "eval_step": "10000000",
            "success_rate": "0.9",
            "collision_rate": None,
            "snqi": float("nan"),
            "partition": "cpu",
            "wandb_url": "not-a-wandb-url",
            "run_summary_sha256": "bad",
        },
        _row(502),
        _row(502),
    ]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    template = _write_configs(tmp_path, (501, 502))

    result = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
    )

    assert result.ready is False
    assert "schema_version must be an issue_2557_reward_curriculum manifest" in result.blockers
    assert "claim_boundary must describe the evidence boundary" in result.blockers
    assert "source_worktrees[0].commit must be a 40-char git sha" in result.blockers
    assert "source_worktrees[0].seeds must be non-empty integer list" in result.blockers
    assert "rows[0] must be an object" in result.blockers
    assert "rows[1].seed must be an integer" in result.blockers
    assert "seed 501: duplicate manifest row" not in result.blockers
    assert "seed 502: duplicate manifest row" in result.blockers
    assert "aggregate.count must match number of rows" in result.blockers


def test_partial_manifest_fails_closed_unless_diagnostic_allowed(tmp_path: Path) -> None:
    """Partial bundles are blocked unless explicitly inspected as diagnostics."""

    manifest = _write_manifest(tmp_path / "seed_summary.json", partial=True)
    _write_configs(tmp_path, (501, 502))
    template = str(tmp_path / "configs/seed{seed}.yaml")

    blocked = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
    )
    diagnostic = assess_issue_2557_replica_readiness(
        manifest,
        repo_root=tmp_path,
        expected_seeds=(501, 502),
        config_template=template,
        allow_partial=True,
    )

    assert blocked.ready is False
    assert any("partial" in blocker for blocker in blocked.blockers)
    assert diagnostic.ready is True
    assert diagnostic.warnings


def test_tracked_issue_2557_summary_remains_diagnostic_only() -> None:
    """The public #2557 summary stays blocked unless partial diagnostics are explicit."""

    blocked = assess_issue_2557_replica_readiness(
        TRACKED_SEED_SUMMARY,
        repo_root=REPO_ROOT,
        expected_seeds=(501, 502, 503, 504, 505, 506, 507, 508),
    )
    diagnostic = assess_issue_2557_replica_readiness(
        TRACKED_SEED_SUMMARY,
        repo_root=REPO_ROOT,
        expected_seeds=(501, 502, 503, 504, 505, 506, 507, 508),
        allow_partial=True,
    )

    assert blocked.ready is False
    assert "manifest still lists incomplete_or_pending_seeds" in blocked.blockers
    assert diagnostic.ready is True
    assert diagnostic.metadata["row_count"] == 14
    assert diagnostic.warnings == (
        "manifest lists incomplete_or_pending_seeds; diagnostic-only readiness",
    )


def test_main_emits_text_summary_for_ready_manifest(tmp_path: Path, capsys) -> None:
    """The in-process CLI path emits the human-readable ready summary."""

    manifest = _write_manifest(tmp_path / "seed_summary.json")
    _write_configs(tmp_path, (501, 502))
    template = str(tmp_path / "configs/seed{seed}.yaml")

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--expected-seeds",
            "501,502",
            "--config-template",
            template,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "issue #2557 replica artifact readiness: ready" in captured.out


def test_main_rejects_reversed_seed_range(tmp_path: Path) -> None:
    """Seed range parsing rejects reversed ranges before readiness assessment."""

    manifest = _write_manifest(tmp_path / "seed_summary.json")

    try:
        main(["--manifest", str(manifest), "--expected-seeds", "502-501"])
    except ValueError as exc:
        assert "seed range start exceeds end" in str(exc)
    else:  # pragma: no cover - defensive assertion message
        raise AssertionError("main should reject reversed seed range")


def test_cli_json_reports_missing_seed(tmp_path: Path) -> None:
    """The CLI emits machine-readable missing-seed blockers."""

    manifest = _write_manifest(tmp_path / "seed_summary.json", seeds=(501,))
    template = _write_configs(tmp_path, (501, 502))

    completed = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--manifest",
            str(manifest),
            "--expected-seeds",
            "501-502",
            "--config-template",
            template,
            "--json",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1, completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["ready"] is False
    assert payload["missing_seeds"] == [502]
